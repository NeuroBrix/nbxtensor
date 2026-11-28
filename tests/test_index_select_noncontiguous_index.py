"""CPU-only regression tests for non-contiguous `index` tensors in `index_select_wrapper`
and `_meta_index` (`aten::index`).

Covers the exact NextDiT / Z-Image-Turbo 3D RoPE pattern where `pos_ids` has shape
`(N, 3)` and `pos_ids.select(1, axis)` yields a 1D non-contiguous index tensor with
stride `(3,)`. Both `index_select_mid_kernel` (`dim != x.ndim - 1`) and
`index_select_kernel` (`dim == x.ndim - 1`) read `index` as a flat contiguous buffer
(`index + i`), so `index_select_wrapper` must materialize `index = index.contiguous()`.
"""
from __future__ import annotations

import ast
import ctypes
from pathlib import Path

import numpy as np
import pytest

from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.ops import dispatch as _dispatch
from nbxtensor.ops import index_select as _isel
from nbxtensor.ops import index_wrappers as _iw
from nbxtensor.tensor import NBXTensor

REPO = Path(__file__).resolve().parents[1]

_DTYPE_MAP = {
    np.dtype("float32"): NBXDtype.float32,
    np.dtype("int32"): NBXDtype.int32,
    np.dtype("int64"): NBXDtype.int64,
    np.dtype("complex64"): NBXDtype.complex64,
}


def _cpu_tensor(arr: np.ndarray) -> NBXTensor:
    arr_c = np.ascontiguousarray(arr)
    t = NBXTensor.empty_cpu(arr_c.shape, _DTYPE_MAP[arr_c.dtype])
    if arr_c.nbytes:
        ctypes.memmove(t.data_ptr(), arr_c.ctypes.data, arr_c.nbytes)
    return t


def _raw_flat_view(t: NBXTensor) -> np.ndarray:
    """Read `t.numel()` elements linearly from `t.data_ptr()` — exactly what the
    Triton kernels (`tl.load(index + i)`) do on GPU! If `t` is non-contiguous,
    this intentionally exposes the wrong strided elements."""
    np_dt = np.int32 if t._dtype == NBXDtype.int32 else (
        np.int64 if t._dtype == NBXDtype.int64 else np.float32
    )
    buf = (ctypes.c_char * (t.numel() * np.dtype(np_dt).itemsize)).from_address(t.data_ptr())
    return np.frombuffer(buf, dtype=np_dt).copy()


def _make_fake_mid_kernel(captured):
    class _FakeMidKernel:
        def __getitem__(self, grid):
            def _launch(inp, out, outer, N, inner, index, index_len,
                        fault_ptr, fault_code, **meta):
                captured.append({
                    "kernel": "mid",
                    "index_contiguous": index.is_contiguous(),
                    "index_strides": tuple(index.stride()),
                })
                # Read index linearly from data_ptr() just like tl.load(index + i)
                idx_flat = _raw_flat_view(index)
                inp_np = _raw_flat_view(inp).reshape(outer, N, inner)
                out_np = np.empty((outer, index_len, inner), dtype=inp_np.dtype)
                for o in range(outer):
                    for i in range(index_len):
                        ii = int(idx_flat[i])
                        if ii < 0:
                            ii += N
                        out_np[o, i, :] = inp_np[o, ii, :]
                ctypes.memmove(out.data_ptr(), out_np.ctypes.data, out._nbytes)
            return _launch
    return _FakeMidKernel()


def _make_fake_last_kernel(captured):
    class _FakeLastKernel:
        def __getitem__(self, grid):
            def _launch(inp, out, M, N, index, index_len,
                        fault_ptr, fault_code, BLOCK_M, BLOCK_N):
                captured.append({
                    "kernel": "last",
                    "index_contiguous": index.is_contiguous(),
                    "index_strides": tuple(index.stride()),
                })
                idx_flat = _raw_flat_view(index)
                inp_np = _raw_flat_view(inp).reshape(M, N)
                out_np = np.empty((M, index_len), dtype=inp_np.dtype)
                for m in range(M):
                    for i in range(index_len):
                        ii = int(idx_flat[i])
                        if ii < 0:
                            ii += N
                        out_np[m, i] = inp_np[m, ii]
                ctypes.memmove(out.data_ptr(), out_np.ctypes.data, out._nbytes)
            return _launch
    return _FakeLastKernel()


@pytest.fixture
def cpu_index_select(monkeypatch):
    captured = []
    _orig_empty = NBXTensor.empty
    def _cpu_empty(shape, dtype, device="cuda"):
        if "cpu" in str(device).lower():
            return NBXTensor.empty_cpu(shape, dtype)
        return _orig_empty(shape, dtype, device=device)
    monkeypatch.setattr(NBXTensor, "empty", staticmethod(_cpu_empty))
    monkeypatch.setattr(_isel, "index_select_mid_kernel", _make_fake_mid_kernel(captured))
    monkeypatch.setattr(_iw, "index_select_kernel", _make_fake_last_kernel(captured))
    monkeypatch.setattr(_iw, "set_device", lambda *a, **kw: None)
    monkeypatch.setattr(_iw, "fault_channel", lambda code, tensor: (0, 0))
    monkeypatch.setattr(_iw, "_ew_meta", lambda n: {"BLOCK_SIZE": 256, "num_warps": 4})
    monkeypatch.setattr(_iw, "_ew_grid", lambda n, m: (1,))
    return captured


@pytest.mark.cardless
def test_zero_torch_ast():
    p = REPO / "src" / "nbxtensor" / "ops" / "index_wrappers.py"
    tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "torch" and not alias.name.startswith("torch.")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert mod != "torch" and not mod.startswith("torch.")


@pytest.mark.cardless
@pytest.mark.parametrize("axis_col", [0, 1, 2])
@pytest.mark.parametrize("idx_dtype", [np.int32, np.int64])
def test_index_select_mid_noncontiguous_select_index_complex64(cpu_index_select, axis_col, idx_dtype):
    """Exact NextDiT 3D RoPE pattern: `freqs_cis[pos_ids[:, col]]` on complex64."""
    rng = np.random.default_rng(123 + axis_col)
    table_re = rng.standard_normal((32, 8)).astype(np.float32)
    table_im = rng.standard_normal((32, 8)).astype(np.float32)
    table_c64 = (table_re + 1j * table_im).astype(np.complex64)

    pos_ids_np = np.stack([
        rng.integers(0, 32, size=19, dtype=idx_dtype),
        rng.integers(0, 32, size=19, dtype=idx_dtype),
        rng.integers(0, 32, size=19, dtype=idx_dtype),
    ], axis=1)  # shape (19, 3), C-contiguous

    table_t = _cpu_tensor(table_c64)
    pos_t = _cpu_tensor(pos_ids_np)
    col_idx = pos_t.select(1, axis_col)  # shape (19,), stride (3,) — non-contiguous!
    assert not col_idx.is_contiguous()
    assert tuple(col_idx.stride()) == (3,)

    out_t = _dispatch._meta_index(table_t, [col_idx])
    assert len(cpu_index_select) == 1
    assert cpu_index_select[0]["kernel"] == "mid"
    assert cpu_index_select[0]["index_contiguous"] is True

    expected = table_c64[pos_ids_np[:, axis_col]]
    out_np = out_t.view_as_real().numpy()
    exp_ri = np.stack([expected.real, expected.imag], axis=-1)
    assert np.array_equal(out_np, exp_ri)


@pytest.mark.cardless
def test_index_select_last_dim_noncontiguous_index(cpu_index_select):
    """Last-axis `index_select_kernel` path with a non-contiguous strided index."""
    rng = np.random.default_rng(7)
    x_np = rng.standard_normal((4, 16)).astype(np.float32)
    idx_2d = rng.integers(0, 16, size=(11, 3), dtype=np.int64)

    x_t = _cpu_tensor(x_np)
    idx_col = _cpu_tensor(idx_2d).select(1, 2)  # stride (3,)
    assert not idx_col.is_contiguous()

    out_t = _iw.index_select_wrapper(x_t, 1, idx_col)
    assert len(cpu_index_select) == 1
    assert cpu_index_select[0]["kernel"] == "last"
    assert cpu_index_select[0]["index_contiguous"] is True
    assert np.array_equal(out_t.numpy(), x_np[:, idx_2d[:, 2]])
