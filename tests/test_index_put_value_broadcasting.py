"""Unit tests for generic `values` broadcasting in `index_put_wrapper` (CPU-only).

Covers PyTorch unidirectional broadcasting rules against `(*idx.shape, *x.shape[1:])`:
  - `(tail,) -> (N, tail)` (e.g. Z-Image-Turbo_v2 `(3840,) -> (60, 3840)`)
  - `(1, tail) -> (N, tail)`
  - `(N, 1) -> (N, tail)`
  - Multi-dim tail `(H, W) -> (N, H, W)`, `(W,) -> (N, H, W)`, `(N, 1, W) -> (N, H, W)`
  - 2D integer index `idx` of shape `(A, B)` with `(H, W) -> (A, B, H, W)`
  - Scalar fast path (`VAL_SCALAR=True`)
  - Exact shape match without unnecessary expansion
  - Boolean mask path lowering to broadcasted `values`
  - Incompatible shapes raising `RuntimeError`
"""
import ctypes

import numpy as np
import pytest

from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.ops import index_wrappers as iw
from nbxtensor.tensor import NBXTensor

_DTYPE_MAP = {
    np.dtype("float32"): NBXDtype.float32,
    np.dtype("int64"): NBXDtype.int64,
    np.dtype("uint8"): NBXDtype.uint8,
    np.dtype("bool"): NBXDtype.bool_,
}


def _cpu_tensor(arr: np.ndarray) -> NBXTensor:
    arr_c = np.ascontiguousarray(arr)
    t = NBXTensor.empty_cpu(arr_c.shape, _DTYPE_MAP[arr_c.dtype])
    if arr_c.nbytes:
        ctypes.memmove(t.data_ptr(), arr_c.ctypes.data, arr_c.nbytes)
    return t


def _make_fake_kernel(captured):
    """CPU stand-in for `index_put_kernel` that records launch arguments and
    executes the scatter on CPU NBXTensors so tests verify both shape/flags
    and bit-exact numerical results without touching any GPU."""

    class _FakeKernel:
        def __getitem__(self, grid):
            def _launch(out, idx, vbuf, T, N, R, fault_ptr, fault_code,
                        *, VAL_SCALAR, ACCUMULATE, **meta):
                captured.append({
                    "out_shape": tuple(out.shape),
                    "idx_shape": tuple(idx.shape),
                    "vbuf_shape": tuple(vbuf.shape),
                    "vbuf_contiguous": vbuf.is_contiguous(),
                    "vbuf_numel": vbuf.numel(),
                    "T": T,
                    "N": N,
                    "R": R,
                    "VAL_SCALAR": VAL_SCALAR,
                    "ACCUMULATE": ACCUMULATE,
                    "vbuf_np": vbuf.numpy().copy(),
                })
                out_np = out.numpy().reshape(R, T)
                idx_np = idx.numpy().reshape(-1)
                val_np = vbuf.numpy().reshape(-1)
                for s, row in enumerate(idx_np):
                    r = int(row)
                    if r < 0:
                        r += R
                    if VAL_SCALAR:
                        slice_val = val_np[0]
                    else:
                        slice_val = val_np[s * T:(s + 1) * T]
                    if ACCUMULATE:
                        out_np[r] += slice_val
                    else:
                        out_np[r] = slice_val
                ctypes.memmove(out.data_ptr(), out_np.ctypes.data, out._nbytes)

            return _launch

    return _FakeKernel()


@pytest.fixture
def cpu_index_put(monkeypatch):
    captured = []
    _orig_clone = NBXTensor.clone

    def _cpu_clone(self):
        if self._device == "cpu":
            src_c = self.contiguous()
            dst = NBXTensor.empty_cpu(src_c._shape, src_c._dtype, pinned=src_c._pinned)
            if src_c._nbytes:
                ctypes.memmove(dst.data_ptr(), src_c.data_ptr(), src_c._nbytes)
            return dst
        return _orig_clone(self)

    monkeypatch.setattr(NBXTensor, "clone", _cpu_clone)
    monkeypatch.setattr(iw, "index_put_kernel", _make_fake_kernel(captured))
    monkeypatch.setattr(iw, "set_device", lambda *a, **kw: None)
    monkeypatch.setattr(iw, "fault_channel", lambda code, tensor: (0, 0))
    monkeypatch.setattr(iw, "_ew_meta", lambda n: {"BLOCK_SIZE": 256})
    return captured


@pytest.mark.cardless
@pytest.mark.parametrize(
    "x_shape, idx_arr, val_shape, accumulate",
    [
        # (tail,) -> (N, tail) — exact Z-Image-Turbo_v2 pattern
        ((10, 8), np.array([1, 4, 7], dtype=np.int64), (8,), False),
        ((10, 8), np.array([1, 4, 7], dtype=np.int64), (8,), True),
        # (1, tail) -> (N, tail)
        ((10, 8), np.array([0, 3, 5, 9], dtype=np.int64), (1, 8), False),
        # (N, 1) -> (N, tail)
        ((10, 8), np.array([2, 6, 8], dtype=np.int64), (3, 1), False),
        # Multi-dim tail: (H, W) -> (N, H, W)
        ((6, 3, 4), np.array([0, 2, 5], dtype=np.int64), (3, 4), False),
        # Multi-dim tail: (W,) -> (N, H, W)
        ((6, 3, 4), np.array([1, 4], dtype=np.int64), (4,), False),
        # Multi-dim tail: (N, 1, W) -> (N, H, W)
        ((6, 3, 4), np.array([1, 3, 5], dtype=np.int64), (3, 1, 4), False),
        # Multi-dim tail: (1, H, 1) -> (N, H, W)
        ((6, 3, 4), np.array([0, 4], dtype=np.int64), (1, 3, 1), False),
        # 2D integer index: (A, B) with (H, W) -> (A, B, H, W)
        ((8, 3, 4), np.array([[0, 1, 2], [5, 6, 7]], dtype=np.int64), (3, 4), False),
        # 2D integer index: (B, 1, W) -> (A, B, H, W)
        ((8, 3, 4), np.array([[0, 1, 2], [5, 6, 7]], dtype=np.int64), (3, 1, 4), False),
        # Exact match (no expand needed)
        ((6, 3, 4), np.array([0, 2, 5], dtype=np.int64), (3, 3, 4), False),
    ],
)
def test_index_put_broadcasts_values_to_target_shape(
    cpu_index_put, x_shape, idx_arr, val_shape, accumulate
):
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal(x_shape).astype(np.float32)
    val_np = rng.standard_normal(val_shape).astype(np.float32)

    x_t = _cpu_tensor(x_np)
    idx_t = _cpu_tensor(idx_arr)
    val_t = _cpu_tensor(val_np)

    out_t = iw.index_put_wrapper(x_t, [idx_t], val_t, accumulate=accumulate)
    assert len(cpu_index_put) == 1
    call = cpu_index_put[0]

    target_shape = idx_arr.shape + x_shape[1:]
    expected_vbuf = np.broadcast_to(val_np, target_shape)

    assert call["VAL_SCALAR"] is False
    assert call["vbuf_contiguous"] is True
    assert call["vbuf_shape"] == target_shape
    assert np.array_equal(call["vbuf_np"], expected_vbuf)

    ref = x_np.copy()
    if accumulate:
        np.add.at(ref, idx_arr, expected_vbuf)
    else:
        ref[idx_arr] = expected_vbuf
    assert np.allclose(out_t.numpy(), ref, atol=0, rtol=0)


@pytest.mark.cardless
def test_index_put_scalar_fast_path(cpu_index_put):
    x_np = np.zeros((5, 4), dtype=np.float32)
    idx_np = np.array([1, 3], dtype=np.int64)
    val_np = np.array([7.5], dtype=np.float32)

    out_t = iw.index_put_wrapper(
        _cpu_tensor(x_np),
        [_cpu_tensor(idx_np)],
        _cpu_tensor(val_np),
    )
    assert len(cpu_index_put) == 1
    assert cpu_index_put[0]["VAL_SCALAR"] is True
    assert cpu_index_put[0]["vbuf_numel"] == 1

    ref = x_np.copy()
    ref[idx_np] = 7.5
    assert np.array_equal(out_t.numpy(), ref)


@pytest.mark.cardless
def test_index_put_bool_mask_broadcasts_row_vector(cpu_index_put, monkeypatch):
    """Exact Z-Image-Turbo_v2 boolean-mask path: `x[mask] = row_vec`."""
    x_np = np.zeros((8, 6), dtype=np.float32)
    mask_np = np.array([False, True, False, True, False, False, True, False], dtype=np.uint8)
    val_np = np.arange(1, 7, dtype=np.float32)

    # Stub nonzero_1d_wrapper on CPU to return the True row indices.
    def _fake_nonzero_1d(m, expected=None):
        assert expected is None
        rows = np.nonzero(m.numpy())[0].astype(np.int64)
        return _cpu_tensor(rows)

    monkeypatch.setattr(iw, "nonzero_1d_wrapper", _fake_nonzero_1d)

    out_t = iw.index_put_wrapper(
        _cpu_tensor(x_np),
        [_cpu_tensor(mask_np)],
        _cpu_tensor(val_np),
    )
    assert len(cpu_index_put) == 1
    assert cpu_index_put[0]["vbuf_shape"] == (3, 6)
    assert cpu_index_put[0]["VAL_SCALAR"] is False

    ref = x_np.copy()
    ref[mask_np.astype(bool)] = val_np
    assert np.array_equal(out_t.numpy(), ref)


@pytest.mark.cardless
@pytest.mark.parametrize(
    "x_shape, idx_shape, val_shape",
    [
        # Wrong leading dim (2 != 3)
        ((10, 8), (3,), (2, 8)),
        # Wrong tail dim (7 != 8)
        ((10, 8), (3,), (7,)),
        # Rank higher than target_shape
        ((10, 8), (3,), (1, 3, 8)),
        # Same numel (12) but non-broadcastable shape (4, 3) vs (3, 4)
        ((10, 4), (3,), (4, 3)),
        # Multi-dim tail mismatch
        ((6, 3, 4), (2,), (3, 5)),
    ],
)
def test_index_put_incompatible_values_shape_raises(
    cpu_index_put, x_shape, idx_shape, val_shape
):
    x_t = _cpu_tensor(np.zeros(x_shape, dtype=np.float32))
    idx_t = _cpu_tensor(np.zeros(idx_shape, dtype=np.int64))
    val_t = _cpu_tensor(np.ones(val_shape, dtype=np.float32))

    with pytest.raises(RuntimeError, match="cannot be broadcast"):
        iw.index_put_wrapper(x_t, [idx_t], val_t)
    assert len(cpu_index_put) == 0
