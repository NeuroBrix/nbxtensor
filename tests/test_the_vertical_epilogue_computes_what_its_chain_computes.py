"""A fusion is an optimisation, never a change of meaning (project lead 23/11/2026): a matmul with its activation fused
as an epilogue (`linalg_wrappers.mm(a, b, _epilogue)` / `addmm(..., _epilogue)`, the fusion_vertical pass) equals, bit
for bit, the chain it replaces — the matmul's stored result, then silu / gelu / gelu-tanh rounded at their own store —
on many rows (the fused kernel) and one row (the decode path), in bf16 and fp32."""
import numpy as np
import pytest

from nbxtensor.dtype.nbx_dtype import NBXDtype as D


def _accelerator():
    from nbxtensor.memory.allocator import DeviceAllocator
    if DeviceAllocator.device_count() == 0:
        pytest.skip("no accelerator on this machine")


@pytest.mark.parametrize("code", [1, 2, 3])
@pytest.mark.parametrize("rows,dt", [(81, D.bfloat16), (1, D.float32), (1, D.bfloat16), (81, D.float32)])
def test_the_matmul_epilogue_is_its_chain_bit_for_bit(rows, dt, code):
    _accelerator()
    from nbxtensor.ops import elementwise_wrappers as EW
    from nbxtensor.ops import linalg_wrappers as LW
    from nbxtensor.tensor import NBXTensor
    rng = np.random.default_rng(rows * 10 + code)
    a = NBXTensor.from_numpy(rng.standard_normal((rows, 256)).astype(np.float32)).to(dt)
    b = NBXTensor.from_numpy((rng.standard_normal((256, 384)) * 0.1).astype(np.float32)).to(dt)
    c = LW.mm(a, b)
    want = EW.silu(c) if code == 1 else EW.gelu(c, approximate="tanh" if code == 3 else "none")
    got = LW.mm(a, b, code)
    assert got.nbx_dtype == want.nbx_dtype
    x, y = got.to(D.float32).numpy().reshape(-1), want.to(D.float32).numpy().reshape(-1)
    assert np.array_equal(x, y), (int((x != y).sum()), x.size, float(np.abs(x - y).max()))


@pytest.mark.parametrize("code", [1, 2, 3])
@pytest.mark.parametrize("rows,dt", [(81, D.bfloat16), (1, D.float32), (81, D.float32)])
def test_the_addmm_epilogue_is_its_chain_bit_for_bit(rows, dt, code):
    _accelerator()
    from nbxtensor.ops import elementwise_wrappers as EW
    from nbxtensor.ops import linalg_wrappers as LW
    from nbxtensor.tensor import NBXTensor
    rng = np.random.default_rng(rows * 7 + code)
    a = NBXTensor.from_numpy(rng.standard_normal((rows, 256)).astype(np.float32)).to(dt)
    b = NBXTensor.from_numpy((rng.standard_normal((256, 384)) * 0.1).astype(np.float32)).to(dt)
    bias = NBXTensor.from_numpy(rng.standard_normal((384,)).astype(np.float32)).to(dt)
    c = LW.addmm(bias, a, b)
    want = EW.silu(c) if code == 1 else EW.gelu(c, approximate="tanh" if code == 3 else "none")
    got = LW.addmm(bias, a, b, _epilogue=code)
    x, y = got.to(D.float32).numpy().reshape(-1), want.to(D.float32).numpy().reshape(-1)
    assert got.nbx_dtype == want.nbx_dtype and np.array_equal(x, y), (int((x != y).sum()), float(np.abs(x - y).max()))
