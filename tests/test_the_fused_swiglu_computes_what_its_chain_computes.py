"""A fusion is an optimisation, never a change of meaning (project lead 23/11/2026): the fused SwiGLU
(`elementwise_wrappers.swiglu_fused_wrapper` -> `fused_moe.silu_mul_split_kernel`) equals, bit for bit, the traced
chain it replaces — silu(gate) then mul(·, up), each op rounded to the stored dtype — on many rows and one row, in the
traced dtype (bf16) and a wider one (fp32: a one-row projection's store)."""
import numpy as np
import pytest

from nbxtensor.dtype.nbx_dtype import NBXDtype as D


def _accelerator():
    from nbxtensor.memory.allocator import DeviceAllocator
    if DeviceAllocator.device_count() == 0:
        pytest.skip("no accelerator on this machine")


@pytest.mark.parametrize("rows,dt", [(81, D.bfloat16), (1, D.float32), (1, D.bfloat16), (81, D.float32)])
def test_the_fused_swiglu_is_its_chain_bit_for_bit(rows, dt):
    _accelerator()
    from nbxtensor.ops import elementwise_wrappers as EW
    from nbxtensor.tensor import NBXTensor
    rng = np.random.default_rng(rows)
    gate = NBXTensor.from_numpy((rng.standard_normal((1, rows, 512)) * 3).astype(np.float32)).to(dt)
    up = NBXTensor.from_numpy(rng.standard_normal((1, rows, 512)).astype(np.float32)).to(dt)
    want = EW.mul(EW.silu(gate), up)
    got = EW.swiglu_fused_wrapper(gate, up)
    assert got.nbx_dtype == want.nbx_dtype
    a, b = got.to(D.float32).numpy().reshape(-1), want.to(D.float32).numpy().reshape(-1)
    assert np.array_equal(a, b), (int((a != b).sum()), float(np.abs(a - b).max()))


@pytest.mark.parametrize("rows,dt", [(81, D.bfloat16), (1, D.float32), (1, D.bfloat16), (81, D.float32)])
def test_the_concatenated_swiglu_is_its_chain_bit_for_bit(rows, dt):
    """`invoke_silu_and_mul` (fused_moe.silu_and_mul_kernel: gate and up as the two halves of one [M, 2N] tensor, the
    MoE experts' form) against silu(gate) * up, each rounded at its store."""
    _accelerator()
    from nbxtensor.ops import elementwise_wrappers as EW
    from nbxtensor.tensor import NBXTensor
    rng = np.random.default_rng(rows + 5)
    both = NBXTensor.from_numpy((rng.standard_normal((rows, 1024)) * 3).astype(np.float32)).to(dt)
    want = EW.mul(EW.silu(both.narrow(1, 0, 512).contiguous()), both.narrow(1, 512, 512).contiguous())
    got = EW.invoke_silu_and_mul(both, rows, 512)
    a, b = got.to(D.float32).numpy().reshape(-1), want.to(D.float32).numpy().reshape(-1)
    assert got.nbx_dtype == want.nbx_dtype and np.array_equal(a, b), (int((a != b).sum()), float(np.abs(a - b).max()))
