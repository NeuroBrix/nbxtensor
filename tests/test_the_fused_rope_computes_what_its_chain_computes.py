"""A fusion is an optimisation, never a change of meaning (project lead 23/11/2026): the fused rotary embedding
(`elementwise_wrappers.rope_fused_wrapper` -> `rope.rope_forward_kernel`) equals, bit for bit, the traced chain it
replaces — mul(x, cos) + mul(rotate_half(x), sin), each op rounded to the stored dtype — on a prefill (81 rows) and a
one-row launch, with a Q of the traced dtype (bf16) and a Q wider than it (fp32: a one-row projection's store).

Red on the base: the kernel's plain expression rounded once where the chain rounds three times (bf16: one ulp off) and
was contracted into an fma (fp32: 2.4e-7 off), and granite's fresh and cached prefills on the compiled Triton sequence
parted (28 / 52 / 52 tokens against the referee's 24 / 24 / 24, the Mac 23/11)."""
import numpy as np
import pytest

from nbxtensor.dtype.nbx_dtype import NBXDtype as D


def _accelerator():
    from nbxtensor.memory.allocator import DeviceAllocator
    if DeviceAllocator.device_count() == 0:
        pytest.skip("no accelerator on this machine")


def _chain(x, cos, sin):
    from nbxtensor.ops import elementwise_wrappers as EW
    from nbxtensor.tensor import NBXTensor
    h = x.shape[-1] // 2
    rot = NBXTensor.cat([EW.neg(x.narrow(3, h, h)), x.narrow(3, 0, h)], 3)
    return EW.add(EW.mul(x, cos), EW.mul(rot, sin))


@pytest.mark.parametrize("rows,qdt", [(81, D.bfloat16), (1, D.float32), (1, D.bfloat16), (81, D.float32)])
def test_the_fused_rope_is_its_chain_bit_for_bit(rows, qdt):
    _accelerator()
    from nbxtensor.ops import elementwise_wrappers as EW
    from nbxtensor.tensor import NBXTensor
    rng = np.random.default_rng(rows)
    q = NBXTensor.from_numpy(rng.standard_normal((1, 16, rows, 64)).astype(np.float32)).to(qdt)
    k = NBXTensor.from_numpy(rng.standard_normal((1, 8, rows, 64)).astype(np.float32)).to(qdt)
    ang = np.arange(81 - rows, 81)[:, None] / (10000 ** (np.arange(0, 64, 2) / 64))[None, :]
    ang = np.concatenate([ang, ang], -1)[None, None]
    cos = NBXTensor.from_numpy(np.cos(ang).astype(np.float32)).to(D.bfloat16)
    sin = NBXTensor.from_numpy(np.sin(ang).astype(np.float32)).to(D.bfloat16)
    want_q, want_k = _chain(q, cos, sin), _chain(k, cos, sin)
    got_q, got_k = EW.rope_fused_wrapper(q, k, cos, sin)
    for got, want in ((got_q, want_q), (got_k, want_k)):
        assert got.nbx_dtype == want.nbx_dtype
        a, b = got.to(D.float32).numpy(), want.to(D.float32).numpy()
        assert np.array_equal(a, b), float(np.abs(a - b).max())
