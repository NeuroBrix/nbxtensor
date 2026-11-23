"""The dead-causal-mask elimination (the engine's sequence pass `_eliminate_dead_causal_mask_ops`) rewrites
SDPA(q, k, v, attn_mask = where(logical_not(tril(ones(N, N))), -inf, 0)) into SDPA(q, k, v, is_causal=True): the
library's causal attention must compute what the masked one computes, bit for bit (project lead 23/11/2026: an
elimination is checked like a fusion). Square prefill (the only shape the [N, N] chain forms), bf16 and fp32."""
import numpy as np
import pytest

from nbxtensor.dtype.nbx_dtype import NBXDtype as D


def _accelerator():
    from nbxtensor.memory.allocator import DeviceAllocator
    if DeviceAllocator.device_count() == 0:
        pytest.skip("no accelerator on this machine")


@pytest.mark.parametrize("n,dt", [(81, D.bfloat16), (81, D.float32), (7, D.bfloat16), (1, D.float32)])
def test_causal_attention_is_the_masked_chain_bit_for_bit(n, dt):
    _accelerator()
    from nbxtensor.ops.attention_wrappers import scaled_dot_product_attention_wrapper as sdpa
    from nbxtensor.tensor import NBXTensor
    rng = np.random.default_rng(n)
    q, k, v = (NBXTensor.from_numpy(rng.standard_normal((1, 16, n, 64)).astype(np.float32)).to(dt) for _ in range(3))
    mask = np.where(np.tril(np.ones((n, n), bool)), 0.0, -np.inf).astype(np.float32)
    masked = sdpa(q, k, v, attn_mask=NBXTensor.from_numpy(mask).to(dt))
    causal = sdpa(q, k, v, is_causal=True)
    a, b = causal.to(D.float32).numpy().reshape(-1), masked.to(D.float32).numpy().reshape(-1)
    assert np.array_equal(a, b), (int((a != b).sum()), a.size, float(np.abs(a - b).max()))
