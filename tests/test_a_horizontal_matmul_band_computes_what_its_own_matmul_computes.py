"""The horizontal fusion (engine pass fusion_horizontal, OPT-IN today: NBX_OPTIM_FUSION_HORIZONTAL=1) runs sibling
projections as ONE wider mm over their concatenated weights and hands each consumer its column band. A fusion computes
what its chain computes (project lead 23/11/2026): each band must equal, bit for bit, the member's own mm — many rows
and one, bf16 and fp32. A size-2 group of the q/k/v class (out widths 1024 + 512)."""
import numpy as np
import pytest

from nbxtensor.dtype.nbx_dtype import NBXDtype as D


def _accelerator():
    from nbxtensor.memory.allocator import DeviceAllocator
    if DeviceAllocator.device_count() == 0:
        pytest.skip("no accelerator on this machine")


@pytest.mark.parametrize("rows,dt", [(81, D.bfloat16), (1, D.float32), (1, D.bfloat16), (81, D.float32)])
def test_each_band_of_the_wider_matmul_is_its_own_matmul_bit_for_bit(rows, dt):
    _accelerator()
    from nbxtensor.ops import linalg_wrappers as LW
    from nbxtensor.tensor import NBXTensor
    rng = np.random.default_rng(rows)
    a = NBXTensor.from_numpy(rng.standard_normal((rows, 1024)).astype(np.float32)).to(dt)
    w1 = (rng.standard_normal((1024, 1024)) * 0.05).astype(np.float32)
    w2 = (rng.standard_normal((1024, 512)) * 0.05).astype(np.float32)
    own = [LW.mm(a, NBXTensor.from_numpy(w).to(dt)) for w in (w1, w2)]
    wide = LW.mm(a, NBXTensor.from_numpy(np.concatenate([w1, w2], 1)).to(dt))
    bands = [wide.narrow(1, 0, 1024), wide.narrow(1, 1024, 512)]
    for band, ref in zip(bands, own):
        x, y = band.contiguous().to(D.float32).numpy().reshape(-1), ref.to(D.float32).numpy().reshape(-1)
        assert np.array_equal(x, y), (int((x != y).sum()), x.size, float(np.abs(x - y).max()))
