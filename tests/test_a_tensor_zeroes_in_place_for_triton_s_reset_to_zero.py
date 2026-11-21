"""An NBXTensor zeroes in place (`zero_`), as Triton's own tuning asks of every argument a kernel declares
`reset_to_zero`, and zeroing a strided view zeroes that view's elements and no others.

Triton's autotuner zeroes each `reset_to_zero` argument with `.zero_()` before every configuration it measures. An
NBXTensor had no `zero_`: the first tuning of such a kernel raised AttributeError (qa on the library, 2026-11-21; no
kernel of the library declared it yet). `fill_(0)` zeroed `nbytes` bytes from the view's first address, which on a
strided view is not the view's elements.

What this test would do if the code were wrong: the tuning raises "'NBXTensor' object has no attribute 'zero_'",
the accumulator holds the sum of every measured configuration, or zeroing a view clears elements outside it."""
import numpy as np
import pytest

from nbxtensor.backend.runtime import gpu_backend

pytestmark = pytest.mark.skipif(gpu_backend() is None, reason="no device")


def test_a_reset_to_zero_kernel_tunes_through_triton_and_counts_one_run():
    triton = pytest.importorskip("triton")
    import triton.language as tl
    from nbxtensor.ops._configs import nbx_autotune
    from nbxtensor.ops.oracles.fp64 import host_values
    from nbxtensor.tensor import NBXTensor
    import nbxtensor.ops._launch_seam  # noqa: F401

    def add_one(acc, N, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        tl.store(acc + offs, tl.load(acc + offs, mask=m) + 1.0, mask=m)

    add_one.__module__ = "nbxtensor.tests"           # the launcher takes only the library's own kernels (`_is_ours`)
    add_one = nbx_autotune(configs=[triton.Config({"BLOCK": 64}), triton.Config({"BLOCK": 128})], key=["N"],
                           reset_to_zero=["acc"])(triton.jit(add_one))

    acc = NBXTensor.from_numpy(np.zeros(256, dtype=np.float32))
    add_one[lambda META: (triton.cdiv(256, META["BLOCK"]),)](acc, 256)
    assert (host_values(acc) == 1.0).all(), "the accumulator kept the sums of the measured configurations"


def test_zeroing_a_strided_view_zeroes_that_view_alone():
    from nbxtensor.ops.oracles.fp64 import host_values
    from nbxtensor.tensor import NBXTensor
    base = np.arange(1, 33, dtype=np.float32).reshape(4, 8)
    t = NBXTensor.from_numpy(base)
    t[:, 2:4].fill_(0)                 # fill_ exists on main too: main shows the wrong value, not a missing method
    want = base.copy()
    want[:, 2:4] = 0
    assert np.array_equal(host_values(t), want)
