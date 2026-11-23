"""On Metal a pinned host buffer (`cudaMallocHost`) is a shared MTLBuffer a kernel binds like any device buffer, so
releasing it while a committed command buffer still runs is a use-after-free on the GPU's side. `free` waited for the
queue before releasing; `free_host` did not (library C, 23/11/2026, orchestrator's line-3 scan: metal_device.py:495).

What this test does if the code is wrong: a kernel is left in flight, then a pinned host buffer is released through
the allocator's own path (`DeviceAllocator.free_host_pinned`); at the moment the buffer leaves the runtime's tables
(`_unexpose`, just before its release) the queue still holds work that could be writing memory.

Run: python -m pytest tests/test_a_pinned_host_buffer_is_released_after_the_work_in_flight.py
"""
import sys

import numpy as np
import pytest

if sys.platform != "darwin":
    pytest.skip("Metal only", allow_module_level=True)
pytest.importorskip("triton")

N = 1 << 22


def _device():
    """The Metal runtime, the allocator and the tensor class, or a skip where no Metal device answers."""
    from nbxtensor.memory.allocator import DeviceAllocator
    from nbxtensor.backend import metal_device as M
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    import nbxtensor.ops.elementwise_wrappers  # noqa: F401  the launcher's seam
    try:
        DeviceAllocator.set_device(0)
        NBXTensor.zeros((1,), dtype=NBXDtype.float32, device="cuda:0")
        M.runtime().sync()
    except Exception as e:                                         # pragma: no cover
        pytest.skip(f"no Metal device: {e}")
    return M, DeviceAllocator, NBXTensor


def test_a_pinned_host_buffer_is_released_only_after_the_queue_retired(monkeypatch):
    M, A, T = _device()
    from nbxtensor.ops import elementwise_wrappers as E
    rt = M.runtime()
    ptr = A.malloc_host_pinned(1 << 20)
    x = T.from_numpy(np.ones(N, dtype=np.float32), device="cuda:0")
    rt.sync()
    seen = []
    real = rt._unexpose
    monkeypatch.setattr(rt, "_unexpose", lambda buf: (seen.append(rt.has_pending_gpu_writes()), real(buf))[1])
    y = E.add(x, x)                                                 # work committed or encoded, not waited for
    assert rt.has_pending_gpu_writes(), "the probe kernel retired before the release: nothing was in flight"
    A.free_host_pinned(ptr)
    assert seen == [False], "a pinned host buffer was released while the queue still held work"
    assert np.all(y.numpy() == 2.0)
