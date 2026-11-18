"""On Metal, a timing event recorded after a launch is reached only once that launch has run: the time between two
events is the GPU time of the work encoded between them, and the benchmark built on it (`launcher.do_bench`, the
certifier's timer) measures the kernel.

Before (the Mac, 2026-11-18): `event_record` committed a command buffer of its own while the launches before it were
still in the stream's open buffer, not yet committed. Both events retired before the kernel ran, so the interval was
two empty buffers apart: every matmul read 0.08 to 0.10 ms whatever its size (2048^3 fp16: 0.08 ms against 3.0 ms by
the wall clock; 1152x32768x5120: 0.10 ms against 63.9 ms, an implied 2700 TFLOP/s on a GPU of about 6). Every Apple
certification ranked its configurations on that number.

What this test does if the code is wrong:
- the interval cell fails when two events around one 2048^3 fp16 matmul read less than half of that matmul's wall
  time with a full synchronisation;
- the bench cell fails when `do_bench` on the same matmul reads less than half of it, or when a matmul with sixteen
  times the work reads less than four times the time.

Run: python -m pytest tests/test_a_metal_event_times_the_kernels_recorded_before_it.py
"""
import sys
import time

import numpy as np
import pytest

if sys.platform != "darwin":
    pytest.skip("Metal only", allow_module_level=True)
pytest.importorskip("triton")

from nbxtensor.backend import launcher as L  # noqa: E402
from nbxtensor.memory.allocator import DeviceAllocator as DA  # noqa: E402
from nbxtensor.ops import linalg_wrappers as LW  # noqa: E402
from nbxtensor.tensor import NBXTensor  # noqa: E402


def _matmul(m, n, k):
    rng = np.random.default_rng(0)
    a = NBXTensor.from_numpy(np.ascontiguousarray((rng.standard_normal((m, k)) * 0.1).astype(np.float16)))
    b = NBXTensor.from_numpy(np.ascontiguousarray((rng.standard_normal((k, n)) * 0.1).astype(np.float16)))
    fn = lambda: LW.mm(a, b)  # noqa: E731
    fn()
    DA.stream_synchronize(0)
    return fn


def _wall_ms(fn, n=5):
    DA.stream_synchronize(0)
    began = time.perf_counter()
    for _ in range(n):
        fn()
    DA.stream_synchronize(0)
    return 1000.0 * (time.perf_counter() - began) / n


def test_two_events_around_a_launch_read_its_time():
    fn = _matmul(2048, 2048, 2048)
    wall = _wall_ms(fn)
    between = min(L._time_ms(fn, 1) for _ in range(3))
    assert between >= 0.5 * wall, f"events read {between:.3f} ms around a matmul the wall clock reads {wall:.3f} ms"


def test_the_benchmark_measures_the_kernel():
    small, big = _matmul(2048, 2048, 2048), _matmul(2048, 32768, 2048)
    wall = _wall_ms(small)
    t_small = L.do_bench(small, warmup=5, rep=20, size_repetitions_by="kernel")
    t_big = L.do_bench(big, warmup=5, rep=20, size_repetitions_by="kernel")
    assert t_small >= 0.5 * wall, f"do_bench read {t_small:.3f} ms for a matmul the wall clock reads {wall:.3f} ms"
    assert t_big >= 4.0 * t_small, f"16x the work read {t_big:.3f} ms against {t_small:.3f} ms"
