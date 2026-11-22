"""The HOST's speeds, measured here beside the device's (`backend.device_measure`): what an application's planner reads
to price a plan whose work or weights the host holds (NeuroBrix's Prism records them in a hardware profile).

- `host_matmul_gflops(n)`: a square float32 GEMM on the host's numpy (its BLAS), 2*n^3 / the best of 3 timed runs
  after one warm run, in GFLOP/s.
- `host_memory_gbps(nbytes)`: one host-memory copy of `nbytes` (read + write bytes) / the best of 3 timed runs after
  one warm run, in GB/s.

Moved from the engine (NeuroBrix `core/prism/facts_probe.py` 70-95, qa 1231-4: numpy measures are the library's),
unchanged: the same operands (`numpy.random.default_rng(0)`), sizes and statistic, so a profile measured before the
move and one measured after state the same quantity. Torch-free.
"""
from __future__ import annotations

import time

import numpy as np

__all__ = ["host_matmul_gflops", "host_memory_gbps"]


def host_matmul_gflops(n: int = 4096) -> float:
    """GFLOP/s of an n x n float32 GEMM on the host."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((n, n), dtype=np.float32)
    b = rng.standard_normal((n, n), dtype=np.float32)
    a @ b
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        a @ b
        best = min(best, time.perf_counter() - t0)
    return 2.0 * n ** 3 / best / 1e9


def host_memory_gbps(nbytes: int = 1 << 30) -> float:
    """GB/s of a host-memory copy of `nbytes` bytes ((read + write) bytes / time)."""
    src = np.ones(int(nbytes), dtype=np.uint8)
    dst = np.empty_like(src)
    np.copyto(dst, src)
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        np.copyto(dst, src)
        best = min(best, time.perf_counter() - t0)
    return 2.0 * int(nbytes) / best / 1e9
