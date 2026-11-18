"""On Metal, a key with no certificate is tuned at run time on the user's own machine, and the times that sweep ranks its
configurations on are the GPU times of those configurations: they follow the work.

The run-time sweep is Triton's autotuner with the library's timer handed to it (`ops/_configs.py`, `nbx_autotune`:
`do_bench` = `launcher.do_bench`), so it goes through the same Metal timing events as the benchmark and the certifier.
Before (the Mac, 2026-11-18) those events retired before the kernel ran: every configuration of every matmul read about
0.1 ms whatever its size, and the sweep kept whichever configuration that number happened to favour.

The sweep is forced with `NBX_AUTOTUNE_CERTIFIED=off` (the certified directory is not applied) and an empty replay
cache of the test's own, so the key is one the machine has never tuned.

What this test does if the code is wrong:
- it fails when the fastest time the sweep recorded for a 2048^3 fp16 matmul is less than half of what the wall clock
  reads for that matmul with a full synchronisation;
- it fails when the sweep's fastest time for a matmul with sixteen times the work is less than four times that.

Run: python -m pytest tests/test_a_runtime_autotune_on_metal_ranks_on_times_that_follow_the_work.py
"""
import sys
import time

import numpy as np
import pytest

if sys.platform != "darwin":
    pytest.skip("Metal only", allow_module_level=True)
pytest.importorskip("triton")

from nbxtensor.memory.allocator import DeviceAllocator as DA  # noqa: E402
from nbxtensor.ops import linalg_wrappers as LW  # noqa: E402
from nbxtensor.ops import matmul as MM  # noqa: E402
from nbxtensor.tensor import NBXTensor  # noqa: E402


@pytest.fixture
def runtime_sweep(monkeypatch, tmp_path):
    monkeypatch.setenv("NBX_AUTOTUNE_CERTIFIED", "off")
    monkeypatch.setenv("NEUROBRIX_REPLAY_CACHE", str(tmp_path / "replay_cache"))
    tuner = MM.matmul_kernel
    kept = dict(tuner.cache)
    tuner.cache.clear()
    yield tuner
    tuner.cache.clear()
    tuner.cache.update(kept)


def _operands(m, n, k):
    rng = np.random.default_rng(0)
    a = NBXTensor.from_numpy(np.ascontiguousarray((rng.standard_normal((m, k)) * 0.1).astype(np.float16)))
    b = NBXTensor.from_numpy(np.ascontiguousarray((rng.standard_normal((k, n)) * 0.1).astype(np.float16)))
    return a, b


def _swept_fastest_ms(tuner, m, n, k):
    """The first call of an untuned key sweeps; the fastest time the sweep recorded, and the wall time of the kept one."""
    a, b = _operands(m, n, k)
    before = set(tuner.cache)
    LW.mm(a, b)
    DA.stream_synchronize(0)
    assert set(tuner.cache) - before, f"{m}x{n}x{k}: the call did not sweep matmul_kernel"
    timings = dict(tuner.configs_timings)
    assert len(timings) > 1, f"{m}x{n}x{k}: one configuration only, nothing was ranked"
    fastest = min((t[0] if isinstance(t, (list, tuple)) else float(t)) for t in timings.values())
    DA.stream_synchronize(0)
    began = time.perf_counter()
    for _ in range(5):
        LW.mm(a, b)
    DA.stream_synchronize(0)
    return fastest, 1000.0 * (time.perf_counter() - began) / 5


def test_the_runtime_sweep_ranks_on_times_that_follow_the_work(runtime_sweep):
    small, wall = _swept_fastest_ms(runtime_sweep, 2048, 2048, 2048)
    big, _ = _swept_fastest_ms(runtime_sweep, 2048, 32768, 2048)
    assert small >= 0.5 * wall, f"the sweep's fastest time {small:.3f} ms for a matmul the wall clock reads {wall:.3f} ms"
    assert big >= 4.0 * small, f"16x the work: the sweep's fastest time {big:.3f} ms against {small:.3f} ms"
