"""The host GEMM and host copy measures are the library's (qa 1231-4: numpy measures moved out of the engine's
`core/prism/facts_probe.py`), beside `backend.device_measure`, card-less. Red on the base: the module does not exist."""
from nbxtensor.backend import host_measure


def test_the_host_gemm_states_gflops_of_2n3_over_its_time(monkeypatch):
    ticks = iter([0.0, 0.5, 1.0, 1.25, 2.0, 2.5])      # three timed runs: 0.5, 0.25, 0.5 s -> the best is 0.25 s
    monkeypatch.setattr(host_measure.time, "perf_counter", lambda: next(ticks))
    assert host_measure.host_matmul_gflops(64) == 2.0 * 64 ** 3 / 0.25 / 1e9


def test_the_host_copy_states_read_plus_write_bytes_over_its_time(monkeypatch):
    ticks = iter([0.0, 0.5, 1.0, 1.1, 2.0, 2.5])
    monkeypatch.setattr(host_measure.time, "perf_counter", lambda: next(ticks))
    assert abs(host_measure.host_memory_gbps(1 << 20) - 2.0 * (1 << 20) / 0.1 / 1e9) < 1e-9


def test_a_real_measure_is_positive():
    assert host_measure.host_matmul_gflops(128) > 0 and host_measure.host_memory_gbps(1 << 22) > 0
