"""nbxtensor.profile.kernel_times on CUDA: every launch and every copy sits between two timing events on its stream;
the table counts launches by kernel name, sums their GPU time and the GPU time between kernels, says what the session
cost the host and how far a figure can overstate its kernel. No GPU: the device is a simulated stream with its clock.

What this test does if the code were wrong:
- the table cells fail when the printed table of a session that fills nothing more than Metal's changes by a byte, or
  when a session that kept the run's cadence does not print the time between kernels and the bound of its figures;
- the session cell fails when a launch is not counted under its kernel's name, when a kernel's GPU time is not the
  interval between its two events, or when the time between kernels is not the device's wait for its next launch;
- the bound cell fails when a kernel shorter than its own submission is not counted as an upper bound, or when the
  bound of the other figures is not the host time from the start event to the launch's return;
- the pool cell fails when a session longer than a device's events loses a launch, counts the gap that holds its own
  read, or creates more events than the pool;
- the device cell fails when two cards share events, or when the card that was current is not current after the close;
- the driver, copy and replay cells fail when a launch or a copy is not timed inside a session, is timed outside one,
  or when a replay launches a device graph inside a session;
- the refusal cells fail when a second session opens, a backend without the mode is not refused by name, or a
  runtime's refused event is swallowed.
"""
from __future__ import annotations

import ctypes

import pytest

from nbxtensor import profile
from nbxtensor.backend import cuda_profile


class _Device:
    """Cards with serial streams and one clock (ns). An event is stamped when its stream reaches it: at its
    submission on an idle stream, else when the work queued before it ends. `work(ns)` queues a kernel."""

    def __init__(self) -> None:
        self.host = 0                                   # the host's clock, moved by `spend`
        self.current = 0
        self.busy = {}                                  # (device, stream) -> when its queued work ends
        self.stamp = {}                                 # event -> its stamp
        self.home = {}                                  # event -> the device it was created on
        self.created = self.destroyed = self.waits = 0
        self.selected = []

    def spend(self, ns: int) -> None:
        self.host += ns

    def now(self) -> int:
        return self.host

    def work(self, ns: int, stream: int = 0) -> None:
        key = (self.current, stream)
        self.busy[key] = max(self.busy.get(key, 0), self.host) + ns

    # --- what a session asks of its clock -------------------------------------------------------------------------
    def device(self) -> int:
        return self.current

    def select(self, device: int) -> None:
        self.current = device
        self.selected.append(device)

    def create(self) -> int:
        self.created += 1
        self.home[self.created] = self.current
        return self.created

    def destroy(self, event: int) -> None:
        self.destroyed += 1

    def record(self, event: int, stream: int) -> None:
        assert self.home[event] == self.current, "an event recorded on a card it was not created on"
        self.spend(1_000)                               # the record's own host time
        self.stamp[event] = max(self.busy.get((self.current, stream), 0), self.host)

    def wait(self, event: int) -> None:
        assert self.home[event] == self.current
        self.waits += 1
        self.host = max(self.host, self.stamp[event])

    def elapsed_ms(self, start: int, end: int) -> float:
        assert self.home[start] == self.home[end] == self.current, "an interval read across cards"
        return (self.stamp[end] - self.stamp[start]) / 1e6


def _session(pairs: int = 8):
    dev = _Device()
    return dev, cuda_profile.Session(dev, now=dev.now, pairs=pairs)


def _launch(dev, session, name: str, kernel_ns: int, submit_ns: int = 2_000, stream: int = 0) -> None:
    """One launch as the driver issues it: start event, the launch call (`submit_ns` of host time), end event."""
    mark = session.begin(stream)
    dev.spend(submit_ns)
    dev.work(kernel_ns, stream)
    mark[0].end(mark, name)


# --- the table ---------------------------------------------------------------------------------------------------------

def test_a_table_filled_as_metal_fills_it_prints_and_writes_what_it_did():
    t = profile.KernelTimes()
    t.add("mul", 2.0)
    t.add("mul", 4.0)
    t.add("blit", 1.0)
    t.wall_s = 0.5
    assert t.format().splitlines() == [
        "[kernel-times] wall 0.500 s, GPU 7.0 ms summed over 3 launches (each launch and copy timed alone: a ranking, "
        "not the step's time)",
        f"[kernel-times]   {'kernel':<48} {'launches':>8} {'total ms':>10} {'mean ms':>8} {'max ms':>8} {'share':>6}",
        f"[kernel-times]   {'mul':<48} {2:>8} {6.0:>10.2f} {3.0:>8.4f} {4.0:>8.3f} {85.7:>5.1f}%",
        f"[kernel-times]   {'blit':<48} {1:>8} {1.0:>10.2f} {1.0:>8.4f} {1.0:>8.3f} {14.3:>5.1f}%"]
    assert t.as_dict() == {
        "wall_s": 0.5, "gpu_ms": 7.0, "unsampled": 0,
        "measures": "every launch and copy alone: the figures rank kernels, not the production step",
        "kernels": [{"name": "mul", "launches": 2, "total_ms": 6.0, "max_ms": 4.0},
                    {"name": "blit", "launches": 1, "total_ms": 1.0, "max_ms": 1.0}]}


def test_a_table_past_its_top_rows_sums_the_rest_in_one_line():
    t = profile.KernelTimes()
    for i in range(5):
        t.add(f"k{i}", float(5 - i))
    lines = t.format(top=2).splitlines()
    assert len(lines) == 5 and "k0" in lines[2] and "k1" in lines[3]
    assert "the rest (3 kernels)" in lines[4] and f"{3:>8} {6.0:>10.2f}" in lines[4]


def test_a_session_that_kept_the_cadence_prints_the_time_between_kernels_and_the_bound_of_its_figures(tmp_path):
    dev, session = _session()
    _launch(dev, session, "gemv", 500_000)
    dev.spend(40_000)
    _launch(dev, session, "gemv", 500_000)
    table = session.close()
    lines = table.format().splitlines()
    assert lines[0].endswith("(each launch and copy between two GPU events on its stream, the run's own cadence kept)")
    assert lines[1].startswith("[kernel-times]   between kernels: ") and " over 1 gaps " in lines[1]
    assert "overstates the kernels by at most" in lines[2] and "unattributed" in lines[3] and lines[4].split()[1] == "kernel"
    path = tmp_path / "t.json"
    assert table.report(str(path)).endswith(f"the whole table: {path}")
    import json
    said = json.loads(path.read_text())
    assert said["cadence"]["gaps"] == 1 and said["kernels"][0]["name"] == "gemv" and "timing events" in said["measures"]


# --- the session -------------------------------------------------------------------------------------------------------

def test_each_launch_is_counted_by_its_name_with_its_own_gpu_time_and_the_wait_between_kernels():
    dev, session = _session()
    for _ in range(3):
        _launch(dev, session, "heavy", 800_000)
        dev.spend(900_000)                              # the host slower than the device: it waits for its next launch
        _launch(dev, session, "light", 50_000)
        dev.spend(900_000)
    table = session.close()
    rows = {n: (c, t) for n, c, t, _m in table.ranked()}
    assert rows["heavy"][0] == 3 and rows["light"][0] == 3 and table.unsampled == 0
    # on an idle device an interval is its kernel and the launch call submitted before it (2 us)
    assert rows["heavy"][1] == pytest.approx(3 * 0.802) and rows["light"][1] == pytest.approx(3 * 0.052)
    assert table.ranked()[0][0] == "heavy"
    c = table.cadence
    assert c["gaps"] == 5 and c["bounded"] == 0
    # heavy's end to light's start: 900 us of host less the 800 us the kernel was still running, and the start's record
    assert c["between_ms"] == pytest.approx(3 * 0.102 + 2 * 0.852)
    assert c["own_ms"] == pytest.approx(6 * 0.002)      # two records of 1 us per launch
    assert c["overstated_ms"] == pytest.approx(6 * 0.003)


def test_launches_queued_behind_a_busy_device_measure_their_kernels_exactly_with_no_time_between_them():
    dev, session = _session()
    _launch(dev, session, "long", 5_000_000)
    for _ in range(4):
        _launch(dev, session, "short", 100_000)         # submitted while `long` still runs
    table = session.close()
    rows = {n: (c, t) for n, c, t, _m in table.ranked()}
    assert rows["short"] == (4, pytest.approx(0.4))
    assert table.cadence["between_ms"] == pytest.approx(0.0) and table.cadence["gaps"] == 4


def test_a_kernel_shorter_than_its_own_submission_is_counted_as_an_upper_bound():
    dev, session = _session()
    _launch(dev, session, "tiny", 500, submit_ns=20_000)
    _launch(dev, session, "big", 1_000_000, submit_ns=20_000)
    table = session.close()
    c = table.cadence
    assert c["bounded"] == 1 and c["bounded_by_kernel"] == {"tiny": 1}
    rows = {n: t for n, _c, t, _m in table.ranked()}
    assert rows["tiny"] == pytest.approx(0.021)         # the pair's submission, not the 0.5 us kernel
    # tiny: its whole figure; big: the host time from its start event to the launch's return
    assert c["overstated_ms"] == pytest.approx(0.021 + 0.021)


def test_a_session_longer_than_a_device_s_events_reads_them_and_reuses_them():
    dev, session = _session(pairs=4)
    for _ in range(11):
        _launch(dev, session, "k", 10_000)
        dev.spend(50_000)
    assert dev.created == 8 and dev.waits == 2          # two reads inside the session
    table = session.close()
    assert table.ranked() == [("k", 11, pytest.approx(11 * 0.012), pytest.approx(0.012))]
    assert table.cadence["gaps"] == 8                   # 10 gaps, less the two that hold a read
    assert dev.created == 8 and dev.destroyed == 8


def test_each_card_has_its_own_events_and_the_current_card_is_current_again_after_the_close():
    dev, session = _session()
    _launch(dev, session, "on0", 100_000)
    dev.current = 1
    _launch(dev, session, "on1", 200_000)
    _launch(dev, session, "on1", 200_000, stream=7)     # another stream of card 1: a boundary, but no time between
    table = session.close()
    assert {n: c for n, c, _t, _m in table.ranked()} == {"on0": 1, "on1": 2}
    assert table.cadence["gaps"] == 1 and table.cadence["between_ms"] == 0.0 and dev.current == 1 and dev.selected == [0, 1]
    assert dev.created == dev.destroyed == 32


# --- the launch, the copies, the replay -------------------------------------------------------------------------------

class _Lib:
    def __init__(self, dev):
        self.dev, self.launched = dev, 0

    def cuLaunchKernel(self, *args):
        self.launched += 1
        self.dev.spend(2_000)
        self.dev.work(300_000)
        return 0


def _driver(dev):
    from nbxtensor.backend.launcher import CudaDriver
    drv = object.__new__(CudaDriver)
    drv.lib, drv._param_counts, drv._arg_buffers = _Lib(dev), {}, {}
    return drv


def test_the_cuda_driver_times_a_launch_under_its_kernel_s_name_inside_a_session_only(monkeypatch):
    dev, session = _session()
    drv = _driver(dev)
    fn = ctypes.c_void_p(0x1234)
    monkeypatch.setitem(cuda_profile.NAMES, 0x1234, "rms_norm_kernel")
    drv.launch(fn, (4,), (128, 1, 1), 0, 0, [("i32", 3)])
    assert drv.lib.launched == 1 and dev.created == 0   # no session: the launch alone, no event
    monkeypatch.setattr(cuda_profile, "SESSION", session)
    drv.launch(fn, (4,), (128, 1, 1), 0, 0, [("i32", 3)])
    drv.launch(fn, (0,), (128, 1, 1), 0, 0, [("i32", 3)])     # no program: nothing launched, nothing timed
    assert drv.lib.launched == 2
    # queued behind the launch before it: the kernel's own 300 us
    assert session.close().ranked() == [("rms_norm_kernel", 1, pytest.approx(0.3), pytest.approx(0.3))]


class _Memory:
    def __init__(self, dev):
        self.dev = dev

    def memcpy(self, dst, src, nbytes, kind):
        self.dev.work(nbytes)
        return 0

    def memset(self, ptr, value, nbytes):
        self.dev.work(nbytes)
        return 0


def test_the_allocator_s_copies_that_reach_a_device_are_timed_by_their_kind_inside_a_session_only(monkeypatch):
    from nbxtensor.backend import runtime
    from nbxtensor.memory.allocator import DeviceAllocator
    dev, session = _session()
    monkeypatch.setattr(runtime, "device_memory", lambda: _Memory(dev))
    DeviceAllocator.memcpy_direct(1, 2, 64, 3)
    assert dev.created == 0
    monkeypatch.setattr(cuda_profile, "SESSION", session)
    DeviceAllocator.memcpy_direct(1, 2, 400_000, 1)
    DeviceAllocator.memcpy_direct(1, 2, 300_000, 2)
    DeviceAllocator.memcpy_direct(1, 2, 200_000, 3)
    DeviceAllocator.memcpy_direct(1, 2, 200_000, 3)
    DeviceAllocator.memcpy_direct(1, 2, 64, 0)          # host to host: no device, not timed
    DeviceAllocator.memset_direct(1, 0, 100_000)
    assert {n: c for n, c, _t, _m in session.close().ranked()} == {
        "copy host to device": 1, "copy device to host": 1, "copy device to device": 2, "set": 1}


def test_a_replay_that_would_launch_device_graphs_runs_its_launches_directly_inside_a_session(monkeypatch):
    from nbxtensor.backend import replay_graph
    from nbxtensor.backend.frozen_plan import FrozenPlan
    ran = []
    monkeypatch.setattr(replay_graph, "capable", lambda: True)          # a device that replays graphs
    monkeypatch.setattr(FrozenPlan, "_scan_bound", lambda self, arena: 0)
    monkeypatch.setattr(FrozenPlan, "_scan_extra", lambda self, arena: 0)
    monkeypatch.setattr(FrozenPlan, "crosses_cards", lambda self: True)
    monkeypatch.setattr(FrozenPlan, "_replay_graph", lambda self, segmented: ran.append("graphs"))
    monkeypatch.setattr(FrozenPlan, "_direct_actions", lambda self, records=None: ran.append("direct"))
    monkeypatch.setattr(FrozenPlan, "hand_out_outputs", lambda self, arena: None)

    class _Arena:
        def restore_from(self, snapshot):
            pass

    plan = object.__new__(FrozenPlan)
    plan.scan_slots, plan.graph_disabled, plan.arena_snapshot = None, False, []
    plan.replay(_Arena())
    monkeypatch.setattr(cuda_profile, "SESSION", _session()[1])
    plan.replay(_Arena())
    assert ran == ["graphs", "direct"]


# --- the refusals ------------------------------------------------------------------------------------------------------

def test_a_second_session_and_a_close_with_none_open_are_refused_by_name(monkeypatch):
    monkeypatch.setattr(cuda_profile, "SESSION", _session()[1])
    with pytest.raises(RuntimeError, match="ZERO FALLBACK: a kernel-time session is already open"):
        cuda_profile.open_session()
    monkeypatch.setattr(cuda_profile, "SESSION", None)
    with pytest.raises(RuntimeError, match="ZERO FALLBACK: no kernel-time session is open"):
        cuda_profile.close_session()


def test_a_backend_without_the_mode_is_refused_by_name(monkeypatch):
    from nbxtensor.backend import runtime
    monkeypatch.setattr(runtime, "gpu_backend", lambda: "hip")
    with pytest.raises(RuntimeError, match="ZERO FALLBACK: per-kernel GPU time is measured on Metal and on CUDA; this "
                                           "backend is 'hip'"):
        with profile.kernel_times():
            pass


def test_on_cuda_the_public_door_opens_the_library_s_session_and_closes_it_on_an_error(monkeypatch):
    from nbxtensor.backend import runtime
    dev = _Device()
    monkeypatch.setattr(runtime, "gpu_backend", lambda: "cuda")
    monkeypatch.setattr(cuda_profile, "RuntimeEvents", lambda rt, table: dev)
    monkeypatch.setattr(runtime, "gpu_runtime", lambda: None)
    monkeypatch.setattr(runtime, "backend_table", lambda: {})
    with pytest.raises(KeyError):
        with profile.kernel_times() as table:
            assert cuda_profile.SESSION is not None and cuda_profile.SESSION.table is table
            raise KeyError("the work failed")
    assert cuda_profile.SESSION is None
    seen = []
    assert profile.ranked_call(lambda: 7, out=seen.append) == 7
    assert seen[0].startswith("[kernel-times] wall ") and cuda_profile.SESSION is None


def test_a_runtime_table_without_the_event_calls_is_refused_by_name():
    with pytest.raises(RuntimeError, match="ZERO FALLBACK: this backend's runtime table names no event_elapsed"):
        cuda_profile.RuntimeEvents(None, {k: k for k in ("event_create", "event_destroy", "event_record",
                                                         "event_sync", "get_device")})


def test_the_runtime_s_calls_are_bound_by_their_table_names_and_a_refused_one_is_an_error(monkeypatch):
    """The binding itself, on the C library every system has: `abs` stands for each call (an int in, an int out), so
    a call given 0 succeeds and a call given a handle is 'refused' with that handle as its error."""
    from nbxtensor.memory.allocator import DeviceAllocator
    monkeypatch.setattr(DeviceAllocator, "clear_last_error", staticmethod(lambda: 0))
    libc = ctypes.CDLL(None)
    events = cuda_profile.RuntimeEvents(libc, {k: "abs" for k in ("event_create", "event_destroy", "event_record",
                                                                 "event_sync", "event_elapsed", "get_device")})
    events.wait(0)
    with pytest.raises(RuntimeError, match=r"ZERO FALLBACK: the kernel-time session's wait for an event was refused "
                                           r"by the runtime \(error 5\)"):
        events.wait(5)
