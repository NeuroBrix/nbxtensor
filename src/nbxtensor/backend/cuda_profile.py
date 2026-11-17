"""Per-kernel GPU time on CUDA: a profiling session puts every launch, and every copy, between two timing events on
its own stream (the public door: `nbxtensor.profile.kernel_times`).

Why events and not a synchronised timer. A host timer around a launch needs the host to wait for the stream before
and after it: every figure then carries the submit call and the wait's wake-up (the size of a small kernel), and the
queue between host and device is gone, so the time the device spends waiting for its next launch — the figure a
launch-bound step is steered by — is replaced by the wait itself. An event is a marker in the stream, stamped by the
device when the stream reaches it: the host never waits, the launches queue as they do without the session, and the
interval between a launch's two events is read on the device's clock.

What the events cost and where they stop, both measured by the session and said in its table:
- the host records two events per launch; the host time of these calls is summed (`own_ms`), because on a step the
  host bounds it lengthens the time between kernels by that much;
- on an idle device the start event is stamped when it is submitted, a little before the kernel starts, and the end
  event no earlier than its own submission: a launch's interval overstates its kernel by at most the host time from
  the start event to the launch's return (`overstated_ms` sums that bound), and a launch whose interval is no longer
  than its whole submission is counted (`bounded`): its time is an upper bound.

The time between kernels is the interval from one launch's end event to the next one's start event on the same stream
of the same device. Events are taken from a pool per device; when it is empty the session waits for that device's last
events, reads its intervals and reuses them — the gap that holds this read is left out of the time between kernels.
A replay that would run as device graphs runs its launches directly inside a session (`frozen_plan`): a graph is one
launch to the device, and its kernels could not be told apart.
"""
from __future__ import annotations

import ctypes
import threading
import time
from typing import Dict, List, Optional

from nbxtensor.profile import KernelTimes

#: the open session, or None: the one flag a launch and a copy read (`launcher.CudaDriver.launch`, the allocator's copies)
SESSION: Optional["Session"] = None

#: kernel function handle -> its name, told by the driver when it loads the function (`launcher.CudaDriver.load`)
NAMES: Dict[int, str] = {}

#: launches a device's pool times before the session waits for the device and reads them (two events each)
POOL_PAIRS = 4096

#: a copy's row, by its kind (the allocator's: 1 host to device, 2 device to host, 3 device to device)
COPY_ROWS = {1: "copy host to device", 2: "copy device to host", 3: "copy device to device"}
SET_ROW = "set"

NOTE = "each launch and copy between two GPU events on its stream, the run's own cadence kept"
MEASURES = ("every launch and copy between two timing events on its stream; an interval overstates its kernel by at "
            "most the host's submission of it (cadence.overstated_ms)")

#: the runtime calls a session needs, by their key in the backend's table (`backend.runtime.backend_table`)
_CALLS = ("event_create", "event_destroy", "event_record", "event_sync", "event_elapsed", "get_device")


class RuntimeEvents:
    """The runtime's event calls, bound once with their signatures (a launch pays two records) and checked: a call
    the runtime refuses is an error that names it."""

    def __init__(self, rt, table: dict) -> None:
        missing = [k for k in _CALLS if not table.get(k)]
        if missing:
            raise RuntimeError(f"ZERO FALLBACK: this backend's runtime table names no {', '.join(missing)}: "
                               f"per-kernel GPU time cannot be measured on it")
        vp = ctypes.c_void_p

        def bound(key, *argtypes):
            return ctypes.CFUNCTYPE(ctypes.c_int, *argtypes)((table[key], rt))
        self._create = bound("event_create", ctypes.POINTER(vp))
        self._destroy = bound("event_destroy", vp)
        self._record = bound("event_record", vp, vp)
        self._wait = bound("event_sync", vp)
        self._elapsed = bound("event_elapsed", ctypes.POINTER(ctypes.c_float), vp, vp)
        self._device = bound("get_device", ctypes.POINTER(ctypes.c_int))
        self._ms, self._dev = ctypes.c_float(), ctypes.c_int()

    @staticmethod
    def _refused(rc: int, what: str):
        from nbxtensor.memory.allocator import DeviceAllocator
        DeviceAllocator.clear_last_error()                  # a refused call must not poison the next one
        return RuntimeError(f"ZERO FALLBACK: the kernel-time session's {what} was refused by the runtime (error {rc})")

    def device(self) -> int:
        rc = self._device(ctypes.byref(self._dev))
        if rc != 0:
            raise self._refused(rc, "read of the current device")
        return self._dev.value

    def select(self, device: int) -> None:
        from nbxtensor.memory.allocator import DeviceAllocator
        DeviceAllocator.set_device(device)

    def create(self) -> int:
        event = ctypes.c_void_p()
        rc = self._create(ctypes.byref(event))
        if rc != 0 or not event.value:
            raise self._refused(rc, "timing event")
        return event.value

    def destroy(self, event: int) -> None:
        self._destroy(event)

    def record(self, event: int, stream: int) -> None:
        rc = self._record(event, stream)
        if rc != 0:
            raise self._refused(rc, f"event record on stream {stream:#x}")

    def wait(self, event: int) -> None:
        rc = self._wait(event)
        if rc != 0:
            raise self._refused(rc, "wait for an event")

    def elapsed_ms(self, start: int, end: int) -> float:
        rc = self._elapsed(ctypes.byref(self._ms), start, end)
        if rc != 0:
            raise self._refused(rc, "read of an interval between two events")
        return float(self._ms.value)


class Session:
    """One profiling session. `clock` gives the device's events (`RuntimeEvents`: device, select, create, destroy,
    record, wait, elapsed_ms); `now` is the host's clock in nanoseconds."""

    def __init__(self, clock, now=time.perf_counter_ns, pairs: int = POOL_PAIRS) -> None:
        self.clock, self.now, self.pairs = clock, now, int(pairs)
        self.table = KernelTimes()
        self.table.note, self.table.measures = NOTE, MEASURES
        self.cadence = self.table.cadence = {"between_ms": 0.0, "gaps": 0, "own_ms": 0.0, "overstated_ms": 0.0,
                                             "bounded": 0, "bounded_by_kernel": {}}
        self.made: Dict[int, List[int]] = {}            # device -> every event created on it
        self.free: Dict[int, List[int]] = {}            # device -> its events not holding a launch
        self.pending: Dict[int, list] = {}              # device -> its launches not read yet, in their order
        self.booked: Dict[int, float] = {}              # device -> ms booked to its rows and its time between launches
        self._lock = threading.Lock()
        self.t0 = now()

    def begin(self, stream) -> tuple:
        """The start event of a launch (or a copy) recorded on `stream`; the mark its `end` takes."""
        entered = self.now()
        stream = int(stream or 0)
        device = self.clock.device()
        free = self.free.get(device)
        if not free:
            free = self._events_for(device)
        start, end = free.pop(), free.pop()
        t0 = self.now()
        self.clock.record(start, stream)
        return self, device, stream, start, end, t0, self.now() - entered

    def end(self, mark: tuple, name: str) -> None:
        """The end event of the launch `mark` opened; its row is `name`."""
        _session, device, stream, start, end, t0, own = mark
        t2 = self.now()
        self.clock.record(end, stream)
        t3 = self.now()
        self.pending[device].append((stream, name, start, end, t2 - t0, t3 - t0, own + t3 - t2))

    def _events_for(self, device: int) -> List[int]:
        """`device`'s free events: its pool created at its first launch, else its launches read and their events
        reused."""
        with self._lock:
            if device not in self.made:
                self.pending[device] = []
                self.made[device] = [self.clock.create() for _ in range(2 * self.pairs)]
                self.free[device] = list(self.made[device])
            elif not self.free[device]:
                self.read(device)
            return self.free[device]

    def read(self, device: int) -> None:
        """The current device's pending launches into the table. Every launch's two events are placed on ONE timeline
        (their offset from the batch's first start event, on the device's clock) and the timeline is booked once:
        the time a launch alone holds goes to its row, the time no launch holds goes to `between_ms`, and the time
        two launches hold together (streams that overlap) goes to the one that started first, so no instant is booked
        twice. A launch whose end the device stamped before its start is an error that names both stamps."""
        pending = self.pending[device]
        for event in {stream: end for stream, _n, _s, end, _l, _w, _o in pending}.values():
            self.clock.wait(event)                          # the last end event of each stream: all before it are reached
        placed = [self._place(pending[0][2], row) for row in pending]
        self.booked[device] = self._book(sorted(placed), self.booked.get(device, 0.0))
        for _stream, _name, start, end, _l, _w, _o in pending:
            self.free[device] += (start, end)
        pending.clear()

    def _place(self, origin: int, row: tuple) -> tuple:
        """(start, end, name) of a launch as offsets in ms from `origin` on the device's clock; what its events cost
        and the bound of its figure counted. An end before its start is an error that names both stamps."""
        stream, name, start, end, lead_ns, window_ns, own_ns = row
        first, last = self.clock.elapsed_ms(origin, start), self.clock.elapsed_ms(origin, end)
        if last < first:
            raise RuntimeError(f"ZERO FALLBACK: the device stamped the end of {name!r} (stream {stream:#x}) at "
                               f"{last!r} ms before its start at {first!r} ms (offsets from the batch's first "
                               f"event): the interval cannot be booked")
        c = self.cadence
        c["own_ms"] += own_ns / 1e6
        if last - first <= window_ns / 1e6:                 # no longer than its own submission: an upper bound
            c["bounded"] += 1
            c["bounded_by_kernel"][name] = c["bounded_by_kernel"].get(name, 0) + 1
            c["overstated_ms"] += last - first
        else:
            c["overstated_ms"] += lead_ns / 1e6
        return first, last, name

    def _book(self, placed: list, booked: float) -> float:
        """The launches in the order they started, booked on one timeline: what a launch alone holds goes to its
        row, the time no launch holds to `between_ms`, the time two hold together to the one that started first.
        Returns `booked` plus everything booked here."""
        c = self.cadence
        frontier = placed[0][0] if placed else 0.0          # the end of what the timeline has booked so far
        for index, (first, last, name) in enumerate(placed):
            c["gaps"] += 1 if index else 0                  # a boundary between two launches, empty or not
            idle = max(0.0, first - frontier)
            held = max(0.0, last - max(first, frontier))
            c["between_ms"] += idle
            self.table.add(name, held)
            booked += idle + held
            frontier = max(frontier, last)
        return booked

    def close(self) -> KernelTimes:
        """Every device's launches read, its events destroyed; the device that was current is current again."""
        current = self.clock.device() if self.made else None
        for device, events in self.made.items():
            if device != current:
                self.clock.select(device)
            self.read(device)
            for event in events:
                self.clock.destroy(event)
            if device != current:
                self.clock.select(current)
        self.table.wall_s = (self.now() - self.t0) / 1e9
        self._unattributed()
        return self.table

    def _unattributed(self) -> None:
        """The ONE figure derived by subtraction: the session's wall time less what its timeline booked on a device.
        Printed as it is, however large; a device booked for more than the wall is an error that names both."""
        wall_ms = self.table.wall_s * 1e3
        by_device = {}
        for device, booked in self.booked.items():
            if booked > wall_ms + 1e-9:
                raise RuntimeError(f"ZERO FALLBACK: device {device} booked {booked!r} ms of non-overlapping intervals "
                                   f"in a session of {wall_ms!r} ms wall: the clocks disagree, nothing is attributed")
            by_device[device] = {"booked_ms": booked, "unattributed_ms": wall_ms - booked}
        c = self.cadence
        c["wall_ms"] = wall_ms
        c["by_device"] = by_device
        c["unattributed_ms"] = wall_ms - max((d["booked_ms"] for d in by_device.values()), default=0.0)


def open_session() -> KernelTimes:
    """Open the process's one session on the CUDA runtime in force. Refused by name when one is open."""
    global SESSION
    if SESSION is not None:
        raise RuntimeError("ZERO FALLBACK: a kernel-time session is already open in this process")
    from nbxtensor.backend import runtime
    SESSION = Session(RuntimeEvents(runtime.gpu_runtime(), runtime.backend_table()))
    return SESSION.table


def close_session() -> KernelTimes:
    """Every launch of the session read into its table; the flag cleared whatever happens."""
    global SESSION
    session = SESSION
    if session is None:
        raise RuntimeError("ZERO FALLBACK: no kernel-time session is open")
    try:
        return session.close()
    finally:
        SESSION = None
