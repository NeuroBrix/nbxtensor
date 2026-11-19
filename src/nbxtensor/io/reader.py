"""The weight reader: parallel reads that measure THEMSELVES on the load they serve (the launch-time cure, 2026-11-19).

A load used to read with a worker count measured ahead of it (`read_rate.workers_for`: trials over the cache, 130 s on
a quiet store, 367 s beside three other loads) and kept in the hardware profile; a busy store gave no stable trial,
nothing was kept, and every later launch paid the trials again. Here nothing is measured ahead and nothing is kept: the
reads of the model's own weights start at ONE in flight and the width doubles while the rate it measures on those very
reads rises; at the first width that does not read faster than the best one before it, the reader settles on the best.
The bytes read are the load's own, read once.

Two shapes of load use it:

* `map(fn, items, nbytes)`: whole pieces (a shard file each) read by `fn` in parallel, `width` in flight; a width is
  judged on one round (as many pieces completed as the width), its rate the bytes they hold over the round's seconds.
* `read_bytes(fd, offset, n)` and `io`: one tensor's bytes split into page-aligned spans across `width` threads
  (`safetensors.read_bytes`), and the same split for the host conversions (`safetensors.cast`...); a width is judged
  on a read it split that many ways (a tensor too small to split says nothing about the width and only counts).

The width never passes `ceiling`, which the caller decides (the reads a plan prices host memory for): the reader holds
no default. `measured()` says what the load measured — the width reached, the rate at each width, bytes, seconds — for
the caller's launch line. Torch-free."""
from __future__ import annotations

import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Callable, Dict, Iterable, Iterator, Optional, Sequence, TypeVar

__all__ = ["Growth", "Reader"]

_END = object()
T = TypeVar("T")
R = TypeVar("R")


class Growth:
    """The width rule, alone: one in flight first; after a width's measure, twice the width (never past `ceiling`)
    while that measure is the best so far; else settled on the best width measured. A measure of a width other than
    the current one (a piece started before the width moved) is not a measure of it and is ignored."""

    def __init__(self, ceiling: int):
        if not isinstance(ceiling, int) or ceiling < 1:
            raise ValueError(f"ZERO FALLBACK: a reader's ceiling is a count of reads >= 1 the caller decides, not "
                             f"{ceiling!r}")
        self.ceiling = ceiling
        self.width = 1
        self.rates: Dict[int, float] = {}
        self.settled = False

    def note(self, width: int, nbytes: int, seconds: float) -> None:
        if self.settled or width != self.width or seconds <= 0 or nbytes <= 0:
            return
        rate = nbytes / seconds
        best_before = max((r for w, r in self.rates.items() if w != width), default=None)
        self.rates[width] = rate
        if best_before is not None and rate <= best_before:
            self.settled, self.width = True, max(self.rates, key=self.rates.get)
        elif width >= self.ceiling:
            self.settled = True
        else:
            self.width = min(2 * width, self.ceiling)


class Reader:
    """Parallel reads of one process's weight loads, their width grown by their own measure (`Growth`) up to
    `ceiling`. One reader serves every load of a process: what it settled on holds for the process's later loads (a
    streamed plan's reloads), and is never written anywhere."""

    def __init__(self, ceiling: int):
        self._growth = Growth(ceiling)
        self._pool = ThreadPoolExecutor(max_workers=ceiling, thread_name_prefix="nbx-read")
        self._lock = threading.Lock()
        self.bytes = 0
        self.seconds = 0.0

    @property
    def ceiling(self) -> int:
        return self._growth.ceiling

    @property
    def width(self) -> int:
        return self._growth.width

    @property
    def io(self):
        """(width, executor): the split `safetensors.read_bytes` / `cast` / `to_bf16_bits` / `split_across_workers`
        take, at the width the reader stands at now."""
        return self._growth.width, self._pool

    def _note(self, width: int, nbytes: int, seconds: float, judged: bool) -> None:
        with self._lock:
            self.bytes += nbytes
            self.seconds += seconds
            if judged:
                self._growth.note(width, nbytes, seconds)

    def read_bytes(self, fd: int, offset: int, nbytes: int):
        """`nbytes` of the file at `offset`, read across the current width (`safetensors.read_bytes`), timed; a read
        split as many ways as the width is that width's measure."""
        from nbxtensor.io import safetensors as _sf
        width = self._growth.width
        t0 = time.perf_counter()
        buf = _sf.read_bytes(fd, offset, nbytes, (width, self._pool))
        dt = time.perf_counter() - t0
        self._note(width, nbytes, dt, len(_sf.spans(nbytes, 1, _sf.PAGE, width)) == width)
        return buf

    def map(self, fn: Callable[[T], R], items: Iterable[T], nbytes: Callable[[T], int]) -> Iterator[R]:
        """`fn(item)` for every item, `width` of them in flight, the results as they complete. A round is as many
        completions as the width it started at; its rate (the bytes `nbytes` gives the items completed in it, over its
        seconds) judges that width, and the next round runs at the width the rule gives. `fn` reads by its own means
        and must not use this reader's `io` (its threads are this call's own)."""
        todo = list(items)
        if not todo:
            return
        it = iter(todo)
        pending: Dict = {}
        with ThreadPoolExecutor(max_workers=self.ceiling, thread_name_prefix="nbx-read-map") as pool:
            def fill():
                while len(pending) < self._growth.width:
                    item = next(it, _END)
                    if item is _END:
                        return
                    pending[pool.submit(fn, item)] = item
            width, started, done, got = self._growth.width, time.perf_counter(), 0, 0
            fill()
            try:
                while pending:
                    finished, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                    for f in finished:
                        item = pending.pop(f)
                        result = f.result()
                        got += int(nbytes(item))
                        done += 1
                        yield result
                    if done >= width:
                        self._note(width, got, time.perf_counter() - started, True)
                        width, started, done, got = self._growth.width, time.perf_counter(), 0, 0
                    fill()
                if done:
                    self._note(width, got, time.perf_counter() - started, False)
            finally:
                for f in pending:
                    f.cancel()

    def measured(self) -> Dict[str, object]:
        """What the loads measured: the width reached and whether it settled, the rate at each width (bytes/s), the
        bytes read and their seconds, the ceiling."""
        g = self._growth
        return {"workers": g.width, "settled": g.settled, "rates": dict(g.rates), "bytes": self.bytes,
                "seconds": self.seconds, "ceiling": g.ceiling}

    def said(self) -> str:
        """The launch line's words for `measured()`."""
        m = self.measured()
        gbps = (m["bytes"] / m["seconds"] / 1e9) if m["seconds"] else 0.0
        steps = ", ".join(f"{w}: {r / 1e9:.2f}" for w, r in sorted(m["rates"].items()))
        return (f"read {m['bytes'] / 1e9:.2f} GB in {m['seconds']:.1f} s ({gbps:.2f} GB/s), {m['workers']} parallel "
                f"read(s){'' if m['settled'] else ' (still growing)'} of at most {m['ceiling']}"
                + (f"; GB/s by width {steps}" if steps else ""))
