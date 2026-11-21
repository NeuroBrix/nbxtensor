"""The weight reader measures itself on the load it serves (`nbxtensor.io.reader`, the launch-time cure 2026-11-19).

No trial ahead of the load, nothing kept: the reads start one in flight and the width doubles while the rate measured
on those same reads rises, settling on the best at the first width that does not read faster; never past the ceiling
the caller decides. What each test would do if the code were wrong: a store that saturates at four readers left short of four (red); a width beyond the ceiling (red); a piece dropped, read twice or its failure swallowed (red);
bytes read wrong through the split (red); a default ceiling taken when none is given (red)."""
import os
import threading

import numpy as np
import pytest

from nbxtensor.io.reader import Growth, Reader


def test_the_width_doubles_while_the_rate_rises_and_settles_on_the_best():
    g = Growth(64)
    for w, rate in ((1, 1.0), (2, 1.9), (4, 3.5), (8, 3.4)):
        assert g.width == w
        g.note(w, int(rate * 1e9), 1.0)
    assert g.settled and g.width == 4 and sorted(g.rates) == [1, 2, 4, 8]


def test_the_width_never_passes_the_ceiling():
    g = Growth(6)
    for w in (1, 2, 4, 6):
        assert g.width == w
        g.note(w, w * 10**9, 1.0)
    assert g.settled and g.width == 6


def test_a_measure_of_another_width_is_ignored():
    g = Growth(8)
    g.note(4, 10**9, 1.0)
    assert g.width == 1 and not g.rates


@pytest.mark.parametrize("bad", [0, -1, None, 2.0])
def test_no_ceiling_no_reader(bad):
    with pytest.raises(ValueError, match="ZERO FALLBACK"):
        Growth(bad)


class _Clock:
    """A clock the reads advance (deterministic: no sleep, no scheduler): a read of `units` at width w takes
    units / min(w, saturation) — a store that gives no more past `saturation` readers."""

    def __init__(self, saturation):
        self.t, self.saturation, self.lock = 0.0, saturation, threading.Lock()

    def __call__(self):
        return self.t

    def take(self, units, width):
        with self.lock:
            self.t += units / min(width, self.saturation)


def test_a_width_kept_by_an_earlier_load_starts_there_and_gives_its_spread(monkeypatch):
    """The next launch starts from the kept figure, never from one again; its rounds measure the figure's spread."""
    import nbxtensor.io.reader as RD
    clock = _Clock(4)
    monkeypatch.setattr(RD.time, "perf_counter", clock)
    r = Reader(32, start=4)

    def read(i):
        clock.take(1.0 + (i % 3) / 10, r.width)
        return i
    list(r.map(read, range(24), lambda i: 1 << 20))
    m = r.measured()
    assert m["start"] == 4 and m["workers"] == 4 and m["settled"] and list(m["draws"]) == [4]
    lo, hi = m["spread"]
    assert 0 < lo < hi and "started at 4" in r.said() and "round(s) at that width" in r.said()
    assert Reader(6, start=16).width == 6                      # never past the ceiling


def test_a_failing_piece_fails_the_load():
    def fn(i):
        if i == 3:
            raise OSError("short read")
        return i
    with pytest.raises(OSError, match="short read"):
        list(Reader(4).map(fn, range(10), lambda i: 1))


def test_the_split_read_returns_the_file_s_bytes(tmp_path):
    data = np.random.default_rng(7).integers(0, 255, size=(9 << 20) + 123, dtype=np.uint8)
    p = tmp_path / "w.bin"
    p.write_bytes(data.tobytes())
    r = Reader(8)
    fd = os.open(p, os.O_RDONLY)
    try:
        for off, n in ((0, 4 << 20), (4 << 20, 5 << 20), ((9 << 20) - 7, 130)):
            assert np.array_equal(r.read_bytes(fd, off, n), data[off:off + n])
    finally:
        os.close(fd)
    m = r.measured()
    assert m["bytes"] == (4 << 20) + (5 << 20) + 130 and m["workers"] <= 8
    assert "parallel read(s)" in r.said()


def test_a_split_read_is_judged_on_rounds_of_reads_the_split_cuts(monkeypatch):
    """A tensor too small for the split at the ceiling is latency, not a width's rate: thousands of 4 KB norms never
    move the width; the large reads, a round of as many as the width, grow it until the store gives no more. (The first
    smoke on a real model judged width 1 on a 4 KB read at 0.001 GB/s.)"""
    import nbxtensor.io.reader as RD
    from nbxtensor.io import safetensors as SF
    clock = _Clock(4)
    monkeypatch.setattr(RD.time, "perf_counter", clock)

    def fake_read(fd, offset, n, io):
        clock.take(2e-4 * io[0] + n / 2e9, io[0])          # latency, then a store that saturates at four readers
        return np.empty(n, dtype=np.uint8)
    monkeypatch.setattr(SF, "read_bytes", fake_read)
    r = Reader(16)
    for _ in range(300):
        r.read_bytes(0, 0, 4096)
    assert r.width == 1 and not r.measured()["rates"]
    for _ in range(64):
        r.read_bytes(0, 0, 4096)
        r.read_bytes(0, 0, 8 << 20)
    m = r.measured()
    assert m["settled"] and m["workers"] == 4, m



def _shared_store_load(reader, sizes, n, per_read=1.0, total=4.0):
    """`reader.map` over `n` reads on a store whose reads in flight SHARE its bandwidth: each moves at
    min(per_read, total / in flight) bytes a second, so `w` readers move min(w * per_read, total). An event clock
    (deterministic, no thread runs a read): a read ends at the next completion its share allows, and its function runs
    at its end, as a read's thread returns then."""
    import itertools
    import nbxtensor.io.reader as RD
    t, live, seq = [0.0], {}, itertools.count()

    class Read:
        def __init__(self, call):
            self.call, self.seq = call, next(seq)

        def result(self):
            return self.value

        def cancel(self):
            pass

    class Pool:
        def __init__(self, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def submit(self, fn, item):
            f = Read(lambda: fn(item))
            live[f] = float(sizes(item))
            return f

    def wait(fs, return_when=None):
        rate = min(per_read, total / len(live))
        f = min(live, key=lambda g: (live[g], g.seq))
        dt = live[f] / rate
        t[0] += dt
        for g in live:
            live[g] -= rate * dt
        del live[f]
        f.value = f.call()
        return {f}, set(fs) - {f}
    return RD, Pool, wait, (lambda: t[0]), lambda: list(reader.map(lambda i: i, range(n), sizes))


@pytest.mark.parametrize("sizes", [lambda i: 1.0, lambda i: 1.0 if i % 2 else 3.0, lambda i: 1.0 + i % 3],
                         ids=["equal", "one-and-three", "one-two-three"])
@pytest.mark.parametrize("ceiling", [32, 4], ids=["below-the-ceiling", "at-the-ceiling"])
def test_each_width_is_judged_on_the_bytes_its_reads_moved(monkeypatch, sizes, ceiling):
    """qa on 1196 and 1197: on a store whose reads in flight share its bandwidth (four readers fill it), every width
    is judged at the rate it reads, min(width, 4), and the reader settles at four, whether the reads are equal or not,
    below the ceiling and when the ceiling is the width. Red before: counting whole completions overrated a wider width
    and with unequal reads settled at one (1196); counting only the reads started in a round underrated every width by
    up to half (1197)."""
    r = Reader(ceiling)
    RD, Pool, wait, clock, load = _shared_store_load(r, sizes, 240)
    monkeypatch.setattr(RD, "ThreadPoolExecutor", Pool)
    monkeypatch.setattr(RD, "wait", wait)
    monkeypatch.setattr(RD.time, "perf_counter", clock)
    assert sorted(load()) == list(range(240))
    m = r.measured()
    for w, rate in m["rates"].items():
        assert abs(rate - min(w, 4)) <= 0.02 * min(w, 4), (w, rate, m["rates"])
    assert m["workers"] == 4, m
    assert m["bytes"] == sum(int(sizes(i)) for i in range(240)), "every read's bytes count in the load's total"

def test_a_closed_reader_starts_no_read():
    r = Reader(2)
    r.close()
    with pytest.raises(RuntimeError):
        r.io[1].submit(int)


def _on_a_store(monkeypatch, reader, sizes, n, per_read, total):
    RD, Pool, wait, clock, load = _shared_store_load(reader, sizes, n, per_read, total)
    monkeypatch.setattr(RD, "ThreadPoolExecutor", Pool)
    monkeypatch.setattr(RD, "wait", wait)
    monkeypatch.setattr(RD.time, "perf_counter", clock)
    assert sorted(load()) == list(range(n))
    return reader.measured()


@pytest.mark.parametrize("store,n", [(8, 8), (8, 9), (8, 22), (8, 23), (16, 50), (16, 51)])
def test_a_window_the_end_of_the_load_reaches_before_it_closes_is_not_judged(monkeypatch, store, n):
    """qa on 1200 (the drain): a window closed at the width-th read ended inside it, whatever was in flight; once the
    load ran out of pieces, an open window measured the load's tail, its width read low and the reader SETTLED a
    doubling or two below the store (store 8, ceiling 8: 8 and 9 pieces settled at 2, 22 and 23 at 4; store 16: 50 and
    51 at 8), a wrong figure the engine keeps. Now such a window only counts in the bytes: every width judged reads at
    the store's rate for it, min(width, store); a load too short to conclude ends unsettled and says the best width it
    measured, never the next one to try."""
    r = Reader(store)
    m = _on_a_store(monkeypatch, r, lambda i: 1.0, n, 1.0, float(store))
    assert m["rates"], m
    for w, ds in m["draws"].items():
        for d in ds:
            assert abs(d - min(w, store)) <= 0.02 * min(w, store), (w, ds, m)
    best = max(m["rates"], key=m["rates"].get)
    assert m["workers"] == best, m
    assert m["settled"] is False or best == store, m
    assert m["spread"] == (min(m["draws"][best]), max(m["draws"][best])), m
    assert m["bytes"] == n


@pytest.mark.parametrize("store", [4, 6, 8])
@pytest.mark.parametrize("n", [240, 241, 247])
def test_a_settled_widths_spread_holds_no_window_of_the_loads_tail(monkeypatch, store, n):
    """qa on 1200: on a long load, the one draw of a settled width more than 2 % off its rate was the load's LAST window
    (from -33 %, 2.67 on a 4-unit store, to +5 %), the tail measured instead of the width. No draw of it now."""
    r = Reader(32)
    m = _on_a_store(monkeypatch, r, lambda i: 1.0 if i % 2 else 3.0, n, 1.0, float(store))
    reach = next(w for w in (1, 2, 4, 8, 16, 32) if w >= store)    # the first doubling that reads at the store's rate
    assert m["settled"] and m["workers"] == reach, m
    for w, ds in m["draws"].items():
        for d in ds:
            assert abs(d - min(w, store)) <= 0.02 * min(w, store), (w, ds, m)
