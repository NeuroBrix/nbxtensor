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


def test_a_store_that_saturates_at_four_readers_is_read_four_wide(monkeypatch):
    import nbxtensor.io.reader as RD
    clock = _Clock(4)
    monkeypatch.setattr(RD.time, "perf_counter", clock)
    r = Reader(32)
    items = list(range(60))

    def read(i):
        clock.take(1.0, r.width)
        return i
    out = list(r.map(read, items, lambda i: 1 << 20))
    assert sorted(out) == items
    m = r.measured()
    assert m["settled"] and m["workers"] == 4 and sorted(m["rates"]) == [1, 2, 4, 8], m
    assert m["bytes"] == 60 << 20 and m["spread"] is not None


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
