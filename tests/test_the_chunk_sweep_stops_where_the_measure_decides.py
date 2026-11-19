"""The chunk probe's sweep stops where its own measure has decided, and decides exactly what the full sweep decides.

The first launch on the 4-card rack (2026-11-19, py-spy) spent 1024 s in 48 replay children, 24 per card class: every
candidate replayed three times to the end, a 2 MiB replay 75-85 s against an R16 OFF bound near 4 s. Now a candidate's
replays stop once more than half lie on one side of OFF's bound (the median is then settled), and with an odd count
each replay stops at the bound itself. What each test would do if the code were wrong: a decision that differs from
the full sweep on any draw (red — the chunk would move), the recorded rack's chunk moving (red), a stopped sweep
running as many replays as the full one (red), a replay stopped short of the bound or not at all (red)."""
import itertools
import random
import statistics

import pytest

from nbxtensor.memory import trace_replay as T

GRAN = 2 << 20
KS = (1, 2, 4, 8, 16, 32, 64)


def _full_sweep(off, draws, reps):
    """The sweep as it was (nbxtensor main 02b8fae): every replay to its end, the first candidate whose median of
    `reps` is within OFF's median plus OFF's spread."""
    bound = statistics.median(off) + (max(off) - statistics.median(off))
    for k in KS:
        if statistics.median(draws[GRAN * k][:reps]) <= bound:
            return GRAN * k
    return None


class _Child:
    """Replays from a table: OFF's draws, then each candidate's in order; a replay told to stop past `stop_after`
    reports the seconds it had reached there (just past the bound) as replay_trace does."""

    def __init__(self, off, draws):
        self.off, self.draws, self.calls, self.stops = list(off), {c: list(v) for c, v in draws.items()}, 0, []

    def __call__(self, trace, device, chunk, stop_after=None):
        self.calls += 1
        self.stops.append(stop_after)
        t = self.off.pop(0) if chunk == 0 else self.draws[chunk].pop(0)
        if stop_after is not None and t > stop_after:
            return {"seconds": stop_after * (1 + 1e-9) + 1e-12, "beyond": True}
        return {"seconds": t}


@pytest.fixture(autouse=True)
def _no_device(monkeypatch):
    monkeypatch.setattr(T, "_give_back", lambda device: None)


def _sweep(off, draws, reps):
    child = _Child(off, draws)
    chunk, _times = T.chunk_by_replay(0, GRAN, KS, reps, trace="t.tsv", child=child)
    return chunk, child


@pytest.mark.parametrize("reps", [1, 2, 3, 4, 5])
def test_the_stopped_sweep_decides_what_the_full_sweep_decides(reps):
    rng = random.Random(20261119 + reps)
    for _ in range(3000):
        off = [rng.uniform(1.0, 4.0) for _ in range(reps)]
        draws = {GRAN * k: [rng.choice([rng.uniform(0.5, 6.0), rng.uniform(1.0, 90.0)]) for _ in range(reps)]
                 for k in KS}
        chunk, child = _sweep(off, draws, reps)
        assert chunk == _full_sweep(off, draws, reps), (off, draws)


def test_ties_at_the_bound_decide_the_same():
    """Seconds equal to the bound count as within it, both ways."""
    for reps in (2, 3, 4):
        off = [2.0] * reps
        for pattern in itertools.product([1.0, 2.0, 3.0], repeat=reps):
            draws = {GRAN * k: list(pattern) if k == 1 else [1.0] * reps for k in KS}
            assert _sweep(off, draws, reps)[0] == _full_sweep(off, draws, reps)


# the child seconds the rack's first launch took per candidate (py-spy, 2026-11-19: window/fl4b.txt, three replays
# each; device 0 is a 16 GB card, device 2 a 32 GB one) — wall seconds of each child, the order they ran in
RACK = {
    "16g": {0: [2.6, 3.2, 4.2], 2: [77.8, 84.7, 84.9], 4: [38.3, 42.2, 44.8], 8: [19.4, 23.8, 24.2],
            16: [11.9, 12.6, 12.9], 32: [6.8, 7.2, 8.8], 64: [4.9, 5.0, 5.5], 128: [3.2, 3.5, 4.2]},
    "32g": {0: [2.4, 2.8, 3.0], 2: [74.5, 74.8, 75.0], 4: [38.3, 39.0, 39.5], 8: [18.6, 20.4, 22.1],
            16: [11.8, 11.9, 12.7], 32: [6.3, 6.5, 8.4], 64: [3.1, 3.4, 4.2], 128: [2.1, 2.5, 3.8]},
}


@pytest.mark.parametrize("card", sorted(RACK))
def test_the_rack_s_recorded_sweep_keeps_its_chunk_with_fewer_seconds(card):
    rec = RACK[card]
    off = rec[0]
    draws = {mb << 20: v for mb, v in rec.items() if mb}
    chunk, child = _sweep(off, draws, 3)
    assert chunk == _full_sweep(off, draws, 3)
    full_s = sum(sum(v) for v in rec.values())
    bound = statistics.median(off) + (max(off) - statistics.median(off))
    # what the stopped sweep spends: OFF in full, each candidate replay up to the bound at most
    spent = sum(off) + sum(min(t, bound) for c in draws for t in draws[c][:3]) if chunk is None else None
    assert child.calls < 3 * (1 + len(KS)), child.calls
    assert all(s == pytest.approx(bound) for s in child.stops[3:])
    if spent is not None:
        assert spent < full_s / 5, (spent, full_s)


def test_an_even_count_never_stops_a_replay():
    """With an even count a half-and-half split needs both middle values: no replay is stopped at the bound."""
    off, draws = [2.0, 2.0], {GRAN * k: [1.0, 9.0] for k in KS}
    chunk, child = _sweep(off, draws, 2)
    assert all(s is None for s in child.stops)
    assert chunk == _full_sweep(off, draws, 2)


def test_a_replay_stops_once_past_its_bound(monkeypatch):
    """replay_trace stops at the first event past `stop_after`, its seconds past the bound."""
    import nbxtensor.memory.allocator as A

    class FakeDA:
        _POOL = type("P", (), {"stats": {}})()
        _cuda_peak_bytes = {}
        mallocs = 0

        @staticmethod
        def set_device(d): pass

        @staticmethod
        def use_expandable(d, c, range_bytes=None): pass

        @staticmethod
        def set_limit(n, d=None): pass

        @classmethod
        def malloc_cuda(cls, n, d, pooled=True):
            cls.mallocs += 1
            return cls.mallocs

        @staticmethod
        def free_cuda(p): pass

        @staticmethod
        def device_synchronize(d): pass

    clock = itertools.count()
    monkeypatch.setattr(A, "DeviceAllocator", FakeDA)
    monkeypatch.setattr(T.time, "perf_counter", lambda: float(next(clock)))
    events = [("M", i, 1 << 20) for i in range(100)]
    r = T.replay_trace(events, 0, GRAN, 1 << 30, stop_after=10.0)
    assert r["beyond"] and r["seconds"] > 10.0 and FakeDA.mallocs < 100
    FakeDA.mallocs = 0
    r = T.replay_trace(events, 0, GRAN, 1 << 30)
    assert "beyond" not in r and FakeDA.mallocs == 100
