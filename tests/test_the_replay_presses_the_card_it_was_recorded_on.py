"""Why the chunk probe replays a recorded run under its card's size (`card_mb`), never under the record's own peak.

The question (launch-time cure, 2026-11-19): the replay maps chunks up to the whole 16 GB card (30 722 map calls at a
2 MiB chunk on the carried SANA-Video trace) — does the record bound that itself, by its peak live bytes? It does not.
The run held MORE from the driver than its peak live bytes (the large class fragments: 18.9 GB uncapped against a
13.9 GB live peak), so under its live peak the same events are REFUSED (10 refusals: the replay is no longer the run)
and map more (32 038 calls), not fewer; and with no cap at all nothing is ever reclaimed, which is the very cost the
chunk is chosen on (the module's doc: "a run's mapping cost is what its refusals' reclaims map again"). The card the
trace was recorded on is the one bound the record states that keeps the run the run.

Card-less: the allocator's own rules replayed (`pool.simulate`, its expandable range and its reclaim). What each test
would do if the claim were wrong: the live peak would replay with no refusal and fewer maps (red), the card-sized replay
would refuse (red), the uncapped replay would reclaim (red)."""
import pytest

from nbxtensor.memory.pool import simulate
from nbxtensor.memory.trace_replay import load_trace, traces

MB = 1 << 20
KIND = {"M": "alloc", "A": "arena", "F": "free"}


@pytest.fixture(scope="module")
def record():
    meta, ev = load_trace(traces()[0])
    events = [(KIND[k], i, n) for k, i, n in ev]
    live = peak = 0
    sizes = {}
    for k, i, n in ev:
        if k == "F":
            live -= sizes.pop(i, 0)
        else:
            sizes[i] = n
            live += n
            peak = max(peak, live)
    return int(meta["card_mb"]) * MB, peak, events


@pytest.mark.parametrize("chunk_mb", [2, 32, 128])
def test_the_live_peak_is_no_bound_the_card_is(record, chunk_mb):
    card, peak, events = record
    on_card = simulate(events, vmm_chunk=chunk_mb * MB, cap=card)
    on_peak = simulate(events, vmm_chunk=chunk_mb * MB, cap=peak)
    free = simulate(events, vmm_chunk=chunk_mb * MB, cap=None)
    assert free.peak_bytes > card > peak                      # the run holds past its live peak, and past the card
    assert on_card.refusals == 0 and on_card.unmaps > 0       # the card reclaims, and the run still runs
    assert on_peak.refusals > 0 and on_peak.maps > on_card.maps
    assert free.unmaps == 0                                   # no card, no reclaim: not the cost being measured


def test_the_map_calls_of_the_carried_record(record):
    """The figures the record states (2 MiB: the 75-85 s replay on the rack)."""
    card, peak, events = record
    got = {mb: simulate(events, vmm_chunk=mb * MB, cap=card).maps for mb in (2, 4, 8, 16, 32, 64, 128)}
    assert got[2] == 30722 and got[128] == 499, got
    assert all(got[a] > got[b] for a, b in zip(sorted(got), sorted(got)[1:]))
