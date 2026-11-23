"""The expandable range's chunk by RULE per memory class (orders row 211): no first-launch sweep on a class the rule
knows; the rule's bound is the allocator's own cost of a too-large chunk, read from the carried trace.

What each test would do if the claim were wrong: no rule (the base) -> ImportError, red; a rule that extrapolated past
the measured grid or ignored the bound would answer 128 MiB, red; a rule that guessed for an unknown class would answer
a chunk where None is due, red; the allocator still refusing a ruled class without a measurement, red."""
import pytest
from types import SimpleNamespace

from nbxtensor.memory import chunk_rule
from nbxtensor.memory.pool import simulate
from nbxtensor.memory.trace_replay import load_trace, traces

MiB = 1 << 20
V100_16 = 16160 * MiB          # the cards the grid was measured on, as the driver states them
V100_32 = 32510 * MiB
CONTEXT = 306 * MiB            # the runtime's own context on those cards (stability_c0.txt:37)
ROOM = {V100_16: V100_16 - CONTEXT, V100_32: V100_32 - CONTEXT}


def rule(card, gran=2 * MiB, arch="volta"):
    return chunk_rule.rule_chunk_bytes(card, gran, ROOM.get(card, card - CONTEXT), arch)


def test_the_memory_class_is_the_whole_gib_above():
    assert chunk_rule._class_of(V100_16) == "16g" and chunk_rule._class_of(V100_32) == "32g"


def test_the_bound_is_what_a_reclaim_can_strand_within_the_slack():
    # carried 16 GB trace: 2 500 MiB over its live peak, 15 large blocks live there -> 33 chunks
    assert 75 * MiB < chunk_rule.bound_bytes(traces()[0].name, 16384 * MiB) < 76 * MiB     # the nominal class size
    # the card's own room (qa 1232-1): 16 160 MiB less its 306 MiB context -> 59.7 MiB, under 64
    assert 59 * MiB < chunk_rule.bound_bytes(traces()[0].name, ROOM[V100_16]) < 60 * MiB


def test_the_16g_card_gets_32_mib_on_its_own_room_and_the_32g_card_64():
    assert rule(V100_16) == 32 * MiB
    assert rule(V100_32) == 64 * MiB


def test_another_make_of_a_ruled_class_is_measured_not_given_the_v100_grid():
    assert rule(V100_16, arch="turing") is None and rule(V100_16, arch="") is None


def test_a_grid_without_off_is_refused_by_name(monkeypatch):
    monkeypatch.setattr(chunk_rule, "_classes", lambda: {"16g": {"architecture": "volta", "trace": traces()[0].name,
                                                                 "grid_seconds": {32: 1.9, 64: 1.8}}})
    with pytest.raises(RuntimeError, match="states no OFF"):
        rule(V100_16)


def test_no_room_over_the_peak_is_refused_by_name():
    chunk_rule.bound_bytes.cache_clear()
    with pytest.raises(RuntimeError, match="states no room for any chunk"):
        chunk_rule.bound_bytes(traces()[0].name, 1024 * MiB)


def test_a_faster_chunk_past_the_bound_is_never_the_answer(monkeypatch):
    grid = {0: 2.0, 32: 1.9, 64: 1.8, 128: 1.0}
    monkeypatch.setattr(chunk_rule, "_classes", lambda: {"16g": {"architecture": "volta", "trace": traces()[0].name,
                                                                 "grid_seconds": grid}})
    assert rule(V100_16, arch="volta") == 32 * MiB and chunk_rule.rule_chunk_bytes(
        V100_16, 2 * MiB, 16384 * MiB, "volta") == 64 * MiB


def test_a_chunk_slower_than_off_or_off_granularity_is_skipped(monkeypatch):
    grid = {0: 2.0, 32: 1.9, 64: 2.5}
    monkeypatch.setattr(chunk_rule, "_classes", lambda: {"16g": {"architecture": "volta", "trace": traces()[0].name,
                                                                 "grid_seconds": grid}})
    assert chunk_rule.rule_chunk_bytes(V100_16, 2 * MiB, 16384 * MiB, "volta") == 32 * MiB
    assert chunk_rule.rule_chunk_bytes(V100_16, 48 * MiB, 16384 * MiB, "volta") is None     # neither 32 nor 64 is whole units of 48 MiB


def test_a_class_with_no_entry_is_measured_not_guessed():
    assert rule(24 * 1024 * MiB) is None
    assert rule(V100_16, gran=0) is None


def test_the_rules_chunk_replays_the_carried_run_without_a_refusal():
    meta, ev = load_trace(traces()[0])
    kind = {"M": "alloc", "A": "arena", "F": "free"}
    sim = simulate([(kind[k], i, n) for k, i, n in ev], vmm_chunk=rule(V100_16), cap=ROOM[V100_16])
    assert sim.refusals == 0
