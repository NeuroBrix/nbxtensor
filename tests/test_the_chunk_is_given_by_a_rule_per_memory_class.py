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


def test_the_memory_class_is_the_whole_gib_above():
    assert chunk_rule._class_of(V100_16) == "16g" and chunk_rule._class_of(V100_32) == "32g"


def test_the_bound_is_what_a_reclaim_can_strand_within_the_slack():
    # carried 16 GB trace: 2 500 MiB over its live peak, 15 large blocks live there -> 33 chunks
    b = chunk_rule.bound_bytes(traces()[0].name)
    assert 75 * MiB < b < 76 * MiB


@pytest.mark.parametrize("card", [V100_16, V100_32])
def test_both_measured_classes_get_64_mib(card):
    assert chunk_rule.rule_chunk_bytes(card, 2 * MiB) == 64 * MiB


def test_a_faster_chunk_past_the_bound_is_never_the_answer(monkeypatch):
    grid = {0: 2.0, 32: 1.9, 64: 1.8, 128: 1.0}
    monkeypatch.setattr(chunk_rule, "_classes", lambda: {"16g": {"trace": traces()[0].name, "grid_seconds": grid}})
    assert chunk_rule.rule_chunk_bytes(V100_16, 2 * MiB) == 64 * MiB


def test_a_chunk_slower_than_off_or_off_granularity_is_skipped(monkeypatch):
    grid = {0: 2.0, 32: 1.9, 64: 2.5}
    monkeypatch.setattr(chunk_rule, "_classes", lambda: {"16g": {"trace": traces()[0].name, "grid_seconds": grid}})
    assert chunk_rule.rule_chunk_bytes(V100_16, 2 * MiB) == 32 * MiB
    assert chunk_rule.rule_chunk_bytes(V100_16, 48 * MiB) is None     # neither 32 nor 64 is whole units of 48 MiB


def test_a_class_with_no_entry_is_measured_not_guessed():
    assert chunk_rule.rule_chunk_bytes(24 * 1024 * MiB, 2 * MiB) is None
    assert chunk_rule.rule_chunk_bytes(V100_16, 0) is None


def test_the_rules_chunk_replays_the_carried_run_without_a_refusal():
    meta, ev = load_trace(traces()[0])
    kind = {"M": "alloc", "A": "arena", "F": "free"}
    sim = simulate([(kind[k], i, n) for k, i, n in ev], vmm_chunk=chunk_rule.rule_chunk_bytes(V100_16, 2 * MiB),
                   cap=int(meta["card_mb"]) * MiB)
    assert sim.refusals == 0 and sim.maps == 989
