"""The chunk rule on a card another process holds (C18, the Dell 23/11: a granite-3.1-1b Triton server, and a
PixArt-XL run, refused with "the carried trace sana-video-2b-720p.16g... peaks at 14559087916 live bytes, at or over
the room this card leaves (13505 MiB)" — the room was total less free, a neighbour's memory read as this card's
context, and a reference run that did not fit refused the user's model).

What this test does if the code is wrong:
1. a 16 GB-class card with 13 GB held by a neighbour is refused, or not given its class's chunk, though its device
   file states its runtime context (the room is the card's own: total less memory.runtime_context_mb);
2. where the reference run truly does not fit the room, the run is refused instead of taking the driver segments for
   THIS RUN ONLY (`ThisRunOnly(0)`), or the allocator then refuses for lack of a measure;
3. that run-only answer reads as "no rule" (None) to the engine, which would sweep and keep a chunk measured under a
   neighbour: nothing is kept — the run-only answer is not None, so no sweep is owed;
4. a class whose device file states null does not say that it reads it, or one that states nothing is not refused;
5. a device file states no memory.runtime_context_mb.
Device-free: the driver's figures are stand-ins.
Run: python -m pytest tests/test_a_card_a_neighbour_holds_gets_a_chunk_never_a_refusal.py
"""
from types import SimpleNamespace

import pytest

from nbxtensor.memory import chunk_rule, trace_replay
from nbxtensor.memory.allocator import DeviceAllocator

MiB = 1 << 20
TOTAL = 16160 * MiB


@pytest.fixture
def card(monkeypatch):
    """A V100-SXM2-16GB: 16 160 MiB; the process holds no block; a neighbour may hold `held` MiB."""
    state = SimpleNamespace(neighbour_mb=0, context_mb=306)
    monkeypatch.setattr(DeviceAllocator, "_cuda_live_bytes", {})
    monkeypatch.setattr(DeviceAllocator, "mem_get_info", staticmethod(
        lambda dev: (TOTAL - (state.context_mb + state.neighbour_mb) * MiB, TOTAL)))
    monkeypatch.setattr(trace_replay, "vmm_granularity", lambda dev: 2 * MiB)
    monkeypatch.setattr("nbxtensor.ops.autotune_certified.active_profile", lambda: ("nvidia", "volta"))
    state.file = {"memory": {"runtime_context_mb": 306}}
    monkeypatch.setattr("nbxtensor.ops._configs.active_vendor_profile", lambda: state.file)
    monkeypatch.setattr(chunk_rule, "_SAID", set())
    monkeypatch.setattr("nbxtensor.ops.wrappers.get_hardware_profile", lambda: SimpleNamespace(devices=[
        SimpleNamespace(index=0, memory_mb=16160, measured={}, architecture="volta")]))
    return state


def test_a_neighbour_holding_13_gb_changes_no_chunk(card, capsys):
    card.neighbour_mb = 13000
    assert chunk_rule.vmm_chunk_bytes(0, TOTAL) == 32 * MiB            # the card's own room, as with no neighbour
    assert DeviceAllocator._vmm_chunk_fact(0) == 32 * MiB
    assert "another process holds about 13000 MiB" in capsys.readouterr().err


def test_a_reference_run_that_does_not_fit_is_the_driver_segments_for_this_run_and_nothing_kept(card, capsys):
    card.file = {"memory": {"runtime_context_mb": 3000}}               # a room (13 160 MiB) under the reference run's 13 884
    got = chunk_rule.vmm_chunk_bytes(0, TOTAL)
    assert isinstance(got, chunk_rule.ThisRunOnly) and got == 0
    assert got is not None                                              # the engine owes no sweep: nothing measured, nothing kept
    assert DeviceAllocator._vmm_chunk_fact(0) == 0                      # the run goes on, on the driver segments
    err = capsys.readouterr().err
    assert "the rule does not answer on this card as it stands" in err and "nothing is kept" in err


def test_a_class_whose_file_states_null_reads_it_once_and_says_so_and_silence_is_refused(card, capsys):
    card.file = {"memory": {"runtime_context_mb": None}}
    assert chunk_rule.vmm_chunk_bytes(0, TOTAL) == 32 * MiB
    assert chunk_rule.vmm_chunk_bytes(0, TOTAL) == 32 * MiB
    assert capsys.readouterr().err.count("states memory.runtime_context_mb: null: read once now") == 1
    card.file = {"memory": {}}
    with pytest.raises(RuntimeError, match="states no memory.runtime_context_mb"):
        chunk_rule.vmm_chunk_bytes(0, TOTAL)


def test_every_device_file_states_its_runtime_context():
    import pathlib
    import yaml
    import nbxtensor
    for f in (pathlib.Path(nbxtensor.__file__).parent / "backend" / "devices").glob("*/*.yml"):
        assert "runtime_context_mb" in (yaml.safe_load(f.read_text()).get("memory") or {}), f


def test_the_volta_file_states_its_context():
    import pathlib
    import yaml
    import nbxtensor
    f = pathlib.Path(nbxtensor.__file__).parent / "backend" / "devices" / "nvidia" / "volta.yml"
    assert yaml.safe_load(f.read_text())["memory"]["runtime_context_mb"] == 306
