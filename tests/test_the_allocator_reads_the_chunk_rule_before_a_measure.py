"""The allocator reads the chunk RULE (`memory.chunk_rule`) for a memory class it knows, where the hardware profile
states no first-launch measure: no sweep is owed on that class. Red on the base: the allocator refuses by name
(ZERO FALLBACK, no measured `vmm_chunk_bytes`) — the sweep was the only source. A class with no rule keeps its measure,
or is refused, as before."""
import pytest
from types import SimpleNamespace

from nbxtensor.memory import trace_replay
from nbxtensor.memory.allocator import DeviceAllocator

MiB = 1 << 20


def _profile(mb, measured):
    return SimpleNamespace(devices=[SimpleNamespace(index=0, memory_mb=mb, measured=measured)])


def test_the_allocator_reads_the_rule_where_the_profile_states_no_measure(monkeypatch):
    monkeypatch.setattr(trace_replay, "vmm_granularity", lambda dev: 2 * MiB)
    monkeypatch.setattr("nbxtensor.ops.wrappers.get_hardware_profile", lambda: _profile(16160, {}))
    assert DeviceAllocator._vmm_chunk_fact(0) == 64 * MiB
    # the rule is the answer for its class, over an older sweep's figure
    monkeypatch.setattr("nbxtensor.ops.wrappers.get_hardware_profile",
                        lambda: _profile(16160, {"vmm_chunk_bytes": 16 * MiB}))
    assert DeviceAllocator._vmm_chunk_fact(0) == 64 * MiB


def test_a_class_without_rule_keeps_its_measure_or_is_refused(monkeypatch):
    monkeypatch.setattr(trace_replay, "vmm_granularity", lambda dev: 2 * MiB)
    monkeypatch.setattr("nbxtensor.ops.wrappers.get_hardware_profile",
                        lambda: _profile(24564, {"vmm_chunk_bytes": 32 * MiB}))
    assert DeviceAllocator._vmm_chunk_fact(0) == 32 * MiB
    monkeypatch.setattr("nbxtensor.ops.wrappers.get_hardware_profile", lambda: _profile(24564, {}))
    with pytest.raises(RuntimeError, match="ZERO FALLBACK"):
        DeviceAllocator._vmm_chunk_fact(0)
