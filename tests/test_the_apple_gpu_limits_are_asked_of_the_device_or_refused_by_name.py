"""`nbxtensor.backend.metal_device.device_limits()`: the Apple GPU's working-set budget
(`recommendedMaxWorkingSetSize`) and largest single buffer (`maxBufferLength`), asked of the device by hardware
detection before any device is selected, without opening the library's Metal runtime.

What this test would catch if the code were wrong: a limit guessed (a fraction of the host's memory, a table) instead
of read, the two limits swapped, a working set above the machine's memory, the runtime opened by a mere question, a
missing Metal answered with numbers or with an anonymous error instead of `MetalUnavailableError` and its reason, or
the question importing heavy modules (Triton, the Objective-C bridge) on its way to a refusal — RED."""
import importlib.abc
import os
import sys

import pytest

from nbxtensor.backend import metal_device
from nbxtensor.backend.metal_device import MetalUnavailableError, device_limits
from nbxtensor import device as _device


class _NoMetal(importlib.abc.MetaPathFinder):
    """Refuses `Metal` (and its submodules): a host whose bindings are not installed."""

    def find_spec(self, name, path=None, target=None):
        """Raise for Metal; leave every other module to the next finder."""
        if name == "Metal" or name.startswith("Metal."):
            raise ModuleNotFoundError(f"No module named {name!r} (refused by the test)", name=name)
        return None


@pytest.mark.skipif(sys.platform != "darwin", reason="an Apple GPU answers only on macOS")
def test_this_mac_s_device_answers_both_limits_within_its_memory():
    opened_before = metal_device._RUNTIME
    limits = device_limits()
    assert set(limits) == {"working_set_bytes", "max_buffer_bytes"}
    assert limits["working_set_bytes"] > 0 and limits["max_buffer_bytes"] > 0
    machine = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    assert limits["working_set_bytes"] <= machine
    assert limits["max_buffer_bytes"] <= limits["working_set_bytes"]
    assert metal_device._RUNTIME is opened_before, "a question about the device opened the Metal runtime"


def test_without_metal_the_answer_is_a_named_refusal_and_nothing_heavy_is_imported(monkeypatch):
    monkeypatch.delitem(sys.modules, "Metal", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_NoMetal(), *sys.meta_path])
    opened_before = metal_device._RUNTIME
    before = set(sys.modules)
    with pytest.raises(MetalUnavailableError) as refused:
        device_limits()
    assert str(refused.value).startswith("ModuleNotFoundError: No module named 'Metal'"), str(refused.value)
    new = set(sys.modules) - before
    assert not new, f"the refusal imported {sorted(new)}"
    assert metal_device._RUNTIME is opened_before, "a refused question opened the Metal runtime"


def test_a_host_with_no_device_is_refused_by_that_reason(monkeypatch):
    fake = type(sys)("Metal")
    fake.MTLCreateSystemDefaultDevice = lambda: None
    monkeypatch.setitem(sys.modules, "Metal", fake)
    with pytest.raises(MetalUnavailableError, match=r"^MTLCreateSystemDefaultDevice\(\) returned no device$"):
        device_limits()


def _answer(call):
    """What a call answers: its value, or its refusal's type and text."""
    try:
        return ("value", call())
    except MetalUnavailableError as exc:
        return ("refused", str(exc))


def test_the_device_door_answers_with_the_one_reading_on_any_host():
    """An application asks the device door (`nbxtensor.device`), which answers from `device_limit` itself: the same
    value on a Mac, the same refusal by name on a host without Metal (qa 1277-1: the test had no macOS gate)."""
    assert _device.MetalUnavailableError is MetalUnavailableError
    for key in ("working_set_bytes", "max_buffer_bytes"):
        assert _answer(lambda: _device.apple_gpu_limit(key)) == _answer(lambda: metal_device.device_limit(key))


def test_a_limit_that_reads_zero_takes_only_itself(monkeypatch):
    """qa 1277 (said): a 0 maxBufferLength refused the working set too; each limit is judged on its own."""
    fake = type(sys)("Metal")
    dev = type("D", (), {"recommendedMaxWorkingSetSize": lambda self: 19069665280, "maxBufferLength": lambda self: 0})()
    fake.MTLCreateSystemDefaultDevice = lambda: dev
    monkeypatch.setitem(sys.modules, "Metal", fake)
    assert metal_device.device_limit("working_set_bytes") == 19069665280
    with pytest.raises(MetalUnavailableError, match=r"^maxBufferLength\(\) = 0$"):
        metal_device.device_limit("max_buffer_bytes")
