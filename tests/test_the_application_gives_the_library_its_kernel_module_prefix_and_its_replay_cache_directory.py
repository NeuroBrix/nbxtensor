"""The library names no application's module prefix: the kernel modules it walks for autotuners are its own
(`nbxtensor.ops.`) and the prefixes an application registers, and the replay cache directory is the one the application
gives (`set_replay_cache_dir`), which wins over every default. The border (qa on 1217: autotune_cache.py named the
engine's `neurobrix.kernels.ops.` module prefix).

Run: python -m pytest tests/<this file>"""
import sys
import types

import pytest

pytest.importorskip("triton")
from triton.runtime.autotuner import Autotuner  # noqa: E402

from nbxtensor.ops import autotune_cache as AC  # noqa: E402


@pytest.fixture(autouse=True)
def _restore():
    prefixes, given = getattr(AC, "_KERNEL_MODULE_PREFIXES", None), getattr(AC, "_REPLAY_DIR", [None])[0]
    yield
    if prefixes is not None:
        AC._KERNEL_MODULE_PREFIXES = prefixes
    if hasattr(AC, "_REPLAY_DIR"):
        AC._REPLAY_DIR[0] = given


def _module_with_a_tuner(monkeypatch, name):
    tuner = object.__new__(Autotuner)           # no kernel needed: the walk only asks what an object is
    mod = types.ModuleType(name)
    mod.some_kernel = tuner
    monkeypatch.setitem(sys.modules, name, mod)
    return tuner


def _walked(tuner):
    return any(obj is tuner for _qual, obj in AC._autotuners())


def test_the_library_walks_only_its_own_prefix_until_an_application_registers_one(monkeypatch):
    own = _module_with_a_tuner(monkeypatch, "nbxtensor.ops.a_kernel_file_of_the_test")
    foreign = _module_with_a_tuner(monkeypatch, "someapp.kernels.ops.a_kernel_file")
    assert _walked(own) and not _walked(foreign)
    assert getattr(AC, "_KERNEL_MODULE_PREFIXES", None) == ("nbxtensor.ops.",)


def test_a_registered_prefix_is_walked_and_registering_twice_changes_nothing(monkeypatch):
    foreign = _module_with_a_tuner(monkeypatch, "someapp.kernels.ops.a_kernel_file")
    AC.register_kernel_module_prefix("someapp.kernels.ops")          # the dot is added
    AC.register_kernel_module_prefix("someapp.kernels.ops.")
    assert _walked(foreign)
    assert AC._KERNEL_MODULE_PREFIXES == ("nbxtensor.ops.", "someapp.kernels.ops.")


@pytest.mark.parametrize("bad", ["", ".", None, 3])
def test_a_prefix_that_is_not_one_is_refused(bad):
    with pytest.raises(ValueError):
        AC.register_kernel_module_prefix(bad)


def test_the_library_source_registers_no_application_prefix_by_default():
    assert AC._KERNEL_MODULE_PREFIXES == ("nbxtensor.ops.",)
    import inspect
    assert "kernels.ops." not in inspect.getsource(AC._autotuners)


def test_the_directory_the_application_gives_wins_over_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("NEUROBRIX_REPLAY_CACHE", str(tmp_path / "from_the_environment"))
    AC.set_replay_cache_dir(None)
    assert AC._dir() == str(tmp_path / "from_the_environment")
    AC.set_replay_cache_dir(tmp_path / "given")
    assert AC._dir() == str(tmp_path / "given")
    AC.set_replay_cache_dir(None)
    assert AC._dir() == str(tmp_path / "from_the_environment")


def test_a_callable_directory_is_read_when_used_not_when_given(monkeypatch, tmp_path):
    where = [tmp_path / "one"]
    AC.set_replay_cache_dir(lambda: where[0])
    assert AC._dir() == str(tmp_path / "one")
    where[0] = tmp_path / "two"
    assert AC._dir() == str(tmp_path / "two")


def test_a_directory_that_is_no_path_is_refused():
    with pytest.raises(TypeError):
        AC.set_replay_cache_dir(7)


def test_the_prefix_the_engine_once_had_in_the_source_is_walked_only_when_the_engine_registers_it(monkeypatch):
    """The module prefix the library carried for the engine (qa on 1217) walks nothing now: the registration is the way."""
    engines = _module_with_a_tuner(monkeypatch, "neurobrix.kernels.ops.a_kernel_file")
    assert not _walked(engines)
    AC.register_kernel_module_prefix("neurobrix.kernels.ops.")
    assert _walked(engines)
