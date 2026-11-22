"""A Triton below the floor the library declares is refused by name when the launcher first prepares a launch, on any
platform: the dependency's floor (`triton>=3.5.0` in pyproject.toml) binds only where pip installs Triton, so on a Mac or
with a source-built Triton a Triton below it failed on an AttributeError at the first launch (qa on 1217).

The floor is read from the declaration (pyproject.toml, or the installed metadata as packaged) and never repeated here or
in the code. The installed version is monkeypatched below it, at it, and above it, in the spellings a source build gives.

What this test does if the code is wrong: no refusal below the floor (red), the refusal without the installed version or
the declared one (red), a local version `3.8.0+git4a15f415.nbx1` taken for something it is not (red), a pre-release
accepted as the release it precedes (red), a refusal at the floor itself (red), a floor spelled a second time in the code
(the declaration is edited by the test and the refusal must follow it).

Run: python -m pytest tests/<this file>"""
import re
from pathlib import Path

import pytest

pytest.importorskip("triton")
from nbxtensor.backend import launcher as L  # noqa: E402
from nbxtensor.backend import triton_floor as TF  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _declared():
    text = (ROOT / "pyproject.toml").read_text()
    return tuple(int(x) for x in re.search(r'"triton>=(\d+)\.(\d+)\.(\d+)', text).groups())


@pytest.fixture(autouse=True)
def _fresh():
    TF.reset()
    yield
    TF.reset()


def _text(parts, suffix=""):
    return ".".join(str(p) for p in parts) + suffix


def _below():
    a, b, c = _declared()
    return (a, b - 1, 9) if b else (a - 1, 9, 9)


def test_the_floor_the_code_checks_is_the_one_the_declaration_carries():
    assert TF._declaration()[0] == _declared()


@pytest.mark.parametrize("suffix", ["", "+git4a15f415.nbx1", ".dev20250101", "rc1"])
def test_a_triton_below_the_floor_is_refused_by_name_with_what_is_installed_and_declared(suffix):
    installed = _text(_below(), suffix)
    with pytest.raises(TF.TritonBelowFloor) as err:
        TF.ensure(installed)
    msg = str(err.value)
    assert installed in msg and _text(_declared()) in msg and "pyproject" in msg


@pytest.mark.parametrize("suffix", ["", "+git4a15f415.nbx1", ".post1"])
def test_the_floor_itself_and_a_local_version_of_it_or_above_are_accepted(suffix):
    TF.ensure(_text(_declared(), suffix))
    a, b, c = _declared()
    TF.ensure(_text((a, b + 3, 0), "+git4a15f415.nbx1"))
    TF.ensure("3.8.0+git4a15f415.nbx1")


@pytest.mark.parametrize("suffix", [".dev20250101", "rc1", "a2"])
def test_a_pre_release_of_the_floor_is_below_it(suffix):
    with pytest.raises(TF.TritonBelowFloor):
        TF.ensure(_text(_declared(), suffix))


def test_the_launcher_refuses_before_it_touches_the_kernel_and_says_the_same_name(monkeypatch):
    import triton
    monkeypatch.setattr(triton, "__version__", _text(_below(), "+git0000000.nbx1"))

    def touched(kernel):                 # what a launch reaches first after the floor: reaching it means no floor check
        raise AssertionError("the launcher reached the kernel before checking the Triton floor")
    monkeypatch.setattr(L, "_binder", touched)   # card-less too: no NoGpuRuntime to turn a missing check into a skip
    with pytest.raises(TF.TritonBelowFloor, match="needs Triton"):
        L.prepare(None, (), {})          # no kernel is needed: the refusal comes first
    with pytest.raises(TF.TritonBelowFloor):      # not remembered as fine: every launch is refused, none falls back
        L.prepare(None, (), {})


def test_the_installed_triton_is_checked_once_and_a_launch_then_pays_a_flag(monkeypatch):
    import triton
    monkeypatch.setattr(triton, "__version__", "3.8.0+git4a15f415.nbx1")
    TF.ensure()
    assert TF._OK[0]
    monkeypatch.setattr(triton, "__version__", _text(_below()))
    TF.ensure()                           # answered once: no second read of the version


def test_the_refusal_follows_the_declaration_not_a_number_in_the_code(monkeypatch):
    monkeypatch.setattr(TF, "_declaration", lambda: ((99, 0, 0), "a declaration of the test"))
    with pytest.raises(TF.TritonBelowFloor, match=r"99\.0\.0"):
        TF.ensure("3.8.0+git4a15f415.nbx1")
    monkeypatch.setattr(TF, "_declaration", lambda: (None, ""))
    TF.ensure("0.0.1")                    # nothing declared, nothing to check against


def test_the_floor_is_read_from_the_installed_metadata_when_no_pyproject_is_beside_the_source(monkeypatch, tmp_path):
    monkeypatch.setattr(TF, "__file__", str(tmp_path / "a" / "b" / "c" / "d" / "triton_floor.py"))
    from importlib import metadata
    monkeypatch.setattr(metadata, "requires",
                        lambda name: ["numpy", "triton>=3.5.0; sys_platform == 'linux' and platform_machine == 'x86_64'"])
    assert TF._declaration() == ((3, 5, 0), "the installed nbxtensor's metadata")
