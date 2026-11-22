"""The library declares the lowest Triton it runs on, and that is the lowest Triton that has what its launch path calls
without a fallback.

The launcher binds a kernel with `JITFunction._pack_args` (first released in Triton 3.5.0) and picks the source language
with `JITFunction.is_gluon` (3.4.0); a Triton below 3.5.0 failed on an AttributeError at the first launch while the
metadata allowed it (qa on 1203). The floor is read from pyproject.toml, not repeated here.

What this test would do if the code were wrong: no floor in the metadata (red), a floor the installed Triton does not
reach (red), a floor below the release that introduced a method the launch path calls (red), a README that states
another floor (red)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# The first release that has `JITFunction._pack_args`: read from the Triton tags v3.4.0 (absent) and v3.5.0 (present).
_PACK_ARGS_FIRST = (3, 5, 0)


def _triton_requirement() -> str:
    deps = re.search(r'^dependencies\s*=\s*\[(.*?)\]\s*$', (ROOT / "pyproject.toml").read_text(), re.M | re.S).group(1)
    found = [d for d in re.findall(r'"([^"]+)"', deps) if re.match(r"triton\b", d)]
    assert len(found) == 1, found
    return found[0]


def _floor() -> tuple:
    m = re.match(r"triton\s*>=\s*(\d+)\.(\d+)\.(\d+)", _triton_requirement())
    assert m, f"the Triton dependency declares no lower bound: {_triton_requirement()!r}"
    return tuple(int(x) for x in m.groups())


def _version(text: str) -> tuple:
    return tuple(int(x) for x in re.match(r"(\d+)\.(\d+)\.(\d+)", text).groups())


def test_the_metadata_declares_a_floor():
    assert _floor()


def test_the_floor_is_not_below_what_the_launch_path_calls():
    assert _floor() >= _PACK_ARGS_FIRST, (_floor(), _PACK_ARGS_FIRST)


def test_the_installed_triton_reaches_the_floor_and_has_what_the_launcher_calls():
    triton = pytest.importorskip("triton")
    from triton.runtime.jit import JITFunction
    assert _version(triton.__version__) >= _floor(), (triton.__version__, _floor())
    for name in ("_pack_args", "is_gluon"):
        assert hasattr(JITFunction, name), name


def test_the_readme_states_the_declared_floor():
    floor = ".".join(str(x) for x in _floor())
    assert f"Triton {floor} or later" in (ROOT / "README.md").read_text()
