"""nbxtensor asks for triton only on the platforms PyPI publishes triton for, so it installs everywhere else. PyPI's
triton files are manylinux x86_64 and aarch64 wheels alone (triton 3.9.0 and every release before it, no sdist): a
plain "triton" made `pip install nbxtensor` ResolutionImpossible on a clean macOS venv (the Mac, 2026-11-06 23:03), and
"triton; platform_system != 'Darwin'" left it with no solution on native Windows (win_amd64, 2026-11-18), the engine
with it. The Metal Triton is built from source (Triton with TRITON_EXT_ENABLED=1 and triton-ext's AppleGPU backend);
where Triton is absent the library refuses by name with what to install (backend/metal_backend).

The requirement is read from the project's own metadata (pyproject.toml, the file the wheel's Requires-Dist is built
from) and its marker is evaluated with the environment of each platform, as pip evaluates it there.

What this test would do if the packaging were wrong: a triton that Windows or macOS must resolve from PyPI (red), a
Linux x86_64 or aarch64 install without triton (red)."""
from __future__ import annotations

from pathlib import Path

import pytest
from packaging.requirements import Requirement

try:
    import tomllib
except ModuleNotFoundError:          # python 3.10
    import tomli as tomllib

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

#: (sys_platform, platform_system, platform_machine) as each platform's python reports them
PLATFORMS_WITHOUT_TRITON = {
    "windows-amd64": ("win32", "Windows", "AMD64"),
    "macos-arm64": ("darwin", "Darwin", "arm64"),
    "macos-x86_64": ("darwin", "Darwin", "x86_64"),
}
PLATFORMS_WITH_TRITON = {
    "linux-x86_64": ("linux", "Linux", "x86_64"),
    "linux-aarch64": ("linux", "Linux", "aarch64"),
}


def _triton_requirement() -> Requirement:
    deps = [Requirement(d) for d in tomllib.loads(PYPROJECT.read_text())["project"]["dependencies"]]
    triton = [r for r in deps if r.name == "triton"]
    assert len(triton) == 1, [str(r) for r in triton]
    return triton[0]


def _asked_for_on(platform) -> bool:
    sys_platform, system, machine = platform
    marker = _triton_requirement().marker
    assert marker is not None, "a triton without a marker is asked for on every platform"
    return marker.evaluate({"sys_platform": sys_platform, "platform_system": system, "platform_machine": machine})


@pytest.mark.parametrize("name", sorted(PLATFORMS_WITHOUT_TRITON))
def test_triton_is_not_asked_for_where_pypi_publishes_none(name):
    assert not _asked_for_on(PLATFORMS_WITHOUT_TRITON[name]), (name, str(_triton_requirement()))


@pytest.mark.parametrize("name", sorted(PLATFORMS_WITH_TRITON))
def test_triton_is_asked_for_where_pypi_publishes_it(name):
    assert _asked_for_on(PLATFORMS_WITH_TRITON[name]), (name, str(_triton_requirement()))
