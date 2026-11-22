"""The Triton floor the library declares, checked once against the Triton that is installed.

The floor is declared in ONE place, the library's own dependency line (`triton>=X.Y.Z` in pyproject.toml), and read from
that declaration: from the pyproject.toml beside the source when the library runs from a source tree, else from the
installed distribution's metadata (`importlib.metadata.requires('nbxtensor')`, the same line as packaged). No second
copy of the number lives in the code.

Why it is read here and not left to pip: the dependency binds only where pip installs Triton (Linux x86_64 / aarch64).
On a Mac, where Triton is built from source, or with a source-built Triton anywhere, nothing stopped a Triton below the
floor, which then failed on an AttributeError at its first launch (the launch path calls Triton names a Triton below the floor does not have).
`ensure()` refuses by name instead: no fallback, no version tolerance, one refusal that says what is installed, what is
declared and where. A tree whose declaration cannot be found (no pyproject beside the source, no metadata carrying a
floor) declares no floor and is not checked: there is nothing to check against.

A version is read as the release it names: a local version (`3.8.0+git4a15f415.nbx1`, how a source build names itself)
is its release, and a pre-release or development version (`3.5.0rc1`, `3.5.0.dev20250101`) is BELOW the release it
precedes, as PEP 440 orders them."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

_OK = [False]


class TritonBelowFloor(RuntimeError):
    """The installed Triton is older than the floor the library declares."""


_REQ = re.compile(r"^\s*triton\s*(?:\[[^\]]*\])?\s*([^;]*)", re.I)
_LOWER = re.compile(r">=\s*([0-9][0-9A-Za-z.+!_-]*)")


def parse_version(text: str) -> Tuple[Tuple[int, ...], bool]:
    """(release numbers, is_pre): `3.8.0+git4a15f415.nbx1` -> ((3, 8, 0), False); `3.5.0rc1` -> ((3, 5, 0), True)."""
    public = str(text).strip().split("+", 1)[0]
    m = re.match(r"(?:\d+!)?(\d+(?:\.\d+)*)(.*)$", public)
    if not m:
        raise ValueError(f"no version in {text!r}")
    rest = m.group(2).lower()
    return tuple(int(x) for x in m.group(1).split(".")), bool(re.search(r"(a|b|c|rc|alpha|beta|pre|preview|dev)", rest))


def _below(installed: Tuple[Tuple[int, ...], bool], floor: Tuple[int, ...]) -> bool:
    release, pre = installed
    width = max(len(release), len(floor))
    pad = lambda t: t + (0,) * (width - len(t))  # noqa: E731
    return (pad(release), 0 if pre else 1) < (pad(floor), 1)


def _floor_of_requirement(req: str) -> Optional[Tuple[int, ...]]:
    m = _REQ.match(req)
    if not m:
        return None
    lower = _LOWER.search(m.group(1))
    return parse_version(lower.group(1))[0] if lower else None


def _declaration() -> Tuple[Optional[Tuple[int, ...]], str]:
    """(floor, where it was read): the pyproject beside the source, else the installed metadata; (None, "") when neither
    declares one."""
    pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
    if pyproject.is_file():
        text = pyproject.read_text(encoding="utf-8")
        if re.search(r'^name\s*=\s*"nbxtensor"', text, re.M):
            dep = re.search(r'^dependencies\s*=\s*\[(.*?)\]\s*$', text, re.M | re.S)
            for d in re.findall(r'"([^"]+)"', dep.group(1)) if dep else []:
                floor = _floor_of_requirement(d)
                if floor:
                    return floor, f"{pyproject}"
    try:
        from importlib import metadata
        for d in metadata.requires("nbxtensor") or []:
            floor = _floor_of_requirement(d)
            if floor:
                return floor, "the installed nbxtensor's metadata"
    except Exception:
        pass
    return None, ""


def ensure(installed: Optional[str] = None) -> None:
    """Refuse by name a Triton below the declared floor; otherwise remember the answer, so a launch pays a flag.
    `installed`: a version text to check instead of the one Triton reports (tests)."""
    explicit = installed is not None
    if _OK[0] and not explicit:
        return
    if not explicit:
        import triton
        installed = triton.__version__
    floor, where = _declaration()
    if floor is not None and _below(parse_version(installed), floor):
        want = ".".join(str(x) for x in floor)
        raise TritonBelowFloor(
            f"nbxtensor needs Triton {want} or later (declared in {where}); Triton {installed} is installed. The floor is "
            f"the first Triton that has the names the launch path calls (JITFunction._pack_args among them): refused "
            f"here, by name, instead of an AttributeError at the first launch. Install Triton {want} or later (on a Mac, rebuild the "
            f"source-built Triton from a {want} or later checkout).")
    if not explicit:
        _OK[0] = True


def reset() -> None:
    """Forget the answer. For tests."""
    _OK[0] = False
