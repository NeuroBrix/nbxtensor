"""The C compiler Triton builds its driver with, found ahead of the first build or refused by name.

Triton compiles a kernel to a binary with no C compiler (ptxas, the vendor's assembler, is all a cubin needs, and the
library's own drivers load that binary through the runtime library). What does need one is Triton's OWN driver: its
backend module builds a C helper (`cuda_utils`, the launcher stub) with `triton/runtime/build.py` the first time the
driver is used, and where no compiler is found it raises "Failed to find C compiler" from inside that first use, which
is minutes into a run when a measure is the first thing to reach it. The library hands launches to Triton's own driver
in two places: a kernel that is not the library's (`launcher.install`, the hand-back), and the differential arm
(`NBX_LAUNCHER=triton`). `ensure()` runs there, before the hand-back, and refuses by name.

The compiler is resolved exactly as Triton resolves it (`triton/runtime/build.py`, `_find_compiler`, the C branch): the
`CC` environment variable when it is set, else `gcc` then `clang` on PATH, gcc first. Triton takes `CC` as written and
fails at the build if it names nothing; here it must resolve to an executable (`shutil.which`, which reads a path or a
name on PATH), so a `CC` naming a missing compiler is refused by its own name. A build implementation registered with
Triton (`triton.knobs.build.impl`) replaces the compiler there, so none is needed here either.

Triton builds only on a cache miss (`triton/runtime/build.py`, `_compile_so`: the cache is read first, `_build` only when
the file is absent), so a machine whose Triton cache already holds the built helper (a pre-warmed image, a compiler
removed after the first run) works with no compiler. The cache is not read here by a second, guessed key: where no
compiler is found, the refusal is registered as Triton's build implementation (`triton.knobs.build.impl`, the hook
Triton's own error names), which Triton reaches only on a miss, and Triton's active driver is made at once, so its C
helper is either loaded from the cache or refused by name now, before the hand-back; a later kernel's launcher stub that
misses the cache is refused by the same name at its build.

Checked only where a Triton build can happen: the Triton backend that serves this process's compile target must be one
whose driver builds with `triton/runtime/build.py` (read from the backend's own driver module, never from a list of
names). A backend that never builds with it is not checked. Answered once per process: a launch then pays a flag."""
from __future__ import annotations

import os
import shutil
import sys
from typing import Optional, Tuple

_OK = [False]

#: the names Triton tries on PATH when CC is not set, in its order of preference (`_find_compiler`: gcc, else clang)
TRITON_NAMES: Tuple[str, ...] = ("gcc", "clang")


class NoCCompiler(RuntimeError):
    """No C compiler for Triton to build its driver with."""


def resolve(environ=None) -> Tuple[Optional[str], str]:
    """(the compiler's executable or None, what was asked for): `CC` when set, else the first of `TRITON_NAMES` on PATH."""
    env = os.environ if environ is None else environ
    cc = env.get("CC")
    if cc is not None:
        return shutil.which(cc), f"CC={cc!r}"
    for name in TRITON_NAMES:
        found = shutil.which(name)
        if found is not None:
            return found, name
    return None, " or ".join(TRITON_NAMES) + " on PATH"


def _build_impl():
    """The build implementation registered with Triton (`triton.knobs.build.impl`), or None."""
    try:
        from triton import knobs
    except ImportError:
        return None
    return getattr(getattr(knobs, "build", None), "impl", None)


def builds_with_a_c_compiler(target=None) -> bool:
    """True when the Triton backend serving `target` (default: this process's compile target, `launcher.target()`)
    has a driver module that builds with `triton/runtime/build.py`."""
    from triton.backends import backends
    if target is None:
        from nbxtensor.backend.launcher import target as _target
        target = _target()
    for backend in backends.values():
        if backend.compiler is None or not backend.compiler.supports_target(target):
            continue
        module = sys.modules.get(getattr(backend.driver, "__module__", ""))
        return any(getattr(v, "__module__", None) == "triton.runtime.build" for v in getattr(module, "__dict__", {}).values())
    return False


def ensure(target=None, environ=None) -> None:
    """Refuse by name a process with no C compiler where Triton's own driver will build one; otherwise remember the
    answer. `target` / `environ`: a compile target and an environment to check instead of this process's (tests)."""
    explicit = target is not None or environ is not None
    if _OK[0] and not explicit:
        return
    if _build_impl() is not None:
        return                          # Triton builds with the registered implementation, not with a compiler
    if not builds_with_a_c_compiler(target):
        _OK[0] = _OK[0] or not explicit  # a backend that never builds with one: nothing to check, answered once
        return
    found, asked = resolve(environ)
    if found is None:
        _refuse_at_build(NoCCompiler(
            f"Triton's driver builds a C helper that its cache does not hold and no C compiler was found ({asked}; "
            f"Triton resolves CC, else {' then '.join(TRITON_NAMES)} on PATH, triton/runtime/build.py): refused by name "
            f"instead of inside the build. Install a C compiler (e.g. gcc) or set CC to one that exists; a Triton cache "
            f"already holding the built helper needs none."))
        _make_driver()                  # the helper loads from Triton's cache, or the refusal fires here
    if not explicit:
        _OK[0] = True


def _refuse_at_build(error: NoCCompiler) -> None:
    """Register `error` as Triton's build implementation: Triton calls it only for a build its cache misses."""
    from triton import knobs

    def refuse(name, src, srcdir, library_dirs, include_dirs, libraries):
        raise error
    knobs.build.impl = refuse


def _make_driver() -> None:
    """Make Triton's active driver now: its C helper is loaded from Triton's cache or built at this point."""
    from triton.runtime import driver
    driver.active.get_current_target()


def reset() -> None:
    """Forget the answer. For tests."""
    _OK[0] = False
