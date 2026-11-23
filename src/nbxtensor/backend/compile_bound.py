"""A sweep candidate is compiled under a bound: a candidate whose compile does not end within it is EXCLUDED and
recorded with its text, never a hang of the user's run (project lead 23/11/2026).

The Volta V100, base def527d: `moe_route_kernel`'s softmax-topk key at T=2 fp32 never returned from Triton's
make_llir for three of its twelve candidates (BLOCK_H 64 at 2 warps, 128 at 2 and 4); the compile time doubled with
each doubling of BLOCK_H / warps (4.2 s at 2, 4.6-5.1 at 4, 6.9-7.3 at 8, 13.9-14.7 at 16) and left the scale at 32
and 64 — the lanes each thread carries, the quantity of the PixArt-XL fault the same day (4096 lanes over 2 warps).
Triton's compile runs in-process and cannot be interrupted, so each candidate is compiled FIRST in the run's proof
worker (`ops/ew_proof_child`, one process for the whole run, started again only after it is killed): the worker writes
Triton's on-disk cache, the parent's own compile is then a cache hit; a worker that does not answer within the bound is
killed and the candidate excluded.

The bound is the vendor profile's `autotune.compile_bound_s` (a value of the device file, never chosen here). A profile
that declares none compiles in-process as before, said once.

Three answers, never confused (C16, the Dell leg of 23/11: every "did not end within 120 s" of moe_route_kernel was a
worker that died at once on an AttributeError, read as a timeout): a compile that ENDS is a cache hit for the parent;
one that does not end within the bound is EXCLUDED (`CompileUnbounded`); a candidate Triton's compiler REFUSES (its own
error, e.g. out of resources) is reported as the in-process compile reports it (`CompileRefused`, counted as a
candidate that could not run); a worker that DIES on anything else is an ERROR of this library, raised with the
worker's own text (`CompileWorkerDied`), never an exclusion.
"""
from __future__ import annotations

import os
import pickle
import sys
import tempfile
from typing import Any, Dict, Optional, Tuple


class CompileUnbounded(RuntimeError):
    """A candidate whose compile did not end within the profile's bound: excluded, with its text."""


class CompileRefused(RuntimeError):
    """A candidate Triton's compiler refused in the worker (its own error, e.g. out of resources): the candidate
    cannot run, as an in-process compile that raises."""


class CompileWorkerDied(RuntimeError):
    """The worker died on the compile for a reason that is not the candidate's (a fault of this library): raised with
    the worker's text, never read as an exclusion."""


#: the file the worker writes when Triton's compiler refuses the candidate
_REFUSED = "refused.txt"


_SAID = [False]


def bound_s() -> Optional[float]:
    """The profile's compile bound in seconds, None when it declares none (said once)."""
    from nbxtensor.ops._configs import active_vendor_profile
    try:
        value = (active_vendor_profile().get("autotune") or {}).get("compile_bound_s")
    except Exception:
        value = None
    if value is None and not _SAID[0]:
        _SAID[0] = True
        print("[autotune] this device profile declares no autotune.compile_bound_s: sweep candidates compile "
              "in-process, unbounded", file=sys.stderr)
    return float(value) if value is not None else None


def _describe(value: Any) -> Tuple[str, Any]:
    """An argument as the worker rebuilds it for a compile: a pointer by its dtype and 16-byte alignment (what the
    binary is specialised on), anything else as it is."""
    if hasattr(value, "data_ptr") and hasattr(value, "dtype"):
        return ("ptr", (value.dtype, int(value.data_ptr()) % 16 == 0, tuple(int(d) for d in value.shape),
                        tuple(int(s) for s in value.stride()), int(value.element_size())))
    return ("val", value)


def _launch_only(kernel) -> set:
    """The keywords a launch takes and a compile does not: the keyword-only parameters of the kernel's own `run`
    (`grid`, `warmup` in Triton 3.x — an autotuner's launch hands them on), read from its signature, never listed
    here; a compile handed `grid` refuses it ("Keyword argument grid was specified but unrecognised", the Mac 23/11,
    the second fault the dead worker of C16 had hidden)."""
    import inspect
    run = getattr(type(kernel), "run", None)
    if run is None:                         # a stand-in that launches nothing: it takes no launch keyword either
        return set()
    return {n for n, p in inspect.signature(run).parameters.items() if p.kind is p.KEYWORD_ONLY}


_UNNAMED_SAID: set = set()


def _named(kernel) -> bool:
    """Whether the worker can find `kernel` by its module and name, as `compile_in_worker` will: the module importable
    and its attribute of that name holding this very function. A kernel written outside a module (a test's inline
    kernel renamed into `nbxtensor.tests`) cannot be handed to a worker: it is compiled in-process, unbounded, said
    once by name — never a worker sent to die on a module it cannot import (C16)."""
    import importlib.util
    module = kernel.fn.__module__
    try:
        found = importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        found = False
    named = found and jit_of_or_none(getattr(sys.modules.get(module), kernel.__name__, None)) is kernel
    if not named and kernel.__name__ not in _UNNAMED_SAID:
        _UNNAMED_SAID.add(kernel.__name__)
        print(f"[autotune] {kernel.__name__}: its module {module!r} does not name it for a worker: its candidates "
              f"compile in-process, unbounded", file=sys.stderr)
    return named


def jit_of_or_none(obj):
    """`jit_of(obj)`, or None when `obj` holds no @triton.jit function."""
    try:
        return jit_of(obj) if obj is not None else None
    except TypeError:
        return None


def precompile(kernel, args: tuple, kwargs: Dict[str, Any]) -> None:
    """Compile `kernel` for this launch in the run's worker, under the profile's bound; raise CompileUnbounded when it
    does not end in time. A no-op where the profile declares no bound."""
    if not hasattr(kernel, "params"):       # not a Triton kernel (a test's stand-in): nothing is compiled to bound
        return
    bound = bound_s()
    if bound is None or not _named(kernel):
        return
    from nbxtensor.ops import ew_proof_child
    where = tempfile.mkdtemp(prefix="nbx_compile_")
    launch_only = _launch_only(kernel)
    spec = {"module": kernel.fn.__module__, "name": kernel.__name__, "args": [_describe(a) for a in args],
            "kwargs": {k: _describe(v) for k, v in kwargs.items() if k not in launch_only}}
    with open(os.path.join(where, "compile.pkl"), "wb") as fh:      # read back only by the worker this process starts
        pickle.dump(spec, fh)
    code, text = ew_proof_child._WORKER.ask({"mode": "compile", "where": where, "start": 0}, timeout=bound)
    if code == ew_proof_child.TIMED_OUT:
        raise CompileUnbounded(f"{kernel.__name__}: the candidate's compile did not end within {bound:g} s "
                               f"(autotune.compile_bound_s) — excluded: {text.strip()[-300:]}")
    if code != 0:
        raise CompileWorkerDied(f"{kernel.__name__}: the compile worker died (exit {code}) — not a timeout, not an "
                                f"exclusion: {text.strip()[-1500:]}")
    refused = os.path.join(where, _REFUSED)
    if os.path.exists(refused):
        raise CompileRefused(f"{kernel.__name__}: Triton's compiler refused the candidate: {open(refused).read()}")


def jit_of(obj):
    """The `@triton.jit` function a module-level name holds: itself, or the one an autotuner or heuristics wrapper
    holds as `.fn`, followed to the end (`moe_route_kernel` is an `Autotuner` around it). Refused by name when there is
    none: the worker then dies with that text, an error, never an exclusion."""
    seen = obj
    while not hasattr(obj, "params"):
        if not hasattr(obj, "fn"):
            raise TypeError(f"compile_bound: {seen!r} holds no @triton.jit function (no .params, no .fn)")
        obj = obj.fn
    return obj


def compile_in_worker(where: str) -> None:
    """The worker's side: rebuild the arguments (pointers at an address of the same alignment) and compile; a refusal
    by Triton's compiler (an exception of a `triton` module) is written for the parent, anything else kills the
    worker with its traceback."""
    import importlib
    from nbxtensor.backend.launcher import prepare
    from nbxtensor.ops.ew_proof_child import _View
    spec = pickle.load(open(os.path.join(where, "compile.pkl"), "rb"))
    kernel = jit_of(getattr(importlib.import_module(spec["module"]), spec["name"]))
    rebuild = lambda d: (_View(0x10000 if d[1][1] else 0x10004, d[1][0], d[1][2], d[1][3], d[1][4])  # noqa: E731
                         if d[0] == "ptr" else d[1])
    try:
        prepare(kernel, tuple(rebuild(d) for d in spec["args"]), {k: rebuild(d) for k, d in spec["kwargs"].items()})
    except Exception as exc:
        if not type(exc).__module__.startswith("triton"):
            raise
        with open(os.path.join(where, _REFUSED), "w") as fh:
            fh.write(f"{type(exc).__name__}: {exc}"[:2000])
