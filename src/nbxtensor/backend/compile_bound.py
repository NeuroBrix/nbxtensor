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
"""
from __future__ import annotations

import os
import pickle
import sys
import tempfile
from typing import Any, Dict, Optional, Tuple


class CompileUnbounded(RuntimeError):
    """A candidate whose compile did not end within the profile's bound (or whose compile failed in the worker)."""


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


def precompile(kernel, args: tuple, kwargs: Dict[str, Any]) -> None:
    """Compile `kernel` for this launch in the run's worker, under the profile's bound; raise CompileUnbounded when it
    does not end in time. A no-op where the profile declares no bound."""
    if not hasattr(kernel, "params"):       # not a Triton kernel (a test's stand-in): nothing is compiled to bound
        return
    bound = bound_s()
    if bound is None:
        return
    from nbxtensor.ops import ew_proof_child
    where = tempfile.mkdtemp(prefix="nbx_compile_")
    spec = {"module": kernel.fn.__module__, "name": kernel.__name__, "args": [_describe(a) for a in args],
            "kwargs": {k: _describe(v) for k, v in kwargs.items()}}
    with open(os.path.join(where, "compile.pkl"), "wb") as fh:      # read back only by the worker this process starts
        pickle.dump(spec, fh)
    code, text = ew_proof_child._WORKER.ask({"mode": "compile", "where": where, "start": 0}, timeout=bound)
    if code != 0:
        raise CompileUnbounded(f"{kernel.__name__}: the candidate's compile did not end within {bound:g} s "
                               f"(autotune.compile_bound_s) — excluded: {text.strip()[-300:]}")


def compile_in_worker(where: str) -> None:
    """The worker's side: rebuild the arguments (pointers at an address of the same alignment) and compile."""
    import importlib
    from nbxtensor.backend.launcher import prepare
    from nbxtensor.ops.ew_proof_child import _View
    spec = pickle.load(open(os.path.join(where, "compile.pkl"), "rb"))
    kernel = getattr(importlib.import_module(spec["module"]), spec["name"])
    rebuild = lambda d: (_View(0x10000 if d[1][1] else 0x10004, d[1][0], d[1][2], d[1][3], d[1][4])  # noqa: E731
                         if d[0] == "ptr" else d[1])
    prepare(kernel, tuple(rebuild(d) for d in spec["args"]), {k: rebuild(d) for k, d in spec["kwargs"].items()})
