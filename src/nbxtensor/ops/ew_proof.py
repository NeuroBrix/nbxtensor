"""The proof of an element-wise class's TILE on each compiled variant it reaches.

A class setting (`elementwise_launch.ew_meta`) is swept on a copy and handed to every element-wise kernel of its
class. It is proven on a VARIANT — one binary: the kernel, its specialization (pointer dtypes and alignment, every
constexpr), its options — before that binary runs on it: the launcher hands every first resolution of a variant
carrying a class tile (`ops/ew_tile.EwTile`) to `settle`, once per compiled variant, never per launch.

The proof (orchestrator rulings 23/11/2026 12:5x-13:1x):
  * every candidate of the class runs the variant on the SAME inputs — a snapshot of the real launch, each pointer
    argument's addressable bytes, overlapping arguments kept overlapping — in a CHILD process with
    CUDA_LAUNCH_BLOCKING=1: an illegal address is sticky and ends the process that meets it. A fault ends the child;
    it is recorded (kernel, variant, tile, the fault's text) and a fresh child resumes at the next candidate;
  * a candidate PASSES when it does not fault and, for a kernel that declares itself tile-independent
    (`ew_tile.tile_independent`), every byte it leaves in the launch's buffers is identical (NaN payloads included)
    to the agreeing set: more than half of the non-faulting candidates AND at least three. A candidate outside that
    set is recorded with its first differing element and both values. A kernel that declares itself tile-dependent
    is proven by the absence of a fault alone, and the record names it;
  * fewer than three non-faulting candidates, or no agreeing set: the variant is not proven — a named error, never a
    silent use of the unproven setting.

WHAT THIS CANNOT SEE: a defect every candidate shares (in the source, or emitted the same for every tile) passes it.
It is a proof of the TILE, not of the kernel; the kernel's own tests are the proof of its values. It is no oracle.

The table names what it was proven with: the device class (as the certified directory: vendor / profile), the
compiler (`autotune_certified.generator_identity`) and the assembler build (ptxas's version on CUDA; the Metal
backend's hash is in the generator identity). A proof made with another assembler is not read — the fault this
exists for was the assembler's. Shipped proofs: `<certified directory>/<vendor>/<profile>/ew_tile_proof.json`;
proofs a run made: the replay cache's `ew_tile_proof.json`, kept per stamp.

At run time a variant no table holds is proven ONCE at its first launch, one printed line saying so (kernel,
variant, bytes snapshotted, seconds), and kept. If the class's winner fails on it, the class moves to the best
candidate (the copy's timing) every proven variant of the class passes, and that is kept too. A confirmation run
(`--certified-only`) proves nothing: an unproven variant is refused by name there.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

HEADER = ("A proof of the TILE on each compiled variant, not of the kernel: no fault, and for a tile-independent "
          "kernel the same bits as the agreeing candidates on the same inputs. A defect every candidate shares passes "
          "it; the kernel's own tests are the proof of its values. Not an oracle.")
FLOOR = 3                   # the agreeing set holds at least this many candidates (ruling 13:1x, condition 2)


class TileUnproven(RuntimeError):
    """A variant of the element-wise class whose tile cannot be proven, or is not proven in a confirmation run."""


# ---------------------------------------------------------------------------
# the stamp: what a proof was made with
# ---------------------------------------------------------------------------
_STAMP: List[Optional[dict]] = [None]


def stamp() -> dict:
    """Device class, compiler and assembler of this process, read once."""
    if _STAMP[0] is None:
        from nbxtensor.ops.autotune_certified import active_profile, generator_identity
        prof = active_profile()
        if prof is None:
            raise TileUnproven("ZERO FALLBACK: the element-wise tile proof names its device class, and no vendor "
                               "profile is in force")
        _STAMP[0] = {"vendor": prof[0], "profile": prof[1], "generator": generator_identity(),
                     "assembler": _assembler()}
    return _STAMP[0]


def _assembler() -> str:
    """The assembler's own version text: ptxas on CUDA; the out-of-tree backend (Metal) is named by the generator's
    backend hash, its assembler is inside it."""
    from nbxtensor.backend.launcher import target
    if getattr(target(), "backend", None) != "cuda":
        return f"in the backend ({getattr(target(), 'backend', '?')}): see generator.backend_hash"
    from triton.backends.nvidia.compiler import get_ptxas_version
    return " ".join(get_ptxas_version().split())


def stamp_key(st: dict) -> str:
    return hashlib.sha256(json.dumps(st, sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# the variant
# ---------------------------------------------------------------------------
def variant_of(kernel, specialization, options, tiles) -> Tuple[str, Tuple[int, int, int]]:
    """(variant id, tile) of a launch: the id holds everything that changes the binary except the tile — the
    kernel, each specialized parameter (class tiles written by role), the options other than warps and stages —
    and the tile is (block, warps, stages)."""
    spec = [("tile", v.role) if _is_tile(v) else (str(t), repr(v)) for t, v in specialization]
    rest = sorted((k, repr(v)) for k, v in options.items() if k not in ("num_warps", "num_stages"))
    qual = f"{getattr(kernel.fn, '__module__', '?')}.{kernel.__name__}"
    vid = json.dumps([qual, spec, rest])
    return vid, (int(tiles["tile"]), int(options.get("num_warps", _default_option(kernel, "num_warps"))),
                 int(options.get("num_stages", _default_option(kernel, "num_stages"))))


def _is_tile(v) -> bool:
    from nbxtensor.ops.ew_tile import EwTile
    return isinstance(v, EwTile)


_DEFAULTS: Dict[tuple, int] = {}


def _default_option(kernel, name: str) -> int:
    """The warps or stages the backend compiles when the launch names none (its own option defaults)."""
    if (id(kernel), name) not in _DEFAULTS:
        from nbxtensor.backend.launcher import _binder
        _DEFAULTS[(id(kernel), name)] = int(getattr(_binder(kernel)[2].parse_options({}), name))
    return _DEFAULTS[(id(kernel), name)]


# ---------------------------------------------------------------------------
# the tables
# ---------------------------------------------------------------------------
class _Table:
    """Shipped proofs and the proofs this machine made, both at this process's stamp."""

    def __init__(self):
        self.key = stamp_key(stamp())
        self.shipped = self._read(self._shipped_path()).get("variants", {})
        kept = self._read(self._kept_path())
        self.kept = dict(kept.get("variants", {}))
        self.timing = dict(kept.get("copy_timing", {}))
        self.seen_classes: Dict[str, set] = {}

    @staticmethod
    def _shipped_path():
        from nbxtensor.ops.autotune_certified import directory
        st = stamp()
        return directory() / st["vendor"] / st["profile"] / "ew_tile_proof.json"

    @staticmethod
    def _kept_path():
        from nbxtensor.ops.autotune_cache import _dir
        return os.path.join(_dir(), "ew_tile_proof.json")

    def _read(self, path) -> Dict[str, dict]:
        """This stamp's block of a table file; a proof made with another compiler or assembler is not read."""
        try:
            with open(path) as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            return {}
        return doc.get("stamps", {}).get(self.key) or {}

    def record(self, vid: str) -> Optional[dict]:
        return self.kept.get(vid) or self.shipped.get(vid)

    def status(self, vid: str, tile) -> Optional[str]:
        rec = self.record(vid)
        if rec is None:
            return None
        return "passed" if list(tile) in rec["passed"] else "failed"

    def keep(self, vid: str, rec: dict) -> None:
        self.kept[vid] = rec
        self._write()

    def keep_timing(self) -> None:
        self._write()

    def _write(self) -> None:
        path = self._kept_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path) as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            doc = {"header": HEADER, "stamps": {}}
        block = doc["stamps"].setdefault(self.key, {"stamp": stamp(), "variants": {}})
        block["variants"].update(self.kept)
        block["copy_timing"] = self.timing
        tmp = f"{path}.{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True)
        os.replace(tmp, path)

    def class_winner(self, cls: int) -> Optional[Tuple[int, int, int]]:
        """The best candidate, by the copy's timing, that every proven variant of class `cls` passes."""
        sets = [set(map(tuple, self.record(v)["passed"])) for v in self.seen_classes.get(str(cls), ())]
        if not sets:
            return None
        common = set.intersection(*sets)
        timing = self.copy_timing(cls, sorted(common))
        return min(common, key=lambda t: timing[t]) if common else None

    def copy_timing(self, cls: int, tiles) -> Dict[tuple, float]:
        """The copy's milliseconds per tile at the class's floor: kept, else measured ONCE in a child, said, kept."""
        known = self.timing.get(str(cls), {})
        missing = [t for t in tiles if repr(list(t)) not in known]
        if missing:
            from nbxtensor.ops import ew_proof_child
            t0 = time.time()
            got = ew_proof_child.time_copy_in_child(cls, missing)
            known.update({repr(list(t)): ms for t, ms in got.items()})
            self.timing[str(cls)] = known
            self.keep_timing()
            print(f"[ew-proof] class {cls}: the copy timed once at {len(missing)} tile(s) to choose among the "
                  f"candidates every proven variant passes ({time.time() - t0:.1f} s), kept", file=sys.stderr)
        return {tuple(t): known[repr(list(t))] for t in tiles}


_TABLE: List[Optional[_Table]] = [None]


def table() -> _Table:
    if _TABLE[0] is None:
        _TABLE[0] = _Table()
    return _TABLE[0]


# ---------------------------------------------------------------------------
# the launcher's door
# ---------------------------------------------------------------------------
def settle(kernel, grid, bound: Dict[str, Any], specialization, options: Dict[str, Any], tiles: dict):
    """None: this launch runs as asked (its tile is proven on its variant). Else (grid, kwargs) of the same launch at
    the class's moved setting. Raises TileUnproven by name where no proven setting exists."""
    vid, tile = variant_of(kernel, specialization, options, tiles)
    tab = table()
    tab.seen_classes.setdefault(str(tiles["tile"].ew_class), set()).add(vid)
    status = tab.status(vid, tile)
    if status is None:
        from nbxtensor.ops.autotune_certified import certified_only
        if certified_only():
            raise TileUnproven(f"ZERO FALLBACK: a confirmation run met element-wise variant {kernel.__name__} "
                               f"{vid} at tile {tile}, which no proof table holds (stamp {tab.key})")
        tab.keep(vid, prove(kernel, grid, bound, options, tiles, vid))
        status = tab.status(vid, tile)
    if status == "passed":
        return None
    return _moved(kernel, grid, bound, options, tiles, tab)


def _moved(kernel, grid, bound, options, tiles, tab):
    """The class's winner failed this variant: move the class to the best common candidate, and this launch."""
    cls = tiles["tile"].ew_class
    best = tab.class_winner(cls)
    if best is None:
        raise TileUnproven(f"ZERO FALLBACK: element-wise class {cls}: no candidate passes every proven variant "
                           f"({sorted(tab.seen_classes.get(str(cls), ()))})")
    from nbxtensor.ops.elementwise_launch import move_class_winner
    meta = move_class_winner(cls, best[0], best[1])
    print(f"[ew-proof] class {cls}: {kernel.__name__} fails the class's tile "
          f"{int(tiles['tile'])}x{int(tiles.get('warps', options.get('num_warps', 0)))}; the class moves to "
          f"{best[0]}x{best[1]} (stages {best[2]}), the best candidate every proven variant passes", file=sys.stderr)
    return retiled(grid, bound, options, int(tiles["tile"]), meta, best[2])


def retiled(grid, bound, options, old_block: int, meta: dict, stages: int):
    """The same launch at the class setting `meta`: every class tile replaced, axis 0 of a fixed grid scaled to cover
    what it covered (an element-wise launch's programs walk axis 0 a tile at a time; a callable grid re-reads it)."""
    from nbxtensor.ops.ew_tile import EwTile
    kw = {k: (meta["BLOCK_SIZE"] if isinstance(v, EwTile) and v.role == "tile" else v) for k, v in bound.items()}
    opts = dict(options, num_warps=meta["num_warps"], num_stages=stages)
    if not callable(grid):
        g = tuple(int(x) for x in grid)
        new = int(meta["BLOCK_SIZE"])
        grid = (-(-g[0] * old_block // new),) + g[1:]
    return grid, {**kw, **opts}


# ---------------------------------------------------------------------------
# the proof
# ---------------------------------------------------------------------------
def prove(kernel, grid, bound, options, tiles, vid) -> dict:
    """Run every candidate of the class on a snapshot of this launch, in child processes; the variant's record."""
    from nbxtensor.ops import ew_proof_child
    from nbxtensor.ops.elementwise_launch import class_candidates
    from nbxtensor.ops.ew_tile import declared_independence
    t0 = time.time()
    grid = grid(bound) if callable(grid) else grid
    spec, nbytes = ew_proof_child.snapshot_launch(kernel, grid, bound, options, int(tiles["tile"]))
    independent = declared_independence(kernel)
    cands = class_candidates()
    results, differ = ew_proof_child.run_candidates(spec, cands)
    rec = judge(independent, cands, results, differ)
    rec.update(seconds=round(time.time() - t0, 2), snapshot_bytes=nbytes, kernel=kernel.__name__)
    print(f"[ew-proof] proved {kernel.__name__} once (variant {stamp_key({'v': vid})}): {len(rec['passed'])} of "
          f"{len(cands)} tiles pass, {len(rec['faults'])} fault, snapshot {nbytes} bytes, {rec['seconds']} s"
          + ("" if independent else " — proven by fault only (" + ("declares itself tile-dependent"
             if independent is False else "declares no tile independence") + ")"), file=sys.stderr)
    if not rec["passed"]:
        raise TileUnproven(f"ZERO FALLBACK: element-wise variant {kernel.__name__}: {rec['refused']}")
    return rec


def judge(independent, cands, results, differ) -> dict:
    """The record of one variant from its candidates' results ({tile: ('fault', text) | ('ran', digests)});
    `differ(digests, agreeing digests)` names a dissenting candidate's first differing element."""
    faults = [{"tile": list(t), "fault": results[t][1]} for t in cands if results[t][0] == "fault"]
    ran = [t for t in cands if results[t][0] == "ran"]
    rec = {"independent": independent, "faults": faults, "dissent": [], "passed": [], "refused": None}
    if len(ran) < FLOOR:
        rec["refused"] = f"{len(ran)} candidate(s) ran without a fault, fewer than {FLOOR}"
        return rec
    if not independent:
        rec["passed"] = [list(t) for t in ran]
        return rec
    groups: Dict[tuple, List[tuple]] = {}
    for t in ran:
        groups.setdefault(tuple(results[t][1]), []).append(t)
    agree = max(groups.values(), key=len)
    if len(agree) * 2 <= len(ran) or len(agree) < FLOOR:
        rec["refused"] = f"no agreeing set: {sorted(len(g) for g in groups.values())} of {len(ran)} non-faulting"
        return rec
    rec["passed"] = [list(t) for t in agree]
    ref = results[agree[0]][1]
    rec["dissent"] = [{"tile": list(t), "first_difference": differ(results[t][1], ref)} for t in ran if t not in agree]
    return rec
