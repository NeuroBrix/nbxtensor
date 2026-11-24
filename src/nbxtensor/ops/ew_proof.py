"""The proof of an element-wise class's TILE on each compiled variant it reaches — a function of the VARIANT alone.

A class setting (`elementwise_launch.ew_meta`) is swept on a copy and handed to every element-wise kernel of its
class. It is proven on a VARIANT — one binary: the kernel, its specialization (pointer dtypes and alignment, integer
widths and divisibility, every constexpr), its options — before that binary runs on it: the launcher hands every
first resolution of a variant carrying a class tile (`ops/ew_tile.EwTile`) to `settle`, once per compiled variant,
never per launch.

The proof (orchestrator rulings 23/11/2026 12:5x-13:1x; the synthetic cure, the project lead 23/11/2026 evening):
  * every candidate of the class runs the variant on the SAME SYNTHETIC launches, made from the variant alone
    (`ops/ew_synthetic.py`): a fixed ladder of counts that always holds masked tails (n = 1, s - 1, L - 1, L + 1,
    2L + 3 for the class's smallest tile s and largest L, each kept in the count's own specialization), layouts drawn
    from the rank (contiguous and permuted), buffers of each pointer's dtype and alignment filled deterministically,
    scalars drawn deterministically. NOTHING of the launch that first met the variant enters the proof — not its
    count, its grid, its shapes nor what memory lay after its buffers — so the record, keyed by the variant, holds
    for every launch of it. A kernel whose signature does not determine its synthetic launch declares its recipe
    where it is defined (`ew_tile.synthetic`), or is refused by name;
  * EVERY BUFFER lies against a GUARD (`memory.guarded`): on CUDA its end at the end of the mapped pages of a VMM
    range whose next granule is unmapped (an access past the extent is an illegal address, every time); on Metal its
    end at the end of a dedicated MTLBuffer, in a worker with shader validation on (an access past it is reported).
    A device where no guard can be made proves NOTHING: the variant is refused by name ("unguarded") — nothing says
    proven for a tail no run met; what guard such a device offers is a census question when it arrives, never a
    default. Before a run's first proof a CANARY (`canary`: one deliberate overrun) must be seen by the guard, or the
    device is unguarded the same way;
  * a tile FAULTS only on a DEVICE fault in the worker: a kernel fault status at its launch or at the synchronize
    after it on CUDA (`backend.faults.KERNEL_FAULTS`: the 700 and its kin), a shader-validation report on Metal. Any
    other end — a refusal of the launcher, an exception before the launch, a signal, a timeout, an unexplained exit —
    means the proof COULD NOT RUN: the variant is refused by name and nothing is written;
  * the candidates run in a CHILD process (`ops/ew_proof_child.py`; one worker serves a burst of proofs, empties
    its pool back to the driver after each job and EXITS after `ew_proof_child.IDLE_SECONDS` (2 s) without a job, so
    it holds nothing on the card while the parent runs): an illegal address is sticky and ends the process that
    meets it. A fault ends the worker; it is recorded (kernel,
    variant, tile, the fault's text) and a fresh worker resumes at the next candidate;
  * a candidate PASSES when it does not fault and, for a kernel that declares itself tile-independent
    (`ew_tile.tile_independent`), every byte it leaves in the synthetic buffers is identical (NaN payloads included)
    to the agreeing set: more than half of the non-faulting candidates AND at least three (FLOOR). A candidate outside
    that set is recorded with its first differing element and both values. A kernel that declares itself
    tile-dependent is proven by the absence of a fault alone, and the record names it;
  * fewer than three non-faulting candidates, or no agreeing set: the variant is not proven — a named error, never a
    silent use of the unproven setting.

The cost is the synthetic buffers' bytes — kilobytes to megabytes, the same for every launch of a variant, never the
launch's bytes.

WHAT THIS CANNOT SEE: a defect every candidate shares (in the source, or emitted the same for every tile) passes it;
an access inside the guard's alignment slack (at most 15 bytes) or before a buffer's start. It is a proof of the TILE,
not of the kernel; the kernel's own tests are the proof of its values. It is no oracle.

The table names what it was proven with: the device class (as the certified directory: vendor / profile), the
compiler (`autotune_certified.generator_identity`), the assembler build (ptxas's version on CUDA; the Metal backend's
hash is in the generator identity) and the proof METHOD (`ew_synthetic.METHOD`). A proof made with another assembler
or another method is not read — the fault this exists for was the assembler's, and a proof of another method is
another fact. Shipped proofs: `<certified directory>/<vendor>/<profile>/ew_tile_proof.json`;
proofs a run made: the replay cache's `ew_tile_proof.json`, kept per stamp.

At run time a variant no table holds is proven ONCE at its first launch, one printed line saying so (kernel,
variant, the ladder, synthetic bytes, seconds), and kept. If the class's winner fails on it, the class moves to the
candidate NEAREST the winner on the class's own ordered ladder (`elementwise_launch.class_candidates`, the copy's
configs in order; a tie to the earlier one) among those every proven variant of the class passes: read, never timed,
so two homes pick the same tile (C17: a timing measured per fresh replay cache re-picked 512x8, 2048x8 or 1024x4 run
to run on the Dell, a search the owner's reset forbids). A confirmation run (`--certified-only`) proves nothing: an
unproven variant is refused by name there.
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
    """Device class, compiler, assembler and proof method of this process, read once: a table made under another
    stamp is not read."""
    if _STAMP[0] is None:
        from nbxtensor.ops.autotune_certified import active_profile, generator_identity
        prof = active_profile()
        if prof is None:
            raise TileUnproven("ZERO FALLBACK: the element-wise tile proof names its device class, and no vendor "
                               "profile is in force")
        from nbxtensor.ops.ew_synthetic import METHOD
        _STAMP[0] = {"vendor": prof[0], "profile": prof[1], "generator": generator_identity(),
                     "assembler": _assembler(), "method": METHOD}
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
    """A short stable key of a stamp dict: the hex sha256 of its sorted json, 16 characters."""
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
    from nbxtensor.ops.ew_tile import stated_stages
    stages = stated_stages(options)
    return vid, (int(tiles["tile"]), int(options.get("num_warps", _default_option(kernel, "num_warps"))),
                 int(_default_option(kernel, "num_stages") if stages is None else stages))


def _is_tile(v) -> bool:
    """Whether a value is an element-wise class tile (`ew_tile.EwTile`)."""
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
        """The shipped and kept proofs at this process's stamp, read once."""
        self.key = stamp_key(stamp())
        self.shipped = self._read(self._shipped_path()).get("variants", {})
        kept = self._read(self._kept_path())
        self.kept = dict(kept.get("variants", {}))
        self.unkept: Dict[str, dict] = {}       # refusals that are no measurement (unguarded, no recipe): this run's
        self.seen_classes: Dict[str, set] = {}

    @staticmethod
    def _shipped_path():
        """The shipped table's path: <certified directory>/<vendor>/<profile>/ew_tile_proof.json."""
        from nbxtensor.ops.autotune_certified import directory
        st = stamp()
        return directory() / st["vendor"] / st["profile"] / "ew_tile_proof.json"

    @staticmethod
    def _kept_path():
        """The kept table's path: the replay cache's ew_tile_proof.json."""
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
        """The variant's record: this run's unkept refusal, else kept, else shipped, else None."""
        return self.unkept.get(vid) or self.kept.get(vid) or self.shipped.get(vid)

    def status(self, vid: str, tile) -> Optional[str]:
        """"passed" / "failed" for a tile of a recorded variant, None when the variant has no record."""
        rec = self.record(vid)
        if rec is None:
            return None
        return "passed" if list(tile) in rec["passed"] else "failed"

    def keep(self, vid: str, rec: dict) -> None:
        """Keep a variant's record and write the kept table."""
        self.kept[vid] = rec
        self._write()

    def _write(self) -> None:
        """Write the kept table whole (beside it, renamed over it), this stamp's block updated."""
        path = self._kept_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path) as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            doc = {"header": HEADER, "stamps": {}}
        block = doc["stamps"].setdefault(self.key, {"stamp": stamp(), "variants": {}})
        block["variants"].update(self.kept)
        block.pop("copy_timing", None)                  # the timing the move once read (before C17): no longer a fact
        tmp = f"{path}.{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True)
        os.replace(tmp, path)

    def class_winner(self, cls: int, winner: Tuple[int, int]) -> Optional[Tuple[int, int, int]]:
        """The candidate class `cls` moves to: among those every proven variant of the class passes, the one nearest
        the class's `winner` (block, warps) on the class's ordered ladder (`elementwise_launch.class_candidates`), a
        tie to the earlier; None when no candidate passes every variant. Read from the ladder and the proofs, never
        timed."""
        sets = [set(map(tuple, self.record(v)["passed"])) for v in self.seen_classes.get(str(cls), ())
                if self.record(v) is not None]
        if not sets:
            return None
        common = set.intersection(*sets)
        from nbxtensor.ops.elementwise_launch import class_candidates
        ladder = class_candidates()
        at = next((i for i, c in enumerate(ladder) if (c[0], c[1]) == tuple(winner)), None)
        if at is None:
            raise TileUnproven(f"ZERO FALLBACK: element-wise class {cls}: its winner {winner[0]}x{winner[1]} is no "
                               f"candidate of the class's ladder {ladder}")
        passing = [i for i, c in enumerate(ladder) if c in common]
        return ladder[min(passing, key=lambda i: (abs(i - at), i))] if passing else None


_TABLE: List[Optional[_Table]] = [None]


def table() -> _Table:
    """The process's proof table (`_Table`), built at its first use."""
    if _TABLE[0] is None:
        _TABLE[0] = _Table()
    return _TABLE[0]


# ---------------------------------------------------------------------------
# the launcher's door
# ---------------------------------------------------------------------------
def settle(kernel, grid, bound: Dict[str, Any], specialization, options: Dict[str, Any], tiles: dict):
    """None: this launch runs as asked (its tile is proven on its variant). Else (grid, kwargs) of the same launch at
    the class's moved setting. Raises TileUnproven by name where no proven setting exists — a variant refused (fewer
    than three candidates ran, no agreeing set, no guard on this device, no synthetic recipe) is refused at every
    launch. The proof itself reads nothing of this launch but its variant (`specialization`, `options`); `grid` and
    `bound` serve only the moved launch."""
    if not callable(grid):
        raise TileUnproven(f"ZERO FALLBACK: {kernel.__name__} carries an element-wise class tile and its grid is fixed "
                           f"by the caller ({tuple(grid)}): a grid computed from the tile it was handed is wrong for "
                           "every other tile; hand it as a function of the tile (elementwise_launch.ew_grid_of)")
    vid, tile = variant_of(kernel, specialization, options, tiles)
    tab = table()
    tab.seen_classes.setdefault(str(tiles["tile"].ew_class), set()).add(vid)
    status = tab.status(vid, tile)
    if status is None:
        from nbxtensor.ops.autotune_certified import certified_only
        if certified_only():
            raise TileUnproven(f"ZERO FALLBACK: a confirmation run met element-wise variant {kernel.__name__} "
                               f"{vid} at tile {tile}, which no proof table holds (stamp {tab.key})")
        rec = prove(kernel, specialization, options, tiles, vid)
        if rec.get("unkept"):
            tab.unkept[vid] = rec           # a refusal that measured nothing: said this run, never written
        else:
            tab.keep(vid, rec)
        status = tab.status(vid, tile)
    rec = tab.record(vid) or {}
    if rec.get("refused"):
        where = "this run's refusal, never written" if rec.get("unkept") else f"kept; stamp {tab.key}"
        raise TileUnproven(f"ZERO FALLBACK: element-wise variant {kernel.__name__} is not proven: {rec['refused']} "
                           f"({where})")
    if status == "passed":
        return None
    return _moved(kernel, grid, bound, options, tiles, tab, vid, tile)


def _why_failed(rec: dict, tile) -> str:
    """What the variant's record says of `tile`: its fault, its dissent, or the record's own word."""
    for f in rec.get("faults", []):
        if tuple(f["tile"]) == tuple(tile):
            return f"faulted: {str(f['fault']).splitlines()[0][:160]}"
    if any(tuple(d["tile"]) == tuple(tile) for d in rec.get("dissent", [])):
        return "its bits differ from the agreeing candidates"
    return rec.get("struck", {}).get(f"{tile[0]}x{tile[1]}", "not among the tiles its record passes")


_SAID: set = set()      # (class, tile) of every move this process has said


def _moved(kernel, grid, bound, options, tiles, tab, vid=None, tile=None):
    """The class's winner failed this variant: move the class to the best common candidate, and this launch. The
    move is said in ONE line per class and tile it moves to, per process (never per launch): the tile the class now
    runs on, the tile it leaves, and why — the variant (kernel and id) whose record fails it, and its fault."""
    cls = tiles["tile"].ew_class
    best = tab.class_winner(cls, (int(tiles["tile"]), int(tiles.get("warps", options.get("num_warps", 0)))))
    if best is None:
        raise TileUnproven(f"ZERO FALLBACK: element-wise class {cls}: no candidate passes every proven variant "
                           f"({sorted(tab.seen_classes.get(str(cls), ()))})")
    from nbxtensor.ops.elementwise_launch import move_class_winner
    meta = move_class_winner(cls, best[0], best[1])
    if (cls, tuple(best)) in _SAID:
        return retiled(grid, bound, options, int(tiles["tile"]), meta, best[2])
    _SAID.add((cls, tuple(best)))
    print(move_line(cls, kernel.__name__, vid, tile if tile is not None else (
        int(tiles["tile"]), int(tiles.get("warps", options.get("num_warps", 0))), None), best,
        _why_failed(tab.record(vid) or {}, tile) if vid is not None else "not among the tiles its record passes"),
        file=sys.stderr)
    return retiled(grid, bound, options, int(tiles["tile"]), meta, best[2])


def move_line(cls, kernel_name: str, vid, left, best, why: str) -> str:
    """The one line a class's move is said in: the tile the class now runs on, and why — the tile it leaves and the
    variant (kernel, short id) whose record fails it, with what that record says (its fault)."""
    variant = stamp_key({"v": vid}) if vid is not None else "?"
    stages = f" stages {left[2]}" if len(left) > 2 and left[2] is not None else ""
    return (f"[ew-proof] class {cls} now runs on tile {best[0]}x{best[1]} (stages {best[2]}), the nearest on its "
            f"ladder every proven variant passes: it left tile {left[0]}x{left[1]}{stages}, which variant {variant} "
            f"of {kernel_name} fails ({why})")


def retiled(grid, bound, options, old_block: int, meta: dict, stages: int):
    """The same launch at the class setting `meta`: every class tile replaced; the grid, a function of the tile
    (`elementwise_launch.ew_grid_of`), is evaluated again at the new one by the launcher."""
    return grid, {**at_tile(bound, meta["BLOCK_SIZE"]), **dict(options, num_warps=meta["num_warps"],
                                                              num_stages=stages)}


def at_tile(bound: Dict[str, Any], block) -> Dict[str, Any]:
    """`bound` with every class tile replaced by `block` (an EwTile of the same class)."""
    from nbxtensor.ops.ew_tile import EwTile
    if not isinstance(block, EwTile):
        cls = next(v.ew_class for v in bound.values() if isinstance(v, EwTile))
        block = EwTile(int(block), cls, "tile")
    return {k: (block if isinstance(v, EwTile) and v.role == "tile" else v) for k, v in bound.items()}


# ---------------------------------------------------------------------------
# the proof
# ---------------------------------------------------------------------------
def prove(kernel, specialization, options, tiles, vid) -> dict:
    """The variant's record: every candidate of the class run on the variant's synthetic launches
    (`ew_synthetic.synthesise`), every buffer guarded, in the worker process; judged (`judge`). A variant that
    cannot be proven here without measuring anything — no guard on this device, no synthetic recipe — gets a refusal
    marked `unkept` (said this run, never written to a table: a later library or device may prove it)."""
    from nbxtensor.memory.guarded import guard_kind
    from nbxtensor.ops import ew_proof_child, ew_synthetic
    from nbxtensor.ops.elementwise_launch import class_candidates
    from nbxtensor.ops.ew_tile import declared_independence
    t0 = time.time()
    cands = class_candidates()
    guard = guard_kind()
    if guard is None:
        return _unkept(kernel, "unguarded", "unguarded: no guard can be made on this device (no VMM range, no shader "
                       "validation), so no run of a masked tail can show a lane touching past its extent; nothing "
                       "says proven for a tail no run met")
    try:
        launches = ew_synthetic.synthesise(kernel, specialization, cands)
    except ew_synthetic.NotSynthesisable as why:
        return _unkept(kernel, guard, str(why))
    unseen = canary(guard)
    if unseen:
        return _unkept(kernel, "unguarded", f"unguarded: {unseen}"
                       + (f". {remedy(kernel.__name__, vid)}" if guard in _CANARY_UNRUN else ""))
    where, nbytes = ew_proof_child.write_job(kernel, launches, cands, options, guard)
    independent = declared_independence(kernel)
    try:
        results, differ = ew_proof_child.run_candidates(where, cands)
        rec = judge(independent, cands, results, differ)
    except ew_proof_child.ProofCouldNotRun as why:
        return _unkept(kernel, guard, f"ZERO FALLBACK: the proof of {kernel.__name__} (variant "
                       f"{stamp_key({'v': vid})}) could not run: {why}. {remedy(kernel.__name__, vid)}")
    finally:
        import shutil
        shutil.rmtree(where, ignore_errors=True)        # the job's files: removed on success and on failure (C6-1)
    counts = sorted({ln.count for ln in launches})
    rec.update(seconds=round(time.time() - t0, 2), synthetic_bytes=nbytes, kernel=kernel.__name__,
               method=ew_synthetic.METHOD, guard=guard, ladder=counts,
               recipe=getattr(ew_synthetic.recipe_of(kernel), "description", "declared"))
    _count(rec["seconds"], nbytes)
    print(f"[ew-proof] proved {kernel.__name__} once on synthetic inputs (variant {stamp_key({'v': vid})}): "
          f"{len(rec['passed'])} of {len(cands)} tiles pass, {len(rec['faults'])} fault, ladder {counts} x "
          f"{len({ln.layout for ln in launches})} layout(s), guard {guard}, {nbytes} synthetic bytes, {rec['seconds']} s"
          + ("" if independent else " — proven by fault only (" + ("declares itself tile-dependent"
             if independent is False else "declares no tile independence") + ")"), file=sys.stderr)
    return rec              # a refused one too: kept, said once, raised by `settle` at every launch


def remedy(kernel_name: str, vid) -> str:
    """What a variant whose proof could not run needs, said with its refusal: room for the worker on a card of the
    launch's class (the worker ran on the freest of the cards the run declared, `ew_proof_child.declare_proof_cards`),
    or the variant's line in the shipped table — named by its signature as the record line prints it (`variant <key>`),
    its stamp and the table's path."""
    from nbxtensor.ops.ew_proof_child import _WORKER
    st = stamp()
    card, free = _WORKER.device
    where = (f"the worker ran on visible card {card}, the freest of the run's declared cards of the class, "
             f"{free // (1024 * 1024)} MiB free at its start" if free >= 0 else
             f"the worker ran on the launch's own card {card}: the run declared no other card of the class "
             f"(`ew_proof_child.declare_proof_cards`), or no free-memory reading was available")
    return (f"REMEDY: free memory on a card of this class ({st['vendor']}/{st['profile']}; {where}), or add the "
            f"variant's line to the shipped table {_Table._shipped_path()} (stamp {stamp_key(st)}, {kernel_name} "
            f"variant {stamp_key({'v': vid})}): prove it where a card of this class has room and merge the kept "
            f"proofs with `python -m nbxtensor.ops.ew_proof export <replay cache>/ew_tile_proof.json <that path>`")


_CANARY: Dict[str, str] = {}
_CANARY_UNRUN: set = set()          # the guards whose canary could not run (its worker found no room, ...)


def canary(guard: str) -> str:
    """'' when the guard is SEEN on this device, else why not — once per process, before its first proof: the
    canary kernel (`ops/ew_canary.py`), which writes one element past its output, runs on its synthetic launches
    against the guard in the worker, and must end as a device fault (a 700 on CUDA, a validation report on Metal).
    Once per process and not per worker: a fault ends the worker that meets it (a CUDA fault is sticky), and what is
    proven is the device's and the library's guard, the same for every worker of the run."""
    if guard not in _CANARY:
        from nbxtensor.ops import ew_proof_child, ew_synthetic
        from nbxtensor.ops.ew_canary import CANARY_SPEC, ew_canary_kernel
        cand = [(256, 4, 3)]
        launches = ew_synthetic.synthesise(ew_canary_kernel, CANARY_SPEC, cand)[:1]
        where, _n = ew_proof_child.write_job(ew_canary_kernel, launches, cand, {}, guard)
        try:
            results, _d = ew_proof_child.run_candidates(where, cand)
            seen = results[cand[0]][0] == "fault"
            _CANARY[guard] = "" if seen else ("the canary's overrun one element past its buffer was NOT seen by the "
                                              f"guard ({guard}): it does not guard on this device")
        except ew_proof_child.ProofCouldNotRun as why:
            _CANARY[guard] = f"the guard's canary could not run ({why})"
            _CANARY_UNRUN.add(guard)
        finally:
            import shutil
            shutil.rmtree(where, ignore_errors=True)
        print(f"[ew-proof] the guard's canary ({guard}): " + ("seen" if not _CANARY[guard] else _CANARY[guard]),
              file=sys.stderr)
    return _CANARY[guard]


def _unkept(kernel, guard: str, why: str) -> dict:
    """A refusal that measured nothing (`prove`): the record `settle` raises, never written to a table."""
    print(f"[ew-proof] {kernel.__name__}: NOT proven — {why}", file=sys.stderr)
    return {"independent": None, "faults": [], "dissent": [], "passed": [], "refused": why, "kernel": kernel.__name__,
            "guard": guard, "unkept": True}


_RUN = {"variants": 0, "seconds": 0.0, "largest": 0}


def _count(seconds: float, nbytes: int) -> None:
    """This run's proofs, said once at its end: what a first launch with no shipped table paid."""
    if _RUN["variants"] == 0:
        import atexit
        atexit.register(_say_run)
    _RUN["variants"] += 1
    _RUN["seconds"] += seconds
    _RUN["largest"] = max(_RUN["largest"], nbytes)


def _say_run() -> None:
    """At exit: the run's proofs in one line (variants, seconds, worker processes, largest synthetic job)."""
    from nbxtensor.ops.ew_proof_child import _Worker
    print(f"[ew-proof] this run proved {_RUN['variants']} element-wise variant(s) in {_RUN['seconds']:.1f} s with "
          f"{_Worker.started} child process(es); largest synthetic job {_RUN['largest']} bytes", file=sys.stderr)


def judge(independent, cands, results, differ) -> dict:
    """The record of one variant from its candidates' results on the same synthetic launches ({tile: ('fault', text)
    | ('ran', digests)}): faults excluded, the agreeing set (more than half of the non-faulting, at least FLOOR) passed
    for a tile-independent kernel, every non-faulting one for another; `differ(digests, agreeing digests)` names a
    dissenting candidate's first differing element."""
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


# ---------------------------------------------------------------------------
# the shipped table, from the proofs runs made
# ---------------------------------------------------------------------------
def export(kept_path: str, shipped_path: str) -> str:
    """Merge the proofs a run kept (`kept_path`, a replay cache's ew_tile_proof.json) into a shipped table
    (`shipped_path`, <certified directory>/<vendor>/<profile>/ew_tile_proof.json), stamp by stamp. A variant the
    two hold differently is refused by name: a table is never silently rewritten. Returns a one-line summary."""
    kept = json.load(open(kept_path))
    try:
        shipped = json.load(open(shipped_path))
    except OSError:
        shipped = {"header": HEADER, "stamps": {}}
    added = 0
    for key, block in kept.get("stamps", {}).items():
        dest = shipped["stamps"].setdefault(key, {"stamp": block["stamp"], "variants": {}})
        for vid, rec in block.get("variants", {}).items():
            have = dest["variants"].get(vid)
            if have is not None and sorted(have["passed"]) != sorted(rec["passed"]):
                raise TileUnproven(f"ZERO FALLBACK: {rec.get('kernel')} variant {vid}: the shipped table passes "
                                   f"{sorted(have['passed'])}, this run's proof {sorted(rec['passed'])}")
            if have is None:
                dest["variants"][vid] = rec
                added += 1
    os.makedirs(os.path.dirname(os.path.abspath(shipped_path)), exist_ok=True)
    with open(shipped_path, "w") as fh:
        json.dump(shipped, fh, indent=1, sort_keys=True)
    dep = sorted({r.get("kernel") for b in shipped["stamps"].values() for r in b["variants"].values()
                  if not r.get("independent")})
    return (f"{added} variant(s) added to {shipped_path}; proven by fault only (tile-dependent or undeclared): "
            f"{dep or 'none'}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "export":
        print(export(sys.argv[2], sys.argv[3]))
    else:
        raise SystemExit("usage: python -m nbxtensor.ops.ew_proof export <kept ew_tile_proof.json> <shipped path>")
