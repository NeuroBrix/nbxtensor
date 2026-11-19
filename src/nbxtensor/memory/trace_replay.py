"""The expandable range's chunk, measured by replaying a REAL run's allocations (the supervisor 2026-11-02 17:15).

A run's mapping cost is what its refusals' reclaims map again (SANA-Video_2B_720p on a 16 GB V100: 89 GB mapped over a
run at a 2 MiB chunk — registres/benchmarks/vmm_v100.md), and only the run's own allocation order produces those
refusals; a synthetic churn does not (it always favours the largest chunk). So the library carries recorded traces as
data (`traces/*.tsv`: every allocation and free of one gate cell, in order, by id and bytes — no model, no pointer),
and the chunk probe replays one through THIS allocator, on the device, under the card size the trace was recorded on
(`card_mb`, the imposed limit — the trace presses that class on any larger card), once with the expandable range off
(the driver segments: R16 OFF) and once per candidate chunk, each in its own process (a clean allocator).

The chunk is the smallest candidate whose replay costs no more than R16 OFF's replay plus OFF's own spread (max -
median over the repetitions): the expandable range then holds what it saves (the 32g replay: 8 GB less at its peak)
without paying for it in time. None when no candidate does — said, and the caller keeps the driver segments.

Torch-free, numpy-free."""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

TRACES_DIR = Path(__file__).parent / "traces"
_MARK = "NBX-TRACE-REPLAY "      # the child's result line, whatever else it prints


def traces() -> List[Path]:
    """The traces the library carries, by name."""
    return sorted(TRACES_DIR.glob("*.tsv"))


def load_trace(path) -> Tuple[Dict[str, str], List[Tuple[str, int, int]]]:
    """(the trace's header facts `# key: value`, its events (kind, id, nbytes)). A trace without `card_mb` or without
    one event is refused by name."""
    meta: Dict[str, str] = {}
    events: List[Tuple[str, int, int]] = []
    for line in open(path):
        if line.startswith("#"):
            key, sep, value = line[1:].strip().partition(": ")
            if sep and " " not in key:
                meta[key] = value.strip()
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 3 and parts[0] in ("M", "A", "F"):
            events.append((parts[0], int(parts[1]), int(parts[2])))
    if "card_mb" not in meta or not events:
        raise RuntimeError(f"ZERO FALLBACK: the allocation record {path} states no card_mb or holds no event")
    return meta, events


def replay_trace(events: Sequence[Tuple[str, int, int]], device: int, chunk: int, limit_bytes: int,
                 stop_after: Optional[float] = None) -> Dict[str, float]:
    """Replay `events` through this process's allocator on `device` under `limit_bytes`, the large class in an
    expandable range of `chunk` bytes (0: the driver segments). The wall seconds, the mapping seconds, the reclaims and
    the peak held. `stop_after`: the seconds past which this replay can no longer change the sweep's decision (the
    bound R16 OFF set, `chunk_by_replay`) — the replay stops there, `beyond` True, its seconds those it had reached
    (always past the bound: the seconds only grow)."""
    from nbxtensor.memory.allocator import DeviceAllocator as DA
    DA.set_device(device)
    DA.use_expandable(device, chunk, range_bytes=limit_bytes)
    DA.set_limit(limit_bytes, device)
    live: Dict[int, int] = {}
    t0 = time.perf_counter()
    for kind, key, nbytes in events:
        if stop_after is not None and time.perf_counter() - t0 > stop_after:
            st = DA._POOL.stats
            return {"seconds": time.perf_counter() - t0, "beyond": True,
                    "map_seconds": st.get("range_map_us", 0) / 1e6, "reclaims": st.get("range_reclaims", 0),
                    "peak_bytes": DA._cuda_peak_bytes.get(device, 0)}
        if kind == "F":
            ptr = live.pop(key, None)
            if ptr is not None:
                DA.free_cuda(ptr)
        elif nbytes:
            live[key] = DA.malloc_cuda(nbytes, device, pooled=(kind == "M"))
    DA.device_synchronize(device)
    seconds = time.perf_counter() - t0
    st = DA._POOL.stats
    return {"seconds": seconds, "map_seconds": st.get("range_map_us", 0) / 1e6,
            "reclaims": st.get("range_reclaims", 0), "peak_bytes": DA._cuda_peak_bytes.get(device, 0)}


def _child(trace: str, device: int, chunk: int, stop_after: Optional[float] = None) -> Dict[str, float]:
    """One replay in a fresh process (no allocator state carried between candidates), shown `device` ALONE (the
    backend's `visible_env`, `backend.runtime.one_device_env`): it replays on its device 0 and never initialises a
    runtime on another card — on a 4-card rack every child saw the four (the 2026-11-19 first launch)."""
    from nbxtensor.backend.runtime import one_device_env
    code = ("import json, sys; from nbxtensor.memory import trace_replay as T; "
            "m, ev = T.load_trace(sys.argv[1]); "
            "print('" + _MARK + "' + json.dumps(T.replay_trace(ev, int(sys.argv[2]), int(sys.argv[3]), "
            "int(m['card_mb']) << 20, float(sys.argv[4]) if sys.argv[4:] else None)))")
    # the replay measures the allocator, never its diagnostics: the parent's run-level switches stay out of it
    env = {k: v for k, v in one_device_env(device).items()
           if not k.startswith(("NBX_ALLOC_STATS", "NBX_MALLOC_TRACE"))}
    r = subprocess.run([sys.executable, "-c", code, trace, "0", str(chunk)]
                       + ([repr(float(stop_after))] if stop_after is not None else []),
                       capture_output=True, text=True, env=env)
    line = next((l for l in r.stdout.splitlines() if l.startswith(_MARK)), None)
    if r.returncode != 0 or line is None:
        raise RuntimeError(f"the allocation-record replay of {Path(trace).name} at chunk {chunk} failed: "
                           f"{(r.stderr or r.stdout)[-400:]}")
    return json.loads(line[len(_MARK):])


def _the_trace(trace: Optional[str]) -> str:
    if trace is None:
        carried = traces()
        if len(carried) != 1:
            raise RuntimeError(f"ZERO FALLBACK: the library carries {len(carried)} allocation records "
                               f"({', '.join(p.name for p in carried)}): the chunk probe names the one it replays")
        trace = str(carried[0])
    return trace


def _give_back(device: int) -> None:
    # every replay runs in a child sized at the recorded card: what THIS process caches on the device is given back
    # first, or the child finds the card short by it (a test process's pools after the kernels suite: 16 GB, the
    # replay out of memory at chunk 0 — order-dependent, 2026-11-03)
    from nbxtensor.memory.allocator import DeviceAllocator as DA
    DA.device_synchronize(device)
    if DA._range_vmm(device) is not None:
        DA._range_reclaim(device, None)
    DA._pool_flush(device)


def vmm_granularity(device: int) -> Optional[int]:
    """The device's VMM granularity (the chunk probe's unit), or None when it has no VMM: nothing to replay."""
    from nbxtensor.backend import runtime as _rt
    from nbxtensor.memory.vmm import Vmm
    lib = Vmm.load(_rt.backend_table().get("driver_libs"))
    v = Vmm(lib, device) if lib is not None else None
    rc, gran = v.granularity() if v is not None and v.supported() else (1, 0)
    return gran if rc == 0 and gran > 0 else None


def first_replay(device: int, trace: Optional[str] = None) -> Tuple[float, float]:
    """The first replay `chunk_by_replay` makes (R16 OFF), made ahead: (its replay seconds, to hand to
    `chunk_by_replay(off_first=)`, and the child's whole wall seconds — what one candidate costs on this machine,
    process start included)."""
    _give_back(device)
    t0 = time.perf_counter()
    seconds = _child(_the_trace(trace), device, 0)["seconds"]
    return seconds, time.perf_counter() - t0


def decided(seconds: Sequence[float], reps: int, bound: float) -> Optional[bool]:
    """Whether `reps` replays whose first ones took `seconds` have a median within `bound` — True, False, or None while
    the replays still to come can change it. The median of `reps` lies within the bound exactly when more than half of
    them do (odd `reps`: the middle one; even: both middle ones); past half on one side, the rest cannot move it. An
    even count split half and half is the average of the two middle values: decided only once all are in."""
    within = sum(1 for t in seconds if t <= bound)
    beyond = len(seconds) - within
    if within > reps // 2:
        return True
    if beyond > reps // 2:
        return False
    if len(seconds) >= reps:
        return statistics.median(seconds) <= bound
    return None


def chunk_by_replay(device: int, granularity: int, ks: Sequence[int], reps: int,
                    trace: Optional[str] = None, off_first: Optional[float] = None, child=None
                    ) -> Tuple[Optional[int], Dict[int, List[float]]]:
    """(the chunk, every candidate's replay seconds — 0 is R16 OFF). The smallest granularity x k whose median replay
    costs no more than OFF's median plus OFF's spread; None when none does. `trace`: one of `traces()` (the only one
    the library carries when it carries one; several = the caller names it). `off_first`: the first OFF replay, made
    ahead by `first_replay`. `child`: the replay (`_child`'s signature); a test hands its own.

    The sweep stops where the measure has decided, never by a constant: a candidate's replays stop once more than half
    of them lie on one side of OFF's bound (`decided`: the median is then settled), and with an odd `reps` each replay
    stops at the bound itself (`replay_trace(stop_after=)`: a replay past it counts as past it whatever it would have
    reached) — the decision is the full sweep's, replay for replay (a 2 MiB replay ran 75-85 s on the rack against a
    bound near 3 s, 2026-11-19). A replay stopped at the bound keeps the seconds it had reached.
    The seconds returned for a candidate are the replays it ran before its stop: a candidate decided early returns
    fewer than `reps` of them (the decision is still its median's)."""
    child = _child if child is None else child
    trace = _the_trace(trace)
    _give_back(device)
    times: Dict[int, List[float]] = {0: ([off_first] if off_first is not None else [])
                                        + [child(trace, device, 0)["seconds"]
                                           for _ in range(reps - (off_first is not None))]}
    off = times[0]
    bound = statistics.median(off) + (max(off) - statistics.median(off))
    stop = bound if reps % 2 else None          # odd: the count alone decides, a replay past the bound need not finish
    for chunk in [granularity * int(k) for k in sorted(ks)]:     # ascending: the first within OFF's noise is the chunk
        times[chunk] = []
        verdict = None
        while verdict is None:
            times[chunk].append(child(trace, device, chunk, stop)["seconds"])
            verdict = decided(times[chunk], reps, bound)
        if verdict:
            return chunk, times
    return None, times
