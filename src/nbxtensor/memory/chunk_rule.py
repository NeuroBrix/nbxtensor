"""The expandable range's chunk BY RULE per memory class, not by a sweep at every first launch (orders row 211,
2026-11-22: the sweep cost 40-48 s of a 62 s first launch and chose 16 MiB once, 32 MiB nine times in ten runs).

THE RULE. On a card of memory class C (whole GiB rounded up: `16g`) with an entry in `chunk_rule.yml`, the chunk
is the LARGEST chunk of C's measured grid that (1) replayed faster than the driver segments (R16 OFF) on a card of C,
(2) is a whole number of the device's VMM granularity, and (3) is within the allocator's bound below. A class with no
entry answers None: it is measured once (`trace_replay.chunk_by_replay`) and kept, as before.

THE BOUND: what a too-large chunk costs the allocator. A refusal's reclaim gives back only the mapped chunks lying
WHOLE inside a free block (`pool.range_reclaimable`); a free block keeps mapped the two partial chunks at its ends, and
the range keeps the partial chunk above its top. At the carried run's live peak, with B large blocks live, there are at
most B + 1 free gaps, so the reclaim can strand up to (2 (B + 1) + 1) chunks. Those must fit in what the card leaves
over the run's live peak (card bytes - peak live bytes), or a run the driver segments serve is refused under the
expandable range. Bound = that slack / (2 (B + 1) + 1), read from the class's carried trace (`traces/`), no figure of
its own. On the carried 16 GB trace: slack 2 500 MiB, B = 15 large blocks live at the peak, 33 chunks -> 75.7 MiB, so
64 MiB (the largest granular power of two under it) holds and 128 MiB does not — on the NOMINAL 16 384 MiB. The bound
is taken on the card's own room (qa 1232-1): a V100-SXM2-16GB leaves 16 160 - 306 MiB, the bound is 59.7 MiB, and
the rule gives it 32 MiB, not 64. The 32g class carries the 16 GB trace (no 32 GB trace was recorded): its bound is that
run's on the 32 GB card's room (qa 1232-3).

THE 64 MiB FIGURES (the grid of `chunk_rule.yml`): 16g 2.114 s against OFF 2.656 s (32 MiB 2.347, 16 MiB 3.105);
32g 1.714 s against OFF 2.027 s (32 MiB 1.983). The allocator's own replay (`pool.simulate`, card-less) of the trace at
64 MiB under the card: 989 maps, 807 unmaps, no refusal (32 MiB: 1 944 / 1 579). Nothing is extrapolated past the
largest chunk measured on a card: an unmeasured chunk is never the rule's answer.

THE ROOM (C18, the Dell 23/11: a granite server refused on a card a neighbour held, "the carried trace sana-video...
peaks at 14559087916 live bytes, at or over the room this card leaves (13505 MiB)"): the card's total less the runtime
context of ITS device class, a value of the library's device file (`memory.runtime_context_mb`), READ — never total
less free, which counts a neighbour's memory as this card's context. Only a class the rule answers reads it, and that
class must carry its measured figure (null or absent is refused by name); a class the rule does not answer reads
nothing (its file's null says: not measured on that class, and nothing asks).
THE RULE THAT CANNOT ANSWER: a class's reference run that does not fit the present room is no reason to refuse a run.
The rule then answers `ThisRunOnly(0)`: this run takes the driver segments, said once, and NOTHING is kept (a chunk
chosen under a neighbour would poison later launches); a user model that truly does not fit is refused by its own
allocation, by name.

Torch-free, numpy-free."""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

RULE_FILE = Path(__file__).with_name("chunk_rule.yml")
MiB = 1 << 20
_GiB = 1 << 30


@functools.lru_cache(maxsize=1)
def _classes() -> Dict[str, dict]:
    return dict((yaml.safe_load(RULE_FILE.read_text()) or {}).get("classes") or {})


def _class_of(card_bytes: int) -> str:
    """A card's memory class: its memory in whole GiB, rounded up (a V100-SXM2-16GB's 16 160 MiB: `16g`)."""
    return f"{-(-int(card_bytes) // _GiB)}g"


class ThisRunOnly(int):
    """A chunk for this run only (0: the driver segments): the rule could not answer on the card as it stands, so the
    value is neither the class's rule nor a measurement, and no caller keeps it."""


class RoomTooSmall(RuntimeError):
    """The class's reference run does not fit the room this card leaves now: the rule cannot bound a chunk."""


_SAID: set = set()


@functools.lru_cache(maxsize=None)
def bound_bytes(trace: str, room_bytes: int) -> int:
    """The largest chunk the allocator can strand without refusing the carried run `trace` on a card that leaves
    `room_bytes` to the allocator: (room - the run's live peak) / (2 (B + 1) + 1), B = the large-class blocks live at
    that peak. The room is the CARD's (its memory less the runtime's context, read from the driver), not the trace's
    nominal class size (qa 1232-1: a V100-SXM2-16GB states 16 160 MiB and holds a 306 MiB context: 59.7 MiB, where the
    nominal 16 384 MiB gave 75.7)."""
    from nbxtensor.memory.pool import POOL_SMALL_BYTES
    from nbxtensor.memory.trace_replay import TRACES_DIR, load_trace
    meta, events = load_trace(TRACES_DIR / trace)
    sizes: Dict[int, Tuple[str, int]] = {}
    live = peak = blocks_at_peak = 0
    for kind, ident, nbytes in events:
        if kind == "F":
            live -= sizes.pop(ident, ("", 0))[1]
            continue
        sizes[ident] = (kind, nbytes)
        live += nbytes
        if live > peak:
            peak = live
            blocks_at_peak = sum(1 for k, n in sizes.values() if k == "M" and n > POOL_SMALL_BYTES)
    slack = int(room_bytes) - peak
    if slack <= 0:
        raise RoomTooSmall(f"the class's reference run {trace} peaks at {peak // MiB} MiB live, at or over the room "
                           f"this card leaves ({int(room_bytes) // MiB} MiB)")
    return slack // (2 * (blocks_at_peak + 1) + 1)


def rule_chunk_bytes(card_bytes: int, granularity: int, room_bytes: int, architecture: str) -> Optional[int]:
    """The chunk the rule gives a card of `card_bytes` and `architecture` whose VMM maps `granularity`-byte units and
    which leaves `room_bytes` to the allocator; None when its memory class has no entry, the entry's grid was measured
    on another architecture (qa 1232-4: the grid is the V100's; a 16 GB card of another make is not proven by it), or
    no measured chunk passes the rule — the class is then measured once and kept."""
    entry = _classes().get(_class_of(card_bytes))
    if not entry or int(granularity) <= 0 or str(entry.get("architecture")) != str(architecture):
        return None
    grid = {int(k): float(v) for k, v in (entry.get("grid_seconds") or {}).items()}
    if 0 not in grid:
        raise RuntimeError(f"ZERO FALLBACK: {RULE_FILE.name} class {_class_of(card_bytes)} states no OFF (0) time: "
                           f"no chunk can be judged faster than the driver segments")
    try:
        bound = bound_bytes(str(entry["trace"]), int(room_bytes))
    except RoomTooSmall as exc:
        if str(exc) not in _SAID:
            _SAID.add(str(exc))
            print(f"[chunk rule] {exc}: the rule does not answer on this card as it stands; this run takes the driver "
                  f"segments, and nothing is kept", file=__import__("sys").stderr)
        return ThisRunOnly(0)
    fit = [mb for mb, s in grid.items() if mb > 0 and s < grid[0] and (mb * MiB) % int(granularity) == 0
           and mb * MiB <= bound]
    return max(fit) * MiB if fit else None


def vmm_chunk_bytes(device: int, card_bytes: int) -> Optional[int]:
    """The rule's chunk for `device` of `card_bytes`, or None: no VMM, or no rule entry for its memory class and
    architecture — the caller measures once (`trace_replay.chunk_by_replay`) and keeps the result. ONE reader decides
    (qa C18-2): the device file in force (`_configs.active_vendor_profile`) gives the class's architecture AND its
    runtime context; the room is the card's total (the driver's) less that context, the same at every ask whatever a
    neighbour holds; a neighbour seen is said once."""
    from nbxtensor.memory.trace_replay import vmm_granularity
    gran = vmm_granularity(device)
    if not gran:
        return None
    from nbxtensor.ops._configs import active_vendor_profile
    profile = active_vendor_profile()
    architecture = str(profile.get("architecture") or "")
    entry = _classes().get(_class_of(card_bytes))
    if not entry or str(entry.get("architecture")) != architecture:
        return None                                  # no rule for this class: nothing is read, the caller measures
    from nbxtensor.memory.allocator import DeviceAllocator
    free_b, total_b = DeviceAllocator.mem_get_info(device)
    context = _context_bytes(profile)
    others = max(0, int(total_b) - int(free_b) - DeviceAllocator._cuda_live_bytes.get(device, 0) - context)
    if others and ("others", device) not in _SAID:
        _SAID.add(("others", device))
        print(f"[chunk rule] device {device}: another process holds about {others // MiB} MiB of this card; the room "
              f"is the card's own (total {int(total_b) // MiB} MiB less its runtime context {context // MiB} MiB)",
              file=__import__("sys").stderr)
    return rule_chunk_bytes(card_bytes, gran, int(total_b) - context, architecture)


def _context_bytes(profile: dict) -> int:
    """The runtime context of the card's device class, READ from its device file `profile` (`memory.
    runtime_context_mb`), for a class the chunk rule answers. That class must carry its measured figure: null (not
    measured on the class) or nothing is refused by name — the rule never measures the context itself (RED_LINES 5)."""
    memory = profile.get("memory") or {}
    stated = memory.get("runtime_context_mb")
    if stated is None:
        raise RuntimeError("ZERO FALLBACK: the chunk rule answers this card's class, and its device file states no "
                           "measured memory.runtime_context_mb (" + ("null" if "runtime_context_mb" in memory else
                           "absent") + "): the figure is added at its source, measured on a card of the class")
    return int(stated) * MiB