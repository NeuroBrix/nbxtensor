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
64 MiB (the largest granular power of two under it) holds and 128 MiB does not.

THE 64 MiB FIGURES (the grid of `chunk_rule.yml`): 16g 2.114 s against OFF 2.656 s (32 MiB 2.347, 16 MiB 3.105);
32g 1.714 s against OFF 2.027 s (32 MiB 1.983). The allocator's own replay (`pool.simulate`, card-less) of the trace at
64 MiB under the card: 989 maps, 807 unmaps, no refusal (32 MiB: 1 944 / 1 579). Nothing is extrapolated past the
largest chunk measured on a card: an unmeasured chunk is never the rule's answer.

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


@functools.lru_cache(maxsize=None)
def bound_bytes(trace: str) -> int:
    """The largest chunk the allocator can strand without refusing the carried run `trace` on the card it was recorded
    on: (card bytes - the run's live peak) / (2 (B + 1) + 1), B = the large-class blocks live at that peak."""
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
    slack = int(meta["card_mb"]) * MiB - peak
    if slack <= 0:
        raise RuntimeError(f"ZERO FALLBACK: the carried trace {trace} peaks at {peak} live bytes, at or over its card "
                           f"({meta['card_mb']} MiB): it states no room for any chunk")
    return slack // (2 * (blocks_at_peak + 1) + 1)


def rule_chunk_bytes(card_bytes: int, granularity: int) -> Optional[int]:
    """The chunk the rule gives a card of `card_bytes` whose VMM maps `granularity`-byte units; None when its memory
    class has no entry (measure once and keep it) or no measured chunk passes the rule."""
    entry = _classes().get(_class_of(card_bytes))
    if not entry or int(granularity) <= 0:
        return None
    grid = {int(k): float(v) for k, v in (entry.get("grid_seconds") or {}).items()}
    if 0 not in grid:
        raise RuntimeError(f"ZERO FALLBACK: {RULE_FILE.name} class {_class_of(card_bytes)} states no OFF (0) time: "
                           f"no chunk can be judged faster than the driver segments")
    bound = bound_bytes(str(entry["trace"]))
    fit = [mb for mb, s in grid.items() if mb > 0 and s < grid[0] and (mb * MiB) % int(granularity) == 0
           and mb * MiB <= bound]
    return max(fit) * MiB if fit else None


def vmm_chunk_bytes(device: int, card_bytes: int) -> Optional[int]:
    """The rule's chunk for `device` (its VMM granularity read from the driver), or None: no VMM, or no rule entry for
    its class — the caller measures once (`trace_replay.chunk_by_replay`) and keeps the result."""
    from nbxtensor.memory.trace_replay import vmm_granularity
    gran = vmm_granularity(device)
    return rule_chunk_bytes(card_bytes, gran) if gran else None
