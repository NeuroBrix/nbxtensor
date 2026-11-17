"""Where a real run's GPU time goes, kernel by kernel: `kernel_times()` opens a profiling session around the work the
caller runs and returns the table when it closes.

    with nbxtensor.profile.kernel_times() as table:
        ...                     # the work to rank
    print(table.format())

On Metal every launch (and every copy, as one row 'blit') runs alone and is timed on the GPU at its own boundaries
(stage-boundary timestamps), which changes the batching: the table RANKS kernels; it is not the production step's
time, and its header says so with the session's wall time beside the summed GPU time. On CUDA every launch and every
copy sits between two timing events on its stream and the run keeps its own cadence (`backend.cuda_profile`): the
table also says the GPU time between kernels and what the events themselves cost. Another backend is refused by name.
One session per process.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Tuple


class KernelTimes:
    """The table a session fills: per kernel name, launches, total / max GPU ms; the session's wall time and summed
    GPU time. Backend-neutral: the session that fills it says how it timed (`note` in the header, `measures` in the
    JSON; Metal's by default: each launch and each copy ran alone, the figures rank kernels and are not the production
    step's time). A session that keeps the run's own cadence also fills `cadence` (CUDA): the GPU time between
    kernels (`between_ms` over `gaps`), the host time of the session's own calls (`own_ms`), the most the figures
    overstate the kernels (`overstated_ms`), the wall time no measured interval holds (`unattributed_ms`) and the
    launches whose time is an upper bound (`bounded`, and per kernel `bounded_by_kernel`)."""

    def __init__(self) -> None:
        self.rows: Dict[str, List[float]] = {}          # name -> [launches, total ms, max ms]
        self.wall_s = 0.0
        self.unsampled = 0                              # launches whose samples the device did not take
        self.note = "each launch and copy timed alone: a ranking, not the step's time"
        self.measures = "every launch and copy alone: the figures rank kernels, not the production step"
        self.cadence: Optional[dict] = None

    def add(self, name: str, ms: float) -> None:
        row = self.rows.setdefault(name, [0, 0.0, 0.0])
        row[0] += 1
        row[1] += ms
        row[2] = max(row[2], ms)

    @property
    def gpu_ms(self) -> float:
        return sum(r[1] for r in self.rows.values())

    def ranked(self) -> List[Tuple[str, int, float, float]]:
        """(name, launches, total ms, max ms), the heaviest total first."""
        return sorted(((n, int(r[0]), r[1], r[2]) for n, r in self.rows.items()), key=lambda r: -r[2])

    def as_dict(self) -> dict:
        said = {"wall_s": self.wall_s, "gpu_ms": self.gpu_ms, "unsampled": self.unsampled, "measures": self.measures}
        if self.cadence is not None:
            said["cadence"] = dict(self.cadence)
        said["kernels"] = [{"name": n, "launches": c, "total_ms": t, "max_ms": m} for n, c, t, m in self.ranked()]
        return said

    def report(self, path=None, top: int = 25) -> str:
        """The printed table; with `path`, the whole table also written there as JSON (said on the last line)."""
        text = self.format(top)
        if path:
            import json
            from pathlib import Path
            Path(path).write_text(json.dumps(self.as_dict(), indent=1))
            text += f"\n[kernel-times] the whole table: {path}"
        return text

    def format(self, top: int = 25) -> str:
        rows = self.ranked()
        gpu = self.gpu_ms or 1.0
        out = [f"[kernel-times] wall {self.wall_s:.3f} s, GPU {self.gpu_ms:.1f} ms summed over "
               f"{sum(r[1] for r in rows)} launches ({self.note})"] + self._cadence_lines() + [
               f"[kernel-times]   {'kernel':<48} {'launches':>8} {'total ms':>10} {'mean ms':>8} {'max ms':>8} {'share':>6}"]
        for n, c, t, m in rows[:top]:
            out.append(f"[kernel-times]   {n[:48]:<48} {c:>8} {t:>10.2f} {t / c:>8.4f} {m:>8.3f} {100 * t / gpu:>5.1f}%")
        rest = rows[top:]
        if rest:
            t = sum(r[2] for r in rest)
            out.append(f"[kernel-times]   {f'the rest ({len(rest)} kernels)':<48} {sum(r[1] for r in rest):>8} "
                       f"{t:>10.2f} {'':>8} {'':>8} {100 * t / gpu:>5.1f}%")
        return "\n".join(out)

    def _cadence_lines(self) -> List[str]:
        """The header's lines of a session that kept the run's cadence: the time between kernels, what the session
        itself cost the host, and how far its figures can overstate the kernels."""
        c = self.cadence
        if c is None:
            return []
        return [f"[kernel-times]   between kernels: {c['between_ms']:.1f} ms of GPU time over {c['gaps']} gaps (the "
                f"device waiting for its next launch); {c['own_ms']:.1f} ms of host time went to the session's own calls",
                f"[kernel-times]   the GPU figure overstates the kernels by at most {c['overstated_ms']:.1f} ms (a "
                f"launch's submission); {c['bounded']} launches measured no longer than their submission"] + (
            [f"[kernel-times]   unattributed: {c['unattributed_ms']:.1f} ms of the {c['wall_ms']:.1f} ms wall (what no "
             f"measured interval holds: host time, a pool's read; the table above and the time between kernels never "
             f"overlap)"] if "unattributed_ms" in c else [])


def ranked_call(fn, path=None, out=print):
    """`fn()` inside a session; its table handed to `out` (and written to `path` as JSON); `fn`'s result returned."""
    with kernel_times() as table:
        result = fn()
    out(table.report(path))
    return result


@contextmanager
def kernel_times() -> Iterator[KernelTimes]:
    """A profiling session around the caller's work; the table is filled when the block ends (also on an error)."""
    opened, close = _session_of_this_backend()
    try:
        yield opened
    finally:
        close()


def _session_of_this_backend():
    """(the table of a session just opened on the backend in force, the call that closes it)."""
    from nbxtensor.backend.runtime import gpu_backend
    backend = gpu_backend()
    if backend == "metal":
        from nbxtensor.backend.metal_device import runtime
        rt = runtime()
        return rt.open_kernel_times(), rt.close_kernel_times
    if backend == "cuda":
        from nbxtensor.backend import cuda_profile
        return cuda_profile.open_session(), cuda_profile.close_session
    raise RuntimeError(f"ZERO FALLBACK: per-kernel GPU time is measured on Metal and on CUDA; this backend is "
                       f"{backend!r}")
