"""The child side of the element-wise tile proof (`ops/ew_proof.py`), and the parent's handling of its children.

A fault on the device (an illegal address) is sticky: the process that meets it can launch nothing after. So each
candidate tile runs in a child process with CUDA_LAUNCH_BLOCKING=1 on a SNAPSHOT of the real launch: every pointer
argument's addressable bytes, arguments that overlap kept in one buffer (an in-place kernel reads what it writes),
each pointer at its own offset (its alignment, which the binary is specialised on, unchanged). Before each candidate
every buffer is restored; after it, every buffer's bytes are digested — the launch's writes are not named, so all of
them are compared. A child that dies mid-candidate names that candidate's fault; a fresh child resumes at the next.

    python -m nbxtensor.ops.ew_proof_child run  <spec dir> <first candidate>
    python -m nbxtensor.ops.ew_proof_child time <spec dir>
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple

# The spec is pickled by the parent into its own private temporary directory (mkdtemp, mode 0700) and read back only
# by the child it starts: it carries dtype objects and constexpr values JSON cannot hold. Nothing outside this
# process pair writes it.

_ALIGN = 256                 # a snapshot starts on this boundary: every pointer keeps its alignment modulo 16
_CHILD_SECONDS = 600         # a candidate that hangs this long is recorded as a fault, not waited on


class _View:
    """A pointer argument rebuilt in the child: what the launcher reads of a tensor (address, dtype, extent)."""

    def __init__(self, address, dtype, shape, strides, elem):
        """A rebuilt pointer argument: address, dtype, shape, strides, element size."""
        self._address, self.dtype, self.shape, self._strides, self._elem = address, dtype, shape, strides, elem

    def data_ptr(self):
        """The rebuilt pointer address, aligned as the original."""
        return self._address

    def stride(self):
        """The strides, in elements."""
        return self._strides

    def element_size(self):
        """The bytes one element of the argument takes."""
        return self._elem


# ---------------------------------------------------------------------------
# parent: the snapshot
# ---------------------------------------------------------------------------
def snapshot_launch(kernel, grids, bound: Dict[str, Any], options: Dict[str, Any]) -> Tuple[str, int]:
    """(spec directory, bytes snapshotted) of this launch."""
    from nbxtensor.backend.launcher import _addressable_extent, _snapshot_ranges
    spans = {}
    for name, v in bound.items():
        if hasattr(v, "data_ptr") and hasattr(v, "stride"):
            ext = _addressable_extent(v)
            if ext is None:
                from nbxtensor.ops.ew_proof import TileUnproven
                raise TileUnproven(f"ZERO FALLBACK: {kernel.__name__}: argument {name} has a negative stride; its "
                                   f"bytes cannot be snapshotted for the tile proof")
            spans[name] = (int(v.data_ptr()), int(v.data_ptr()) + max(int(ext), 1))
    groups = _merge(sorted(spans.values()))
    where = tempfile.mkdtemp(prefix="nbx_ew_proof_")
    params, total = {}, 0
    for gi, (lo, hi) in enumerate(groups):
        blob = _snapshot_ranges(lo, [(0, hi - lo)])
        total += len(blob)
        with open(os.path.join(where, f"g{gi}.bin"), "wb") as fh:
            fh.write(blob)
    for name, v in bound.items():
        if name in spans:
            gi, lo = next((i, g[0]) for i, g in enumerate(groups) if g[0] <= spans[name][0] < g[1])
            params[name] = ("ptr", gi, spans[name][0] - lo, v.dtype, tuple(int(d) for d in v.shape),
                            tuple(int(s) for s in v.stride()), int(v.element_size()))
        else:
            params[name] = ("val", v)
    spec = {"module": kernel.fn.__module__, "name": kernel.__name__, "params": params,
            "groups": [hi - lo for lo, hi in groups], "grids": [list(g) for g in grids],
            "options": {k: v for k, v in options.items() if k not in ("num_warps", "num_stages")}}
    with open(os.path.join(where, "spec.pkl"), "wb") as fh:
        pickle.dump(spec, fh)
    return where, total


def _merge(spans):
    """Overlapping or touching spans as one group each, its start on the alignment boundary."""
    out: List[List[int]] = []
    for lo, hi in spans:
        lo = lo - lo % _ALIGN
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [tuple(g) for g in out]


# ---------------------------------------------------------------------------
# parent: the children
# ---------------------------------------------------------------------------
def run_candidates(where: str, cands) -> Tuple[Dict[tuple, tuple], Any]:
    """{tile: ('fault', text) | ('ran', digests)} over every candidate, and the differ of two digest tuples."""
    results: Dict[tuple, tuple] = {}
    start = 0
    while start < len(cands):
        with open(os.path.join(where, "cands.json"), "w") as fh:
            json.dump([list(c) for c in cands], fh)
        rc, err = _child(["run", where, str(start)])
        opened, any_started = _read_out(where, start, results, cands)
        if len(results) == len(cands):
            break
        if not any_started:
            from nbxtensor.ops.ew_proof import TileUnproven
            raise TileUnproven("ZERO FALLBACK: the tile proof's child process could not start (no device context "
                               f"beside the loaded model?); the unproven tile is not used. Child said: {err[-800:]}")
        at = opened if opened is not None else len(results)
        results[tuple(cands[at])] = ("fault", err[-1200:].strip() or f"child exit {rc}")
        start = at + 1
    return results, lambda a, b: first_difference(where, a, b)


#: `_Worker.ask`'s code for a job the worker did not answer within its time (the worker killed): a TIMEOUT, told apart
#: from the exit code of a worker that died on the job (a signal's is -1 to -64)
TIMED_OUT = -1000


class _Worker:
    """ONE child process that serves every proof of the run (orchestrator 23/11: a process and a device context per
    variant would put seconds times the variant count into a first launch). Started at the first proof; started
    again only after a fault killed it. Requests and replies are JSON lines on its stdin / a duplicate of its
    stdout; its own prints go to stderr, kept in a file so a fault's text can be read."""
    started = 0                      # children started in this run (the record's measure)

    def __init__(self):
        """No process yet: it starts at the first job."""
        self.proc = None
        self.err = None

    def ask(self, job: dict, timeout: float = None) -> Tuple[int, str]:
        """(0, '') when the job finished; (exit code, its stderr since the job began) when the child died on it."""
        import selectors
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        mark = self.err.tell() if self.err else 0
        self.proc.stdin.write(json.dumps(job) + "\n")
        self.proc.stdin.flush()
        sel = selectors.DefaultSelector()
        sel.register(self.proc.stdout, selectors.EVENT_READ)
        limit = _CHILD_SECONDS if timeout is None else timeout
        reply = self.proc.stdout.readline() if sel.select(timeout=limit) else None
        if reply:
            return 0, ""
        if self.proc.poll() is None:
            self.proc.kill()
        code = self.proc.wait()
        self.err.seek(mark)
        text = self.err.read()
        self.err.seek(0, 2)
        return (code if reply is not None else TIMED_OUT), (text if reply is not None
                                                     else f"timed out after {limit:g} s: {text[-400:]}")

    def _start(self):
        """Start the worker process, its stderr kept in a temporary file."""
        self.err = tempfile.TemporaryFile("w+")
        env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1")
        self.proc = subprocess.Popen([sys.executable, "-m", "nbxtensor.ops.ew_proof_child", "serve"], env=env,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, text=True)
        _Worker.started += 1


_WORKER = _Worker()


def _child(argv) -> Tuple[int, str]:
    """One job for the run's worker: ['run', spec dir, first candidate] or ['time', spec dir]."""
    return _WORKER.ask({"mode": argv[0], "where": argv[1], "start": int(argv[2]) if len(argv) > 2 else 0})


def _read_out(where, start, results, cands):
    """(the candidate this child started and did not finish, or None; whether it started any), its finished
    candidates entered in `results`."""
    open_one, any_started = None, False
    path = os.path.join(where, f"out{start}.jsonl")
    if not os.path.exists(path):
        return None, False
    with open(path) as fh:
        for line in fh:
            ev = json.loads(line)
            if "start" in ev:
                open_one, any_started = ev["start"], True
            else:
                results[tuple(cands[ev["done"]])] = ("ran", tuple(ev["digests"]))
                open_one = None
    return open_one, any_started


def first_difference(where: str, a, b) -> dict:
    """The first element where digest tuple `a` differs from `b`: group, element index, both values (bits)."""
    spec = pickle.load(open(os.path.join(where, "spec.pkl"), "rb"))
    for gi, (da, db) in enumerate(zip(a, b)):
        if da == db:
            continue
        xa = open(os.path.join(where, f"r{gi}_{da}.bin"), "rb").read()
        xb = open(os.path.join(where, f"r{gi}_{db}.bin"), "rb").read()
        byte = next(i for i in range(min(len(xa), len(xb))) if xa[i] != xb[i])
        elem = next((p[6] for p in spec["params"].values() if p[0] == "ptr" and p[1] == gi), 1)
        at = byte - byte % elem
        return {"group": gi, "byte": byte, "element_bytes": elem, "candidate": xa[at:at + elem].hex(),
                "agreeing": xb[at:at + elem].hex()}
    return {}


def time_copy_in_child(cls: int, tiles) -> Dict[tuple, float]:
    """The copy's milliseconds at each tile, at the class's floor, measured in a child (said by the caller)."""
    where = tempfile.mkdtemp(prefix="nbx_ew_time_")
    with open(os.path.join(where, "time.json"), "w") as fh:
        json.dump({"cls": int(cls), "tiles": [list(t) for t in tiles]}, fh)
    rc, err = _child(["time", where])
    if rc != 0:
        from nbxtensor.ops.ew_proof import TileUnproven
        raise TileUnproven(f"ZERO FALLBACK: the copy's timing child failed for class {cls}: {err[-800:]}")
    got = json.load(open(os.path.join(where, "timing.json")))
    return {tuple(t): ms for t, ms in got}


# ---------------------------------------------------------------------------
# child
# ---------------------------------------------------------------------------
def _device_buffer(nbytes: int):
    """A device buffer of `nbytes` bytes (uint8), at least one."""
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.tensor import NBXTensor
    return NBXTensor.empty((max(nbytes, 1),), dtype=NBXDtype.uint8, device="cuda")


def _restore(buf, blob: bytes) -> None:
    """Copy a snapshot's bytes back into its device buffer."""
    import ctypes
    from nbxtensor.memory.allocator import DeviceAllocator
    host = (ctypes.c_char * len(blob)).from_buffer_copy(blob)
    DeviceAllocator.memcpy(int(buf.data_ptr()), ctypes.addressof(host), len(blob), kind=1)


def _child_run(where: str, start: int) -> None:
    """The worker's proof job: every candidate from `start` on the snapshot, each restored before, its buffers digested after."""
    import importlib
    from nbxtensor.backend.launcher import _snapshot_ranges, launch
    from nbxtensor.device import synchronize
    spec = pickle.load(open(os.path.join(where, "spec.pkl"), "rb"))
    cands = json.load(open(os.path.join(where, "cands.json")))
    kernel = getattr(importlib.import_module(spec["module"]), spec["name"])
    blobs = [open(os.path.join(where, f"g{gi}.bin"), "rb").read() for gi in range(len(spec["groups"]))]
    out = open(os.path.join(where, f"out{start}.jsonl"), "a")
    bufs = [_device_buffer(len(b)) for b in blobs]
    for i in range(start, len(cands)):
        block, warps, stages = cands[i]
        out.write(json.dumps({"start": i}) + "\n")
        out.flush()
        for buf, blob in zip(bufs, blobs):
            _restore(buf, blob)
        kw = _kwargs(spec, bufs, block)
        grid = tuple(spec["grids"][i])           # the candidate's own grid, evaluated by the parent at its tile
        launch(kernel, grid, **kw, **spec["options"], num_warps=warps, num_stages=stages)
        synchronize()
        digests = []
        for gi, buf in enumerate(bufs):
            data = _snapshot_ranges(int(buf.data_ptr()), [(0, len(blobs[gi]))])
            d = hashlib.sha256(data).hexdigest()[:24]
            path = os.path.join(where, f"r{gi}_{d}.bin")
            if not os.path.exists(path):
                with open(path, "wb") as fh:
                    fh.write(data)
            digests.append(d)
        out.write(json.dumps({"done": i, "digests": digests}) + "\n")
        out.flush()


def _kwargs(spec, bufs, block: int) -> Dict[str, Any]:
    """The launch's keyword arguments rebuilt in the worker: pointers as views of the snapshot's buffers, the tile at the candidate's."""
    from nbxtensor.ops.ew_tile import EwTile
    kw = {}
    for name, p in spec["params"].items():
        if p[0] == "ptr":
            kw[name] = _View(int(bufs[p[1]].data_ptr()) + p[2], p[3], p[4], p[5], p[6])
        elif isinstance(p[1], EwTile) and p[1].role == "tile":
            kw[name] = block
        else:
            kw[name] = int(p[1]) if isinstance(p[1], EwTile) else p[1]
    return kw


def _child_time(where: str) -> None:
    """The worker's timing job: the copy kernel at each tile on the class floor, milliseconds per launch."""
    import triton
    from nbxtensor.backend.launcher import launch
    from nbxtensor.device import synchronize
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.ops.autotune_bucket import class_floor
    from nbxtensor.ops.elementwise_launch import ew_launch_kernel
    from nbxtensor.tensor import NBXTensor
    job = json.load(open(os.path.join(where, "time.json")))
    rep = class_floor("elements", job["cls"])
    buf = NBXTensor.empty((rep,), dtype=NBXDtype.float32, device="cuda")
    raw = ew_launch_kernel.fn
    got = []
    for block, warps, stages in job["tiles"]:
        grid = (triton.cdiv(rep, block),)
        for _ in range(3):
            launch(raw, grid, buf, buf, rep, job["cls"], BLOCK_SIZE=block, num_warps=warps, num_stages=stages)
        synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            launch(raw, grid, buf, buf, rep, job["cls"], BLOCK_SIZE=block, num_warps=warps, num_stages=stages)
        synchronize()
        got.append([[block, warps, stages], (time.perf_counter() - t0) * 1000.0 / 20])
    json.dump(got, open(os.path.join(where, "timing.json"), "w"))


def _serve() -> None:
    """The worker: one job per line on stdin, one reply line on the protocol stream when it finished."""
    proto = os.fdopen(os.dup(1), "w")
    sys.stdout = sys.stderr                    # a kernel's or the library's print never reaches the protocol
    for line in sys.stdin:
        job = json.loads(line)
        if job["mode"] == "run":
            _child_run(job["where"], job["start"])
        elif job["mode"] == "time":
            _child_time(job["where"])
        elif job["mode"] == "compile":
            from nbxtensor.backend.compile_bound import compile_in_worker
            compile_in_worker(job["where"])
        else:
            raise SystemExit(f"unknown job {job['mode']!r}")
        proto.write("{}\n")
        proto.flush()


if __name__ == "__main__":
    if sys.argv[1] == "serve":
        _serve()
    else:
        raise SystemExit(f"unknown mode {sys.argv[1]!r}")
