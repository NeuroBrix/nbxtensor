"""The child side of the element-wise tile proof (`ops/ew_proof.py`), and the parent's handling of its children.

The proof runs a variant on SYNTHETIC launches made from the variant alone (`ops/ew_synthetic.py`): the parent writes
them as a job (`write_job`: each launch's buffers — type, extent, alignment, fill — its scalars, and its grid at every
candidate tile), never a byte of a real launch. A fault on the device (an illegal address) is sticky: the process that
meets it can launch nothing after. So the candidates run in a worker process (CUDA_LAUNCH_BLOCKING=1; on Metal,
shader validation on, `memory.metal.SHADER_VALIDATION`). For each candidate, for each synthetic launch, the worker
allocates every buffer GUARDED (`memory.guarded`: its end against an unmapped granule on CUDA, against the end of a
dedicated MTLBuffer checked by shader validation on Metal), fills it, launches, waits, and digests every buffer
(the launch's writes are not named, so all of them are compared), then frees them: the worker holds one launch's
buffers at a time, kilobytes to megabytes. A lane that touches past its extent faults (CUDA) or is reported by
validation (Metal): the worker ends on it, the parent records that candidate's fault, and a fresh worker resumes at
the next candidate.

The worker holds nothing on the card while the parent runs: after each job it waits for the device and empties its
pool back to the driver, and it EXITS once no job came for IDLE_SECONDS (the idle bound, 2 s); the parent's next
job starts a fresh one.

    python -m nbxtensor.ops.ew_proof_child serve [idle seconds]
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Tuple

# The job is pickled by the parent into its own private temporary directory (mkdtemp, mode 0700) and read back only
# by the worker it starts: it carries dtype objects and constexpr values JSON cannot hold. Nothing outside this
# process pair writes it.

_CHILD_SECONDS = 600         # a candidate that hangs this long is recorded as a fault, not waited on
#: THE IDLE BOUND: a worker that receives no job for this many seconds EXITS — its device context (about 300 MB on a
#: V100) and anything it held leave the card while the parent runs (Flex.1-alpha, 16 GB V100, 23/11/2026: a worker
#: kept alive between proofs held 2.2 to 2.7 GB the parent's generation then lacked). A first launch proves its
#: variants in bursts; the next burst starts a fresh worker.
IDLE_SECONDS = 2.0
#: the exit code of a worker that ended on its idle bound: it read no job, so the job the parent sent is sent again
IDLE_EXIT = 75


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
# parent: the job
# ---------------------------------------------------------------------------
def write_job(kernel, launches, cands, options: Dict[str, Any], guard: str) -> Tuple[str, int]:
    """(job directory, synthetic bytes): the variant's synthetic launches (`ew_synthetic.synthesise`) written for
    the worker — each buffer as its description (type, elements, alignment, fill, seed), each scalar and constexpr
    as its value, each launch's grid evaluated at every candidate tile — with the guard the worker must build."""
    from nbxtensor.ops.elementwise_launch import ew_grid_of
    from nbxtensor.ops.ew_synthetic import Buf
    from nbxtensor.ops.ew_tile import EwTile
    jobs, total = [], 0
    for ln in launches:
        params, bufs = {}, []
        for name, v in ln.args.items():
            if isinstance(v, Buf):
                params[name] = ("buf", len(bufs))
                bufs.append((v.ty, v.elements, v.elem, v.aligned, v.fill, v.seed))
                total += v.nbytes
            else:
                params[name] = ("val", v)
        grid = ew_grid_of(*ln.axes)
        grids = [list(int(g) for g in grid({"_": EwTile(c[0], 0, "tile")})) for c in cands]
        jobs.append({"params": params, "bufs": bufs, "grids": grids, "count": ln.count, "layout": ln.layout})
    spec = {"module": kernel.fn.__module__, "name": kernel.__name__, "launches": jobs, "guard": guard,
            "options": {k: v for k, v in options.items() if k not in ("num_warps", "num_stages")}}
    where = tempfile.mkdtemp(prefix="nbx_ew_proof_")
    with open(os.path.join(where, "spec.pkl"), "wb") as fh:
        pickle.dump(spec, fh)
    return where, total


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
    """ONE child process that serves a burst of proofs (orchestrator 23/11: a process and a device context per
    variant would put seconds times the variant count into a first launch) and holds NOTHING on the card while the
    parent runs: after every job it returns every block to the driver (`_release_device`), and it exits once no job
    came for IDLE_SECONDS. Started at a job when none is alive (the first proof, after a fault killed it, after its
    idle exit). Requests and replies are JSON lines on its stdin / a duplicate of its stdout; its own prints go to
    stderr, kept in a file so a fault's text can be read."""
    started = 0                      # children started in this run (the record's measure)

    def __init__(self, idle: float = None):
        """No process yet: it starts at the first job. `idle`: its idle bound (default IDLE_SECONDS)."""
        self.proc = None
        self.err = None
        self.idle = IDLE_SECONDS if idle is None else float(idle)

    def ask(self, job: dict, timeout: float = None) -> Tuple[int, str]:
        """(0, '') when the job finished; (exit code, its stderr since the job began) when the child died on it. A
        worker that had ended on its idle bound when the job reached it read nothing: a fresh one is asked."""
        code, text = self._ask(job, timeout)
        if code == IDLE_EXIT:
            code, text = self._ask(job, timeout)
        return code, text

    def _ask(self, job: dict, timeout: float = None) -> Tuple[int, str]:
        """One sending of `job` to the live worker (started if none is)."""
        import selectors
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        mark = self.err.tell() if self.err else 0
        try:
            self.proc.stdin.write(json.dumps(job) + "\n")
            self.proc.stdin.flush()
        except BrokenPipeError:                   # it ended on its idle bound before the line reached it
            return self.proc.wait(), ""
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
        from nbxtensor.memory.metal import SHADER_VALIDATION
        env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1")
        if _metal():
            env[SHADER_VALIDATION] = "1"         # every access checked against its buffer's length (the guard)
        self.proc = subprocess.Popen([sys.executable, "-m", "nbxtensor.ops.ew_proof_child", "serve", repr(self.idle)],
                                     env=env,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, text=True)
        _Worker.started += 1


_WORKER = _Worker()


def _child(argv) -> Tuple[int, str]:
    """One job for the run's worker: ['run', spec dir, first candidate]."""
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
    """The first element where digest tuple `a` differs from `b`: the synthetic launch (its count and layout), the
    parameter, the byte, both values (bits)."""
    spec = pickle.load(open(os.path.join(where, "spec.pkl"), "rb"))
    flat = [(li, name, ln) for li, ln in enumerate(spec["launches"])
            for name, p in ln["params"].items() if p[0] == "buf"]
    for k, (da, db) in enumerate(zip(a, b)):
        if da == db:
            continue
        li, name, ln = flat[k]
        xa = open(os.path.join(where, f"r{k}_{da}.bin"), "rb").read()
        xb = open(os.path.join(where, f"r{k}_{db}.bin"), "rb").read()
        byte = next(i for i in range(min(len(xa), len(xb))) if xa[i] != xb[i])
        elem = ln["bufs"][ln["params"][name][1]][2]
        at = byte - byte % elem
        return {"launch": li, "count": ln["count"], "layout": ln["layout"], "parameter": name, "byte": byte,
                "element_bytes": elem, "candidate": xa[at:at + elem].hex(), "agreeing": xb[at:at + elem].hex()}
    return {}


def _metal() -> bool:
    """Whether this process's device is Metal (its worker then runs with shader validation)."""
    from nbxtensor.backend import runtime as _rt
    try:
        return _rt.gpu_backend() == "metal"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# child
# ---------------------------------------------------------------------------
def _child_run(where: str, start: int) -> None:
    """The worker's proof job: every candidate from `start`, each running every synthetic launch on freshly guarded
    buffers (allocated, filled, launched, waited, digested, freed); a validation report (Metal) ends the worker on that
    candidate as a fault ends it on CUDA."""
    import importlib
    from nbxtensor.backend.launcher import _snapshot_ranges, launch
    from nbxtensor.device import synchronize
    from nbxtensor.ops.ew_synthetic import Buf, fill_bytes
    spec = pickle.load(open(os.path.join(where, "spec.pkl"), "rb"))
    cands = json.load(open(os.path.join(where, "cands.json")))
    kernel = getattr(importlib.import_module(spec["module"]), spec["name"])
    blobs = [[fill_bytes(Buf("", b[0], b[1], b[2], b[3], b[4], b[5])) for b in ln["bufs"]] for ln in spec["launches"]]
    out = open(os.path.join(where, f"out{start}.jsonl"), "a")
    for i in range(start, len(cands)):
        block, warps, stages = cands[i]
        out.write(json.dumps({"start": i}) + "\n")
        out.flush()
        digests: List[str] = []
        for li, ln in enumerate(spec["launches"]):
            held = _guarded(spec["guard"], ln["bufs"], blobs[li])
            try:
                kw = _kwargs(ln, held, block)
                launch(kernel, tuple(ln["grids"][i]), **kw, **spec["options"], num_warps=warps, num_stages=stages)
                synchronize()
                _raise_reported(spec["guard"], spec["name"], ln, cands[i])
                for (ty, elements, elem, *_r), g in zip(ln["bufs"], held):
                    data = _snapshot_ranges(int(g.address), [(0, max(elements, 1) * elem)])
                    d = hashlib.sha256(data).hexdigest()[:24]
                    path = os.path.join(where, f"r{len(digests)}_{d}.bin")
                    if not os.path.exists(path):
                        with open(path, "wb") as fh:
                            fh.write(data)
                    digests.append(d)
            finally:
                for g in held:
                    g.free()
        out.write(json.dumps({"done": i, "digests": digests}) + "\n")
        out.flush()


def _guarded(kind: str, bufs, blobs) -> list:
    """Every buffer of one synthetic launch, guarded (`memory.guarded.guarded`) and filled with its bytes."""
    import ctypes
    from nbxtensor.memory.allocator import DeviceAllocator
    from nbxtensor.memory.guarded import guard_kind, guarded
    if guard_kind() != kind:
        raise RuntimeError(f"ZERO FALLBACK: the proof's worker has guard {guard_kind()!r}, its job asks {kind!r}")
    held = []
    try:
        for (ty, elements, elem, aligned, _fill, _seed), blob in zip(bufs, blobs):
            g = guarded(len(blob), elem, aligned)
            held.append(g)
            host = (ctypes.c_char * len(blob)).from_buffer_copy(blob)
            DeviceAllocator.memcpy(int(g.address), ctypes.addressof(host), len(blob), kind=1)
    except BaseException:
        for g in held:
            g.free()
        raise
    return held


def _raise_reported(kind: str, name: str, ln: dict, cand) -> None:
    """On Metal: raise what shader validation reported for the launch just waited on (an access past a buffer's
    length), so the worker ends on this candidate as a CUDA fault ends it."""
    if kind != "metal-validation":
        return
    from nbxtensor.backend.metal_device import runtime
    reports = runtime().take_validation_reports()
    if reports:
        raise RuntimeError(f"shader validation: {name} at tile {tuple(cand)} touched past an extent (synthetic launch "
                           f"count {ln['count']}, {ln['layout']}): {reports[0][:600]}")


def _kwargs(ln: dict, held, block: int) -> Dict[str, Any]:
    """One synthetic launch's keyword arguments in the worker: pointers as views of the guarded buffers, the tile at
    the candidate's."""
    from nbxtensor.ops.ew_synthetic import dtype_of
    from nbxtensor.ops.ew_tile import EwTile
    kw = {}
    for name, p in ln["params"].items():
        if p[0] == "buf":
            ty, elements, elem = ln["bufs"][p[1]][:3]
            kw[name] = _View(int(held[p[1]].address), dtype_of(ty)[0], (max(elements, 1),), (1,), elem)
        elif isinstance(p[1], EwTile) and p[1].role == "tile":
            kw[name] = block
        else:
            kw[name] = int(p[1]) if isinstance(p[1], EwTile) else p[1]
    return kw


def _release_device() -> int:
    """After a job: wait for the device, return every wholly free pool segment to the driver (the pool EMPTIED, not
    kept as a cache), and answer the bytes this process still holds through the account (0 after a proof: every
    synthetic buffer was freed after its launch)."""
    from nbxtensor.memory.allocator import DeviceAllocator as DA
    if not DA._cuda_live_bytes:
        return 0                                # no device touched by this job (a compile): nothing to release
    DA.device_synchronize()
    DA._pool_flush(None)
    return sum(DA._cuda_live_bytes.values())


def _serve(idle: float = IDLE_SECONDS) -> None:
    """The worker: one job per line on stdin, one reply line on the protocol stream when it finished; the device
    released after each job (`_release_device`); an exit (IDLE_EXIT) once no job came for `idle` seconds."""
    import select
    proto = os.fdopen(os.dup(1), "w")
    sys.stdout = sys.stderr                    # a kernel's or the library's print never reaches the protocol
    while True:
        ready, _w, _x = select.select([sys.stdin], [], [], idle)
        if not ready:
            proto.close()
            os._exit(IDLE_EXIT)                # its context and everything it held leave the card with it
        line = sys.stdin.readline()
        if not line:
            return
        job = json.loads(line)
        if job["mode"] == "run":
            _child_run(job["where"], job["start"])
        elif job["mode"] == "compile":
            from nbxtensor.backend.compile_bound import compile_in_worker
            compile_in_worker(job["where"])
        else:
            raise SystemExit(f"unknown job {job['mode']!r}")
        held = _release_device()
        if held:
            print(f"[ew-proof] worker: {held} bytes still held through the account after a job", file=sys.stderr)
        proto.write("{}\n")
        proto.flush()


if __name__ == "__main__":
    if sys.argv[1] == "serve":
        _serve(float(sys.argv[2]) if len(sys.argv) > 2 else IDLE_SECONDS)
    else:
        raise SystemExit(f"unknown mode {sys.argv[1]!r}")
