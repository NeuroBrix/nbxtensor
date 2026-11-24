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

THE WORKER'S CARD (the project lead, 24/11/2026; register row 444; qa 24/11: never a card the run was not given, never
a context to ask): each worker the proof starts runs on the FREEST OF THE RUN'S OWN CARDS OF THE LAUNCH'S STAMP
(`proof_device`). The candidates are the cards the APPLICATION DECLARED for this run (`declare_proof_cards`: NeuroBrix
declares its plan's cards and the run's `--proof-card`), the launch's own card among them; a run that declared none (a
library test, a tool) has the launch's own card alone — never "every visible card": on a shared machine a visible card
may be someone else's. Among them, the cards that compile the same binary (the launch card's compute capability, a
device attribute read without a context), the one with the most free memory, read from NVML (`nvmlDeviceGetMemoryInfo`,
the card found by its PCI bus id, `cuDeviceGetPCIBusId` of this process's ordinal: NVML numbers the boards, the
process its visible cards) — no context is opened on any card to ask. Without NVML (no management library, HIP) the
launch's own card alone, said once by name. The worker's environment carries the choice (`worker_env`: the backend's
visibility variable narrowed to that one card), so the worker's context opens there and nowhere else. A model that
filled its card (deepseek-moe-16b on V100 32 GB: its card at 32 169 MiB used, the other at 709 MiB) no longer puts the
proof's context on the full card when the run declares the spare one.

The worker holds nothing on the card while the parent runs: after each job it waits for the device and empties its
pool back to the driver, and it EXITS once no job came for IDLE_SECONDS (the idle bound, 2 s); the parent's next
job starts a fresh one.

    python -m nbxtensor.ops.ew_proof_child serve [idle seconds]
"""
from __future__ import annotations

import ctypes
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
#: the exit code of a worker that met a DEVICE fault (a kernel fault status at the launch or the synchronize after it,
#: `backend.faults.KERNEL_FAULTS`; a shader-validation report on Metal): the candidate's fault, recorded
DEVICE_EXIT = 76
#: the exit code of a worker whose job could not run (any other exception): the proof could not run, never a fault
UNRUN_EXIT = 77


class ProofCouldNotRun(RuntimeError):
    """The proof's worker did not reach a verdict on a candidate: an exception that is no device fault (a refusal of
    the launcher, a recipe error, anything raised before the launch), a signal, a timeout, an unexplained exit. The
    variant is refused by name; nothing is recorded of the candidate."""


class DeviceFault(RuntimeError):
    """In the worker: a device fault of the candidate running (a kernel fault status at its launch or at the
    synchronize after it, or a shader-validation report on Metal) — the one outcome recorded as a tile's fault."""


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
    """{tile: ('fault', text) | ('ran', digests)} over every candidate, and the differ of two digest tuples. A
    candidate is a FAULT only when its worker recorded a device fault (`DeviceFault`, exit DEVICE_EXIT); any other end
    of a worker on a candidate — an exception it recorded as not run, a signal, a timeout, an exit with nothing
    recorded — raises ProofCouldNotRun naming it: no verdict on the tile."""
    results: Dict[tuple, tuple] = {}
    start = 0
    while start < len(cands):
        with open(os.path.join(where, "cands.json"), "w") as fh:
            json.dump([list(c) for c in cands], fh)
        rc, err = _child(["run", where, str(start)])
        ended = _read_out(where, start, results, cands)
        if len(results) == len(cands) and rc == 0:
            break
        if ended is not None and ended[0] == "fault" and rc == DEVICE_EXIT:
            at = ended[1]
            results[tuple(cands[at])] = ("fault", ended[2])
            start = at + 1
            continue
        why = (ended[2] if ended is not None and ended[0] == "unrun" else
               ("timed out" if rc == TIMED_OUT else f"signal {-rc}" if rc < 0 else f"exit {rc}")
               + f", nothing recorded: {err[-600:].strip()}")
        at = ended[1] if ended is not None else start
        raise ProofCouldNotRun(f"candidate {tuple(cands[at]) if at is not None and at < len(cands) else '?'}: "
                               f"{why}")
    return results, lambda a, b: first_difference(where, a, b)


#: `_Worker.ask`'s code for a job the worker did not answer within its time (the worker killed): a TIMEOUT, told apart
#: from the exit code of a worker that died on the job (a signal's is -1 to -64)
TIMED_OUT = -1000


def _card_arch(ordinal: int):
    """The compute capability of the visible card `ordinal` (70 for sm_70) on a CUDA process; None on a backend whose
    cards are not told apart here (that card is then never chosen for the proof but the launch's own)."""
    from nbxtensor.backend import runtime as _rt
    if _rt.gpu_backend() != "cuda":
        return None
    from nbxtensor.backend.launcher import CudaDriver
    return CudaDriver.instance().capability_of(int(ordinal))


_DECLARED_CARDS = None              # the run's cards (visible ordinals) an application declared; None: none declared
_SAID: set = set()                  # the fallbacks already said, once each


def declare_proof_cards(ordinals) -> None:
    """The application states the cards THIS RUN WAS GIVEN, as visible ordinals of this process (NeuroBrix: its plan's
    cards and the run's `--proof-card`): the only cards a proof's worker may run on, the launch's own card always among
    them. None forgets the declaration (the launch's own card alone). A negative ordinal is refused by name."""
    global _DECLARED_CARDS
    if ordinals is None:
        _DECLARED_CARDS = None
        return
    cards = sorted({int(i) for i in ordinals})
    if any(i < 0 for i in cards):
        raise ValueError(f"declare_proof_cards: {cards}: a visible ordinal is 0 or more")
    _DECLARED_CARDS = tuple(cards)


def declared_proof_cards():
    """The run's cards as declared (`declare_proof_cards`), a sorted tuple of visible ordinals; None when none was."""
    return _DECLARED_CARDS


def _say_once(key: str, text: str) -> None:
    """Print one `[ew-proof]` line on stderr the first time `key` is met in this process, never again."""
    if key not in _SAID:
        _SAID.add(key)
        print(f"[ew-proof] {text}", file=sys.stderr)


class _NvmlMemory(ctypes.Structure):
    """NVML's nvmlMemory_t: the card's total, free and used bytes, as `nvmlDeviceGetMemoryInfo` fills it."""
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


def _card_free_bytes(ordinals) -> "Dict[int, int] | None":
    """{visible ordinal: free bytes} of each card, from NVML (`nvmlDeviceGetHandleByPciBusId_v2`, then
    `nvmlDeviceGetMemoryInfo`), each card found by its PCI bus id (`CudaDriver.pci_bus_id_of`, a device query): no
    context is opened on any card. None when NVML does not answer (no management library, a backend without one, a
    card NVML does not find): the caller then has the launch's card alone."""
    from nbxtensor.backend import runtime as _rt
    from nbxtensor.backend.process_memory import _nvml
    if _rt.gpu_backend() != "cuda":
        return None
    lib = _nvml(_rt.backend_table().get("management_libs"))
    if lib is None:
        return None
    from nbxtensor.backend.launcher import CudaDriver
    drv = CudaDriver.instance()
    out = {}
    for i in ordinals:
        bus = ctypes.create_string_buffer(drv.pci_bus_id_of(int(i)).encode())
        handle, mem = ctypes.c_void_p(), _NvmlMemory()
        if lib.nvmlDeviceGetHandleByPciBusId_v2(bus, ctypes.byref(handle)) != 0:
            return None
        if lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem)) != 0:
            return None
        out[int(i)] = int(mem.free)
    return out


def proof_device() -> Tuple[int, int]:
    """(visible ordinal, its free bytes or -1 unread) of the card the proof's next worker runs on: the FREEST OF THE
    RUN'S DECLARED CARDS of the launch's stamp (`declare_proof_cards`; none declared: the launch's own card alone) —
    those whose compute capability is the launch card's, the launch's own card included, the most free memory by NVML
    (`_card_free_bytes`, no context opened), the lowest ordinal on a tie. Without NVML, the launch's own card, said
    once. A declared card this process does not see is refused by name. With one device, that device (-1)."""
    from nbxtensor.memory.allocator import DeviceAllocator as DA
    seen = DA.device_count()
    if seen <= 1:
        return 0, -1
    own = int(DA.get_device())
    declared = _DECLARED_CARDS or ()
    unseen = [i for i in declared if not 0 <= i < seen]
    if unseen:
        raise ProofCouldNotRun(f"the run declared card(s) {unseen} for the proof (`declare_proof_cards`), and this "
                               f"process sees {seen} card(s), ordinals 0 to {seen - 1}")
    given = sorted({own, *declared})
    if len(given) == 1:
        return own, -1
    arch = _card_arch(own)
    same = [i for i in given if i == own or (arch is not None and _card_arch(i) == arch)]
    if len(same) == 1:
        return own, -1
    free = _card_free_bytes(same)
    if free is None:
        from nbxtensor.backend import runtime as _rt
        _say_once("no-nvml", f"no free-memory reading without opening a context (NVML, "
                  f"{list(_rt.backend_table().get('management_libs') or []) or 'none for this backend'}, does not "
                  f"answer): the proof's worker runs on the launch's own card {own} alone, not on the run's cards {same}")
        return own, -1
    best = max(same, key=lambda i: (free[i], -i))
    return best, free[best]


def worker_env(ordinal: int, base: Dict[str, str] = None) -> Dict[str, str]:
    """The environment of a worker on the visible card `ordinal`: the backend's visibility variable (its table's
    `visible_env`; the innermost one this process has set, else the last the backend names) narrowed to the one entry
    that IS that card in this process's numbering. A backend with no such variable (Metal: one device) is unchanged.
    An ordinal past the entries this process's variable lists is refused by name."""
    from nbxtensor.backend import runtime as _rt
    env = dict(os.environ if base is None else base)
    names = tuple(_rt.backend_table().get("visible_env") or ())
    if not names:
        return env
    present = [n for n in names if env.get(n) is not None]
    var = present[-1] if present else names[-1]
    if var not in env:
        env[var] = str(int(ordinal))
        return env
    entries = [e.strip() for e in env[var].split(",") if e.strip()]
    if not 0 <= int(ordinal) < len(entries):
        raise ProofCouldNotRun(f"the proof's card {ordinal} is not among the {len(entries)} card(s) {var}="
                               f"{env[var]!r} lists")
    env[var] = entries[int(ordinal)]
    return env


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
        self.device: Tuple[int, int] = (-1, -1)      # (visible ordinal, free bytes at the choice) of the live worker

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
        """Start the worker process on the freest of the run's declared cards of the launch's stamp (`proof_device`,
        carried by its environment: `worker_env`), its stderr kept in a temporary file."""
        self.err = tempfile.TemporaryFile("w+")
        from nbxtensor.memory.metal import SHADER_VALIDATION
        self.device = proof_device()
        from nbxtensor.memory.allocator import DeviceAllocator
        env = worker_env(self.device[0]) if DeviceAllocator.device_count() > 1 else dict(os.environ)    # one: as is
        env["CUDA_LAUNCH_BLOCKING"] = "1"
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
    """The worker's record of a job begun at `start`: its finished candidates entered in `results`, and how it ended
    on a candidate it did not finish — ('fault', index, text) for a device fault, ('unrun', index, text) for any
    other exception it caught, ('open', index, '') for one it began and recorded nothing of; None when it began none."""
    path = os.path.join(where, f"out{start}.jsonl")
    ended = None
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        for line in fh:
            ev = json.loads(line)
            if "start" in ev:
                ended = ("open", ev["start"], "")
            elif "done" in ev:
                results[tuple(cands[ev["done"]])] = ("ran", tuple(ev["digests"]))
                ended = None
            elif "fault" in ev:
                ended = ("fault", ev["fault"], ev["text"])
            elif "unrun" in ev:
                ended = ("unrun", ev["unrun"], ev["text"])
    return ended


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
    buffers (allocated, filled, launched, waited, digested, freed). Its end on a candidate is RECORDED before the
    worker exits: a device fault (`DeviceFault`) as the candidate's fault (exit DEVICE_EXIT), any other exception as
    "not run" (exit UNRUN_EXIT) — the parent never reads a host error as a fault."""
    import traceback
    out = open(os.path.join(where, f"out{start}.jsonl"), "a")
    at = None
    try:
        import importlib
        from nbxtensor.ops.ew_synthetic import Buf, fill_bytes
        spec = pickle.load(open(os.path.join(where, "spec.pkl"), "rb"))
        cands = json.load(open(os.path.join(where, "cands.json")))
        kernel = getattr(importlib.import_module(spec["module"]), spec["name"])
        blobs = [[fill_bytes(Buf("", *b)) for b in ln["bufs"]] for ln in spec["launches"]]
        for i in range(start, len(cands)):
            at = i
            out.write(json.dumps({"start": i}) + "\n")
            out.flush()
            digests = _candidate(where, spec, kernel, blobs, cands[i], i)
            out.write(json.dumps({"done": i, "digests": digests}) + "\n")
            out.flush()
    except DeviceFault as fault:
        out.write(json.dumps({"fault": at, "text": str(fault)[-1200:]}) + "\n")
        out.flush()
        os._exit(DEVICE_EXIT)                   # a CUDA fault is sticky: this worker launches nothing more
    except BaseException as exc:                # noqa: BLE001  every other end is "could not run", recorded so
        out.write(json.dumps({"unrun": at, "text": f"{type(exc).__name__}: {exc}"[:600] + " | "
                              + traceback.format_exc()[-600:]}) + "\n")
        out.flush()
        os._exit(UNRUN_EXIT)


def _candidate(where: str, spec: dict, kernel, blobs, cand, i: int) -> List[str]:
    """One candidate over every synthetic launch of the job: each buffer guarded and filled, the launch made and
    waited (`_launch_waited`: its device faults raised as `DeviceFault`), every buffer digested, then freed; the
    digests."""
    from nbxtensor.backend.launcher import _snapshot_ranges
    block, warps, stages = cand
    digests: List[str] = []
    for li, ln in enumerate(spec["launches"]):
        held = _guarded(spec["guard"], ln["bufs"], blobs[li])
        try:
            kw = _kwargs(ln, held, block)
            _launch_waited(kernel, tuple(ln["grids"][i]), kw, spec, warps, stages)
            _raise_reported(spec["guard"], spec["name"], ln, cand)
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
    return digests


def _launch_waited(kernel, grid, kw: dict, spec: dict, warps: int, stages: int) -> None:
    """The launch and the synchronize after it; a kernel fault status of either (`faults.KERNEL_FAULTS`: the 700 and
    its kin) raised as `DeviceFault`. Every other exception — the launcher refusing an argument, a compile error —
    goes up as it is: not a device fault."""
    from nbxtensor.backend.faults import DeviceStatusError
    from nbxtensor.backend.launcher import launch
    from nbxtensor.device import synchronize
    try:
        launch(kernel, grid, **kw, **spec["options"], num_warps=warps, num_stages=stages)
        synchronize()
    except DeviceStatusError as err:
        if err.kernel_fault():
            raise DeviceFault(f"{type(err).__name__} at {err.call}: {err}") from err
        raise


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
        raise DeviceFault(f"shader validation: {name} at tile {tuple(cand)} touched past an extent (synthetic launch "
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
