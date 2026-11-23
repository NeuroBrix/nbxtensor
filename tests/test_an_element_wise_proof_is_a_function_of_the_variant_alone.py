"""The element-wise tile proof is a function of the VARIANT alone (`ops/ew_proof.py`, `ops/ew_synthetic.py`,
`memory/guarded.py`; the project lead's cure, 23/11/2026): every candidate runs the variant on synthetic launches —
a fixed ladder of counts with masked tails, contiguous and permuted layouts — with every buffer against a guard.

The fault it cures: variant 6be93ab046480f3a (`mul_scalar_strided_nd_kernel` fp32, NDIM 3, V100) faulted at tile
(4096, 2, 3) on one launch's snapshot and passed on another's, and the record, keyed by the variant, then served every
launch; the snapshot cost the launch's bytes.

What each test does if the code is wrong:
1. the ladder misses a masked tail, or a count leaves its specialization (another binary);
2. a guarded buffer's end is not at the guard, or its start loses its alignment residue, or the VMM range maps the
   guard granule;
3. a synthetic launch reads the launch that first met the variant (two launches of different counts give two
   proofs or two records), or the proof's job holds more than the ladder's bytes;
4. a kernel whose signature does not determine its launch is proven anyway instead of refused by name; a device with
   no guard says "proven";
5. a table of the old stamp (no proof method) is read; the export accepts a disagreement;
7. the proof's worker keeps blocks after a job, or stays alive past its idle bound (holding its device context while
   the parent runs: Flex.1-alpha on a 16 GB V100 ran out of memory beside it);
6. on a device (Metal: shader validation; CUDA: the VMM guard), a kernel that touches one element past its extent at
   one tile is not caught at exactly that tile.
Tests 1-5 and 7 are device-free; 6 needs a device.
Run: python -m pytest tests/test_an_element_wise_proof_is_a_function_of_the_variant_alone.py
"""
import json
import os
import pickle
import sys

import pytest

from nbxtensor.ops import ew_proof, ew_proof_child, ew_synthetic
from nbxtensor.ops.ew_tile import EwTile, synthetic

CANDS = [(b, w, 3) for b in (256, 512, 1024, 2048, 4096) for w in (2, 4, 8)]
TILE = EwTile(4096, 65536, "tile")
RED = [("*fp32", "D"), ("*fp32", "D"), ("i32", "D"), ("fp32", None), ("*i64", "D"), ("*i64", "D"),
       ("constexpr", TILE), ("constexpr", 3)]                  # variant 6be93ab046480f3a's specialization


def _spec_of(v):
    from nbxtensor.backend.launcher import specialize_arg
    return specialize_arg(v, True, True)


# --- 1. the ladder ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("attr", ["D", ""])
def test_the_ladder_holds_masked_tails_in_the_counts_own_specialization(attr):
    counts = ew_synthetic.ladder(attr, CANDS)
    want = {"D": [16, 240, 4080, 4112, 8240], "": [2, 255, 4095, 4097, 8195]}[attr]
    assert counts == want
    assert all(_spec_of(n) == ("i32", attr) for n in counts)            # the same binary at every count
    for block in {c[0] for c in CANDS}:
        assert any(n < block for n in counts)                            # a program that is all tail
        assert any(n > block and n % block for n in counts)              # full programs, then a tail
    assert max(counts) > 2 * max(c[0] for c in CANDS)
    assert ew_synthetic.ladder(("constexpr", 1), CANDS) == [1]           # a count the variant fixed


def test_a_64_bit_count_walks_the_same_small_counts_as_a_wide_binary():
    from nbxtensor.ops.mul import mul_scalar_kernel
    spec = [("*fp32", "D"), ("*fp32", "D"), ("i64", "D"), ("fp32", None), ("constexpr", TILE)]
    launches = ew_synthetic.synthesise(mul_scalar_kernel, spec, CANDS)
    assert [int(ln.count) for ln in launches] == [16, 240, 4080, 4112, 8240]
    assert all(_spec_of(ln.args["n_elements"]) == ("i64", "D") for ln in launches)


# --- 2. the guard -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("nbytes,elem", [(4, 4), (64, 4), (4095 * 4, 4), (255 * 2, 2), (17, 1), (24, 8)])
@pytest.mark.parametrize("aligned", [True, False])
def test_a_guarded_buffer_ends_at_its_guard_and_keeps_its_alignment(nbytes, elem, aligned):
    from nbxtensor.memory.guarded import UNIT, place
    end = 0x7F0000200000
    start = place(end, nbytes, elem, aligned)
    assert start + nbytes <= end and end - (start + nbytes) < UNIT       # at most 15 bytes of slack
    assert start % elem == 0 and (start % UNIT == 0) == aligned


class _FakeVmm:
    def __init__(self, gran=2 << 20):
        self.gran, self.calls, self.mapped = gran, [], []

    def granularity(self):
        return 0, self.gran

    def reserve(self, n):
        self.calls.append(("reserve", n))
        return 0, 0x7F0000000000

    def create(self, n):
        self.calls.append(("create", n))
        return 0, 7

    def map(self, va, n, h):
        self.mapped.append((va, va + n))
        return 0

    def unmap(self, va, n):
        self.calls.append(("unmap", va, n))
        return 0

    def release(self, h):
        self.calls.append(("release", h))
        return 0

    def unreserve(self, va, n):
        self.calls.append(("unreserve", va, n))
        return 0


@pytest.mark.parametrize("nbytes", [64, (2 << 20) - 8, 3 << 20])
def test_the_vmm_range_maps_only_the_buffer_and_leaves_the_next_granule_unmapped(nbytes):
    from nbxtensor.memory.vmm import GuardedRange
    v = _FakeVmm()
    g = GuardedRange(v, nbytes, 4, True)
    (lo, hi), = v.mapped
    reserved = dict((c[0], c[1]) for c in v.calls if len(c) == 2)["reserve"]
    assert hi == g.guard and lo <= g.address and g.address + nbytes <= hi and hi - (g.address + nbytes) < 16
    assert reserved == (hi - lo) + v.gran                               # one granule more than mapped: the guard
    assert (hi - lo) == -(-(nbytes + 16) // v.gran) * v.gran            # only the pages the buffer needs
    g.free()
    assert [c[0] for c in v.calls[-3:]] == ["unmap", "release", "unreserve"]


def test_no_guard_on_a_device_without_one_means_not_proven(monkeypatch):
    from nbxtensor.ops.binary_strided import mul_scalar_strided_nd_kernel as k
    monkeypatch.setattr("nbxtensor.memory.guarded.guard_kind", lambda: None)
    monkeypatch.setattr(ew_proof_child, "run_candidates", lambda *a: pytest.fail("a candidate ran unguarded"))
    vid, _ = ew_proof.variant_of(k, RED, {"num_warps": 2, "num_stages": 3}, {"tile": TILE})
    rec = ew_proof.prove(k, RED, {"num_warps": 2, "num_stages": 3}, {"tile": TILE}, vid)
    assert rec["passed"] == [] and rec["refused"].startswith("unguarded") and rec["unkept"]
    tab = _MemTable()
    monkeypatch.setattr(ew_proof, "table", lambda: tab)
    with pytest.raises(ew_proof.TileUnproven, match="unguarded"):
        ew_proof.settle(k, lambda m: (1,), {}, RED, {"num_warps": EwTile(2, 65536, "warps"), "num_stages": 3},
                        {"tile": TILE, "warps": EwTile(2, 65536, "warps")})
    assert tab.written == {} and vid in tab.unkept                     # said this run, never written


# --- 3. a function of the variant alone ---------------------------------------------------------------------------
def test_the_red_variant_synthesises_on_its_ladder_in_both_layouts_within_a_megabyte():
    from nbxtensor.ops.binary_strided import mul_scalar_strided_nd_kernel as k
    launches = ew_synthetic.synthesise(k, RED, CANDS)
    assert sorted({ln.count for ln in launches}) == [16, 240, 4080, 4112, 8240]
    assert {ln.layout for ln in launches} == {"contiguous", "permuted"} and len(launches) == 10
    for ln in launches:
        shape = ln.args["shape_ptr"].fill[1]
        strides = ln.args["x_stride_ptr"].fill[1]
        assert len(shape) == 3 and _prod(shape) == ln.count
        reach = 1 + sum((d - 1) * s for d, s in zip(shape, strides))
        assert ln.args["x_ptr"].elements == reach == ln.count           # a permutation: the same extent
        assert ln.args["output_ptr"].elements == ln.count
    where, nbytes = ew_proof_child.write_job(k, launches, CANDS, {"debug": False}, "vmm")
    try:
        spec = pickle.load(open(os.path.join(where, "spec.pkl"), "rb"))
        per_launch = [sum(b[1] * b[2] for b in ln["bufs"]) for ln in spec["launches"]]
        assert nbytes == sum(per_launch) < 1 << 20                      # the ladder's bytes, never a launch's
        assert max(per_launch) < 100_000                                # what the worker holds at once
        assert all(len(ew_synthetic.fill_bytes(ew_synthetic.Buf("", *b)))
                   == max(b[1], 1) * b[2] for ln in spec["launches"] for b in ln["bufs"])
        assert spec["launches"][0]["grids"][CANDS.index((4096, 2, 3))] == [1]
    finally:
        import shutil
        shutil.rmtree(where)


def _prod(xs):
    out = 1
    for x in xs:
        out *= x
    return out


class _MemTable:
    """A proof table in memory: what settle keeps, and the variants it met."""
    key = "test"

    def __init__(self):
        self.written, self.unkept, self.seen_classes, self.shipped = {}, {}, {}, {}

    @property
    def kept(self):
        return self.written

    def record(self, vid):
        return self.unkept.get(vid) or self.written.get(vid)

    def status(self, vid, tile):
        rec = self.record(vid)
        return None if rec is None else ("passed" if list(tile) in rec["passed"] else "failed")

    def keep(self, vid, rec):
        self.written[vid] = rec


def test_two_launches_of_different_counts_give_one_proof_and_the_same_record(monkeypatch):
    """The same variant met by a launch of 64 elements and one of 100 000 (both 'D'): ONE proof, its job made of the
    variant alone, the second launch served by the record."""
    from nbxtensor.ops.mul import mul_scalar_kernel as k
    monkeypatch.setattr("nbxtensor.memory.guarded.guard_kind", lambda: "vmm")
    monkeypatch.setattr("nbxtensor.ops.elementwise_launch.class_candidates", lambda: list(CANDS))
    jobs = []

    def run(where, cands):
        jobs.append(open(os.path.join(where, "spec.pkl"), "rb").read())
        return {tuple(c): ("ran", ("same",)) for c in cands}, lambda a, b: {}
    monkeypatch.setattr(ew_proof_child, "run_candidates", run)
    tab = _MemTable()
    monkeypatch.setattr(ew_proof, "table", lambda: tab)
    warps = EwTile(2, 65536, "warps")
    tiles = {"tile": TILE, "warps": warps}
    spec = [("*fp32", "D"), ("*fp32", "D"), ("i32", "D"), ("fp32", None), ("constexpr", TILE)]
    for n in (64, 100_000):
        bound = {"x_ptr": object(), "output_ptr": object(), "n_elements": n, "scalar": 3.0, "BLOCK_SIZE": TILE}
        assert ew_proof.settle(k, lambda m, n=n: (-(-n // 4096),), bound, spec,
                               {"num_warps": warps, "num_stages": 3}, tiles) is None
    assert len(jobs) == 1 and len(tab.written) == 1
    (rec,) = tab.written.values()
    assert rec["ladder"] == [16, 240, 4080, 4112, 8240] and rec["method"] == ew_synthetic.METHOD
    assert len(rec["passed"]) == len(CANDS) and "snapshot_bytes" not in rec
    # and the job is the one a direct proof of the variant writes: nothing of either launch is in it
    vid, _ = ew_proof.variant_of(k, spec, {"num_warps": warps, "num_stages": 3}, tiles)
    again = ew_proof.prove(k, spec, {"num_warps": warps, "num_stages": 3}, tiles, vid)
    assert jobs[1] == jobs[0] and again["passed"] == rec["passed"] and again["ladder"] == rec["ladder"]


# --- 4. a recipe, or a refusal by name ----------------------------------------------------------------------------
def test_a_kernel_its_signature_does_not_determine_is_refused_by_name_until_it_declares_a_recipe(monkeypatch):
    import triton
    import triton.language as tl

    @triton.jit
    def two_counts_kernel(x_ptr, y_ptr, n, width, BLOCK: tl.constexpr):
        o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(y_ptr + o, tl.load(x_ptr + o % width, mask=o < n), mask=o < n)

    spec = [("*fp32", "D"), ("*fp32", "D"), ("i32", ""), ("i32", "D"), ("constexpr", TILE)]
    with pytest.raises(ew_synthetic.NotSynthesisable, match="not provable on synthetic inputs.*declare its recipe"):
        ew_synthetic.synthesise(two_counts_kernel, spec, CANDS)
    monkeypatch.setattr("nbxtensor.memory.guarded.guard_kind", lambda: "vmm")
    rec = ew_proof.prove(two_counts_kernel, spec, {"num_warps": 2, "num_stages": 3}, {"tile": TILE}, "v")
    assert rec["unkept"] and "declare its recipe" in rec["refused"] and rec["passed"] == []

    def recipe(S):
        w = S.int("width", 32)
        return {"n": S.n, "width": w, "x_ptr": S.buf("x_ptr", w)}, None
    recipe.count = "n"
    launches = ew_synthetic.synthesise(synthetic(recipe)(two_counts_kernel), spec, CANDS)
    assert [ln.count for ln in launches] == [2, 255, 4095, 4097, 8195]
    assert all(ln.args["x_ptr"].elements == 32 and ln.args["y_ptr"].elements == ln.count for ln in launches)


def test_a_recipe_that_makes_another_binary_is_refused_by_name():
    from nbxtensor.ops.mul import mul_scalar_kernel as k
    bad = ew_synthetic.flat("n_elements")
    spec = [("*fp32", "D"), ("*fp32", ""), ("i32", "D"), ("fp32", None), ("constexpr", TILE)]
    ok = ew_synthetic.synthesise(k, spec, CANDS, recipe=bad)                 # the unaligned output kept unaligned
    assert all(not ln.args["output_ptr"].aligned for ln in ok)

    def wrong(S):
        return {"n_elements": S.n + 1}, None                                 # leaves the count's 'D'
    wrong.count = "n_elements"
    with pytest.raises(ew_synthetic.NotSynthesisable, match="specialises otherwise than the variant"):
        ew_synthetic.synthesise(k, spec, CANDS, recipe=wrong)


def test_every_class_tiled_kernel_with_several_integers_declares_its_recipe():
    """The census of the kernels the class tile reaches (`ew_meta` at the call site): each one either is generic
    (one runtime integer) or declares its recipe where it is defined."""
    import ast
    import pathlib
    root = pathlib.Path(ew_proof.__file__).resolve().parents[1]
    defs, launched = {}, set()
    for p in root.rglob("*.py"):
        tree = ast.parse(p.read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef) and any("jit" in ast.unparse(d) for d in n.decorator_list):
                runtime = [a.arg for a in n.args.args
                           if not (a.annotation is not None and "constexpr" in ast.unparse(a.annotation))]
                defs[n.name] = (p, n, runtime)
        for f in ast.walk(tree):
            if isinstance(f, ast.FunctionDef):
                for c in ast.walk(f):
                    if isinstance(c, ast.Call) and isinstance(c.func, ast.Subscript) and any(
                            "ew_meta(" in ast.unparse(k) or "m['BLOCK_SIZE']" in ast.unparse(k)
                            or 'm["BLOCK_SIZE"]' in ast.unparse(k) for k in c.args + [kw.value for kw in c.keywords]):
                        v = c.func.value
                        launched.add(v.id if isinstance(v, ast.Name) else getattr(v, "attr", None))
    missing = []
    for name in sorted(launched & set(defs)):
        p, node, runtime = defs[name]
        declared = any("synthetic(" in ast.unparse(d) for d in node.decorator_list)
        ints = [a for a in runtime if not a.endswith("_ptr") and not a.startswith("ptr_")]
        if not declared and len(ints) > 1:
            missing.append(f"{name} ({p.name}): {ints}")
    assert missing == [], missing


# --- 5. the stamp and the export ----------------------------------------------------------------------------------
def test_a_table_of_the_old_stamp_is_not_read(monkeypatch, tmp_path):
    new = {"vendor": "v", "profile": "p", "generator": "g", "assembler": "a", "method": ew_synthetic.METHOD}
    old = {k: v for k, v in new.items() if k != "method"}                    # every table made before the cure
    monkeypatch.setattr(ew_proof, "_STAMP", [new])
    rec = {"passed": [[4096, 2, 3]], "kernel": "k"}
    doc = {"header": ew_proof.HEADER, "stamps": {ew_proof.stamp_key(old): {"stamp": old, "variants": {"vid": rec}}}}
    path = tmp_path / "ew_tile_proof.json"
    path.write_text(json.dumps(doc))
    monkeypatch.setattr(ew_proof._Table, "_shipped_path", staticmethod(lambda: str(path)))
    monkeypatch.setattr(ew_proof._Table, "_kept_path", staticmethod(lambda: str(path)))
    tab = ew_proof._Table()
    assert ew_proof.stamp()["method"] == ew_synthetic.METHOD
    assert tab.key != ew_proof.stamp_key(old) and tab.record("vid") is None and tab.status("vid", (4096, 2, 3)) is None
    doc["stamps"][tab.key] = {"stamp": new, "variants": {"vid": rec}}
    path.write_text(json.dumps(doc))
    assert ew_proof._Table().status("vid", (4096, 2, 3)) == "passed"


def test_the_export_refuses_a_disagreement_and_adds_an_agreement(tmp_path):
    def table(passed):
        return {"header": ew_proof.HEADER, "stamps": {"s": {"stamp": {}, "variants": {
            "vid": {"kernel": "mul_scalar_strided_nd_kernel", "independent": True, "passed": passed}}}}}
    all15 = [list(c) for c in CANDS]
    shipped, kept = tmp_path / "shipped.json", tmp_path / "kept.json"
    shipped.write_text(json.dumps(table([c for c in all15 if c != [4096, 2, 3]])))
    kept.write_text(json.dumps(table(all15)))
    with pytest.raises(ew_proof.TileUnproven, match="the shipped table passes"):
        ew_proof.export(str(kept), str(shipped))
    kept.write_text(json.dumps(table([c for c in all15 if c != [4096, 2, 3]])))
    assert ew_proof.export(str(kept), str(shipped)).startswith("0 variant(s) added")
    other = tmp_path / "other.json"
    assert ew_proof.export(str(kept), str(other)).startswith("1 variant(s) added")


# --- 6. the guard on a device -------------------------------------------------------------------------------------
def _device_guard():
    from nbxtensor.memory.allocator import DeviceAllocator
    from nbxtensor.memory.guarded import guard_kind
    try:
        if DeviceAllocator.device_count() == 0:
            return None
    except Exception:
        return None
    return guard_kind()


@pytest.fixture
def _own_worker(monkeypatch):
    """A proof worker of this test's own that can import the test kernels (this directory on its path)."""
    here = os.path.dirname(os.path.abspath(__file__))
    monkeypatch.setenv("PYTHONPATH", here + os.pathsep + os.environ.get("PYTHONPATH", ""))
    if here not in sys.path:
        monkeypatch.syspath_prepend(here)
    worker = ew_proof_child._Worker()
    monkeypatch.setattr(ew_proof_child, "_WORKER", worker)
    yield worker
    if worker.proc is not None and worker.proc.poll() is None:
        worker.proc.kill()
        worker.proc.wait()


def test_a_lane_past_its_extent_is_caught_at_exactly_its_tile(_own_worker):
    """On this device's guard (Metal: shader validation on dedicated buffers; CUDA: the VMM's unmapped granule): a copy
    that writes ONE element past its output at tile 4096 faults at the three 4096 candidates and nowhere else; the
    correct copy passes every candidate with the same bits."""
    kind = _device_guard()
    if kind is None:
        pytest.skip("no device with a guard on this machine")
    import _ew_overrun_kernel as K
    from nbxtensor.ops.elementwise_launch import class_candidates
    cands = class_candidates()
    spec = [("*fp32", "D"), ("*fp32", "D"), ("i32", "D"), ("constexpr", EwTile(256, 65536, "tile"))]
    opts = {"num_warps": 2, "num_stages": 3}
    bad = ew_proof.prove(K.overrun_at_4096_kernel, spec, opts, {}, "bad")
    assert bad["guard"] == kind and bad["refused"] is None
    assert sorted(tuple(f["tile"]) for f in bad["faults"]) == sorted(c for c in cands if c[0] == 4096)
    assert sorted(map(tuple, bad["passed"])) == sorted(c for c in cands if c[0] != 4096)
    good = ew_proof.prove(K.masked_copy_kernel, spec, opts, {}, "good")
    assert good["faults"] == [] and good["dissent"] == [] and len(good["passed"]) == len(cands)


def _volta():
    try:
        from nbxtensor.backend.launcher import target
        t = target()
        return getattr(t, "backend", None) == "cuda" and int(t.arch) == 70
    except Exception:
        return False


@pytest.mark.skipif(not _volta(), reason="the red cell is a V100 assembler's build of tile (4096, 2, 3)")
def test_the_red_variant_faults_at_its_tile_on_the_synthetic_ladder_alone():
    """Variant 6be93ab046480f3a, no model and no launch of a caller: the synthetic ladder finds the 700 at
    (4096, 2, 3) and variant 6859212e3972bf7a (add_strided_nd fp16, NDIM 4) faults at some tile (`tools/prove_an_element_wise_variant.py --red`)."""
    import importlib.util
    import pathlib
    tool = pathlib.Path(__file__).resolve().parents[1] / "tools" / "prove_an_element_wise_variant.py"
    spec = importlib.util.spec_from_file_location("prove_an_element_wise_variant", tool)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--red"]) == 0


# --- 7. the worker holds nothing on the card while the parent runs ------------------------------------------------
def test_after_a_job_the_worker_returns_every_block_and_holds_nothing(monkeypatch):
    """A worker's account after a proof, counted: blocks a job freed but the pool cached go back to the driver, and
    what it still holds is 0."""
    from nbxtensor.memory.allocator import DeviceAllocator as DA
    pool = {"cached": 6 << 20}
    seen = []
    monkeypatch.setattr(DA, "_cuda_live_bytes", {0: 0})
    monkeypatch.setattr(DA, "device_synchronize", staticmethod(lambda device_idx=None: seen.append("sync")))

    def flush(dev=None):
        seen.append("flush")
        freed, pool["cached"] = pool["cached"], 0
        return freed
    monkeypatch.setattr(DA, "_pool_flush", staticmethod(flush))
    assert ew_proof_child._release_device() == 0
    assert seen == ["sync", "flush"] and pool["cached"] == 0


def test_the_worker_exits_on_its_idle_bound_and_a_later_job_starts_a_fresh_one():
    """Device-free: a worker with a 0.3 s bound is gone after it; the next job is answered by a fresh worker (the
    job sent to the one that ended is sent again, never recorded as a death)."""
    import time
    w = ew_proof_child._Worker(idle=0.3)
    from nbxtensor.ops.mul import mul_scalar_kernel as k
    spec = [("*fp32", "D"), ("*fp32", "D"), ("i32", "D"), ("fp32", None), ("constexpr", TILE)]
    launches = ew_synthetic.synthesise(k, spec, CANDS)
    job, _n = ew_proof_child.write_job(k, launches, CANDS, {}, "vmm")
    with open(os.path.join(job, "cands.json"), "w") as fh:
        json.dump([], fh)                                       # no candidate: the job runs nothing on a device
    try:
        assert w.ask({"mode": "run", "where": job, "start": 0}) == (0, "")
        first = w.proc
        deadline = time.time() + 10
        while first.poll() is None and time.time() < deadline:
            time.sleep(0.05)
        assert first.poll() == ew_proof_child.IDLE_EXIT          # gone: nothing of it left on any card
        assert w.ask({"mode": "run", "where": job, "start": 0}) == (0, "")
        assert w.proc is not first and ew_proof_child._Worker.started >= 2
    finally:
        if w.proc is not None and w.proc.poll() is None:
            w.proc.kill()
        import shutil
        shutil.rmtree(job, ignore_errors=True)
