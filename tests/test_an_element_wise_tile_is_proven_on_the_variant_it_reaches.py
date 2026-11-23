"""An element-wise class tile is proven on each compiled variant it reaches before that binary runs on it
(`ops/ew_tile.py`, `ops/ew_proof.py`; orchestrator rulings 23/11/2026 12:5x-13:1x). PixArt-XL on a V100: the class
setting BLOCK_SIZE 4096 / 2 warps, swept on the copy, faulted (700) in `mul_scalar_strided_nd_kernel` fp32 — the
assembler's -O3 build of that tile.

What each test would do on the base: `ops.ew_tile` / `ops.ew_proof` do not exist (ImportError, red); the CUDA cell
faults with an illegal address (red)."""
import json
import math
import pickle

import numpy as np
import pytest

from nbxtensor.ops import ew_proof
from nbxtensor.ops.ew_tile import EwTile, class_tiles, declared_independence, tile_dependent, tile_independent


def test_a_class_tile_is_its_number_and_keys_apart_from_it():
    t = EwTile(4096, 65536, "tile")
    assert t == 4096 and int(t) == 4096 and str(t) == "4096" and math.ceil(64 / t) == 1
    assert hash(t) != hash(4096)                      # its own entry in the launcher's cache: the proof runs there
    assert {**{"BLOCK_SIZE": t}}["BLOCK_SIZE"].ew_class == 65536     # survives a ** splat and a key read
    back = pickle.loads(pickle.dumps(t))
    assert isinstance(back, EwTile) and back.role == "tile" and back == 4096
    assert class_tiles([t, 3, EwTile(2, 65536, "warps")]).keys() == {"tile", "warps"}


def test_the_class_meta_hands_out_marked_tiles():
    from nbxtensor.ops.elementwise_launch import class_meta
    meta = class_meta(65536, 4096, 2)
    assert isinstance(meta["BLOCK_SIZE"], EwTile) and meta["BLOCK_SIZE"].role == "tile"
    assert isinstance(meta["num_warps"], EwTile) and meta["num_warps"] == 2


def test_a_kernel_declares_its_tile_independence_where_it_is_defined():
    from nbxtensor.ops.binary_strided import mul_scalar_strided_nd_kernel
    from nbxtensor.ops.index_wrappers import scatter_add_kernel
    assert declared_independence(mul_scalar_strided_nd_kernel) is True
    assert declared_independence(scatter_add_kernel) is False

    class Fn:
        params = ()
    k = tile_dependent("an accumulate")(Fn())
    assert declared_independence(k) is False and k.nbx_tile_dependent_why == "an accumulate"
    assert declared_independence(tile_independent(Fn())) is True


CANDS = [(256, 2, 3), (256, 8, 3), (1024, 4, 3), (4096, 2, 3), (4096, 8, 3)]


def _ran(*digests):
    return ("ran", tuple(digests))


def test_a_faulting_tile_is_excluded_and_recorded():
    res = {c: _ran("a") for c in CANDS}
    res[(4096, 2, 3)] = ("fault", "cuLaunchKernel failed (700: an illegal memory access was encountered)")
    rec = ew_proof.judge(True, CANDS, res, lambda a, b: {})
    assert [4096, 2, 3] not in rec["passed"] and len(rec["passed"]) == 4
    assert rec["faults"] == [{"tile": [4096, 2, 3], "fault": res[(4096, 2, 3)][1]}]


def test_a_tile_that_leaves_other_bits_is_excluded_with_its_first_difference():
    res = {c: _ran("a") for c in CANDS}
    res[(1024, 4, 3)] = _ran("b")
    rec = ew_proof.judge(True, CANDS, res, lambda a, b: {"byte": 8, "candidate": a, "agreeing": b})
    assert [1024, 4, 3] not in rec["passed"]
    assert rec["dissent"] == [{"tile": [1024, 4, 3], "first_difference": {"byte": 8, "candidate": ("b",),
                                                                          "agreeing": ("a",)}}]


def test_no_majority_or_fewer_than_three_is_not_a_proof():
    res = {c: _ran(str(i % 2)) for i, c in enumerate(CANDS[:4])}
    assert ew_proof.judge(True, CANDS[:4], res, lambda a, b: {})["passed"] == []         # 2 against 2
    res = {c: ("fault", "700") for c in CANDS}
    res[CANDS[0]], res[CANDS[1]] = _ran("a"), _ran("a")
    rec = ew_proof.judge(True, CANDS, res, lambda a, b: {})
    assert rec["passed"] == [] and "fewer than 3" in rec["refused"]


def test_a_tile_dependent_kernel_is_proven_by_fault_alone():
    res = {c: _ran(str(i)) for i, c in enumerate(CANDS)}                     # every candidate different bits
    res[(4096, 2, 3)] = ("fault", "700")
    rec = ew_proof.judge(False, CANDS, res, lambda a, b: {})
    assert len(rec["passed"]) == 4 and rec["dissent"] == []


def test_the_grid_is_a_function_of_the_tile_on_every_axis():
    """The token bitmask's grid (rows, cdiv(vocab, tile)): its TILE axis is axis 1 — a grid computed once by the
    caller, or rescaled on axis 0, ran the candidates and the moved launch on a wrong grid (V100, 23/11)."""
    from nbxtensor.ops.elementwise_launch import ew_grid_of, per_tile
    grid = ew_grid_of(2, per_tile(151936))
    assert grid({"BLOCK": EwTile(2048, 1048576, "tile")}) == (2, 75)
    assert grid({"BLOCK": EwTile(256, 1048576, "tile")}) == (2, 594)
    assert ew_grid_of(per_tile(10, times=6))({"B": EwTile(4, 16, "tile")}) == (18,)


def test_the_moved_launch_keeps_its_grid_function_and_takes_the_new_tile():
    from nbxtensor.ops.elementwise_launch import ew_grid_of, per_tile
    meta = {"BLOCK_SIZE": EwTile(256, 65536, "tile"), "num_warps": EwTile(8, 65536, "warps")}
    bound = {"x": "x", "n": 64, "BLOCK": EwTile(2048, 65536, "tile"), "NDIM": 3}
    fn = ew_grid_of(2, per_tile(32000))
    grid, kw = ew_proof.retiled(fn, bound, {"num_warps": EwTile(2, 65536, "warps")}, 2048, meta, 3)
    assert grid is fn and grid(kw) == (2, 125)
    assert kw["BLOCK"] == 256 and kw["num_warps"] == 8 and kw["num_stages"] == 3 and kw["NDIM"] == 3


def test_a_class_tile_launch_with_a_fixed_grid_is_refused_by_name():
    class K:
        __name__ = "k"
        fn = type("F", (), {"__module__": "m"})
    tiles = {"tile": EwTile(2048, 65536, "tile")}
    with pytest.raises(ew_proof.TileUnproven, match="grid is fixed by the caller"):
        ew_proof.settle(K(), (2, 16), {}, [], {"num_warps": 8}, tiles)


class _Empty:
    key = "test"

    def __init__(self):
        self.seen_classes = {}

    def status(self, vid, tile):
        return None


def test_a_confirmation_run_refuses_an_unproven_variant_by_name(monkeypatch):
    monkeypatch.setenv("NBX_AUTOTUNE_CERTIFIED_ONLY", "1")
    monkeypatch.setattr(ew_proof, "table", lambda: _Empty())

    class K:
        __name__ = "mul_scalar_strided_nd_kernel"
        fn = type("F", (), {"__module__": "nbxtensor.ops.binary_strided"})
    tiles = {"tile": EwTile(4096, 65536, "tile"), "warps": EwTile(2, 65536, "warps")}
    spec = [("*fp32", "D"), ("constexpr", tiles["tile"]), ("constexpr", 3)]
    with pytest.raises(ew_proof.TileUnproven, match="no proof table holds"):
        ew_proof.settle(K(), lambda meta: (1,), {}, spec, {"num_warps": tiles["warps"], "num_stages": 3}, tiles)


def test_the_variant_holds_warps_stages_and_every_constexpr():
    class K:
        __name__ = "k"
        fn = type("F", (), {"__module__": "m"})
    tiles = {"tile": EwTile(4096, 65536, "tile")}
    spec = [("*fp32", "D"), ("constexpr", tiles["tile"]), ("constexpr", 3)]
    vid, tile = ew_proof.variant_of(K(), spec, {"num_warps": 2, "num_stages": 3}, tiles)
    assert tile == (4096, 2, 3)
    other, _ = ew_proof.variant_of(K(), spec[:2] + [("constexpr", 2)], {"num_warps": 2, "num_stages": 3}, tiles)
    assert vid != other and json.loads(vid)[1][1] == ["tile", "tile"]


def _cuda():
    try:
        from nbxtensor.backend.launcher import target
        return getattr(target(), "backend", None) == "cuda"
    except Exception:
        return False


@pytest.mark.skipif(not _cuda(), reason="the 700 is a CUDA assembler's fault (V100, ptxas -O3)")
def test_the_pixart_multiply_runs_where_the_class_tile_faults(monkeypatch, tmp_path):
    """The red cell: x = column 0 of a (64, 576) fp32 tensor viewed (64, 1, 1), strides (576, 1, 1), times 0.0,
    at the class setting 4096 / 2 warps. Without the proof it faults (700); with it the variant is proven on its
    synthetic launches in the worker process (nothing of this launch enters the proof), the class moves, and the
    product is zeros."""
    monkeypatch.setenv("NEUROBRIX_REPLAY_CACHE", str(tmp_path))
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.ops import elementwise_wrappers as EW
    from nbxtensor.ops.elementwise_launch import class_meta
    monkeypatch.setattr(ew_proof, "_TABLE", [None])
    monkeypatch.setattr(EW, "_ew_meta", lambda n: class_meta(65536, 4096, 2))
    base = NBXTensor.from_numpy(np.arange(1, 64 * 576 + 1, dtype=np.float32).reshape(64, 576), device="cuda:0")
    x = base.narrow(1, 0, 1).unsqueeze(2)
    out = EW.mul(x, 0.0)
    assert out.numpy().shape == (64, 1, 1) and (out.numpy() == 0.0).all()
    kept = json.load(open(tmp_path / "ew_tile_proof.json"))
    recs = [r for b in kept["stamps"].values() for r in b["variants"].values()
            if r["kernel"] == "mul_scalar_strided_nd_kernel"]
    assert recs and [4096, 2, 3] not in recs[0]["passed"]


def _accelerator():
    from nbxtensor.memory.allocator import DeviceAllocator
    if DeviceAllocator.device_count() == 0:
        pytest.skip("no accelerator on this machine")


def _fresh_proofs(monkeypatch, tmp_path):
    monkeypatch.setenv("NEUROBRIX_REPLAY_CACHE", str(tmp_path))
    monkeypatch.setattr(ew_proof, "_TABLE", [None])
    from nbxtensor.backend import launcher
    launcher.reset_caches()


@pytest.mark.parametrize("vocab", [32000, 151936])
def test_the_token_bitmask_runs_at_2048x8_and_after_a_forced_move(vocab, monkeypatch, tmp_path):
    """The regression of the V100 rerun (23/11): (rows, cdiv(vocab, tile)) at the class tile 2048 x 8 — 32000 had no
    agreeing set, 151936 left its row untouched after a move. Every candidate agrees on the right grid; after a move
    the masked logits are -inf over the whole row, the allowed ones keep their bytes."""
    _accelerator()
    _fresh_proofs(monkeypatch, tmp_path)
    from nbxtensor.ops import token_bitmask as TB
    from nbxtensor.ops.elementwise_launch import class_meta, ew_bucket
    from nbxtensor.tensor import NBXTensor
    cls = ew_bucket(vocab)
    from nbxtensor.ops import elementwise_launch as EL
    monkeypatch.setattr(EL, "ew_meta", lambda n: class_meta(cls, 2048, 8))      # token_bitmask imports it at use
    rng = np.random.default_rng(vocab)
    x = rng.standard_normal((2, vocab)).astype(np.float32)
    allowed = [sorted(rng.choice(vocab, 17, replace=False)), [0, vocab - 1]]
    mask = np.concatenate([TB.token_bitmask(a, vocab) for a in allowed])

    def check():
        t = NBXTensor.from_numpy(x.copy())
        TB.apply_token_bitmask(t, mask)
        got = t.numpy()
        for r, a in enumerate(allowed):
            keep = np.zeros(vocab, bool)
            keep[a] = True
            assert np.array_equal(got[r][keep], x[r][keep]) and np.all(np.isneginf(got[r][~keep]))
    check()
    tab = ew_proof.table()
    (vid, rec), = [(v, r) for v, r in tab.kept.items() if r["kernel"] == "_token_bitmask_kernel"]
    assert rec["refused"] is None and len(rec["passed"]) >= 3
    assert rec["dissent"] == [] and rec["faults"] == []
    struck = [t for t in rec["passed"] if t[:2] == [2048, 8]]
    rec["passed"] = [t for t in rec["passed"] if t[:2] != [2048, 8]]      # force the move: the class tile fails
    rec["struck"] = {f"{t[0]}x{t[1]}": "struck by the test (no fault: the move is forced)" for t in struck}
    from nbxtensor.backend import launcher
    launcher.reset_caches()
    check()


def test_a_one_d_kernel_far_above_the_largest_tile_agrees_on_every_candidate(monkeypatch, tmp_path):
    _accelerator()
    _fresh_proofs(monkeypatch, tmp_path)
    from nbxtensor.ops import elementwise_wrappers as EW
    from nbxtensor.ops.elementwise_launch import class_meta, ew_bucket
    from nbxtensor.tensor import NBXTensor
    n = 100003
    monkeypatch.setattr(EW, "_ew_meta", lambda k: class_meta(ew_bucket(n), 1024, 4))
    x = np.arange(n, dtype=np.float32)
    out = EW.mul(NBXTensor.from_numpy(x), 3.0).numpy()
    assert np.array_equal(out, x * 3.0)
    recs = [r for r in ew_proof.table().kept.values() if r["kernel"] == "mul_scalar_kernel"]
    assert recs and recs[0]["dissent"] == [] and recs[0]["refused"] is None
