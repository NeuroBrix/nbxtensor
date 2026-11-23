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


def test_the_moved_launch_covers_what_it_covered():
    meta = {"BLOCK_SIZE": EwTile(256, 65536, "tile"), "num_warps": EwTile(8, 65536, "warps")}
    bound = {"x": "x", "n": 64, "BLOCK_SIZE": EwTile(4096, 65536, "tile"), "NDIM": 3}
    grid, kw = ew_proof.retiled((5, 2), bound, {"num_warps": EwTile(2, 65536, "warps")}, 4096, meta, 3)
    assert grid == (80, 2) and kw["BLOCK_SIZE"] == 256 and kw["num_warps"] == 8 and kw["num_stages"] == 3
    assert kw["NDIM"] == 3 and kw["n"] == 64


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
        ew_proof.settle(K(), (1,), {}, spec, {"num_warps": tiles["warps"], "num_stages": 3}, tiles)


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
    at the class setting 4096 / 2 warps. On the base it faults (700); with the proof the variant is proven in a child
    process, the class moves, and the product is zeros."""
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
