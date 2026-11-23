"""A launch at its class's proven tile settles at once: no move, no line (`ops/ew_proof.settle`).

The class's setting (`elementwise_launch.class_meta`) stated the tile and the warps, not the stages. A launch's options
then omitted num_stages and the asked tile fell to the BACKEND'S DEFAULT (Metal 1, HIP 2), while every record holds the
ladder's stages (3, `class_candidates`). On the Mac every launch of the class said "fails the class's tile 1024x8 ...
the class moves to 1024x8 (stages 3)" — 1,867 lines on one TinyLlama --triton first launch, at every launch, and the
move redone each time. CUDA's default is the ladder's 3, which hid it there.

What this test does if the code is wrong:
1. the class's setting states no stages, so a launch omitting num_stages on a backend whose default differs is moved
   (settle returns a launch, a "[ew-proof] class" line is printed) although its tile is proven;
2. a class that truly moves says so at every launch instead of once per class and process;
3. (Metal) 50 launches of one small element-wise op print "[ew-proof] class" lines after the first proof.
Run: python -m pytest tests/test_a_launch_at_a_proven_tile_settles_without_a_move.py
"""
import os
import subprocess
import sys

import pytest

from nbxtensor.ops import elementwise_launch as EL
from nbxtensor.ops import ew_proof

LADDER = [(256, 2, 3), (512, 4, 3), (1024, 4, 3), (1024, 8, 3), (2048, 8, 3)]
CLS = 65536


class _K:
    """A kernel as `settle` reads it: a name and the module of its function."""
    __name__ = "copy_kernel"
    fn = type("F", (), {"__module__": "nbxtensor.tensor"})


def _home(passed):
    """A proof table whose every variant passes the tiles `passed`, at no stamp of this machine."""
    tab = object.__new__(ew_proof._Table)
    tab.key = "test"
    tab.kept, tab.shipped, tab.unkept, tab.seen_classes = {}, {}, {}, {}
    tab.record = lambda vid: {"passed": [list(t) for t in passed]}
    return tab


@pytest.fixture(autouse=True)
def _backend_default_differs(monkeypatch):
    """The ladder holds stages 3; the backend compiles stages 1 where a launch names none (Metal's default)."""
    monkeypatch.setattr(EL, "class_candidates", lambda: list(LADDER))
    monkeypatch.setattr(EL, "_MEMO", {})
    monkeypatch.setattr(ew_proof, "_default_option", lambda kernel, name: {"num_warps": 4, "num_stages": 1}[name])
    monkeypatch.setattr(ew_proof, "_SAID", set(), raising=False)


def _launch(meta):
    """The arguments `settle` receives for a launch handed `meta` by `**ew_meta(n)` and naming no num_stages."""
    from nbxtensor.ops.ew_tile import class_tiles
    options = {k: v for k, v in meta.items() if k != "BLOCK_SIZE"}
    bound = {"x": 0, "n": 4096, "BLOCK_SIZE": meta["BLOCK_SIZE"]}
    spec = [("*fp32", "D"), ("i32", ""), ("constexpr", meta["BLOCK_SIZE"])]
    return EL.ew_grid(4096), bound, spec, options, class_tiles(list(bound.values()) + list(options.values()))


def test_the_class_setting_states_the_stages_its_ladder_proves():
    """The class's warps carry the ladder's stages, so the asked tile is keyed at 3, not the backend's 1."""
    meta = EL.class_meta(CLS, 1024, 8)
    assert ew_proof.variant_of(_K(), [], {"num_warps": meta["num_warps"]}, {"tile": meta["BLOCK_SIZE"]})[1] \
        == (1024, 8, 3)


def test_a_launch_omitting_stages_at_a_proven_tile_settles_without_a_move(monkeypatch, capsys):
    """A launch naming no num_stages at a proven tile runs as asked: no move, no line."""
    monkeypatch.setattr(ew_proof, "table", lambda: _home([LADDER[3]]))
    grid, bound, spec, options, tiles = _launch(EL.class_meta(CLS, 1024, 8))
    assert "num_stages" not in options
    assert ew_proof.settle(_K(), grid, bound, spec, options, tiles) is None
    assert "[ew-proof] class" not in capsys.readouterr().err


def test_a_real_move_is_said_once_per_class_however_many_launches(monkeypatch, capsys):
    """Twenty launches of a class that truly moves print its move line once."""
    monkeypatch.setattr(ew_proof, "table", lambda: _home([LADDER[1]]))      # 1024x8 fails: the class moves to 512x4
    monkeypatch.setattr(EL, "move_class_winner", lambda nb, b, w: EL.class_meta(nb, b, w))   # every launch moves
    for _ in range(20):
        grid, bound, spec, options, tiles = _launch(EL.class_meta(CLS, 1024, 8))
        moved = ew_proof.settle(_K(), grid, bound, spec, options, tiles)
        assert moved is not None and moved[1]["BLOCK_SIZE"] == 512 and moved[1]["num_stages"] == 3
    assert capsys.readouterr().err.count("[ew-proof] class") == 1


_FIFTY = r"""
import sys
from nbxtensor.tensor import NBXTensor
from nbxtensor.dtype.nbx_dtype import NBXDtype
x = NBXTensor.empty((40000,), dtype=NBXDtype.float32, device='cuda')
for _ in range(50):
    x.fill_(1.0)
print("fifty done", file=sys.stderr)
"""


def _metal():
    """Whether this host launches on Metal."""
    try:
        from nbxtensor.backend.launcher import target
        return getattr(target(), "backend", None) == "mps"
    except Exception:
        return False


@pytest.mark.skipif(not _metal(), reason="a real Metal launch")
def test_fifty_metal_launches_of_one_op_print_no_class_line(tmp_path):
    """Fifty real Metal fills, on a proving run and on a read-back run, print no class move line."""
    env = dict(os.environ, NEUROBRIX_REPLAY_CACHE=str(tmp_path))
    for run in range(2):                            # the first run proves; the second reads its proof back
        out = subprocess.run([sys.executable, "-c", _FIFTY], env=env, capture_output=True, text=True, timeout=900)
        assert "fifty done" in out.stderr, out.stderr[-2000:]
        assert out.stderr.count("[ew-proof] class") == 0, out.stderr[-2000:]
