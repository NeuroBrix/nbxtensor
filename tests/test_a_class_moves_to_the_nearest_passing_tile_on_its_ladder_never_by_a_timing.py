"""When a variant fails its element-wise class's winner, the class moves to the candidate NEAREST the winner on the
class's ordered ladder (`elementwise_launch.class_candidates`) among those every proven variant passes, a tie to the
earlier one: read from the ladder and the proofs, never timed (C17, orchestrator 17:41 23/11: the copy's timing,
measured per fresh replay cache, re-picked 512x8, 2048x8 or 1024x4 run to run on the Dell — a search).

What this test does if the code is wrong:
1. two fresh homes holding the same proofs move the class to different tiles, or a timing child is asked;
2. the move is not the nearest passing candidate on the ladder (or a tie does not go to the earlier one);
3. a winner that is no candidate of the ladder is not refused by name; no common passing candidate is not None.
Device-free.
Run: python -m pytest tests/test_a_class_moves_to_the_nearest_passing_tile_on_its_ladder_never_by_a_timing.py
"""
import pytest

from nbxtensor.ops import elementwise_launch as EL
from nbxtensor.ops import ew_proof, ew_proof_child

LADDER = [(256, 2, 3), (512, 4, 3), (1024, 4, 3), (2048, 8, 3), (4096, 8, 3), (4096, 16, 3)]


def _home(passed_by_variant):
    """A fresh home's proof table holding these variants' passing tiles for class 65536."""
    tab = object.__new__(ew_proof._Table)
    tab.kept = {vid: {"passed": [list(t) for t in passed]} for vid, passed in passed_by_variant.items()}
    tab.shipped = {}
    tab.unkept = {}
    tab.seen_classes = {"65536": set(passed_by_variant)}
    return tab


@pytest.fixture(autouse=True)
def _ladder(monkeypatch):
    monkeypatch.setattr(EL, "class_candidates", lambda: list(LADDER))
    monkeypatch.setattr(ew_proof_child, "_child", lambda argv: pytest.fail(f"a child was asked: {argv}"))


def test_two_fresh_homes_holding_the_same_proofs_pick_the_same_tile():
    proofs = {"a": [LADDER[0], LADDER[1], LADDER[4]], "b": [LADDER[0], LADDER[1], LADDER[2], LADDER[4]]}
    # the winner 2048x8 (index 3) fails; indexes 0, 1 and 4 pass every variant: 4 is the nearest (distance 1)
    assert {_home(proofs).class_winner(65536, (2048, 8)) for _ in range(2)} == {LADDER[4]}


def test_the_nearest_on_the_ladder_and_a_tie_goes_to_the_earlier():
    tie = {"a": [LADDER[2], LADDER[4]]}             # both at distance 1 from the winner at index 3
    assert _home(tie).class_winner(65536, (2048, 8)) == LADDER[2]
    far = {"a": [LADDER[0], LADDER[5]], "b": [LADDER[0], LADDER[5]]}
    assert _home(far).class_winner(65536, (2048, 8)) == LADDER[5]      # distance 2 beats 3


def test_a_winner_off_the_ladder_is_refused_and_no_common_tile_is_none():
    with pytest.raises(ew_proof.TileUnproven, match="is no candidate of the class's ladder"):
        _home({"a": [LADDER[0]]}).class_winner(65536, (8192, 32))
    assert _home({"a": [LADDER[0]], "b": [LADDER[1]]}).class_winner(65536, (2048, 8)) is None
