"""A confirmation run (certified-only) and a flash launch with fewer query rows than keys (a continuing turn's prefill)
whose own key the directory does not hold (task 203 choice 3, 2026-11-15: the certified square IS the served key).

The launch is SERVED by the certified square key of the same key length (Tk, Tk): a certified setting, so no sweep and
no replay cache, and the door's two promises hold. It is refused, by name, only when the square key is not certified
either.

What each test does if the code is wrong:
* the square default removed from the launch site: `test_a_turn_prefill_flash_is_confirmed_by_its_certified_square_key`
  fails (KeyNotCertified);
* the refusal lost behind the default: `test_a_turn_prefill_flash_without_a_certified_square_is_refused` fails (the
  launch runs).
No card: the directory and the tuner are stand-ins."""
import pytest

pytest.importorskip("triton")


class _FlashTuner:
    """A stand-in autotuner at the launch site (`_configs._announce_first_sweep`): no card, the launch is counted."""
    arg_names, keys = [], ["CACHE_KEY_SEQLEN_Q", "CACHE_KEY_SEQLEN_K", "headdim"]

    def __init__(self):
        self.cache, self.launched = {}, []

    def run(self, *args, **kwargs):
        self.launched.append(kwargs)
        return "launched"


def _confirmation_launch(monkeypatch, held):
    """One q < Tk flash launch (20 query rows, 128 keys, head dim 64) through the launch site in a confirmation run
    whose directory holds `held`; returns the tuner and the launch thunk."""
    from nbxtensor.ops import _configs, autotune_cache, autotune_certified as C, launch_keys as L
    monkeypatch.setenv("NBX_AUTOTUNE_CERTIFIED_ONLY", "1")
    monkeypatch.delenv("NBX_AUTOTUNE_CERTIFIED", raising=False)
    monkeypatch.delenv("NBX_KEY_RECORD", raising=False)
    monkeypatch.setattr(autotune_cache, "qual_of", lambda at: L.FLASH)
    monkeypatch.setattr(C, "lookup", lambda qual, tuner, key, **kw: held.get(tuple(key)))
    monkeypatch.setattr(C, "executing_memory_gb", lambda *a, **k: 18)
    monkeypatch.setattr(C, "active_profile", lambda: ("a-vendor", "a-profile"))
    t = _configs._announce_first_sweep(_FlashTuner())
    return t, lambda: t.run(CACHE_KEY_SEQLEN_Q=20, CACHE_KEY_SEQLEN_K=128, headdim=64)


def test_a_turn_prefill_flash_is_confirmed_by_its_certified_square_key(monkeypatch):
    square = {"config": {"kwargs": {"BLOCK_M": 32, "BLOCK_N": 32}, "num_warps": 4, "num_stages": 1}}
    t, launch = _confirmation_launch(monkeypatch, {(128, 128, 64): square})
    assert launch() == "launched" and len(t.launched) == 1
    assert t.cache[(20, 128, 64)].kwargs == {"BLOCK_M": 32, "BLOCK_N": 32}, "served, but not the certified square key"


def test_a_turn_prefill_flash_without_a_certified_square_is_refused(monkeypatch):
    from nbxtensor.ops.autotune_certified import KeyNotCertified
    t, launch = _confirmation_launch(monkeypatch, {})
    with pytest.raises(KeyNotCertified, match="CERTIFIED-ONLY: no certified setting for"):
        launch()
    assert t.launched == [] and t.cache == {}, "a confirmation run launched (swept) a key nothing certifies"
