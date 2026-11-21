"""A tuning keyword the installed Triton's autotuner does not take is refused by name, never dropped.

`nbx_autotune` dropped any keyword the installed `triton.autotune` lacked, with a warning. `restore_value` and
`reset_to_zero` decide what each measured configuration starts from: dropped, an overwriting or accumulating kernel
(the chunked gated delta rule's state kernel declares `restore_value=['u_ptr']`) would be tuned on its previous
configuration's output, and the run would go on (qa on the library, 2026-11-19). `cache_results` alone is the
library's to drop: it persists its sweeps itself.

What this test would do if the code were wrong: the decoration returns with a warning instead of raising.
It needs no device."""
import warnings

import pytest

triton = pytest.importorskip("triton")

from nbxtensor.ops import _configs


def _an_old_autotune(configs, key, prune_configs_by=None, reset_to_zero=None, warmup=None, rep=None):
    return lambda fn: fn


def test_a_keyword_the_installed_autotuner_lacks_is_refused_by_name(monkeypatch):
    monkeypatch.setattr(triton, "autotune", _an_old_autotune)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TypeError, match=r"does not take \['restore_value'\]"):
            _configs.nbx_autotune(configs=[], key=[], restore_value=["u_ptr"])


def test_the_library_drops_triton_s_own_file_cache_and_nothing_else(monkeypatch):
    seen = {}

    def autotune(configs, key, restore_value=None):
        seen.update(configs=configs, key=key, restore_value=restore_value)
        return lambda fn: fn

    monkeypatch.setattr(triton, "autotune", autotune)
    _configs.nbx_autotune(configs=[], key=["N"], cache_results=True, restore_value=["u_ptr"])
    assert seen == {"configs": [], "key": ["N"], "restore_value": ["u_ptr"]}
