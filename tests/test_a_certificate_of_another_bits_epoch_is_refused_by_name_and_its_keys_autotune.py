"""A certificate the library wrote before a change that moved the bits (its bits epoch) proves settings for bits this
library no longer computes. On a user's machine that certified with the old library, the upgrade must not serve them:
at load a certificate of another bits epoch, or with none, is REFUSED BY NAME (the file, its epoch, the library's) and
its keys go back to the normal autotune path. A certificate of the library's own epoch is still served. The library's
own placement rule (`file_certification`, which every certifying tool files through) stamps the epoch on every proof
it places and drops what an older epoch left in the file, so a recertification writes the new epoch with no change of
its own. Card-less: no kernel runs."""
import copy
import json

import pytest

from nbxtensor.ops import autotune_certified as C

KEY = "(64, 64, 64, 'fp32')"


def _proof(epoch):
    proof = {"date": "2026-11-22T11:40:00", "engine_version": "test", "backend": {"triton": "3.4.0"}, "shape": [64, 64, 64, "fp32"],
             "deviation": 1e-7, "tolerance": 1e-5, "machine": {"device": {"memory_mb": 16384}}, "oracle": "fp64",
             "built": True}
    if epoch is not None:
        proof["bits_epoch"] = epoch
    return proof


def _doc(epoch):
    return {"format": C.FORMAT, "vendor": "testvendor", "profile": "testprofile", "kernel": "nbxtensor.ops.k.mm_kernel",
            "dtype": "fp32", "entries": {KEY: {"config": {"kwargs": {"BLOCK_M": 32}, "num_warps": 4, "num_stages": 2},
                                              "proof": _proof(epoch), "excluded": []}}}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROBRIX_AUTOTUNE_CERTIFIED_DIR", str(tmp_path))
    monkeypatch.setattr(C, "_LOADED", {})
    monkeypatch.setattr(C, "_REFUSED", {})
    path = tmp_path / "testvendor" / "testprofile" / "mm_kernel.fp32.json"
    path.parent.mkdir(parents=True)
    return path


def _served(path, doc):
    path.write_text(json.dumps(doc), encoding="utf-8")
    return C._load("testvendor", "testprofile", "nbxtensor.ops.k.mm_kernel", "fp32")


def test_a_certificate_of_the_library_s_epoch_is_served(store):
    assert C.validate(_doc(C.BITS_EPOCH), store) == []
    assert KEY in (_served(store, _doc(C.BITS_EPOCH)) or {})


@pytest.mark.parametrize("old", [C.BITS_EPOCH - 1, None])
def test_a_certificate_of_another_epoch_or_none_is_refused_by_name_and_its_keys_autotune(store, old):
    assert _served(store, _doc(old)) is None, "a certificate of an older bits epoch was served after the upgrade"
    reason = C._REFUSED[str(store)]
    assert f"bits epoch {'none' if old is None else old}" in reason and f"bits epoch {C.BITS_EPOCH}" in reason


def test_the_placement_rule_stamps_the_epoch_and_drops_what_an_older_epoch_left(store):
    entries = copy.deepcopy(_doc(None)["entries"])
    entries["(128, 128, 128, 'fp32')"] = entries[KEY]
    fresh = copy.deepcopy(_doc(None)["entries"][KEY])
    C.file_certification(entries, KEY, fresh)
    assert list(entries) == [KEY], "an entry of an older bits epoch stayed in the file the certifier writes"
    assert entries[KEY]["proof"]["bits_epoch"] == C.BITS_EPOCH
    doc = {**_doc(None), "format": C.format_for(entries), "entries": entries}
    assert C.validate(doc, store) == []
