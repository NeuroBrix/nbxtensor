"""The machine's replay artifact (the configurations its own sweeps chose, seeded back on the next run) holds settings
proven against the bits of the library that wrote it, certified keys included. After an upgrade that raises the bits
epoch, the artifact the older library wrote must not seed them back: it is named by the epoch as well as by the arch
and the code generator, so the upgraded library reads none of it and its keys take the normal autotune path. Card-less:
the arch and the generator are stubbed, no kernel runs."""
import json

from nbxtensor.ops import autotune_cache as A
from nbxtensor.ops import autotune_certified as C


def test_an_artifact_written_at_the_previous_epoch_is_not_read_after_the_upgrade(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "_dir", lambda: str(tmp_path))
    monkeypatch.setattr(A, "_arch_fingerprint", lambda: "arch0")
    monkeypatch.setattr(A, "_generator_tag", lambda: "gen0")
    monkeypatch.setattr(A, "_STORED", [None])
    monkeypatch.setattr(C, "BITS_EPOCH", C.BITS_EPOCH - 1)
    old = A._artifact_path()
    with open(old, "w") as f:
        json.dump({"nbxtensor.ops.k.mm_kernel": {"(64, 64, 64, 'fp32')": {"kwargs": {"BLOCK_M": 32}}}}, f)
    assert A._stored(), "the stub artifact was not read at the epoch it was written in"
    monkeypatch.undo()
    monkeypatch.setattr(A, "_dir", lambda: str(tmp_path))
    monkeypatch.setattr(A, "_arch_fingerprint", lambda: "arch0")
    monkeypatch.setattr(A, "_generator_tag", lambda: "gen0")
    monkeypatch.setattr(A, "_STORED", [None])
    assert A._artifact_path() != old, "the replay artifact is not named by the bits epoch"
    assert A._stored() == {}, "the upgraded library reads the replay artifact the older epoch wrote"
