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


def _bind(monkeypatch, tmp_path):
    monkeypatch.setattr(A, "_dir", lambda: str(tmp_path))
    monkeypatch.setattr(A, "_arch_fingerprint", lambda: "arch0")
    monkeypatch.setattr(A, "_generator_tag", lambda: "gen0")
    monkeypatch.setattr(A, "_STORED", [None])


def test_an_artifact_under_the_old_name_with_no_epoch_is_not_read_and_is_named(tmp_path, monkeypatch, caplog):
    """What every machine holds today: `autotune_configs_<arch>-<gen>.json`, written before the epoch existed."""
    _bind(monkeypatch, tmp_path)
    old = tmp_path / "autotune_configs_arch0-gen0.json"
    old.write_text(json.dumps({"nbxtensor.ops.k.mm_kernel": {"(64, 64, 64, 'fp32')": {"kwargs": {"BLOCK_M": 32}}}}))
    with caplog.at_level("WARNING", logger=A.__name__):
        assert A._stored() == {}, "the upgraded library reads the replay artifact written before the bits epoch"
    said = [r.getMessage() for r in caplog.records if str(old) in r.getMessage()]
    assert len(said) == 1, "the replay artifact no longer read is not named once"
    assert "bits epoch none" in said[0] and f"bits epoch {C.BITS_EPOCH}" in said[0] and "sweep" in said[0]


def test_an_artifact_of_the_previous_epoch_is_named_once(tmp_path, monkeypatch, caplog):
    _bind(monkeypatch, tmp_path)
    old = tmp_path / f"autotune_configs_arch0-gen0-e{C.BITS_EPOCH - 1}.json"
    old.write_text("{}")
    with caplog.at_level("WARNING", logger=A.__name__):
        A._stored()
        A._STORED[0] = None
        A._stored()
    said = [r.getMessage() for r in caplog.records if str(old) in r.getMessage()]
    assert len(said) == 1 and f"bits epoch {C.BITS_EPOCH - 1}" in said[0]
