"""The certification harness's sweep (ops/autotune_harness.LaunchUnderTest.sweep) and the compile bound keep their
three answers apart, and leave nothing behind (qa C16-1, C12-1, C6-1 on library C 7ac63c81, 23/11/2026: the sweep
wrapped precompile in "except Exception", so a dead compile worker, a timeout and a compiler refusal ALL became
"unrun", which the engine's certifier writes into the certificate as could_not_run: a dead worker certified around).

What this test does if the code is wrong:
1. a dying compile worker in the sweep is counted as a candidate that could not run instead of raised;
2. a timeout and a compiler refusal are not told apart in the sweep's record (their `kind`);
3. a device file in force that states no compile bound, or a profile read error, is read as "no bound";
4. a compile spec directory or an element-wise proof snapshot is left in TMPDIR after its run, on success or failure.
Device-free.
Run: python -m pytest tests/test_a_certificate_never_holds_a_dead_compile_worker.py
"""
import os
from types import SimpleNamespace

import pytest

from nbxtensor.backend import compile_bound as CB
from nbxtensor.ops import autotune_harness as H
from nbxtensor.ops import ew_proof_child


class _Cfg:
    def __init__(self, n):
        self.n = n

    def all_kwargs(self):
        return {"BLOCK": self.n}


def _sweep(monkeypatch, outcome):
    """The harness's sweep of three candidates, the middle one's precompile answering `outcome` (an exception)."""
    run = object.__new__(H.LaunchUnderTest)
    run.tuner, run.args, run.kwargs, run.out = SimpleNamespace(fn=None), (), {}, None
    run.poison = lambda: None
    run.launch = lambda cfg: None
    monkeypatch.setattr(H, "_sync", lambda: None)
    monkeypatch.setattr(H.F, "deviation_against", lambda out, oracle: 0.0)
    monkeypatch.setattr(H, "config_dict", lambda cfg: {"BLOCK": cfg.n})

    def pre(fn, args, kwargs):
        if kwargs["BLOCK"] == 2:
            raise outcome
    monkeypatch.setattr(CB, "precompile", pre)
    return run.sweep([_Cfg(1), _Cfg(2), _Cfg(3)], oracle=None, tolerance=1.0)


def test_a_dead_worker_is_raised_and_the_other_two_keep_their_kind(monkeypatch):
    with pytest.raises(CB.CompileWorkerDied):
        _sweep(monkeypatch, CB.CompileWorkerDied("k: the compile worker died (exit 1)"))
    results, excluded, unrun = _sweep(monkeypatch, CB.CompileUnbounded("k: did not end within 120 s"))
    assert len(results) == 2 and [u["kind"] for u in unrun] == ["compile did not end"]
    results, excluded, unrun = _sweep(monkeypatch, CB.CompileRefused("k: Triton's compiler refused"))
    assert [u["kind"] for u in unrun] == ["compiler refused"]
    assert unrun[0]["error"].startswith("compiler refused: ")


def test_an_undeclared_bound_and_a_read_error_are_errors(monkeypatch):
    monkeypatch.setattr("nbxtensor.ops._configs.active_vendor_profile", lambda: {"autotune": {}})
    with pytest.raises(CB.CompileBoundUndeclared, match="states no autotune.compile_bound_s"):
        CB.bound_s()

    def broken():
        raise OSError("the device file cannot be read")
    monkeypatch.setattr("nbxtensor.ops._configs.active_vendor_profile", broken)
    with pytest.raises(OSError, match="cannot be read"):
        CB.bound_s()


def test_every_device_file_states_its_compile_bound():
    import pathlib
    import yaml
    import nbxtensor
    files = list((pathlib.Path(nbxtensor.__file__).parent / "backend" / "devices").glob("*/*.yml"))
    assert files
    for f in files:
        assert "compile_bound_s" in (yaml.safe_load(f.read_text()).get("autotune") or {}), f


class _K:
    __name__ = "k"
    params = ()
    fn = type("F", (), {"__module__": "nbxtensor.ops.moe_route"})


def _left(tmp):
    return sorted(n for n in os.listdir(tmp) if n.startswith("nbx_"))


@pytest.mark.parametrize("code", [0, ew_proof_child.TIMED_OUT, 1])
def test_no_compile_spec_directory_is_left_behind(tmp_path, monkeypatch, code):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    monkeypatch.setattr(CB, "bound_s", lambda: 120.0)
    monkeypatch.setattr(CB, "_named", lambda k: True)
    monkeypatch.setattr(ew_proof_child._WORKER, "ask", lambda job, timeout=None: (code, "text"))
    try:
        CB.precompile(_K(), (), {"BLOCK": 1})
    except (CB.CompileUnbounded, CB.CompileWorkerDied):
        pass
    assert _left(tmp_path) == []


def test_no_proof_snapshot_is_left_behind_when_the_proof_fails(tmp_path, monkeypatch):
    import tempfile
    from nbxtensor.ops import ew_proof
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    monkeypatch.setattr("nbxtensor.ops.elementwise_launch.class_candidates", lambda: [(64, 2, 3)])
    monkeypatch.setattr(ew_proof, "at_tile", lambda bound, block: bound)
    monkeypatch.setattr(ew_proof_child, "snapshot_launch", lambda *a: (tempfile.mkdtemp(prefix="nbx_ew_proof_"), 8))

    def dies(spec, cands):
        raise RuntimeError("the proof's worker died")
    monkeypatch.setattr(ew_proof_child, "run_candidates", dies)
    with pytest.raises(RuntimeError, match="worker died"):
        ew_proof.prove(_K(), lambda meta: (1,), {}, {}, {}, "vid")
    assert _left(tmp_path) == []
