"""A sweep candidate is compiled under the profile's bound (`backend/compile_bound.py`): one whose compile does not end
in time is excluded with its text, never a hang (the Dell V100 23/11: three of moe_route's twelve candidates never
returned from Triton's make_llir; the run sat 30 minutes). Device-free: the worker is replaced by a fake that times out.
Red on the base: `compile_bound` does not exist and the screen calls no precompile."""
import pytest

from nbxtensor.backend import compile_bound as CB


class _Fn:
    __name__ = "k"
    fn = type("F", (), {"__module__": "m"})


def test_a_candidate_whose_compile_does_not_end_is_refused_by_name(monkeypatch):
    from nbxtensor.ops import ew_proof_child
    monkeypatch.setattr(CB, "bound_s", lambda: 2.0)
    monkeypatch.setattr(ew_proof_child._WORKER, "ask", lambda job, timeout=None: (-1, f"timed out after {timeout:g} s"))
    with pytest.raises(CB.CompileUnbounded, match="did not end within 2 s"):
        CB.precompile(_Fn(), (), {"BLOCK_H": 64, "num_warps": 2})


def test_a_compile_that_ends_passes_and_no_bound_means_in_process(monkeypatch):
    from nbxtensor.ops import ew_proof_child
    seen = []
    monkeypatch.setattr(CB, "bound_s", lambda: 2.0)
    monkeypatch.setattr(ew_proof_child._WORKER, "ask", lambda job, timeout=None: seen.append(timeout) or (0, ""))
    CB.precompile(_Fn(), (), {"BLOCK_H": 16})
    assert seen == [2.0]
    monkeypatch.setattr(CB, "bound_s", lambda: None)
    CB.precompile(_Fn(), (), {"BLOCK_H": 16})
    assert seen == [2.0]


def test_the_volta_profile_declares_its_bound():
    import pathlib
    import yaml
    import nbxtensor
    f = pathlib.Path(nbxtensor.__file__).parent / "backend" / "devices" / "nvidia" / "volta.yml"
    assert yaml.safe_load(f.read_text())["autotune"]["compile_bound_s"] > 0
