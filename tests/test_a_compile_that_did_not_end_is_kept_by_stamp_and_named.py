"""A candidate whose compile did not end within the profile's bound is KEPT, under the proofs' stamp (device class,
compiler identity, assembler build), beside the screen's other exclusions, and a later sweep skips it by name; and the
line names the candidate (C18, the Dell 23/11: the timeout line stopped after its colon, and the MoE prefill file paid
nine true 120 s timeouts again at every fresh sweep).

What this test does if the code is wrong:
1. the timeout line does not name the candidate's settings;
2. a recorded timeout is not found again under its stamp, or is found under another compiler's stamp;
3. the screen does not consult the record before compiling, or does not say it skips.
Device-free.
Run: python -m pytest tests/test_a_compile_that_did_not_end_is_kept_by_stamp_and_named.py
"""
import inspect

import pytest

from nbxtensor.backend import compile_bound as CB
from nbxtensor.ops import autotune_cache as AC
from nbxtensor.ops import ew_proof_child


class _Param:
    def __init__(self, name, constexpr):
        self.name, self.is_constexpr = name, constexpr


class _Kernel:
    __name__ = "moe_route_kernel"
    params = (_Param("x_ptr", False), _Param("T", False), _Param("BLOCK_H", True))
    fn = type("F", (), {"__module__": "nbxtensor.ops.moe_route"})


def test_the_timeout_line_names_the_candidate(monkeypatch):
    monkeypatch.setattr(CB, "bound_s", lambda: 120.0)
    monkeypatch.setattr(CB, "_named", lambda k: True)
    monkeypatch.setattr(ew_proof_child._WORKER, "ask",
                        lambda job, timeout=None: (ew_proof_child.TIMED_OUT, "timed out after 120 s"))
    with pytest.raises(CB.CompileUnbounded, match="moe_route_kernel BLOCK_H=64 num_warps=2 num_stages=1: the "
                                                  "candidate's compile did not end within 120 s"):
        CB.precompile(_Kernel(), (), {"T": 2, "BLOCK_H": 64, "num_warps": 2, "num_stages": 1})


def test_a_timeout_is_kept_by_stamp(tmp_path, monkeypatch):
    monkeypatch.setattr(AC, "_exclusions_path", lambda: str(tmp_path / "autotune_exclusions_x.json"))
    monkeypatch.setattr(AC, "_dir", lambda: str(tmp_path))
    assert not AC.compile_timed_out("moe_route_kernel", (8, 64, 2), "BLOCK_H: 64, num_warps: 2", "stampA")
    assert AC.record_compile_timeout("moe_route_kernel", (8, 64, 2), "BLOCK_H: 64, num_warps: 2", "stampA") == 1
    assert AC.compile_timed_out("moe_route_kernel", (8, 64, 2), "BLOCK_H: 64, num_warps: 2", "stampA")
    assert not AC.compile_timed_out("moe_route_kernel", (8, 64, 2), "BLOCK_H: 64, num_warps: 2", "stampB")
    assert AC.record_compile_timeout("moe_route_kernel", (8, 64, 2), "BLOCK_H: 64, num_warps: 2", "stampA") == 0


def test_the_screen_skips_a_recorded_timeout_before_compiling_and_says_so():
    from nbxtensor.backend import launcher
    src = inspect.getsource(launcher._screen_configs)
    skip, compile_ = src.index("compile_timed_out("), src.index("precompile(tuner.fn")
    assert skip < compile_ and "skipped — its compile was recorded as not ending" in src
    assert "record_compile_timeout(" in src
