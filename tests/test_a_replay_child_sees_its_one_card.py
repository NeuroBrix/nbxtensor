"""The chunk probe's replay child is shown its ONE card (the first launch on the 4-card rack, 2026-11-19: 48 replay
children each started with every card visible).

`backend.install_help.one_device_env` narrows the variable the backend's runtime reads to hide devices (its table's
`visible_env`) to the parent's device; `trace_replay._child` runs the replay under it, on the child's device 0.
What each test would do if the code were wrong: the child's environment keeps the four cards (red), the parent's
index is passed where the parent already narrows (the wrong physical card: red), a device the parent does not see
gives an empty or wrong variable instead of a refusal (red)."""
import json
import subprocess

import pytest

from nbxtensor.backend import runtime as R
from nbxtensor.backend.install_help import one_device_env
from nbxtensor.memory import trace_replay as T

CUDA = {"visible_env": ("CUDA_VISIBLE_DEVICES",)}
HIP = {"visible_env": ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")}


def _table(monkeypatch, table):
    monkeypatch.setattr(R, "backend_table", lambda: table)
    monkeypatch.setattr(R, "gpu_backend", lambda: "test")


def test_nothing_hidden_the_child_sees_the_device_index(monkeypatch):
    _table(monkeypatch, CUDA)
    env = one_device_env(2, {})
    assert env["CUDA_VISIBLE_DEVICES"] == "2"


def test_a_parent_that_sees_some_cards_hands_down_the_physical_one(monkeypatch):
    _table(monkeypatch, CUDA)
    env = one_device_env(1, {"CUDA_VISIBLE_DEVICES": "2,3"})
    assert env["CUDA_VISIBLE_DEVICES"] == "3"


def test_the_innermost_variable_set_is_the_one_narrowed(monkeypatch):
    _table(monkeypatch, HIP)
    env = one_device_env(1, {"ROCR_VISIBLE_DEVICES": "0,2,3", "HIP_VISIBLE_DEVICES": "1,2"})
    assert env["ROCR_VISIBLE_DEVICES"] == "0,2,3" and env["HIP_VISIBLE_DEVICES"] == "2"


def test_a_device_the_parent_does_not_see_is_refused(monkeypatch):
    _table(monkeypatch, CUDA)
    with pytest.raises(RuntimeError, match="ZERO FALLBACK"):
        one_device_env(2, {"CUDA_VISIBLE_DEVICES": "0,1"})
    with pytest.raises(RuntimeError, match="ZERO FALLBACK"):
        one_device_env(0, {"CUDA_VISIBLE_DEVICES": ""})


def test_a_backend_that_names_no_variable_is_refused(monkeypatch):
    _table(monkeypatch, {})
    with pytest.raises(RuntimeError, match="hides devices"):
        one_device_env(0, {})


def test_the_replay_child_runs_on_its_one_card(monkeypatch):
    _table(monkeypatch, CUDA)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    seen = {}

    def run(argv, capture_output, text, env):
        seen["argv"], seen["env"] = argv, env
        return subprocess.CompletedProcess(argv, 0, stdout=T._MARK + json.dumps({"seconds": 1.0}), stderr="")
    monkeypatch.setattr(T.subprocess, "run", run)
    T._child("trace.tsv", 2, 1 << 21)
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert seen["argv"][-3:] == ["trace.tsv", "0", str(1 << 21)]
