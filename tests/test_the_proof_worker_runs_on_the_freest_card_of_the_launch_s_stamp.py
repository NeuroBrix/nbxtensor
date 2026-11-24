"""The element-wise proof's worker runs on the FREEST visible card of the launch's stamp, card-less.

Register row 444 (the Dell pair, 24/11/2026): deepseek-moe-16b-chat, a Triton `run` on two V100 32 GB
(CUDA_VISIBLE_DEVICES=2,3). The plan put the whole model on card 2 (32 169 MiB used of 32 510); card 3 held 709 MiB.
The proof of variant 863fce847ea79e39 (mul_scalar_strided_nd_kernel) could not run on card 2: "cuLaunchKernel failed
(2: out of memory)" — its worker inherited the launch's environment and opened its context on the process's first
visible card, the full one. The worker must start on card 3.

The device facts (count, current card, free bytes, compute capability) are patched, the run's cards declared (the
cure of qa 24/11 is tested in test_the_proof_worker_runs_only_on_the_run_s_declared_cards_read_by_nvml.py); no card
is touched, and the worker process is never started (Popen is replaced by a recorder of its environment)."""
import os

import pytest

from nbxtensor.backend import runtime as _rt
from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.ops import ew_proof, ew_proof_child

MIB = 1024 * 1024
TOTAL = 32510 * MIB


def _cards(monkeypatch, used_mib, arches, own=0, backend="cuda", visible=("CUDA_VISIBLE_DEVICES",), declared="all"):
    """Patch the library's device facts: one card per entry of `used_mib`, its compute capability in `arches`; the
    free memory is NVML's reading (`_card_free_bytes`, patched), the runtime's per-card reading refused (no context
    is opened to ask); the run declares `declared` (default: every card)."""
    monkeypatch.setattr(_rt, "gpu_backend", lambda: backend)
    monkeypatch.setattr(_rt, "backend_table", lambda: {"visible_env": visible})
    monkeypatch.setattr(DeviceAllocator, "device_count", staticmethod(lambda: len(used_mib)))
    monkeypatch.setattr(DeviceAllocator, "get_device", staticmethod(lambda: own))

    def _no_context(*_a, **_k):
        """A per-card runtime reading would select the card and open a context on it: refused."""
        raise AssertionError("a context was opened on a card to read its free memory")

    monkeypatch.setattr(DeviceAllocator, "mem_get_info", classmethod(_no_context))
    monkeypatch.setattr(DeviceAllocator, "set_device", staticmethod(_no_context))
    monkeypatch.setattr(ew_proof_child, "_card_free_bytes",
                        lambda cards: {i: TOTAL - used_mib[i] * MIB for i in cards}, raising=False)
    monkeypatch.setattr(ew_proof_child, "_card_arch", lambda i: arches[i], raising=False)
    monkeypatch.setattr(ew_proof_child, "_metal", lambda: False)
    monkeypatch.setattr(ew_proof_child, "_DECLARED_CARDS",
                        tuple(range(len(used_mib))) if declared == "all" else declared, raising=False)


def _started_env(monkeypatch):
    """Start a worker with Popen recorded: the environment it was given."""
    seen = {}

    class _Popen:
        """A stand-in for the worker process: nothing is started."""

        def __init__(self, argv, env=None, **_kw):
            """Record the environment the worker would be started with."""
            seen["env"] = env

    monkeypatch.setattr(ew_proof_child.subprocess, "Popen", _Popen)
    monkeypatch.setattr(ew_proof_child._Worker, "started", 0)
    ew_proof_child._Worker()._start()
    return seen["env"]


def test_row_444_the_worker_starts_on_the_empty_card_never_on_the_launch_s_full_one(monkeypatch):
    """Register row 444: card 2 (the launch's own, ordinal 0) at 32 169 MiB used, card 3 at 709 MiB, both sm_70 —
    the worker's environment names card 3 alone. Red on eaf453205d83: the worker inherited "2,3" and opened on 2."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    _cards(monkeypatch, [32169, 709], [70, 70], own=0)
    env = _started_env(monkeypatch)
    assert env["CUDA_VISIBLE_DEVICES"] == "3", f"the worker opens on {env['CUDA_VISIBLE_DEVICES']!r}, not card 3"
    assert env["CUDA_LAUNCH_BLOCKING"] == "1"
    assert ew_proof_child.proof_device() == (1, (32510 - 709) * MIB)


def test_the_freest_card_of_the_stamp_wins_the_launch_s_own_included(monkeypatch):
    """Four cards of one class, none spare: the freest wins, the launch's own when it is the freest."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _cards(monkeypatch, [20000, 31000, 25000, 30000], [70, 70, 70, 70], own=0)
    assert ew_proof_child.proof_device()[0] == 0
    assert _started_env(monkeypatch)["CUDA_VISIBLE_DEVICES"] == "0"
    _cards(monkeypatch, [20000, 31000, 15000, 30000], [70, 70, 70, 70], own=1)
    assert ew_proof_child.proof_device()[0] == 2


def test_a_card_of_another_stamp_is_never_chosen(monkeypatch):
    """The freest card compiles another binary (sm_80 beside an sm_70 launch): the freest of the launch's own
    capability is chosen; with no other card of it, the launch's own."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    _cards(monkeypatch, [30000, 100, 20000], [70, 80, 70], own=0)
    assert ew_proof_child.proof_device()[0] == 2
    assert _started_env(monkeypatch)["CUDA_VISIBLE_DEVICES"] == "2"
    _cards(monkeypatch, [30000, 100], [70, 80], own=0)
    assert ew_proof_child.proof_device()[0] == 0


def test_with_one_device_the_choice_is_that_device_and_the_environment_is_as_it_was(monkeypatch):
    """One visible device (a Mac's one Metal device, a one-card host): that device, the environment unchanged."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    _cards(monkeypatch, [1000], [70], own=0, backend="metal", visible=())
    assert ew_proof_child.proof_device() == (0, -1)
    env = _started_env(monkeypatch)
    assert env["CUDA_VISIBLE_DEVICES"] == "5"
    assert {k: v for k, v in env.items() if k != "CUDA_LAUNCH_BLOCKING"} == {
        k: v for k, v in os.environ.items() if k != "CUDA_LAUNCH_BLOCKING"}


def test_the_worker_environment_names_the_card_in_the_process_s_own_numbering(monkeypatch):
    """The visibility variable narrowed to the chosen entry (a board number or a UUID as listed); unset, the ordinal;
    an ordinal past the listed entries refused by name; a backend with no such variable unchanged."""
    monkeypatch.setattr(_rt, "backend_table", lambda: {"visible_env": ("CUDA_VISIBLE_DEVICES",)})
    assert ew_proof_child.worker_env(1, {"CUDA_VISIBLE_DEVICES": "GPU-aa, GPU-bb"})["CUDA_VISIBLE_DEVICES"] == "GPU-bb"
    assert ew_proof_child.worker_env(3, {})["CUDA_VISIBLE_DEVICES"] == "3"
    with pytest.raises(ew_proof_child.ProofCouldNotRun, match="not among the 2 card"):
        ew_proof_child.worker_env(2, {"CUDA_VISIBLE_DEVICES": "2,3"})
    monkeypatch.setattr(_rt, "backend_table", lambda: {"visible_env": ()})
    assert ew_proof_child.worker_env(1, {"X": "1"}) == {"X": "1"}


def test_a_proof_that_could_not_run_says_the_remedy_and_the_variant_s_signature(monkeypatch):
    """The refusal names the remedy: room on a card of the class (the card the worker had and its free memory), or
    the variant's line in the shipped table, by its signature as the record line prints it."""
    monkeypatch.setattr(ew_proof, "stamp", lambda: {"vendor": "nvidia", "profile": "volta", "generator": "g",
                                                    "assembler": "a", "method": "m"})
    monkeypatch.setattr(ew_proof._Table, "_shipped_path", staticmethod(lambda: "/certified/nvidia/volta/t.json"))
    monkeypatch.setattr(ew_proof_child._WORKER, "device", (1, 31801 * MIB))
    vid = "a variant id"
    text = ew_proof.remedy("mul_scalar_strided_nd_kernel", vid)
    assert text.startswith("REMEDY: free memory on a card of this class (nvidia/volta")
    assert "visible card 1, the freest of the run's declared cards of the class, 31801 MiB free" in text
    assert f"mul_scalar_strided_nd_kernel variant {ew_proof.stamp_key({'v': vid})}" in text
    assert "/certified/nvidia/volta/t.json" in text and "python -m nbxtensor.ops.ew_proof export" in text
