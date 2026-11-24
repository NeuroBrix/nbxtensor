"""The element-wise proof's worker runs only on the cards the RUN WAS GIVEN, their free memory read from NVML, card-less.

qa 24/11/2026 on 45f30438 (major): the worker's card was chosen among every VISIBLE card, each card's free memory asked
with `set_device` + `mem_get_info` — a parent CUDA context opened on every visible card (about 300 MB left on each),
and on a shared machine a visible card may be someone else's. The cure:
  * the candidates are the cards the application declared for the run (`ew_proof_child.declare_proof_cards`), the
    launch's own card among them; a run that declared none has the launch's own card alone;
  * the free memory is NVML's (`nvmlDeviceGetHandleByPciBusId_v2`, `nvmlDeviceGetMemoryInfo`), each card found by its
    PCI bus id (`cuDeviceGet`, `cuDeviceGetPCIBusId`, device queries): no context is opened to ask;
  * without NVML, the launch's own card alone, said once by name.

The CUDA driver and NVML are fakes whose context calls fail; the runtime's per-card reading (`set_device`,
`mem_get_info`) fails too. No card is touched and no worker is started (Popen records its environment)."""
import ctypes

import pytest

from nbxtensor.backend import process_memory
from nbxtensor.backend import runtime as _rt
from nbxtensor.backend.launcher import CudaDriver
from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.ops import ew_proof_child

MIB = 1024 * 1024
TOTAL = 32510 * MIB
#: the boards of a four-card host, NVML's numbering: PCI bus id and MiB used
BOARDS = {0: ("0000:1A:00.0", 31000), 1: ("0000:1C:00.0", 30000), 2: ("0000:3B:00.0", 32169), 3: ("0000:3D:00.0", 709)}


class _FakeCuda:
    """The CUDA driver's device queries for the visible boards `visible` (CUDA_VISIBLE_DEVICES, in order); every
    context call fails the test."""

    def __init__(self, visible):
        """Visible ordinal i is board visible[i]."""
        self.visible = list(visible)
        self.asked = []

    def cuDeviceGet(self, ref, ordinal):
        """CUdevice of a visible ordinal: the ordinal itself, as the driver numbers them."""
        ref._obj.value = int(ordinal.value)
        return 0

    def cuDeviceGetPCIBusId(self, buf, _n, dev):
        """The PCI bus id of the board behind visible device `dev`."""
        self.asked.append(int(dev.value))
        buf.value = BOARDS[self.visible[int(dev.value)]][0].encode()
        return 0

    def cuDeviceGetAttribute(self, ref, attr, _dev):
        """Every board is sm_70 (the major 7 for attribute 75, the minor 0 for 76)."""
        ref._obj.value = {CudaDriver.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: 7,
                          CudaDriver.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: 0}[attr]
        return 0

    def __getattr__(self, name):
        """A context call (cuCtx*, cuDevicePrimaryCtx*) fails the test; any other call is not expected."""
        if name.startswith(("cuCtx", "cuDevicePrimaryCtx")):
            def _refused(*_a):
                """A context would be opened or made current: refused."""
                raise AssertionError(f"{name}: a context was asked of the driver to read a card's free memory")
            return _refused
        raise AttributeError(name)


class _FakeNvml:
    """NVML over BOARDS: a handle per PCI bus id (the board's index), its memory info."""

    def __init__(self):
        """No handle asked yet."""
        self.asked = []

    def nvmlDeviceGetHandleByPciBusId_v2(self, bus, ref):
        """The handle of the board at `bus` (its NVML index + 1, a non-null pointer)."""
        board = next(b for b, (pci, _u) in BOARDS.items() if pci.encode() == bus.value)
        self.asked.append(bus.value.decode())
        ref._obj.value = board + 1
        return 0

    def nvmlDeviceGetMemoryInfo(self, handle, ref):
        """The board's total, free and used bytes."""
        used = BOARDS[int(handle.value) - 1][1] * MIB
        ref._obj.total, ref._obj.free, ref._obj.used = TOTAL, TOTAL - used, used
        return 0


@pytest.fixture
def host(monkeypatch):
    """A CUDA process under CUDA_VISIBLE_DEVICES=2,3,0,1 (visible 0..3 = boards 2, 3, 0, 1), launched on visible 0
    (board 2, the full one); the driver and NVML faked, the runtime's per-card reading refused, no card declared."""
    visible = [2, 3, 0, 1]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(map(str, visible)))
    monkeypatch.setattr(_rt, "gpu_backend", lambda: "cuda")
    monkeypatch.setattr(_rt, "backend_table", lambda: {"visible_env": ("CUDA_VISIBLE_DEVICES",),
                                                       "management_libs": ["libnvidia-ml.so.1"]})
    monkeypatch.setattr(DeviceAllocator, "device_count", staticmethod(lambda: len(visible)))
    monkeypatch.setattr(DeviceAllocator, "get_device", staticmethod(lambda: 0))

    def _no_context(*_a, **_k):
        """The runtime's per-card reading selects the card and opens a context on it: refused."""
        raise AssertionError("set_device/mem_get_info: a context was opened on a card to read its free memory")

    monkeypatch.setattr(DeviceAllocator, "set_device", staticmethod(_no_context))
    monkeypatch.setattr(DeviceAllocator, "mem_get_info", classmethod(_no_context))
    drv = CudaDriver.__new__(CudaDriver)
    drv.lib = _FakeCuda(visible)
    monkeypatch.setattr(CudaDriver, "instance", classmethod(lambda cls: drv))
    nvml = _FakeNvml()
    monkeypatch.setattr(process_memory, "_nvml", lambda libs: nvml if libs else None)
    monkeypatch.setattr(ew_proof_child, "_metal", lambda: False)
    monkeypatch.setattr(ew_proof_child, "_DECLARED_CARDS", None, raising=False)
    monkeypatch.setattr(ew_proof_child, "_SAID", set(), raising=False)
    return drv, nvml


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


def test_the_candidates_are_only_the_cards_the_run_declared(host, monkeypatch):
    """Undeclared, the launch's own card alone (board 2), though board 3 beside it is nearly empty — it may be
    someone else's; the run declares visible 0 and 2 (boards 2 and 0): board 0 is chosen, never the undeclared
    emptier board 3. Red on 45f30438: every visible card was a candidate."""
    assert ew_proof_child.proof_device() == (0, -1), "an undeclared run reached past the launch's own card"
    assert _started_env(monkeypatch)["CUDA_VISIBLE_DEVICES"] == "2"
    ew_proof_child.declare_proof_cards([2])
    assert ew_proof_child.declared_proof_cards() == (2,)
    assert ew_proof_child.proof_device() == (2, (32510 - 31000) * MIB)
    assert _started_env(monkeypatch)["CUDA_VISIBLE_DEVICES"] == "0"
    ew_proof_child.declare_proof_cards([0, 1])
    assert ew_proof_child.proof_device() == (1, (32510 - 709) * MIB)
    with pytest.raises(ValueError, match="0 or more"):
        ew_proof_child.declare_proof_cards([-1])
    ew_proof_child.declare_proof_cards([7])
    with pytest.raises(ew_proof_child.ProofCouldNotRun, match=r"declared card\(s\) \[7\].*sees 4 card"):
        ew_proof_child.proof_device()
    ew_proof_child.declare_proof_cards(None)
    assert ew_proof_child.proof_device() == (0, -1)


def test_the_free_memory_is_nvml_s_and_no_context_is_opened_to_ask(host, monkeypatch):
    """Register row 444 with the spare card declared: every context call of the driver and the runtime's per-card
    reading fail the test; the choice is board 3 by NVML's free bytes. Red on 45f30438: mem_get_info per card."""
    drv, nvml = host
    ew_proof_child.declare_proof_cards([0, 1, 2, 3])
    assert ew_proof_child.proof_device() == (1, (32510 - 709) * MIB)
    assert sorted(drv.lib.asked) == [0, 1, 2, 3]
    assert sorted(nvml.asked) == sorted(pci for pci, _u in BOARDS.values())
    assert _started_env(monkeypatch)["CUDA_VISIBLE_DEVICES"] == "3"


def test_a_visible_ordinal_reaches_nvml_by_its_pci_bus_id(host):
    """Visible 0..3 are boards 2, 3, 0, 1: NVML numbers the boards, so each card is found by its bus id, never by
    the ordinal — the free bytes come back keyed by the process's ordinals."""
    drv, _nvml = host
    assert drv.pci_bus_id_of(1) == "0000:3D:00.0" and drv.pci_bus_id_of(2) == "0000:1A:00.0"
    free = ew_proof_child._card_free_bytes([0, 1, 2, 3])
    assert free == {i: TOTAL - BOARDS[b][1] * MIB for i, b in enumerate([2, 3, 0, 1])}


def test_without_nvml_the_launch_s_card_alone_said_once_by_name(host, monkeypatch, capsys):
    """No management library opens: the launch's own card alone, the environment narrowed to it, said once with the
    library's names; never a context to ask. Red on 45f30438: mem_get_info per card."""
    monkeypatch.setattr(process_memory, "_nvml", lambda libs: None)
    ew_proof_child.declare_proof_cards([0, 1, 2, 3])
    assert ew_proof_child.proof_device() == (0, -1)
    assert _started_env(monkeypatch)["CUDA_VISIBLE_DEVICES"] == "2"
    err = capsys.readouterr().err
    assert err.count("[ew-proof] ") == 1 and "NVML" in err and "libnvidia-ml.so.1" in err
    assert "launch's own card 0 alone" in err
    assert ew_proof_child.proof_device() == (0, -1)
    assert capsys.readouterr().err == ""
