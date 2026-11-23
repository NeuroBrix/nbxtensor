"""Both MoE paths move a tensor onto the device that runs its part through ONE function (`moe_multi_device.on_device`,
the `needs_move` rule): a host tensor always moves, whatever device index it carries (zero3 offload keeps host weights
whose `_device_idx` is 0). The fused path had its own copy (`moe_fused._xfer`) that compared the index alone and
returned such a host tensor unmoved onto device 0 — a host address handed to a kernel (orchestrator's line-3 scan,
library C, 23/11/2026; red line 3: one function per memory job).

What this test does if the code is wrong:
1. the fused path's transfer, given a host tensor whose index equals the target device, returns it without a
   transfer;
2. the fused module defines a transfer of its own beside the multi-device one.
No device is needed: the transfer itself is replaced by a recorder.

Run: python -m pytest tests/test_a_host_tensor_crossing_to_a_moe_device_is_always_moved.py
"""
import ast
import inspect

from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.ops import moe_fused, moe_multi_device
from nbxtensor.tensor import NBXTensor


def _fused_transfer():
    """The function the fused path moves a tensor with (its own `_xfer` before library C, the shared one after)."""
    return getattr(moe_fused, "on_device", None) or getattr(moe_fused, "_xfer")


def test_a_host_tensor_whose_index_is_the_target_is_moved(monkeypatch):
    moved = []
    record = lambda t, dev: (moved.append((t, dev)), "moved")[1]  # noqa: E731
    monkeypatch.setattr(moe_multi_device, "transfer_tensor", record)
    monkeypatch.setattr(moe_fused, "transfer_tensor", record, raising=False)
    host = NBXTensor.empty_cpu((4,), NBXDtype.float32)
    host._device_idx = 0
    assert host.is_cpu
    assert _fused_transfer()(host, 0) == "moved", "a host tensor with index 0 was handed to device 0 unmoved"
    assert moved == [(host, 0)]


def test_the_fused_path_keeps_no_transfer_of_its_own():
    tree = ast.parse(inspect.getsource(moe_fused))
    own = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_xfer", "_on", "on_device")]
    assert own == [], f"moe_fused defines its own transfer {own}: one function per memory job (red line 3)"
    assert moe_fused.on_device is moe_multi_device.on_device
