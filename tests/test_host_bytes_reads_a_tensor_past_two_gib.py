"""`interop.host_bytes` (the engine's PyTorch-side hand-off reads a tensor through it, kernels/torch_interop.py) read
the bytes with `ctypes.string_at`, whose size is a C int: a tensor of 2 GiB or more failed with "Negative size passed
to PyBytes_FromStringAndSize" (the same fault `NBXTensor.numpy` recorded on 2026-09-07; orchestrator's line-3 scan,
library C, 23/11/2026).

What this test does if the code is wrong:
1. the read goes through `ctypes.string_at` (patched to refuse here), so a small tensor already fails: the path, proved
   on any machine;
2. a host tensor of 2 GiB + 8 bytes is read back whole, its first and last bytes as written: the size itself, run
   where the host has the room (about 4.3 GB at the peak: the tensor and its copy).

Run: python -m pytest tests/test_host_bytes_reads_a_tensor_past_two_gib.py
"""
import ctypes

import numpy as np
import pytest

from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.interop import handoff
from nbxtensor.tensor import NBXTensor


def test_host_bytes_never_goes_through_string_at(monkeypatch):
    def refused(*a, **k):
        raise AssertionError("host_bytes read through ctypes.string_at, whose size is a C int")
    monkeypatch.setattr(handoff.ctypes, "string_at", refused)
    t = NBXTensor.empty_cpu((3, 4), NBXDtype.float32)
    src = np.arange(12, dtype=np.float32)
    ctypes.memmove(t.data_ptr(), src.ctypes.data, src.nbytes)
    got = np.frombuffer(handoff.host_bytes(t), dtype=np.float32).reshape(3, 4)
    assert np.array_equal(got, np.arange(12, dtype=np.float32).reshape(3, 4))


def test_a_tensor_of_two_gib_and_more_is_read_whole():
    n = (1 << 31) + 8
    t = NBXTensor.empty_cpu((n,), NBXDtype.uint8)
    assert t.is_cpu
    ctypes.memset(t.data_ptr(), 7, 8)
    ctypes.memset(t.data_ptr() + n - 8, 9, 8)
    out = handoff.host_bytes(t)
    assert len(out) == n
    assert out[:8] == b"\x07" * 8 and out[-8:] == b"\x09" * 8
