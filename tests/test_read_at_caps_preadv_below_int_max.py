"""Regression test for macOS preadv/pread EINVAL on tensors >= 2 GiB (stat_runs id=854).

On macOS (Darwin), `os.preadv(fd, [view], offset)` and `os.pread(fd, n, offset)`
fail with `OSError(22, 'Invalid argument')` whenever the requested byte count
exceeds `INT_MAX` (0x7FFFFFFF). On Linux, `preadv` silently caps at `MAX_RW_COUNT`
and returns a short read. `_read_at` must cap every single positioned read to
`_MAX_PREAD_BYTES` (< INT_MAX, page-aligned) so `read_bytes` loops cleanly across
all OSes.
"""
from __future__ import annotations

import errno
import mmap
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from nbxtensor.io import safetensors as sf


def test_max_pread_bytes_is_below_int_max_and_page_aligned():
    assert 0 < sf._MAX_PREAD_BYTES <= 0x7FFFFFFF
    assert sf._MAX_PREAD_BYTES % mmap.PAGESIZE == 0
    # Apple Silicon uses 16 KiB pages; keep the cap aligned there too.
    assert sf._MAX_PREAD_BYTES % 16384 == 0


def test_read_bytes_loops_under_cap_with_strict_preadv(tmp_path, monkeypatch):
    """Simulate macOS EINVAL above cap using a small cap so the test runs fast on any host."""
    cap = 4096
    payload = bytes((i * 31 + 7) & 0xFF for i in range(cap * 3 + 123))
    p = tmp_path / "weights.bin"
    p.write_bytes(payload)

    calls = []
    real_preadv = getattr(os, "preadv", None)

    def strict_preadv(fd, buffers, offset):
        total = sum(len(b) for b in buffers)
        calls.append((offset, total))
        if total > cap:
            raise OSError(errno.EINVAL, "Invalid argument")
        if real_preadv is not None:
            return real_preadv(fd, buffers, offset)
        data = os.pread(fd, total, offset)
        buffers[0][:len(data)] = data
        return len(data)

    monkeypatch.setattr(sf, "_MAX_PREAD_BYTES", cap)
    monkeypatch.setattr(os, "preadv", strict_preadv, raising=False)

    fd = os.open(str(p), os.O_RDONLY)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            got = sf.read_bytes(fd, 0, len(payload), (1, pool))
    finally:
        os.close(fd)

    assert np.array_equal(got, np.frombuffer(payload, dtype=np.uint8))
    assert len(calls) == 4
    assert all(n <= cap for _, n in calls)


def test_read_bytes_loops_under_cap_with_strict_pread_fallback(tmp_path, monkeypatch):
    """When os.preadv is absent, os.pread must also cap per-call length <= _MAX_PREAD_BYTES."""
    cap = 4096
    payload = bytes((i * 17 + 3) & 0xFF for i in range(cap * 2 + 55))
    p = tmp_path / "weights_pread.bin"
    p.write_bytes(payload)

    calls = []
    real_pread = os.pread

    def strict_pread(fd, n, offset):
        calls.append((offset, n))
        if n > cap:
            raise OSError(errno.EINVAL, "Invalid argument")
        return real_pread(fd, n, offset)

    monkeypatch.setattr(sf, "_MAX_PREAD_BYTES", cap)
    monkeypatch.delattr(os, "preadv", raising=False)
    monkeypatch.setattr(os, "pread", strict_pread)

    fd = os.open(str(p), os.O_RDONLY)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            got = sf.read_bytes(fd, 0, len(payload), (1, pool))
    finally:
        os.close(fd)

    assert np.array_equal(got, np.frombuffer(payload, dtype=np.uint8))
    assert len(calls) == 3
    assert all(n <= cap for _, n in calls)
