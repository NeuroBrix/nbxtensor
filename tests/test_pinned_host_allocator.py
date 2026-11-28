"""Cardless unit tests for CachingHostAllocator and async DMA stream/event primitives.

100% CPU-only (no CUDA driver required): uses injected mock backend alloc/free
and stream/event/memcpy_async ops. Also enforces the zero-torch AST invariant
across `src/nbxtensor/`.
"""
from __future__ import annotations

import ast
import gc
import threading
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.memory import (
    CachingHostAllocator,
    NBXEvent,
    NBXStream,
    copy_async,
    empty_host_pinned_cache,
    get_pinned_host_allocator,
    host_pinned_pool_stats,
)
from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.memory.pinned_pool import set_pinned_host_allocator
from nbxtensor.tensor import NBXTensor


# ---------------------------------------------------------------------------
# Helpers — CPU-only deterministic mock backends
# ---------------------------------------------------------------------------

class FakeHostBackend:
    """Thread-safe mock for `cudaMallocHost` / `cudaFreeHost`."""

    def __init__(self, fail_on_call: int | None = None) -> None:
        self._lock = threading.Lock()
        self._next_ptr = 0x1000_0000
        self.live: Dict[int, int] = {}
        self.alloc_calls: List[int] = []
        self.free_calls: List[int] = []
        self._fail_on_call = fail_on_call
        self._failed_once = False

    def alloc(self, nbytes: int) -> int:
        with self._lock:
            call_idx = len(self.alloc_calls) + 1
            if self._fail_on_call is not None and call_idx == self._fail_on_call and not self._failed_once:
                self._failed_once = True
                raise RuntimeError(f"Simulated OOM on call {call_idx} for {nbytes} bytes")
            ptr = self._next_ptr
            self._next_ptr += max(nbytes, 4096) + 0x1000
            self.live[ptr] = nbytes
            self.alloc_calls.append(nbytes)
            return ptr

    def free(self, ptr: int) -> None:
        with self._lock:
            assert ptr in self.live, f"Double-free or unknown pointer {hex(ptr)}"
            del self.live[ptr]
            self.free_calls.append(ptr)


class FakeStreamOps:
    """Mock DeviceAllocator stream/event/memcpy_async operations for cardless tests."""

    def __init__(self) -> None:
        self._next_stream = 100
        self._next_event = 500
        self.current_device = 0
        self.log: List[Tuple] = []

    def get_device(self) -> int:
        return self.current_device

    def set_device(self, idx: int) -> None:
        self.current_device = idx
        self.log.append(("set_device", idx))

    def restore_device(self, idx: int) -> None:
        self.current_device = idx
        self.log.append(("restore_device", idx))

    def create_stream(self) -> int:
        h = self._next_stream
        self._next_stream += 1
        self.log.append(("create_stream", h, self.current_device))
        return h

    def destroy_stream(self, h: int) -> None:
        self.log.append(("destroy_stream", h))

    def stream_synchronize(self, h: int) -> None:
        self.log.append(("stream_synchronize", h))

    def create_event(self, timing: bool = True) -> int:
        h = self._next_event
        self._next_event += 1
        self.log.append(("create_event", h, timing))
        return h

    def destroy_event(self, h: int) -> None:
        self.log.append(("destroy_event", h))

    def record_event(self, ev: int, stream: int = 0) -> None:
        self.log.append(("record_event", ev, stream))

    def stream_wait_event(self, stream: int, ev: int, flags: int = 0) -> None:
        self.log.append(("stream_wait_event", stream, ev))

    def event_synchronize(self, ev: int) -> None:
        self.log.append(("event_synchronize", ev))

    def event_elapsed_ms(self, start: int, end: int) -> float:
        self.log.append(("event_elapsed_ms", start, end))
        return 1.25

    def memcpy_async(self, dst: int, src: int, nbytes: int, kind: int = 1, stream: int = 0) -> None:
        self.log.append(("memcpy_async", dst, src, nbytes, kind, stream))

    def memcpy(self, dst: int, src: int, nbytes: int, kind: int = 3) -> None:
        self.log.append(("memcpy", dst, src, nbytes, kind))


# ---------------------------------------------------------------------------
# 1. CachingHostAllocator tests
# ---------------------------------------------------------------------------

@pytest.mark.cardless
def test_bucket_rounding():
    backend = FakeHostBackend()
    pool = CachingHostAllocator(alloc_fn=backend.alloc, free_fn=backend.free)

    assert pool.bucket_size(0) == 0
    assert pool.bucket_size(1) == 4096
    assert pool.bucket_size(4096) == 4096
    assert pool.bucket_size(4097) == 8192
    assert pool.bucket_size(6000) == 8192
    assert pool.bucket_size(64 * 1024 * 1024) == 64 * 1024 * 1024
    # Above 64 MiB rounds to 4 MiB multiples rather than 2x power-of-two waste.
    assert pool.bucket_size(64 * 1024 * 1024 + 1) == 68 * 1024 * 1024
    with pytest.raises(ValueError):
        pool.bucket_size(-1)


@pytest.mark.cardless
def test_bucket_reuse_and_stats():
    backend = FakeHostBackend()
    pool = CachingHostAllocator(alloc_fn=backend.alloc, free_fn=backend.free)

    assert pool.alloc(0) == 0
    p1 = pool.alloc(1024)  # bucket 4096 -> miss
    st = pool.stats()
    assert st["cache_misses"] == 1
    assert st["cache_hits"] == 0
    assert st["allocated_bytes"] == 4096
    assert st["reserved_bytes"] == 4096
    assert st["cached_bytes"] == 0

    pool.free(p1, 1024)
    st = pool.stats()
    assert st["allocated_bytes"] == 0
    assert st["reserved_bytes"] == 4096
    assert st["cached_bytes"] == 4096
    assert len(backend.free_calls) == 0  # kept in pool

    # Different requested size in same 4096 bucket -> cache hit, identical pointer.
    p2 = pool.alloc(3000)
    assert p2 == p1
    st = pool.stats()
    assert st["cache_hits"] == 1
    assert st["cache_misses"] == 1
    assert st["allocated_bytes"] == 4096
    assert st["reserved_bytes"] == 4096
    assert st["cached_bytes"] == 0
    assert len(backend.alloc_calls) == 1

    # Freeing an already-freed / unknown pointer is a safe no-op.
    pool.free(p2, 3000)
    pool.free(p2, 3000)
    assert pool.stats()["cached_bytes"] == 4096


@pytest.mark.cardless
def test_max_pool_bytes_lru_eviction():
    backend = FakeHostBackend()
    pool = CachingHostAllocator(
        alloc_fn=backend.alloc,
        free_fn=backend.free,
        max_pool_bytes=8192,
    )

    p1 = pool.alloc(4096)
    p2 = pool.alloc(4096)
    p3 = pool.alloc(4096)
    assert pool.stats()["reserved_bytes"] == 12288

    # Park p1 and p2 (total cached = 8192 == cap).
    pool.free(p1, 4096)
    pool.free(p2, 4096)
    assert pool.stats()["cached_bytes"] == 8192
    assert backend.free_calls == []

    # Parking p3 pushes cached to 12288 > 8192 -> oldest (p1) is evicted.
    pool.free(p3, 4096)
    st = pool.stats()
    assert st["cached_bytes"] == 8192
    assert st["reserved_bytes"] == 8192
    assert st["evictions"] == 1
    assert backend.free_calls == [p1]

    # Allocation larger than max_pool_bytes is released directly on free without caching.
    big = pool.alloc(16384)
    pool.free(big, 16384)
    assert big in backend.free_calls
    assert pool.stats()["cached_bytes"] == 8192
    assert pool.stats()["evictions"] == 2

    # Shrinking max_pool_bytes evicts excess immediately.
    freed = pool.set_max_pool_bytes(4096)
    assert freed == 4096
    assert pool.stats()["cached_bytes"] == 4096
    assert p2 in backend.free_calls


@pytest.mark.cardless
def test_empty_cache_frees_all_cached_and_preserves_live():
    backend = FakeHostBackend()
    pool = CachingHostAllocator(alloc_fn=backend.alloc, free_fn=backend.free)

    p_live = pool.alloc(4096)
    p_cached_a = pool.alloc(4096)
    p_cached_b = pool.alloc(8192)
    pool.free(p_cached_a, 4096)
    pool.free(p_cached_b, 8192)

    freed = pool.empty_cache()
    assert freed == 4096 + 8192
    assert set(backend.free_calls) == {p_cached_a, p_cached_b}
    assert p_live in backend.live

    st = pool.stats()
    assert st["allocated_bytes"] == 4096
    assert st["reserved_bytes"] == 4096
    assert st["cached_bytes"] == 0

    # Second empty_cache is a no-op.
    assert pool.empty_cache() == 0

    pool.free(p_live, 4096)
    assert pool.empty_cache() == 4096
    assert backend.live == {}


@pytest.mark.cardless
def test_oom_flushes_cache_and_retries():
    # Fail on the 2nd backend alloc call; cache flush allows retry (3rd call) to succeed.
    backend = FakeHostBackend(fail_on_call=2)
    pool = CachingHostAllocator(alloc_fn=backend.alloc, free_fn=backend.free)

    p1 = pool.alloc(4096)
    pool.free(p1, 4096)
    assert pool.stats()["cached_bytes"] == 4096

    p2 = pool.alloc(8192)
    assert p2 in backend.live
    assert p1 in backend.free_calls
    assert pool.stats()["cached_bytes"] == 0
    assert pool.stats()["allocated_bytes"] == 8192


@pytest.mark.cardless
def test_thread_safety_concurrent_alloc_free():
    backend = FakeHostBackend()
    pool = CachingHostAllocator(
        alloc_fn=backend.alloc,
        free_fn=backend.free,
        max_pool_bytes=64 * 1024,
    )
    errors: List[BaseException] = []

    def worker(tid: int) -> None:
        try:
            sizes = [512, 4096, 6000, 12000, 16384]
            for i in range(80):
                sz = sizes[(tid + i) % len(sizes)]
                ptr = pool.alloc(sz)
                assert ptr != 0
                if i % 15 == 0:
                    pool.empty_cache()
                pool.free(ptr, sz)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent errors: {errors}"
    st = pool.stats()
    assert st["allocated_bytes"] == 0
    assert st["reserved_bytes"] == st["cached_bytes"]
    assert st["cached_bytes"] <= 64 * 1024
    assert st["cache_hits"] > 0

    pool.empty_cache()
    assert backend.live == {}
    assert pool.stats()["reserved_bytes"] == 0


# ---------------------------------------------------------------------------
# 2. NBXTensor pinned lifecycle integration (empty_cpu + finalization)
# ---------------------------------------------------------------------------

@pytest.mark.cardless
def test_tensor_empty_cpu_pinned_and_finalization_reuses_pool():
    backend = FakeHostBackend()
    pool = CachingHostAllocator(alloc_fn=backend.alloc, free_fn=backend.free)
    prev = set_pinned_host_allocator(pool)
    try:
        assert get_pinned_host_allocator() is pool

        t1 = NBXTensor.empty_cpu((1024,), NBXDtype.float32, pinned=True)
        assert t1.is_pinned
        ptr1 = t1.data_ptr()
        assert host_pinned_pool_stats()["cache_misses"] == 1
        assert DeviceAllocator.host_pinned_allocated() == 4096

        # Finalizing t1 returns its pinned buffer back into CachingHostAllocator.
        del t1
        gc.collect()
        assert DeviceAllocator.host_pinned_allocated() == 0
        assert host_pinned_pool_stats()["cached_bytes"] == 4096
        assert backend.free_calls == []

        # Re-allocating via pin_memory=True keyword reuses the exact pooled buffer.
        t2 = NBXTensor.empty_cpu((1024,), NBXDtype.float32, pin_memory=True)
        assert t2.is_pinned
        assert t2.data_ptr() == ptr1
        assert host_pinned_pool_stats()["cache_hits"] == 1

        del t2
        gc.collect()
        freed = empty_host_pinned_cache()
        assert freed == 4096
        assert backend.live == {}
    finally:
        set_pinned_host_allocator(prev)


# ---------------------------------------------------------------------------
# 3. NBXStream, NBXEvent, copy_async, and NBXTensor.to(non_blocking=True)
# ---------------------------------------------------------------------------

@pytest.mark.cardless
def test_stream_and_event_primitives():
    ops = FakeStreamOps()
    s1 = NBXStream(device_idx=1, _ops=ops)
    s2 = NBXStream(device_idx=0, _ops=ops)
    assert s1.device_idx == 1
    assert int(s1) == s1.handle
    assert ops.current_device == 0  # restored after creating s1 on device 1

    ev = s1.record_event()
    assert isinstance(ev, NBXEvent)
    assert ev.enable_timing is False
    assert ("record_event", ev.handle, s1.handle) in ops.log

    s2.wait_event(ev)
    assert ("stream_wait_event", s2.handle, ev.handle) in ops.log

    s2.wait_stream(s1)
    s2.synchronize()
    assert ("stream_synchronize", s2.handle) in ops.log

    # Non-timing event refuses elapsed_time.
    with pytest.raises(RuntimeError):
        ev.elapsed_time(ev)

    t_start = NBXEvent(enable_timing=True, _ops=ops)
    t_end = NBXEvent(enable_timing=True, _ops=ops)
    t_start.record(s1)
    t_end.record(s1)
    t_end.synchronize()
    assert t_start.elapsed_time(t_end) == pytest.approx(1.25)

    # Idempotent close.
    with s1 as ctx:
        assert ctx is s1
    s1.close()
    s1.close()
    assert ops.log.count(("destroy_stream", ctx.handle or 100)) == 1
    ev.close()
    ev.close()
    t_start.close()
    t_end.close()
    s2.close()


@pytest.mark.cardless
def test_copy_async_h2d_d2h_d2d_and_validation():
    ops = FakeStreamOps()
    s = NBXStream(handle=777, _ops=ops)

    cpu_t = NBXTensor(0x2000, (16,), (1,), NBXDtype.float32, "cpu", owns_data=False, pinned=True)
    gpu_a = NBXTensor(0x3000, (16,), (1,), NBXDtype.float32, "cuda", owns_data=False, device_idx=0)
    gpu_b = NBXTensor(0x4000, (16,), (1,), NBXDtype.float32, "cuda", owns_data=False, device_idx=1)

    # H2D async on stream
    assert copy_async(cpu_t, gpu_a, stream=s, _ops=ops) is gpu_a
    assert ("memcpy_async", 0x3000, 0x2000, 64, 1, 777) in ops.log

    # D2H async on stream
    assert copy_async(gpu_a, cpu_t, stream=s, _ops=ops) is cpu_t
    assert ("memcpy_async", 0x2000, 0x3000, 64, 2, 777) in ops.log

    # D2D async on stream
    assert copy_async(gpu_a, gpu_b, stream=s, _ops=ops) is gpu_b
    assert ("memcpy_async", 0x4000, 0x3000, 64, 3, 777) in ops.log

    # Synchronous path when non_blocking=False and stream=None
    assert copy_async(cpu_t, gpu_a, non_blocking=False, _ops=ops) is gpu_a
    assert ("memcpy", 0x3000, 0x2000, 64, 1) in ops.log

    # Size mismatch refused
    small = NBXTensor(0x5000, (4,), (1,), NBXDtype.float32, "cuda", owns_data=False, device_idx=0)
    with pytest.raises(ValueError, match="byte-size mismatch"):
        copy_async(cpu_t, small, stream=s, _ops=ops)

    # Non-dense destination refused
    non_dense_dst = NBXTensor(0x6000, (8, 2), (4, 1), NBXDtype.float32, "cuda", owns_data=False, device_idx=0)
    with pytest.raises(ValueError, match="dense memory window"):
        copy_async(cpu_t, non_dense_dst, stream=s, _ops=ops)


@pytest.mark.cardless
def test_tensor_to_non_blocking_with_stream(monkeypatch):
    ops = FakeStreamOps()
    s = NBXStream(handle=888, _ops=ops)

    monkeypatch.setattr(DeviceAllocator, "set_device", staticmethod(ops.set_device))
    monkeypatch.setattr(DeviceAllocator, "malloc_cuda", staticmethod(lambda n: 0x9000))
    monkeypatch.setattr(DeviceAllocator, "free_cuda", staticmethod(lambda p: None))
    monkeypatch.setattr(DeviceAllocator, "memcpy_async", staticmethod(ops.memcpy_async))

    cpu_t = NBXTensor(0x2000, (8,), (1,), NBXDtype.float32, "cpu", owns_data=False, pinned=True)

    # Legacy `.to("cuda:0")` without non_blocking/stream preserves existing no-op contract.
    assert cpu_t.to("cuda:0") is cpu_t

    # Explicit non_blocking=True + stream dispatches async H2D copy on that stream.
    gpu_t = cpu_t.to("cuda:1", non_blocking=True, stream=s)
    assert gpu_t._device == "cuda"
    assert gpu_t._device_idx == 1
    assert ("set_device", 1) in ops.log
    assert ("memcpy_async", 0x9000, 0x2000, 32, 1, 888) in ops.log

    # Explicit D2H via `.to("cpu", non_blocking=True, stream=s)` uses pinned host pool + async memcpy.
    backend = FakeHostBackend()
    pool = CachingHostAllocator(alloc_fn=backend.alloc, free_fn=backend.free)
    prev = set_pinned_host_allocator(pool)
    try:
        back = gpu_t.to("cpu", non_blocking=True, stream=s)
        assert back._device == "cpu"
        assert back.is_pinned
        assert ("memcpy_async", back.data_ptr(), 0x9000, 32, 2, 888) in ops.log
        del back
        gc.collect()
        assert pool.empty_cache() == 4096
    finally:
        set_pinned_host_allocator(prev)


# ---------------------------------------------------------------------------
# 4. Zero-Torch AST invariant check
# ---------------------------------------------------------------------------

@pytest.mark.cardless
def test_zero_torch_imports_in_nbxtensor():
    src_root = Path(__file__).resolve().parent.parent / "src" / "nbxtensor"
    offenders: List[str] = []
    for py_file in sorted(src_root.rglob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "torch" or alias.name.startswith("torch."):
                        offenders.append(f"{py_file.relative_to(src_root)}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod == "torch" or mod.startswith("torch."):
                    offenders.append(f"{py_file.relative_to(src_root)}:{node.lineno}: from {mod} import ...")
    assert not offenders, "Forbidden torch imports in nbxtensor:\n" + "\n".join(offenders)
