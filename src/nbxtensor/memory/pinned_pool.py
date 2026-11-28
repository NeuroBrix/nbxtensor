"""Reusable pinned host memory caching allocator (`CachingHostAllocator`).

Pure generic tensor/memory primitive (PyTorch `CachingHostAllocator` equivalent):
pools page-locked host allocations (`cudaMallocHost` / `hipHostMalloc`, or any
injected `alloc_fn` / `free_fn` backend) by rounded size bucket so repeated
pinned host allocations and frees avoid repeated OS page-locking syscalls.

Thread-safe, bounded by an optional `max_pool_bytes` LRU eviction cap, and
100% Torch-free.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Callable, Dict, List, Optional

#: Minimum bucket granularity (one 4 KiB OS page, matching page-lock granularity).
DEFAULT_MIN_BUCKET_BYTES: int = 4096
#: Power-of-two bucketing threshold; larger allocations round to `_LARGE_STEP_BYTES`.
_POW2_MAX_BYTES: int = 64 * 1024 * 1024  # 64 MiB
_LARGE_STEP_BYTES: int = 4 * 1024 * 1024  # 4 MiB


def _default_alloc(nbytes: int) -> int:
    from nbxtensor.memory.allocator import DeviceAllocator
    return DeviceAllocator._raw_malloc_host_pinned(nbytes)


def _default_free(ptr: int) -> None:
    from nbxtensor.memory.allocator import DeviceAllocator
    DeviceAllocator._raw_free_host_pinned(ptr)


class CachingHostAllocator:
    """Thread-safe bucketed cache for page-locked host memory allocations.

    Parameters
    ----------
    alloc_fn:
        Callable ``(bucket_bytes: int) -> int`` returning a raw host pointer.
        Defaults to ``DeviceAllocator._raw_malloc_host_pinned``.
    free_fn:
        Callable ``(ptr: int) -> None`` releasing a pointer back to the driver.
        Defaults to ``DeviceAllocator._raw_free_host_pinned``.
    max_pool_bytes:
        Optional upper bound on bytes parked in the free cache. ``None`` means
        unbounded; ``0`` disables caching (every ``free`` immediately releases).
    min_bucket_bytes:
        Minimum allocation bucket size in bytes (must be a positive integer,
        default 4096).
    """

    def __init__(
        self,
        *,
        alloc_fn: Optional[Callable[[int], int]] = None,
        free_fn: Optional[Callable[[int], None]] = None,
        max_pool_bytes: Optional[int] = None,
        min_bucket_bytes: int = DEFAULT_MIN_BUCKET_BYTES,
    ) -> None:
        if min_bucket_bytes <= 0:
            raise ValueError(f"min_bucket_bytes must be positive, got {min_bucket_bytes}")
        if max_pool_bytes is not None and max_pool_bytes < 0:
            raise ValueError(f"max_pool_bytes must be >= 0 or None, got {max_pool_bytes}")

        self._alloc_fn: Callable[[int], int] = alloc_fn if alloc_fn is not None else _default_alloc
        self._free_fn: Callable[[int], None] = free_fn if free_fn is not None else _default_free
        self._max_pool_bytes: Optional[int] = max_pool_bytes
        self._min_bucket_bytes: int = int(min_bucket_bytes)

        self._lock = threading.Lock()
        # Live checked-out pointers: ptr -> bucket_bytes
        self._live: Dict[int, int] = {}
        # Cached free pointers per bucket: bucket_bytes -> LIFO list of ptrs
        self._free_buckets: Dict[int, List[int]] = {}
        # LRU order of all cached free pointers: ptr -> bucket_bytes (oldest first)
        self._lru: "OrderedDict[int, int]" = OrderedDict()

        self._allocated_bytes: int = 0
        self._cached_bytes: int = 0
        self._reserved_bytes: int = 0
        self._peak_allocated_bytes: int = 0
        self._peak_reserved_bytes: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._evictions: int = 0

    def bucket_size(self, nbytes: int) -> int:
        """Round ``nbytes`` up to its pool bucket size."""
        if nbytes < 0:
            raise ValueError(f"nbytes must be non-negative, got {nbytes}")
        if nbytes == 0:
            return 0
        if nbytes <= self._min_bucket_bytes:
            return self._min_bucket_bytes
        if nbytes <= _POW2_MAX_BYTES:
            return 1 << (nbytes - 1).bit_length()
        return ((nbytes + _LARGE_STEP_BYTES - 1) // _LARGE_STEP_BYTES) * _LARGE_STEP_BYTES

    def alloc(self, nbytes: int) -> int:
        """Allocate at least ``nbytes`` of pinned host memory, reusing a cached
        bucket buffer when available."""
        if nbytes < 0:
            raise ValueError(f"nbytes must be non-negative, got {nbytes}")
        if nbytes == 0:
            return 0

        bucket = self.bucket_size(nbytes)

        with self._lock:
            bucket_list = self._free_buckets.get(bucket)
            if bucket_list:
                ptr = bucket_list.pop()
                if not bucket_list:
                    del self._free_buckets[bucket]
                self._lru.pop(ptr, None)
                self._cached_bytes -= bucket
                self._live[ptr] = bucket
                self._allocated_bytes += bucket
                if self._allocated_bytes > self._peak_allocated_bytes:
                    self._peak_allocated_bytes = self._allocated_bytes
                self._cache_hits += 1
                return ptr
            self._cache_misses += 1

        # Cache miss: invoke backend allocator outside the lock.
        try:
            ptr = int(self._alloc_fn(bucket))
        except Exception:
            # Flush cached buffers from other buckets and retry once (OOM recovery).
            freed = self.empty_cache()
            if freed <= 0:
                raise
            ptr = int(self._alloc_fn(bucket))

        if not ptr:
            raise RuntimeError(f"Pinned host allocator returned null pointer for {bucket} bytes")

        with self._lock:
            self._live[ptr] = bucket
            self._allocated_bytes += bucket
            self._reserved_bytes += bucket
            if self._allocated_bytes > self._peak_allocated_bytes:
                self._peak_allocated_bytes = self._allocated_bytes
            if self._reserved_bytes > self._peak_reserved_bytes:
                self._peak_reserved_bytes = self._reserved_bytes
        return ptr

    def free(self, ptr: int, nbytes: Optional[int] = None) -> None:
        """Return a pinned host pointer to the pool (or release via ``free_fn``
        if it exceeds ``max_pool_bytes``)."""
        if not ptr:
            return
        ptr = int(ptr)
        to_release: List[int] = []

        with self._lock:
            bucket = self._live.pop(ptr, None)
            if bucket is None:
                return
            self._allocated_bytes -= bucket

            cap = self._max_pool_bytes
            if cap is not None and bucket > cap:
                # Buffer alone exceeds pool cap: release directly without parking.
                self._reserved_bytes -= bucket
                self._evictions += 1
                to_release.append(ptr)
            else:
                self._free_buckets.setdefault(bucket, []).append(ptr)
                self._lru[ptr] = bucket
                self._cached_bytes += bucket

                if cap is not None:
                    while self._cached_bytes > cap and self._lru:
                        old_ptr, old_bucket = self._lru.popitem(last=False)
                        blist = self._free_buckets.get(old_bucket)
                        if blist is not None:
                            try:
                                blist.remove(old_ptr)
                            except ValueError:
                                pass
                            if not blist:
                                del self._free_buckets[old_bucket]
                        self._cached_bytes -= old_bucket
                        self._reserved_bytes -= old_bucket
                        self._evictions += 1
                        to_release.append(old_ptr)

        for dead_ptr in to_release:
            self._free_fn(dead_ptr)

    def empty_cache(self) -> int:
        """Release all cached (unused) host buffers via ``free_fn``.

        Returns the total number of bytes freed back to the backend.
        """
        to_release: List[int] = []
        freed_bytes = 0

        with self._lock:
            if not self._lru:
                return 0
            to_release = list(self._lru.keys())
            freed_bytes = self._cached_bytes
            self._lru.clear()
            self._free_buckets.clear()
            self._cached_bytes = 0
            self._reserved_bytes -= freed_bytes

        for ptr in to_release:
            self._free_fn(ptr)
        return freed_bytes

    def set_max_pool_bytes(self, max_pool_bytes: Optional[int]) -> int:
        """Update the cached pool byte cap and evict excess cached buffers."""
        if max_pool_bytes is not None and max_pool_bytes < 0:
            raise ValueError(f"max_pool_bytes must be >= 0 or None, got {max_pool_bytes}")
        to_release: List[int] = []
        freed_bytes = 0

        with self._lock:
            self._max_pool_bytes = max_pool_bytes
            if max_pool_bytes is not None:
                while self._cached_bytes > max_pool_bytes and self._lru:
                    old_ptr, old_bucket = self._lru.popitem(last=False)
                    blist = self._free_buckets.get(old_bucket)
                    if blist is not None:
                        try:
                            blist.remove(old_ptr)
                        except ValueError:
                            pass
                        if not blist:
                            del self._free_buckets[old_bucket]
                    self._cached_bytes -= old_bucket
                    self._reserved_bytes -= old_bucket
                    self._evictions += 1
                    freed_bytes += old_bucket
                    to_release.append(old_ptr)

        for ptr in to_release:
            self._free_fn(ptr)
        return freed_bytes

    def stats(self) -> Dict[str, Optional[int]]:
        """Return a snapshot of allocator counters and byte totals."""
        with self._lock:
            return {
                "allocated_bytes": self._allocated_bytes,
                "reserved_bytes": self._reserved_bytes,
                "cached_bytes": self._cached_bytes,
                "peak_allocated_bytes": self._peak_allocated_bytes,
                "peak_reserved_bytes": self._peak_reserved_bytes,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "evictions": self._evictions,
                "max_pool_bytes": self._max_pool_bytes,
            }


# ---------------------------------------------------------------------------
# Process-wide default pinned host allocator
# ---------------------------------------------------------------------------

_DEFAULT_LOCK = threading.Lock()
_DEFAULT_ALLOCATOR: Optional[CachingHostAllocator] = None


def _env_max_pool_bytes() -> Optional[int]:
    raw = os.environ.get("NBX_HOST_PINNED_MAX_BYTES", "").strip()
    if not raw:
        return None
    val = int(raw)
    return val if val >= 0 else None


def get_pinned_host_allocator() -> CachingHostAllocator:
    """Return the process-wide default ``CachingHostAllocator``."""
    global _DEFAULT_ALLOCATOR
    if _DEFAULT_ALLOCATOR is None:
        with _DEFAULT_LOCK:
            if _DEFAULT_ALLOCATOR is None:
                _DEFAULT_ALLOCATOR = CachingHostAllocator(
                    max_pool_bytes=_env_max_pool_bytes(),
                )
    return _DEFAULT_ALLOCATOR


def set_pinned_host_allocator(allocator: Optional[CachingHostAllocator]) -> Optional[CachingHostAllocator]:
    """Install a custom process-wide ``CachingHostAllocator`` (or ``None`` to reset).

    Returns the previously installed allocator. Primarily useful for tests and
    custom memory-capped environments.
    """
    global _DEFAULT_ALLOCATOR
    with _DEFAULT_LOCK:
        prev = _DEFAULT_ALLOCATOR
        _DEFAULT_ALLOCATOR = allocator
        return prev


def empty_host_pinned_cache() -> int:
    """Release all cached buffers held by the default ``CachingHostAllocator``."""
    if _DEFAULT_ALLOCATOR is None:
        return 0
    return _DEFAULT_ALLOCATOR.empty_cache()


def host_pinned_pool_stats() -> Dict[str, Optional[int]]:
    """Return ``stats()`` for the default ``CachingHostAllocator``."""
    return get_pinned_host_allocator().stats()
