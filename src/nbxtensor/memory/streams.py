"""Generic async DMA copy stream, event, and tensor copy primitives.

Pure `torch.cuda.Stream` / `torch.cuda.Event` / `copy_async` equivalents over
`DeviceAllocator` stream/event/memcpy_async APIs. Zero serving, scheduling, or
model logic; 100% Torch-free.
"""
from __future__ import annotations

import sys
from typing import Any, Optional, Union

_is_finalizing = getattr(sys, "is_finalizing", lambda: False)


def _default_ops():
    from nbxtensor.memory.allocator import DeviceAllocator
    return DeviceAllocator


def _stream_handle(stream: Optional[Union["NBXStream", int]]) -> int:
    if stream is None:
        return 0
    return int(stream)


def _event_handle(event: Union["NBXEvent", int]) -> int:
    return int(event)


class NBXEvent:
    """Generic GPU ordering/timing event (equivalent to ``torch.cuda.Event``).

    Defaults to ``enable_timing=False`` to avoid timing-event overhead when only
    cross-stream ordering is needed.
    """

    __slots__ = ("_handle", "_enable_timing", "_owns", "_ops", "_closed")

    def __init__(
        self,
        *,
        enable_timing: bool = False,
        handle: Optional[int] = None,
        _ops: Any = None,
    ) -> None:
        ops = _ops if _ops is not None else _default_ops()
        self._ops = ops
        self._enable_timing = bool(enable_timing)
        self._closed = False
        if handle is None:
            self._handle = int(ops.create_event(timing=self._enable_timing))
            self._owns = True
        else:
            self._handle = int(handle)
            self._owns = False

    @property
    def handle(self) -> int:
        return self._handle

    @property
    def enable_timing(self) -> bool:
        return self._enable_timing

    def __int__(self) -> int:
        return self._handle

    def record(self, stream: Optional[Union["NBXStream", int]] = None) -> None:
        """Record this event on ``stream`` (default stream 0 if omitted)."""
        self._ops.record_event(self._handle, _stream_handle(stream))

    def synchronize(self) -> None:
        """Block the host thread until this event completes."""
        self._ops.event_synchronize(self._handle)

    def elapsed_time(self, end_event: "NBXEvent") -> float:
        """Return elapsed milliseconds from ``self`` to ``end_event``."""
        if not self._enable_timing or not getattr(end_event, "_enable_timing", True):
            raise RuntimeError(
                "elapsed_time requires both NBXEvents to be created with enable_timing=True"
            )
        return float(self._ops.event_elapsed_ms(self._handle, _event_handle(end_event)))

    def close(self) -> None:
        """Destroy the underlying event handle if owned (idempotent)."""
        if self._closed:
            return
        self._closed = True
        if self._owns and self._handle:
            self._ops.destroy_event(self._handle)
            self._handle = 0

    def __del__(self) -> None:
        if _is_finalizing():
            return
        try:
            self.close()
        except Exception:
            pass


class NBXStream:
    """Generic GPU execution/copy stream (equivalent to ``torch.cuda.Stream``)."""

    __slots__ = ("_handle", "_device_idx", "_owns", "_ops", "_closed")

    def __init__(
        self,
        device_idx: Optional[int] = None,
        *,
        handle: Optional[int] = None,
        _ops: Any = None,
    ) -> None:
        ops = _ops if _ops is not None else _default_ops()
        self._ops = ops
        self._device_idx = device_idx
        self._closed = False
        if handle is None:
            prev = None
            if device_idx is not None:
                prev = ops.get_device()
                if prev != device_idx:
                    ops.set_device(device_idx)
            try:
                self._handle = int(ops.create_stream())
            finally:
                if prev is not None and prev != device_idx:
                    ops.restore_device(prev)
            self._owns = True
        else:
            self._handle = int(handle)
            self._owns = False

    @property
    def handle(self) -> int:
        return self._handle

    @property
    def device_idx(self) -> Optional[int]:
        return self._device_idx

    def __int__(self) -> int:
        return self._handle

    def synchronize(self) -> None:
        """Block the host thread until all enqueued work on this stream finishes."""
        self._ops.stream_synchronize(self._handle)

    def record_event(self, event: Optional[NBXEvent] = None) -> NBXEvent:
        """Record ``event`` (or a newly created ordering event) on this stream."""
        if event is None:
            event = NBXEvent(enable_timing=False, _ops=self._ops)
        event.record(self)
        return event

    def wait_event(self, event: Union[NBXEvent, int]) -> None:
        """Make this stream wait for ``event`` without blocking the host."""
        self._ops.stream_wait_event(self._handle, _event_handle(event))

    def wait_stream(self, other: "NBXStream") -> None:
        """Order all future work on ``self`` after all currently enqueued work on ``other``."""
        ev = other.record_event()
        try:
            self.wait_event(ev)
        finally:
            ev.close()

    def close(self) -> None:
        """Destroy the underlying stream handle if owned (idempotent)."""
        if self._closed:
            return
        self._closed = True
        if self._owns and self._handle:
            self._ops.destroy_stream(self._handle)
            self._handle = 0

    def __enter__(self) -> "NBXStream":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def __del__(self) -> None:
        if _is_finalizing():
            return
        try:
            self.close()
        except Exception:
            pass


def copy_async(
    src,
    dst,
    *,
    stream: Optional[Union[NBXStream, int]] = None,
    non_blocking: bool = True,
    _ops: Any = None,
):
    """Copy raw bytes between two ``NBXTensor`` instances (H2D, D2H, D2D, or H2H).

    Enqueues an asynchronous DMA transfer on ``stream`` when ``non_blocking=True``
    or ``stream`` is provided; otherwise runs a synchronous ``memcpy``.

    Returns ``dst``.
    """
    if src._nbytes != dst._nbytes:
        raise ValueError(
            f"copy_async byte-size mismatch: src has {src._nbytes} bytes "
            f"(shape={src._shape}, dtype={src._dtype}), dst has {dst._nbytes} bytes "
            f"(shape={dst._shape}, dtype={dst._dtype})"
        )
    if not dst.spans_densely():
        raise ValueError("copy_async destination tensor must span a single dense memory window")

    src_dense = src if src.spans_densely() else src.contiguous()
    if src_dense._nbytes == 0:
        return dst

    src_dev = getattr(src_dense, "_device", "cuda")
    dst_dev = getattr(dst, "_device", "cuda")
    if src_dev == "cpu" and dst_dev == "cpu":
        kind = 0  # H2H
    elif src_dev == "cpu" and dst_dev == "cuda":
        kind = 1  # H2D
    elif src_dev == "cuda" and dst_dev == "cpu":
        kind = 2  # D2H
    else:
        kind = 3  # D2D

    ops = _ops if _ops is not None else _default_ops()
    if non_blocking or stream is not None:
        ops.memcpy_async(
            dst.data_ptr(),
            src_dense.data_ptr(),
            src_dense._nbytes,
            kind=kind,
            stream=_stream_handle(stream),
        )
    else:
        ops.memcpy(
            dst.data_ptr(),
            src_dense.data_ptr(),
            src_dense._nbytes,
            kind=kind,
        )
    return dst
