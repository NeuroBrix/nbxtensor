"""safetensors -> NBXTensor primitives (moved from the engine's Triton weight loader, M2): the dtype table of the
format, the parallel read of a tensor's bytes, the host-side conversions to the dtype it is held at, its placement in a
device arena or in page-locked host memory. The loading POLICY — which weight goes to which device or stays on the host,
the dtype a stored weight takes under a compute dtype, whether an fp16 target is widened — is the application's: it
passes each decision in. So are the I/O workers: every function that splits its work takes `io = (workers, executor)`,
the application's pool (its size is a fact of the machine the application owns).

Torch-free.
"""
from __future__ import annotations

import ctypes
import json
import math
import mmap
import os
import struct
from typing import NamedTuple, Optional, Sequence, Union

import numpy as np

from nbxtensor.tensor import NBXTensor
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.memory.allocator import DeviceAllocator
from nbxtensor.dtype.itemsize import FP16_MAX
from nbxtensor.dtype.convert import bf16_abs_max, bf16_bits_to_f16, f32_to_bf16_bits
from nbxtensor.shape import contiguous_strides, numel
from nbxtensor.memory.memory_pool import ComponentArena, HostPinnedArena

_STEP = 1 << 18
PAGE = mmap.PAGESIZE      # os.sysconf does not exist on Windows (its import stopped every plan there)


def bf16_abs_maxima(path, header: dict, data_start: int):
    """(key, element count, largest magnitude) of every bfloat16 tensor of the safetensors file at `path`, read as its
    stored 16-bit patterns through a read-only map — no tensor is built, no tensor library opens the file. `header` and
    `data_start`: the file's header and the offset its data starts at. A caller compares the magnitude to a narrower
    dtype's `finite_max` to know whether the weights fit it."""
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    for key, entry in header.items():
        if key == "__metadata__" or entry.get("dtype") != "BF16":
            continue
        begin, end = entry["data_offsets"]
        yield key, numel(entry["shape"]), bf16_abs_max(raw[data_start + begin:data_start + end].view(np.uint16))


class Fp16RangeScan(NamedTuple):
    """What `fp16_range_scan` found. `fits`: every scanned floating value is within +-FP16_MAX. `scanned`: the number of
    floating values read (0 = the files hold none: the caller says what that means, the library does not). When not
    `fits`: `path`, `key`, `abs_max` name the first tensor holding a value outside the range and its largest magnitude."""
    fits: bool
    scanned: int
    path: Optional[str] = None
    key: Optional[str] = None
    abs_max: Optional[float] = None


_SCAN_NUMPY = {"F16": np.float16, "F32": np.float32, "F64": np.float64}


def _abs_max_numpy(arr: np.ndarray) -> float:
    top = 0.0
    for i in range(0, arr.size, _STEP):
        v = np.abs(arr[i:i + _STEP].astype(np.float32 if arr.dtype != np.float64 else np.float64))
        v = v[~np.isnan(v)]
        if v.size:
            top = max(top, float(v.max()))
    return top


def fp16_range_scan(paths) -> Fp16RangeScan:
    """Do all floating weights (BF16, F16, F32, F64) of the safetensors files `paths` fit float16's range (+-FP16_MAX)?
    The decision a caller takes from it — load directly in the narrow type when they fit, fp32 when not — is the
    caller's. Files are read by this library through a read-only map, one tensor at a time in fixed steps (_STEP
    elements): the peak extra memory is a few steps of float32 (about 3 MB) whatever the model's size, never the model
    widened to fp32. Non-finite values: NaN is ignored, an infinity is outside the range. Stops at the first tensor
    outside the range, in the order of `paths` then of the file's header. Integer and other dtypes are not floating and
    are skipped; a bad header raises."""
    scanned = 0
    for path in paths:
        path = str(path)
        with open(path, "rb") as f:
            (hsize,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(hsize))
        raw = np.memmap(path, dtype=np.uint8, mode="r")
        for key, entry in header.items():
            if key == "__metadata__":
                continue
            dt = entry.get("dtype")
            begin, end = entry["data_offsets"]
            body = raw[8 + hsize + begin:8 + hsize + end]
            if dt == "BF16":
                top = bf16_abs_max(body.view(np.uint16))
            elif dt in _SCAN_NUMPY:
                top = _abs_max_numpy(body.view(_SCAN_NUMPY[dt]))
            else:
                continue
            scanned += numel(entry["shape"])
            if top > FP16_MAX:
                return Fp16RangeScan(False, scanned, path, key, top)
        del raw
    return Fp16RangeScan(True, scanned)


def spans(n: int, unit: int, align: int, workers: int):
    """[start, end) element spans of an n-element run, one per worker, each a multiple of `align`
    elements; a run under one page per worker stays whole (a split costs more than it shares)."""
    if workers < 2 or n * unit < workers * PAGE:
        return [(0, n)]
    step = -(-n // workers)
    step = -(-step // align) * align
    return [(a, min(a + step, n)) for a in range(0, n, step)]


# macOS/BSD preadv(2) and pread(2) reject counts > INT_MAX (0x7FFFFFFF) with EINVAL
# ([Errno 22] Invalid argument), whereas Linux silently caps at MAX_RW_COUNT (0x7FFFF000).
# Cap every single positioned read to a 64 KiB-aligned size strictly below INT_MAX so
# tensors >= 2 GiB (e.g. UMT5-XXL fp32 shared.weight = 4.20 GB) loop cleanly on all OSes.
_MAX_PREAD_BYTES = 0x7FFF0000


def _read_at(fd: int, view: memoryview, offset: int) -> int:
    """Read into `view` from the file's byte `offset`, at that offset and never through a shared file position (the
    workers read one file at once): `os.preadv` where the system has it (Linux, macOS); on Windows, which has neither
    preadv nor pread, `ReadFile` with an OVERLAPPED offset on the descriptor's handle — a positioned read. Returns the
    bytes read (0 at the end of the file)."""
    if hasattr(os, "preadv"):
        chunk = view[:_MAX_PREAD_BYTES] if len(view) > _MAX_PREAD_BYTES else view
        return os.preadv(fd, [chunk], offset)
    if hasattr(os, "pread"):
        n = min(len(view), _MAX_PREAD_BYTES)
        data = os.pread(fd, n, offset)
        view[:len(data)] = data
        return len(data)
    import msvcrt
    from ctypes import wintypes

    class _Overlapped(ctypes.Structure):
        _fields_ = [("Internal", ctypes.c_void_p), ("InternalHigh", ctypes.c_void_p), ("Offset", wintypes.DWORD),
                    ("OffsetHigh", wintypes.DWORD), ("hEvent", wintypes.HANDLE)]

    n = min(len(view), 0x7FFFFFFF)                   # one ReadFile reads at most a DWORD's worth
    ov = _Overlapped(0, 0, offset & 0xFFFFFFFF, offset >> 32, None)
    got = wintypes.DWORD(0)
    buf = (ctypes.c_char * n).from_buffer(view)
    ok = ctypes.windll.kernel32.ReadFile(wintypes.HANDLE(msvcrt.get_osfhandle(fd)), buf, n, ctypes.byref(got),
                                         ctypes.byref(ov))
    if not ok:
        err = ctypes.windll.kernel32.GetLastError()
        if err == 38:                                # ERROR_HANDLE_EOF: a read at or past the end
            return 0
        raise OSError(err, f"ReadFile at offset {offset} of fd {fd}")
    return int(got.value)


def read_bytes(fd: int, offset: int, nbytes: int, io) -> np.ndarray:
    """nbytes of the file at offset, read by the loader's workers into one buffer (page-aligned
    spans; `preadv` may return short, so each span reads until it is full)."""
    buf = np.empty(nbytes, dtype=np.uint8)
    view = memoryview(buf)

    def _one(span):
        a, b = span
        while a < b:
            got = _read_at(fd, view[a:b], offset + a)
            if got <= 0:
                raise RuntimeError(f"weight read: {nbytes} bytes at offset {offset} of fd {fd} ended "
                                   f"{b - a} bytes short (file truncated?)")
            a += got

    list(io[1].map(_one, spans(nbytes, 1, PAGE, io[0])))
    return buf


def split_across_workers(src: np.ndarray, out: np.ndarray, convert, io) -> np.ndarray:
    """`convert(src[a:b], out=flat_out[a:b])` over the loader's worker spans: one conversion of the library run in
    parallel on a host-held weight — the widening ran on one thread for every host-held bf16 weight (a third of
    Qwen3-30B-A3B's 447 s load, py-spy 2026-10-27). The split is the loader's; the conversion is nbxtensor's."""
    flat_src = np.ascontiguousarray(src).reshape(-1)
    flat = out.reshape(-1)

    def _one(span):
        a, b = span
        convert(flat_src[a:b], out=flat[a:b])

    list(io[1].map(_one, spans(flat_src.size, flat_src.itemsize, 1, io[0])))
    return out


def cast(arr: np.ndarray, want, io) -> np.ndarray:
    """`arr.astype(want)`, C-contiguous, the elements split across the loader's workers."""
    src = np.ascontiguousarray(arr).reshape(-1)
    out = np.empty(arr.shape, dtype=want)
    flat = out.reshape(-1)

    def _one(span):
        np.copyto(flat[span[0]:span[1]], src[span[0]:span[1]], casting="unsafe")

    list(io[1].map(_one, spans(src.size, src.itemsize, 1, io[0])))
    return out


def to_bf16_bits(arr: np.ndarray, io) -> np.ndarray:
    """float32 or float16 `arr` rounded to bfloat16 (`f32_to_bf16_bits`, uint16 patterns), the elements split across
    the loader's workers like `cast`: each span is widened (float16 -> float32 is exact) and rounded into its slice."""
    src = np.ascontiguousarray(arr).reshape(-1)
    out = np.empty(arr.shape, dtype=np.uint16)
    flat = out.reshape(-1)

    def _one(span):
        f32_to_bf16_bits(src[span[0]:span[1]].astype(np.float32, copy=False), out=flat[span[0]:span[1]])

    list(io[1].map(_one, spans(src.size, src.itemsize, 1, io[0])))
    return out


# safetensors dtype → (numpy dtype for reading, NBXDtype, bytes per element)
SF_DTYPE_INFO = {
    "F16":  (np.float16,  NBXDtype.float16,  2),
    "BF16": (None,        NBXDtype.bfloat16, 2),
    "F32":  (np.float32,  NBXDtype.float32,  4),
    "F64":  (np.float64,  NBXDtype.float64,  8),
    "I8":   (np.int8,     NBXDtype.int8,     1),
    "I16":  (np.int16,    NBXDtype.int16,    2),
    "I32":  (np.int32,    NBXDtype.int32,    4),
    "I64":  (np.int64,    NBXDtype.int64,    8),
    "U8":   (np.uint8,    NBXDtype.uint8,    1),
    "BOOL": (np.bool_,    NBXDtype.bool_,    1),
}


def mapped_view(mapping, offset: int, shape: tuple, sf_dtype: str) -> NBXTensor:
    """A host tensor over the stored bytes of one weight inside a read-only mapped shard (`mmap.ACCESS_READ`): its
    address read from the mapping without a writable export, no byte copied, the mapping held as the tensor's base so
    the pages live as long as the view (2026-11-13: built by the engine's executor until then)."""
    address = np.frombuffer(mapping, dtype=np.uint8).ctypes.data
    return NBXTensor.from_raw(address + int(offset), tuple(shape), SF_DTYPE_INFO[sf_dtype][1], device="cpu",
                              owns_data=False, base=mapping)


class ShardRef(NamedTuple):
    """One stored tensor: the safetensors file `path` and the tensor's `name` in its header (the header `fp16_range_scan`
    and `bf16_abs_maxima` read: 8-byte little-endian size, then the JSON)."""
    path: str
    name: str


class ShardConcat(NamedTuple):
    """A declaration: `sources` joined along existing axis `axis`. A source is a `ShardRef` or another declaration."""
    sources: tuple
    axis: int


class ShardStack(NamedTuple):
    """A declaration: `sources` stacked along a NEW axis `axis` (0..ndim of a source). A source is a `ShardRef` or
    another declaration."""
    sources: tuple
    axis: int


ShardSource = Union[ShardRef, ShardConcat, ShardStack]


class ShardsRefused(ValueError):
    """A declaration the library will not build (a missing tensor, shapes or dtypes that disagree, a bad axis, a
    conversion it does not do); the message names the sources involved."""


def _shard_label(node) -> str:
    while not isinstance(node, ShardRef):
        node = node.sources[0]
    return f"{node.path}:{node.name}"


def _shard_header(path, cache: dict) -> tuple:
    path = str(path)
    if path not in cache:
        with open(path, "rb") as f:
            (hsize,) = struct.unpack("<Q", f.read(8))
            cache[path] = (json.loads(f.read(hsize)), 8 + hsize)
    return cache[path]


def _shard_plan(node, cache: dict):
    """(shape, safetensors dtype code) of a source, every declared source checked, nothing read but headers."""
    if isinstance(node, ShardRef):
        header, _ = _shard_header(node.path, cache)
        entry = header.get(node.name)
        if node.name == "__metadata__" or entry is None:
            raise ShardsRefused(f"{node.path} holds no tensor named {node.name!r}: name a tensor of its header")
        if entry["dtype"] not in SF_DTYPE_INFO:
            raise ShardsRefused(f"{node.path}:{node.name} is stored as {entry['dtype']}, which has no dtype here")
        return tuple(entry["shape"]), entry["dtype"]
    if not isinstance(node, (ShardConcat, ShardStack)) or not len(node.sources):
        raise ShardsRefused(f"a source is a ShardRef or a non-empty ShardConcat / ShardStack, got {node!r}")
    plans = [_shard_plan(s, cache) for s in node.sources]
    first_shape, first_dt = plans[0]
    stacking = isinstance(node, ShardStack)
    nd = len(first_shape) + (1 if stacking else 0)
    if not -nd <= node.axis < nd:
        raise ShardsRefused(f"axis {node.axis} is outside a {nd}-dimensional result (from {_shard_label(node)})")
    axis = node.axis % nd
    total = 0
    for src, (shape, dt) in zip(node.sources[1:], plans[1:]):
        a, b = _shard_label(node), _shard_label(src)
        if dt != first_dt:
            raise ShardsRefused(f"sources {a} ({first_dt}) and {b} ({dt}) differ in dtype: no source is widened or "
                                f"converted, store them alike")
        rest = lambda sh: sh if stacking else sh[:axis] + sh[axis + 1:]
        if len(shape) != len(first_shape) or rest(shape) != rest(first_shape):
            raise ShardsRefused(f"sources {a} {first_shape} and {b} {shape} do not agree on every axis but "
                                f"{'the new one' if stacking else axis}")
    if stacking:
        return first_shape[:axis] + (len(plans),) + first_shape[axis:], first_dt
    total = sum(sh[axis] for sh, _ in plans)
    return first_shape[:axis] + (total,) + first_shape[axis + 1:], first_dt


def _shard_fill(node, dest: np.ndarray, cache: dict) -> None:
    """Write `node` into `dest`, a view of the result: a leaf copies its one tensor from a read-only map, a declaration
    hands each part its own slice of `dest`, so a nested declaration never builds an intermediate."""
    if isinstance(node, ShardRef):
        header, start = _shard_header(node.path, cache)
        entry = header[node.name]
        begin, end = entry["data_offsets"]
        raw = np.memmap(str(node.path), dtype=np.uint8, mode="r")
        dest[...] = raw[start + begin:start + end].view(dest.dtype).reshape(dest.shape)
        del raw                                    # the map of one source is gone before the next opens
        return
    stacking = isinstance(node, ShardStack)
    axis = node.axis % dest.ndim
    at = 0
    for src in node.sources:
        index = [slice(None)] * dest.ndim
        if stacking:
            index[axis] = at
            at += 1
        else:
            width = _shard_plan(src, cache)[0][axis]
            index[axis] = slice(at, at + width)
            at += width
        _shard_fill(src, dest[tuple(index)], cache)


def _build_shards(node, out_dtype) -> NBXTensor:
    cache: dict = {}
    shape, code = _shard_plan(node, cache)
    np_dtype, nbx_dtype, size = SF_DTYPE_INFO[code]
    if out_dtype is not None and out_dtype != nbx_dtype:
        raise ShardsRefused(f"the sources of {_shard_label(node)} are stored as {nbx_dtype!r} and out_dtype is "
                            f"{out_dtype!r}: this function never converts a stored weight, load it alike or convert "
                            f"it with the loader")
    out = NBXTensor.empty_cpu(shape, nbx_dtype)
    if out.numel():
        bits = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[size]
        dest = np.ctypeslib.as_array((ctypes.c_uint8 * out.nbytes()).from_address(out.data_ptr()))
        _shard_fill(node, dest.view(bits).reshape(shape), cache)
    return out


def stack_shards(sources: Sequence[ShardSource], axis: int, out_dtype: Optional[NBXDtype] = None) -> NBXTensor:
    """ONE host tensor: the stored tensors `sources` (each a `ShardRef`, or a `ShardConcat` / `ShardStack` declaration)
    stacked along a new `axis`, bit-equal to `NBXTensor.stack` of the same tensors read whole. Peak memory = the result
    plus ONE source: the result is allocated once, from the headers, and each source is copied from a read-only map of
    its file straight into its slice of it, in the order given; a nested declaration writes into the slice of its
    parent, so `ShardStack` of `ShardConcat`s holds no intermediate. Sources keep their stored dtype; `out_dtype` None
    means that dtype, and a different one is refused (nothing is widened or converted). Sources that differ in dtype or
    shape, a missing tensor and a bad axis raise `ShardsRefused` naming the sources, before any byte is read."""
    return _build_shards(ShardStack(tuple(sources), axis), out_dtype)


def concat_shards(sources: Sequence[ShardSource], axis: int, out_dtype: Optional[NBXDtype] = None) -> NBXTensor:
    """The same as `stack_shards`, the sources joined along their existing `axis` (bit-equal to `NBXTensor.cat`)."""
    return _build_shards(ShardConcat(tuple(sources), axis), out_dtype)


def half_in_place(ptr: int, count: int, device_idx: int):
    """The loader's raw bf16 weight buffer at `ptr` rewritten in place as float16 by nbxtensor's one device
    conversion (`ops.wrappers.bf16_to_fp16_gpu` with `out=` the same buffer)."""
    from nbxtensor.ops.elementwise_wrappers import bf16_to_fp16_gpu

    DeviceAllocator.set_device(device_idx)
    DeviceAllocator.ensure_triton_device(device_idx)
    flat = (count,)
    strides = contiguous_strides(flat)
    bits = NBXTensor(ptr, flat, strides, NBXDtype.int16, 'cuda', owns_data=False, device_idx=device_idx)
    half = NBXTensor(ptr, flat, strides, NBXDtype.float16, 'cuda', owns_data=False, device_idx=device_idx)
    bf16_to_fp16_gpu(bits, out=half)


def load_to_pinned_host(
    raw: bytes,
    shape: tuple,
    source_dtype: NBXDtype,
    target_dtype: NBXDtype,
    io,
    host_arena: Optional[HostPinnedArena] = None,
) -> NBXTensor:
    """Decode safetensors raw bytes into a pinned-host NBXTensor.

    Applies the standard dtype remap chain (bf16→fp16, fp16→bf16) via
    numpy before copying to pinned memory. bf16→fp16 uses the same
    uint16-bit reinterpret path the GPU arena loader uses, so the
    resulting values match exactly.
    """
    if source_dtype == NBXDtype.bfloat16 and target_dtype == NBXDtype.float16:
        # The widening written straight into the destination by the workers: no per-weight temporary (each one over
        # the host mmap threshold an mmap, its page faults and an munmap) and no copy after it — 18 864 of
        # Qwen3-30B-A3B's weights, 57 GB, take this path on a card without bfloat16 (2026-10-30).
        dst = _host_destination(shape, target_dtype, host_arena)
        if len(raw) != dst.nbytes():
            raise RuntimeError(f"weight staging would overflow: {len(raw)} bfloat16 bytes for shape {tuple(shape)} "
                               f"into a {dst.nbytes()}-byte float16 buffer")
        if dst.numel():
            out = np.ctypeslib.as_array((ctypes.c_uint16 * dst.numel()).from_address(dst.data_ptr())).view(np.float16)
            split_across_workers(np.frombuffer(raw, dtype=np.uint16), out, bf16_bits_to_f16, io)
        return dst

    # Decode source bytes to a numpy array matching target_dtype.
    if source_dtype == NBXDtype.bfloat16:
        # Raw bf16 bits stored as uint16. Stays uint16 if target is
        # bf16, or gets bit-shifted to fp32 then cast to fp16 if target
        # is fp16.
        raw_u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
        if target_dtype == NBXDtype.float16:
            arr = split_across_workers(raw_u16, np.empty(raw_u16.shape, dtype=np.float16), bf16_bits_to_f16, io)
        else:
            arr = np.ascontiguousarray(raw_u16)
    else:
        np_dtype = {
            NBXDtype.float16: np.float16,
            NBXDtype.float32: np.float32,
            NBXDtype.float64: np.float64,
            NBXDtype.int64: np.int64,
            NBXDtype.int32: np.int32,
            NBXDtype.int16: np.int16,
            NBXDtype.int8: np.int8,
            NBXDtype.uint8: np.uint8,
            NBXDtype.bool_: np.bool_,
        }[source_dtype]
        arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
        arr = np.ascontiguousarray(arr)
        if target_dtype == NBXDtype.bfloat16 and source_dtype == NBXDtype.float16:
            # fp16 → bf16, rounded to nearest even (fp16 → fp32 is exact)
            arr = to_bf16_bits(arr, io)
        elif target_dtype == NBXDtype.bfloat16 and source_dtype == NBXDtype.float32:
            # fp32 → bf16 rounded to nearest even, the SAME transform the
            # GPU arena placement uses (`place_in_arena`), so the zero3
            # host-staged copy of a weight is bit-identical to its arena copy.
            # An fp32 encoder (PixArt / CogVideoX T5) staged to a bf16 compute
            # target reached here still 4 bytes wide and overflowed the 2-byte
            # pinned buffer — _NUMPY_FOR has no bf16 entry (numpy cannot
            # represent it). The target itself is decided by
            # stored_dtype_in_compute, the one rule shared with the census
            # shadow, so completing this pair keeps the shadow's dtypes and the
            # real run's identical rather than diverging.
            arr = to_bf16_bits(arr, io)

    # Any remaining source->target pair: cast through numpy.
    #
    # The two bit-level paths above cover bf16, which numpy cannot represent.
    # Everything else is an ordinary cast, and it MUST happen: before
    # 2026-09-03 only the two bf16 pairs were converted, so a float32 weight
    # with a float16 target reached the copy still 4 bytes wide and wrote
    # `arr.nbytes` into a buffer sized for the TARGET dtype — 1024 bytes into
    # 512. Kokoro-82M's decoder is exactly that case (a (256,) fp32 tensor),
    # and the full-zoo gate recorded it as a 240 s TIMEOUT rather than as the
    # overflow it is, because the run never got far enough to fail.
    _NUMPY_FOR = {
        NBXDtype.float16: np.float16, NBXDtype.float32: np.float32,
        NBXDtype.float64: np.float64, NBXDtype.int64: np.int64,
        NBXDtype.int32: np.int32, NBXDtype.int16: np.int16,
        NBXDtype.int8: np.int8, NBXDtype.uint8: np.uint8,
        NBXDtype.bool_: np.bool_,
    }
    want = _NUMPY_FOR.get(target_dtype)
    if want is not None and arr.dtype != want:
        arr = cast(arr, want, io)

    dst = _host_destination(shape, target_dtype, host_arena)

    # The copy is sized from the SOURCE and the buffer from the TARGET, so a
    # dtype pair nobody converted is a write past the end of a cudaMallocHost
    # region. That is only caught at all because a pinned range is registered
    # with the driver; the same defect on ordinary host memory corrupts the
    # heap in silence. Refuse here, naming the pair, rather than rely on the
    # driver noticing.
    if arr.nbytes != dst.nbytes():
        raise RuntimeError(
            f"weight staging would overflow: {source_dtype!r} -> "
            f"{target_dtype!r} for shape {tuple(shape)} produced "
            f"{arr.nbytes} bytes for a {dst.nbytes()}-byte buffer. "
            f"This dtype pair has no conversion in load_to_pinned_host — add "
            f"one; never widen the buffer to match."
        )
    DeviceAllocator.memcpy(dst.data_ptr(), arr.ctypes.data,
                           arr.nbytes, kind=0)
    return dst


def _host_destination(shape: tuple, target_dtype: NBXDtype, host_arena: Optional[HostPinnedArena]) -> NBXTensor:
    """A host weight's destination: a slice of the component's one block when the load sized it, else its own
    page-locked block."""
    if host_arena is not None:
        from nbxtensor.dtype.nbx_dtype import dtype_size as _dsize
        ptr = host_arena.alloc((math.prod(shape) if shape else 1) * _dsize(target_dtype))
        return NBXTensor(ptr, tuple(shape), contiguous_strides(tuple(shape)), target_dtype, 'cpu',
                         owns_data=False, device_idx=0, base=host_arena, pinned=True)
    return NBXTensor.empty_cpu(shape, target_dtype, pinned=True)


def host_weight_layout(arr: np.ndarray, krsc: bool):
    """The host array to copy to the device and the element strides of the logical tensor over
    it. A KRSC weight (`launch_keys.conv_weight_krsc`) is written channels-innermost, once, here,
    and stays a logical (K, C, R, S) tensor: the original order is never on the device."""
    if not krsc:
        return arr, contiguous_strides(arr.shape)
    K, C, R, S = arr.shape
    return np.ascontiguousarray(arr.transpose(0, 2, 3, 1)), (R * S * C, 1, S * C, C)


def place_in_arena(raw, shape: tuple, nbx_dtype: NBXDtype, np_dtype, target_dtype: NBXDtype, krsc: bool,
                   arena: ComponentArena, target_dev: int, io) -> NBXTensor:
    """One safetensors tensor's raw bytes (stored as `nbx_dtype`, read through `np_dtype`) placed in `arena` on
    `target_dev` at `target_dtype` — channels-innermost when `krsc` (`host_weight_layout`) — and the NBXTensor over
    the arena's sub-allocation. The current device is the caller's to set."""
    nbytes = len(raw)
    # GPU-accelerated bf16→fp16 path (same buffer, 2 bytes/elem).
    if nbx_dtype == NBXDtype.bfloat16 and target_dtype == NBXDtype.float16:
        ptr = arena.alloc(nbytes)
        arr_u16, strides = host_weight_layout(
            np.frombuffer(raw, dtype=np.uint16).reshape(shape), krsc)
        DeviceAllocator.memcpy(ptr, arr_u16.ctypes.data, nbytes, kind=1)
        half_in_place(ptr, arr_u16.size, target_dev)
        return NBXTensor(ptr, shape, strides, NBXDtype.float16, 'cuda',
                         owns_data=False, device_idx=target_dev)

    # Standard path: numpy → H2D.
    if nbx_dtype == NBXDtype.bfloat16:
        # bf16 bits in uint16 containers, kept as they are
        arr = np.ascontiguousarray(np.frombuffer(raw, dtype=np.uint16).reshape(shape))
        final_dtype = target_dtype
    else:
        arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
        arr = np.ascontiguousarray(arr)
        if (nbx_dtype == NBXDtype.float32
              and target_dtype == NBXDtype.float16):
            # fp32 → fp16 downcast. Required when an encoder ships
            # fp32 on disk (PixArt T5, SDXL text_encoder_2, ...)
            # but the compute path wants fp16. Native's
            # WeightLoader(torch_dtype=fp16) does this implicitly;
            # triton needs to do it explicitly so arena sizing
            # and the kernel-side dtype agree.
            arr = cast(arr, np.float16, io)
            final_dtype = NBXDtype.float16
        elif (nbx_dtype == NBXDtype.float32
              and target_dtype == NBXDtype.bfloat16):
            # fp32 → bf16, rounded to nearest even.
            arr = to_bf16_bits(arr, io)
            final_dtype = NBXDtype.bfloat16
        elif target_dtype != nbx_dtype:
            # fp16 → bf16, rounded to nearest even (fp16 → fp32 is exact).
            arr = to_bf16_bits(arr, io)
            final_dtype = target_dtype
        else:
            final_dtype = target_dtype

    arr, strides = host_weight_layout(arr, krsc)
    arr_bytes = arr.nbytes
    ptr = arena.alloc(arr_bytes)
    DeviceAllocator.memcpy(ptr, arr.ctypes.data, arr_bytes, kind=1)  # H2D

    # Create NBXTensor wrapping the arena sub-allocation (KRSC: the logical shape over
    # the channels-innermost memory)
    return NBXTensor(ptr, shape, strides, final_dtype, 'cuda',
                     owns_data=False, device_idx=target_dev)
