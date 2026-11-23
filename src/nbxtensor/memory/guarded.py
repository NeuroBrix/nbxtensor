"""Guarded device buffers: a buffer whose END touches memory no access may reach, so that any lane touching past its
extent is SEEN, every time — the element-wise tile proof's guard (`ops/ew_proof.py`).

A buffer carved from a pool, an arena or a driver segment has mapped bytes after it: a kernel that touches past its
extent reads or writes them and nothing says so. One variant of one kernel faulted on one launch and passed on
another for exactly that reason (V100, 23/11/2026: what lay after the buffer differed). A guarded buffer removes the
chance: what lies after it is, by construction, a guard.

What each backend offers:

  * CUDA: the driver's virtual memory management (`memory.vmm.GuardedRange`): a reserved range one granule longer
    than the mapped pages, the buffer's end at the mapped end, the next granule UNMAPPED — an access past the extent
    is an illegal address (700). Needs the VMM attribute of the device; a device without it has no guard here.
  * Metal: a DEDICATED MTLBuffer per buffer, of a page-multiple length, the buffer's end at the MTLBuffer's end
    (`GuardedMetalBuffer`), in a process that runs with Metal's shader validation on (`MTL_SHADER_VALIDATION=1`,
    read by Metal when the process opens its device): every load and store a kernel makes is checked against the
    bound MTLBuffer's length, an access past it is NOT made and is reported on the command buffer
    (`MTLCommandBuffer.logs`, "Invalid load/store ... at offset ..."), which the runtime collects
    (`backend.metal_device.MetalRuntime.take_validation_reports`). Without shader validation an Apple GPU does not
    fault past a buffer: there is no guard.
  * Any other device (HIP, the CPU backend, a CUDA device without VMM): NO guard (`guard_kind()` is None). The proof
    says so and proves nothing there.

The guard's resolution is the alignment unit: a buffer whose start keeps a residue modulo 16 (Triton's 'D' pointer
attribute, or its absence) is placed at most 15 bytes before the guard (`place`): an overrun INSIDE that slack is NOT
seen. So the proof's ladder holds sizes with ZERO slack (`ops/ew_synthetic.ladder`): with a count specialised 'D'
every rung's count-sized buffers end exactly at the guard; with a count specialised '' the rung L + 8 does, for every
element of 2 bytes or more. An aligned buffer of 1-byte elements and a '' count always keeps slack (its bytes are
not a multiple of 16): an overrun of less than one 16-byte unit there is not seen. Bytes before the buffer's start are
not guarded (an underflow is not what this is for).

A device where no guard can be made (`guard_kind()` None) proves nothing: the proof refuses by name there. That is a
census question to answer when such a device arrives (what guard it offers), never a default taken for it.

Torch-free, numpy-free.
"""
from __future__ import annotations

from typing import Optional

#: Triton's divisibility unit: a pointer is 'D' when its address is divisible by it, an integer when its value is.
UNIT = 16


def place(end: int, nbytes: int, elem: int, aligned: bool) -> int:
    """The start of a buffer of `nbytes` bytes (elements of `elem` bytes) whose end is as close to `end` as its
    alignment allows: `end - nbytes`, moved down to a multiple of UNIT when `aligned`, or down by one element when it
    must NOT be a multiple of UNIT and is; never more than UNIT - 1 bytes of slack. `end` is a page or granule
    boundary (a multiple of UNIT), so the start keeps the element's natural alignment."""
    if end % UNIT:
        raise ValueError(f"a guard boundary is a multiple of {UNIT}: {end:#x}")
    start = end - int(nbytes)
    if aligned:
        start -= start % UNIT
    elif start % UNIT == 0:
        if elem % UNIT == 0:
            raise ValueError(f"an element of {elem} bytes cannot start off the {UNIT}-byte alignment")
        start -= elem
    return start


def guard_kind() -> Optional[str]:
    """What guards a buffer's end on this process's device: 'vmm' (CUDA with the VMM attribute), 'metal-validation'
    (Metal: dedicated buffers checked by shader validation), or None — no guard can be made here."""
    from nbxtensor.backend import runtime as _rt
    try:
        name = _rt.gpu_backend()
    except Exception:
        return None
    if name == "metal":
        return "metal-validation"
    if name == "cuda":
        return "vmm" if _cuda_vmm() is not None else None
    return None


_VMM = []


def _cuda_vmm():
    """The current CUDA device's VMM calls (`memory.vmm.Vmm`) when it has the attribute, else None (read once)."""
    if not _VMM:
        from nbxtensor.backend import runtime as _rt
        from nbxtensor.memory.allocator import DeviceAllocator
        from nbxtensor.memory.vmm import Vmm
        lib = Vmm.load(_rt.backend_table().get("driver_libs"))
        vmm = Vmm(lib, int(DeviceAllocator.get_device())) if lib is not None else None
        _VMM.append(vmm if vmm is not None and vmm.supported() else None)
    return _VMM[0]


_CONTEXT = []


def _runtime_context() -> None:
    """Make sure the runtime's primary context of the current device exists and is current on this thread before the
    first VMM call (the driver's reserve and map need a current context; the runtime makes its own at its first
    allocation): one unpooled byte allocated and freed, once per process."""
    if not _CONTEXT:
        from nbxtensor.memory.allocator import DeviceAllocator
        DeviceAllocator.free_cuda(DeviceAllocator.malloc_cuda(1, pooled=False))
        _CONTEXT.append(True)


def guarded(nbytes: int, elem: int, aligned: bool):
    """A guarded buffer of `nbytes` bytes on this process's device: an object with `.address` (its first byte),
    `.nbytes` and `.free()`. Refused by name where `guard_kind()` is None."""
    kind = guard_kind()
    if kind == "vmm":
        from nbxtensor.memory.vmm import GuardedRange
        _runtime_context()
        return GuardedRange(_cuda_vmm(), nbytes, elem, aligned)
    if kind == "metal-validation":
        from nbxtensor.memory.metal import GuardedMetalBuffer
        return GuardedMetalBuffer(nbytes, elem, aligned)
    raise RuntimeError("ZERO FALLBACK: no guarded buffer can be made on this device (no VMM, no shader validation): "
                       "an access past an extent cannot be seen here")
