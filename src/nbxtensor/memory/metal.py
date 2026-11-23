"""Apple Metal's memory primitives (`DeviceMemory`) — the Metal device behind the one account.

The account (`memory.allocator.DeviceAllocator`) counts every byte this process holds, refuses past the device's
budget and reports. This object only does what the account asks of the device, over the Metal runtime
(`backend.metal_device.MetalRuntime`): shared-storage buffers whose one address the CPU and the GPU both use, their
release once the GPU is done with them, the in-place binding of host pages, copies and fills in the one address
space. It keeps NO account of its own: until 2026-10-29 the runtime kept a second one (`_live_bytes`) beside the
account's, two figures of one fact.

- `budget()`: `MTLDevice.recommendedMaxWorkingSetSize`, Apple's own ceiling for the process on unified memory. Metal
  hands out buffers past it, backed by memory the whole machine shares, until the system pages: the ACCOUNT refuses
  past it (`supports("budget_refusal")`), with this backend's allocation-refused status (`ALLOC_REFUSED`).
- `mem_get_info()`: (budget less what the account holds, budget) — `MTLDevice.currentAllocatedSize` is cumulative and
  never comes down (measured 2026-09-05), so the headroom is read from the account, never from the device.
- `free(ptr)`: the buffer is released once the command buffers in flight are done with it; the account counts it
  freed at once.

Torch-free (R33).
"""
from __future__ import annotations

import ctypes
from typing import Tuple

from .device_memory import DeviceMemory

#: this backend's status for an allocation refused (the runtime's `_ERR_ALLOC`): the account returns it when the budget
#: refuses, so a refusal reads the same whichever of the two said no
ALLOC_REFUSED = 1

_PRIMITIVES = frozenset({"malloc", "free", "mem_get_info", "memset", "memcpy", "host_alloc", "host_free",
                         "set_device", "get_device", "device_count", "budget", "map_host", "unmap_host",
                         "budget_refusal"})


class MetalMemory(DeviceMemory):
    """The memory primitives of the one Metal device of this process, over its runtime."""

    __slots__ = ("_rt",)
    ALLOC_REFUSED = ALLOC_REFUSED

    def __init__(self, runtime):
        self._rt = runtime

    def supports(self, primitive):
        return primitive in _PRIMITIVES

    def malloc(self, nbytes):
        p = ctypes.c_void_p()
        rc = self._rt.malloc(ctypes.byref(p), ctypes.c_size_t(int(nbytes)))
        return rc, (p.value or 0)

    def free(self, ptr):
        return self._rt.free(ctypes.c_void_p(int(ptr)))

    def budget(self) -> Tuple[int, int]:
        return 0, int(self._rt.budget_bytes())

    def mem_get_info(self):
        from .allocator import DeviceAllocator as DA
        total = int(self._rt.budget_bytes())
        held = DA._cuda_live_bytes.get(0, 0) + DA._host_pinned_live_bytes
        return 0, max(0, total - held), total

    def memset(self, ptr, value, nbytes):
        return self._rt.memset(int(ptr), int(value), int(nbytes))

    def memcpy(self, dst, src, nbytes, kind):
        return self._rt.memcpy(int(dst), int(src), int(nbytes), int(kind))

    def host_alloc(self, nbytes):
        p = ctypes.c_void_p()
        rc = self._rt.malloc_host(ctypes.byref(p), ctypes.c_size_t(int(nbytes)))
        return rc, (p.value or 0)

    def host_free(self, ptr):
        return self._rt.free_host(ctypes.c_void_p(int(ptr)))

    def set_device(self, index):
        return self._rt.set_device(int(index))

    def get_device(self):
        d = ctypes.c_int()
        rc = self._rt.get_device(ctypes.byref(d))
        return rc, d.value

    def device_count(self):
        n = ctypes.c_int()
        rc = self._rt.device_count(ctypes.byref(n))
        return rc, n.value

    def map_host(self, ptr, nbytes):
        return self._rt.bind_mapped(int(ptr), int(nbytes))

    def unmap_host(self, start):
        self._rt.unbind_mapped(start)


#: the switch Metal reads when the process opens its device: every load and store of a pipeline is checked against
#: its bound buffer's length (shader validation). Set by the process's parent, never flipped in a running process.
SHADER_VALIDATION = "MTL_SHADER_VALIDATION"


def shader_validation_on() -> bool:
    """Whether this process runs with Metal's shader validation (`MTL_SHADER_VALIDATION=1` in its environment)."""
    import os
    return os.environ.get(SHADER_VALIDATION) == "1"


class GuardedMetalBuffer:
    """A Metal buffer whose END is the end of a DEDICATED MTLBuffer, in a process with shader validation on: a load
    or a store past the extent is past the bound buffer's length, so Metal does not make it and reports it on the
    command buffer (`MetalRuntime.take_validation_reports`) — the Metal side of `memory.guarded`.

    The MTLBuffer is allocated through the one account (`DeviceAllocator.malloc_cuda`, unpooled: never a pool block
    shared with another tensor) at a page-multiple length — the runtime's own rounding, so its length is exactly
    what was asked (checked) — and the buffer is placed at its end (`memory.guarded.place`). A launch binds it as
    (that MTLBuffer, the buffer's offset in it): the length validation checks is the guard. Refused by name in a
    process without shader validation: there an Apple GPU reads past a buffer without a fault."""

    def __init__(self, nbytes: int, elem: int, aligned: bool) -> None:
        """Allocate the dedicated MTLBuffer and place a buffer of `nbytes` bytes (elements of `elem` bytes) at its
        end; `aligned`: its start divisible by 16."""
        from nbxtensor.backend.metal_device import runtime
        from nbxtensor.memory.allocator import DeviceAllocator
        from nbxtensor.memory.guarded import UNIT, place
        if not shader_validation_on():
            raise RuntimeError(f"ZERO FALLBACK: a guarded Metal buffer needs shader validation ({SHADER_VALIDATION}=1 "
                               "when the process starts): without it a read past a buffer is not seen")
        rt = runtime()
        page = int(rt._page_size)
        self.nbytes = int(nbytes)
        self.length = -(-(self.nbytes + UNIT) // page) * page
        self.base = int(DeviceAllocator.malloc_cuda(self.length, pooled=False))
        buffer, offset = rt.buffer_for_pointer(self.base)
        if buffer is None or offset != 0 or int(buffer.length()) != self.length:
            DeviceAllocator.free_cuda(self.base)
            raise RuntimeError(f"ZERO FALLBACK: a guarded Metal buffer of {self.length} bytes was not a dedicated "
                               f"MTLBuffer of that length (offset {offset}); its end is no guard")
        self.address = place(self.base + self.length, self.nbytes, int(elem), bool(aligned))

    @property
    def guard(self) -> int:
        """The first address past the MTLBuffer: the guard."""
        return self.base + self.length

    def free(self) -> None:
        """Give the dedicated MTLBuffer back through the account."""
        from nbxtensor.memory.allocator import DeviceAllocator
        DeviceAllocator.free_cuda(self.base)
