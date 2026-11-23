"""The device's virtual memory management primitives (CUDA's driver API: cuMemAddressReserve, cuMemCreate, cuMemMap,
cuMemSetAccess, cuMemUnmap, cuMemRelease) — what an expandable range (`memory.expandable`) builds on: ONE virtual
address range per device, physical chunks mapped into it as it grows and unmapped where a free block covers them whole
(R16, the supervisor's GO 2026-11-02 01:30; the method of torch's `expandable_segments`, PR #96995).

Reached through ctypes on the driver library the backend table names (`driver_libs`); a backend without the API (HIP,
Metal) or a device without the attribute (`CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED`) answers
`supported() == False` and the pool keeps its driver segments. Every call returns the driver's status code; the caller
names a failure. Measured on V100 (registres/benchmarks/vmm_v100.md, 2026-11-02): the granularity is 2 MiB; a map of a
chunk (create + map + set access) costs what an ordinary allocation of the same size costs (~0.36 ms), an unmap ~0.1 ms.

`GuardedRange` builds on the same calls a buffer whose end touches an unmapped granule, so that any access past its
extent faults every time: the element-wise tile proof's guard (`ops/ew_proof.py`).

Torch-free, numpy-free.
"""
from __future__ import annotations

import ctypes
from typing import Optional, Tuple

#: cuMemAllocationProp.type: CU_MEM_ALLOCATION_TYPE_PINNED; .location.type: CU_MEM_LOCATION_TYPE_DEVICE
_ALLOC_PINNED, _LOC_DEVICE = 1, 1
#: CU_MEM_ACCESS_FLAGS_PROT_READWRITE
_PROT_RW = 3
#: CU_MEM_ALLOC_GRANULARITY_MINIMUM
_GRAN_MINIMUM = 0
#: CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
_ATTR_VMM = 102


class _Loc(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _AllocFlags(ctypes.Structure):
    _fields_ = [("compressionType", ctypes.c_ubyte), ("gpuDirectRDMACapable", ctypes.c_ubyte),
                ("usage", ctypes.c_ushort), ("reserved", ctypes.c_ubyte * 4)]


class _Prop(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("requestedHandleTypes", ctypes.c_int), ("location", _Loc),
                ("win32HandleMetaData", ctypes.c_void_p), ("allocFlags", _AllocFlags)]


class _Access(ctypes.Structure):
    _fields_ = [("location", _Loc), ("flags", ctypes.c_int)]


class Vmm:
    """One device's VMM calls. `lib`: the driver library (ctypes CDLL); `device`: the device ordinal the process sees."""

    def __init__(self, lib, device: int) -> None:
        self.lib = lib
        self.device = int(device)
        self._prop = _Prop()
        self._prop.type = _ALLOC_PINNED
        self._prop.location = _Loc(_LOC_DEVICE, self.device)
        self._access = self._reachable_from()

    def _reachable_from(self):
        """The access grant of every chunk this device maps: the device itself, then every other device of the process
        that can reach it as a peer (asked through the allocator's one door, `DeviceAllocator.can_access_peer`). A VMM
        mapping is readable ONLY from the devices its grant names — enabling peer access between two contexts does
        nothing for it — so a chunk granted to its owner alone could not be copied card to card: `cudaMemcpy` answered
        "invalid value" for any tensor of the expandable range (a dense model split over two cards died at its first
        hand-off of more than a pool block, 2026-11-11; NVIDIA's statement of the rule: forums.developer.nvidia.com/t/
        148875). A question that fails is refused by name: a grant narrowed to the owner because the question could not
        be asked is the same card-to-card failure, later and unnamed."""
        count = ctypes.c_int(0)
        self._driver("cuDeviceGetCount", self.lib.cuDeviceGetCount(ctypes.byref(count)))
        peers = [other for other in range(count.value) if other != self.device and self._is_peer(other)]
        return (_Access * (1 + len(peers)))(*(_Access(_Loc(_LOC_DEVICE, d), _PROT_RW) for d in (self.device, *peers)))

    def _is_peer(self, other: int) -> bool:
        """Whether device `other` can reach this device's memory as a peer — the runtime's answer, never a guess."""
        from nbxtensor.memory.allocator import DeviceAllocator
        try:
            return DeviceAllocator.can_access_peer(other, self.device)
        except RuntimeError as refused:
            raise RuntimeError(f"{refused}{self._while}") from refused

    @property
    def _while(self) -> str:
        return (f" while device {self.device}'s expandable range asked which devices reach it as peers; a chunk "
                f"granted to its owner alone cannot be copied card to card")

    def _driver(self, call: str, rc: int) -> None:
        """Refuse, by name, a driver call the access grant is built from when it fails."""
        if rc != 0:
            raise RuntimeError(f"ZERO FALLBACK: the driver refused {call} (status {rc}){self._while}")

    @staticmethod
    def load(libs) -> Optional[object]:
        """The first driver library of `libs` that loads, initialised; None when none does."""
        for name in libs or ():
            try:
                lib = ctypes.CDLL(name)
            except OSError:
                continue
            if lib.cuInit(0) == 0:
                return lib
        return None

    def supported(self) -> bool:
        """The device answers the VMM attribute with 1."""
        dev, val = ctypes.c_int(), ctypes.c_int()
        if self.lib.cuDeviceGet(ctypes.byref(dev), self.device) != 0:
            return False
        if self.lib.cuDeviceGetAttribute(ctypes.byref(val), _ATTR_VMM, dev) != 0:
            return False
        return val.value == 1

    def granularity(self) -> Tuple[int, int]:
        """(rc, the minimum allocation granularity in bytes)."""
        g = ctypes.c_size_t()
        rc = self.lib.cuMemGetAllocationGranularity(ctypes.byref(g), ctypes.byref(self._prop), _GRAN_MINIMUM)
        return rc, int(g.value)

    def reserve(self, nbytes: int) -> Tuple[int, int]:
        """(rc, the base of a reserved virtual range of `nbytes`)."""
        va = ctypes.c_uint64()
        rc = self.lib.cuMemAddressReserve(ctypes.byref(va), ctypes.c_size_t(nbytes), ctypes.c_size_t(0),
                                          ctypes.c_uint64(0), ctypes.c_ulonglong(0))
        return rc, int(va.value)

    def unreserve(self, va: int, nbytes: int) -> int:
        """Give a reserved virtual range back (cuMemAddressFree); the driver's status."""
        return self.lib.cuMemAddressFree(ctypes.c_uint64(va), ctypes.c_size_t(nbytes))

    def create(self, nbytes: int) -> Tuple[int, int]:
        """(rc, a physical allocation handle of `nbytes` on the device)."""
        h = ctypes.c_ulonglong()
        rc = self.lib.cuMemCreate(ctypes.byref(h), ctypes.c_size_t(nbytes), ctypes.byref(self._prop),
                                  ctypes.c_ulonglong(0))
        return rc, int(h.value)

    def map(self, va: int, nbytes: int, handle: int) -> int:
        """Map `handle` at `va` and open it to the device; on a failed access grant the mapping is undone."""
        rc = self.lib.cuMemMap(ctypes.c_uint64(va), ctypes.c_size_t(nbytes), ctypes.c_size_t(0),
                               ctypes.c_ulonglong(handle), ctypes.c_ulonglong(0))
        if rc != 0:
            return rc
        rc = self.lib.cuMemSetAccess(ctypes.c_uint64(va), ctypes.c_size_t(nbytes), self._access,
                                     ctypes.c_size_t(len(self._access)))
        if rc != 0:
            self.lib.cuMemUnmap(ctypes.c_uint64(va), ctypes.c_size_t(nbytes))
        return rc

    def unmap(self, va: int, nbytes: int) -> int:
        """Unmap `nbytes` at `va` (cuMemUnmap); the driver's status."""
        return self.lib.cuMemUnmap(ctypes.c_uint64(va), ctypes.c_size_t(nbytes))

    def release(self, handle: int) -> int:
        """Release a physical allocation handle (cuMemRelease); the driver's status."""
        return self.lib.cuMemRelease(ctypes.c_ulonglong(handle))


class GuardedRange:
    """A device buffer whose END touches an UNMAPPED granule: any access past its extent faults, every time.

    The element-wise tile proof (`ops/ew_proof.py`) runs a variant on synthetic inputs and asks one question of each
    candidate tile: does a lane touch past the extent it was given? A buffer carved from a pool or from a driver
    segment has mapped bytes after it, so an overrun there reads or writes them and passes unseen (the same variant
    faulted on one launch and passed on another, V100, 23/11/2026). Here the range is built from the VMM calls:

      1. reserve a virtual range of `mapped + granularity` bytes, `mapped` = the buffer's bytes plus one alignment
         unit (16 bytes, Triton's divisibility), rounded up to the granularity;
      2. create a physical allocation of `mapped` bytes and map it at the range's base, opened to this device;
      3. leave the last granule of the range UNMAPPED, and place the buffer so that its end is the mapped end
         (`memory.guarded.place`: at most 15 bytes of slack where its start must keep an alignment residue).

    The first byte past the buffer (past the slack) is unmapped: a load or a store there is an illegal address
    (CUDA 700), sticky for the context, so the proof meets it in a child process. Bytes BEFORE the buffer's start
    are mapped (the granule's head): an underflow is not what this guards. `free()` unmaps, releases and gives the
    range back. Every failed call is refused by name."""

    def __init__(self, vmm: "Vmm", nbytes: int, elem: int, aligned: bool) -> None:
        """Reserve, map and place a guarded buffer of `nbytes` bytes of elements of `elem` bytes; `aligned`: its
        start divisible by 16 (Triton's 'D' pointer attribute), else not."""
        from nbxtensor.memory.guarded import UNIT, place
        self.vmm, self.nbytes = vmm, int(nbytes)
        rc, gran = vmm.granularity()
        self._check("cuMemGetAllocationGranularity", rc)
        self.mapped = -(-(self.nbytes + UNIT) // gran) * gran
        self.reserved = self.mapped + gran                  # the last granule stays unmapped: the guard
        rc, self.base = vmm.reserve(self.reserved)
        self._check("cuMemAddressReserve", rc)
        rc, self.handle = vmm.create(self.mapped)
        if rc != 0:
            vmm.unreserve(self.base, self.reserved)
            self._check("cuMemCreate", rc)
        rc = vmm.map(self.base, self.mapped, self.handle)
        if rc != 0:
            vmm.release(self.handle)
            vmm.unreserve(self.base, self.reserved)
            self._check("cuMemMap/cuMemSetAccess", rc)
        self.address = place(self.base + self.mapped, self.nbytes, int(elem), bool(aligned))

    def _check(self, call: str, rc: int) -> None:
        """Refuse, by name, a VMM call the guard is built from when it fails."""
        if rc != 0:
            raise RuntimeError(f"ZERO FALLBACK: the driver refused {call} (status {rc}) while building a guarded "
                               f"buffer of {self.nbytes} bytes (an extent against an unmapped granule)")

    @property
    def guard(self) -> int:
        """The first unmapped address: the mapped end."""
        return self.base + self.mapped

    def free(self) -> None:
        """Unmap the buffer, release its physical allocation, give its virtual range back."""
        self.vmm.unmap(self.base, self.mapped)
        self.vmm.release(self.handle)
        self.vmm.unreserve(self.base, self.reserved)
