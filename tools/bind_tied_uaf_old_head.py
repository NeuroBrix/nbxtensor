"""The tie's use-after-free, shown on a card: a source loaded the engine's way (a safetensors shard written here, sized
by `WeightArenas`, placed by the library's `place_in_arena` into one `ComponentArena`, "_arenas" the arena's only
reference), a head tied to it, the source released while the head still holds the alias, then two checks that do not
rely on luck:
  1. the alias's address lies inside a block the allocator still holds (`memory.blocks_in_use()`);
  2. a same-size block is allocated and filled with a sentinel (freed memory is handed out again first), and the
     alias still reads the shard's values.
Check 2 runs only when check 1 holds: reading memory the allocator freed is the fault itself, not a measure.

Device-agnostic: the library's current device (CUDA on a V100, Metal on a Mac).

    # a library at e1c52d95, the old signature bind_tied(weights, param, source):
    PYTHONPATH=src python tools/bind_tied_uaf_old_head.py
    # a library at 2088738e or later, the new one bind_tied(weights, param, source_weights, source_key):
    PYTHONPATH=src python tools/bind_tied_uaf_old_head.py --new

Exit 1 with "USE-AFTER-FREE SHOWN" when a check fails, 0 with "NO USE-AFTER-FREE" when both hold."""
import argparse
import gc
import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

EMBED = np.arange(4096, dtype=np.float32).reshape(256, 16)


def write_shard(path, arrays):
    """A safetensors file at `path` holding `arrays` ({name: float32 array}), written as the format says."""
    header, blobs, offset = {}, [], 0
    for name, a in arrays.items():
        raw = np.ascontiguousarray(a, dtype=np.float32).tobytes()
        header[name] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    text = json.dumps(header).encode()
    Path(path).write_bytes(struct.pack("<Q", len(text)) + text + b"".join(blobs))
    return path


def load_shard(path):
    """The shard loaded as the engine's loader does: `WeightArenas` sizing, one `ComponentArena` on the current
    device, each tensor placed by `place_in_arena`, the arenas under "_arenas" (their only reference)."""
    from nbxtensor import memory
    from nbxtensor.io import safetensors as sf
    from nbxtensor.memory.allocator import DeviceAllocator as DA
    data = Path(path).read_bytes()
    (hsize,) = struct.unpack("<Q", data[:8])
    header, body = json.loads(data[8:8 + hsize]), data[8 + hsize:]
    dev = DA.get_device()
    arenas = memory.weight_arenas()
    for info in header.values():
        arenas.reserve(dev, info["data_offsets"][1] - info["data_offsets"][0])
    arenas.allocate()
    table = {}
    for name, info in header.items():
        np_dtype, nbx_dtype, _ = sf.SF_DTYPE_INFO[info["dtype"]]
        start, end = info["data_offsets"]
        table[name] = sf.place_in_arena(body[start:end], tuple(info["shape"]), nbx_dtype, np_dtype, nbx_dtype,
                                        False, arenas.devices[dev], dev, None)
    table["_arenas"] = arenas.devices
    return table


def live(ptr) -> bool:
    """Whether `ptr` lies inside a block the allocator still holds for a tensor (not freed, not parked)."""
    from nbxtensor import memory
    return any(base <= ptr < base + n for base, n in memory.blocks_in_use().items())


def main() -> int:
    """Load, tie, release the source under the alias, check; the exit code and the last line say the verdict."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--new", action="store_true", help="the new signature bind_tied(weights, param, table, key)")
    args = ap.parse_args()
    from nbxtensor import memory
    from nbxtensor.memory.allocator import DeviceAllocator as DA
    from nbxtensor.memory.memory_pool import ComponentArena
    if DA.device_count() == 0:
        print("no accelerator on this machine: nothing shown")
        return 2
    tmp = Path(tempfile.mkdtemp(prefix="bind_tied_uaf_"))
    lm = load_shard(write_shard(tmp / "lm.safetensors", {"embed_tokens.weight": EMBED,
                                                         "norm.weight": np.ones(16, np.float32)}))
    head = load_shard(write_shard(tmp / "head.safetensors", {"proj.weight": np.ones((16, 16), np.float32),
                                                             "lm_head.weight": np.zeros(EMBED.shape, np.float32)}))
    if args.new:
        memory.bind_tied(head, "lm_head.weight", lm, "embed_tokens.weight")
    else:
        memory.bind_tied(head, "lm_head.weight", lm["embed_tokens.weight"])
    alias = head["lm_head.weight"]
    capacity = next(a.capacity for k, a in lm["_arenas"].items() if k != "_tied")
    print(f"device {DA.get_device()}; signature {'new' if args.new else 'old'}; alias at {alias.data_ptr():#x}")
    memory.release_weights(lm)                     # the source released while the head holds the alias
    gc.collect()
    ok_address = live(alias.data_ptr())
    print(f"check 1, the alias's address in a live block: {ok_address}")
    if not ok_address:
        print("USE-AFTER-FREE SHOWN: the source's arena was freed under a loaded alias", flush=True)
        return 1
    filler = ComponentArena(capacity, DA.get_device())
    sentinel = np.full(capacity // 4, -7.0, np.float32)
    DA.memcpy(filler.alloc(capacity), sentinel.ctypes.data, sentinel.nbytes, kind=1)
    DA.sync_device()
    ok_values = bool(np.array_equal(alias.numpy(), EMBED))
    print(f"check 2, the alias reads the shard after a same-size sentinel fill: {ok_values}")
    if not ok_values:
        print("USE-AFTER-FREE SHOWN: the alias reads what another allocation wrote", flush=True)
        return 1
    del filler
    print("NO USE-AFTER-FREE: the alias keeps the source's arena")
    return 0


if __name__ == "__main__":
    sys.exit(main())
