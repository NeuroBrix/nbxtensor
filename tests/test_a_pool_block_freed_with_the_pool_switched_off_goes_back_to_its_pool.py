"""A block the pool handed out goes back to the pool when it is freed, whatever the pool switch says at that moment.

The free path chose by the switch in force: with `_pool_enabled` False, a pool block was handed to the driver alone.
The driver refuses a block that is not its segment's base (rc 3, "cudaFree rc=3 ... sticky async error", the pool's
bookkeeping keeping a block that no longer frees), and RELEASES THE WHOLE SEGMENT for the block that is its base, while
the segment's other blocks are still live and still listed. The next write to one of them reaches memory that is gone.
Mac M4 Pro, 2026-11-19: the engine suite segfaulted 3 runs of 3 in `NBXTensor.from_numpy` (a memmove at
metal_device.py:588) right after a test fixture that empties the pool and switches it off (`_no_pool`); its 30 refused
frees were all blocks earlier tests had taken from the pool.

What this test would do if the code were wrong: the child process writes into a live block of a segment the driver
has already released (a segfault, or the bytes not read back), or logs "[NBX-CUDA-ERROR]" for the base or the block."""
import subprocess
import sys
import textwrap

import pytest

from nbxtensor.backend.runtime import gpu_backend

pytestmark = pytest.mark.skipif(gpu_backend() is None, reason="no device")

CHILD = textwrap.dedent("""
    import numpy as np
    from nbxtensor.memory.allocator import DeviceAllocator as DA
    DA.set_device(0)
    DA._maybe_init_pool()
    DA._pool_enabled = True
    a, b = DA.malloc_cuda(4096), DA.malloc_cuda(4096)
    blocks = DA._POOL.blocks
    base, other = (a, b) if blocks[a][1] is None else (b, a)
    p = base
    while p is not None and p != other:          # the two blocks are carved from one segment
        p = blocks[p][2]
    assert blocks[base][1] is None and p == other, "the two blocks are not one segment's"
    # what a test fixture does to measure with the pool off: empty it, switch it off, then a live block is freed
    DA.empty_cache_pool()
    DA._pool_enabled = False
    DA.free_cuda(base)
    DA._pool_enabled = True
    host = np.arange(1024, dtype=np.float32)
    DA.memcpy(other, host.ctypes.data, host.nbytes, kind=1)
    back = np.empty_like(host)
    DA.memcpy(back.ctypes.data, other, host.nbytes, kind=2)
    assert (back == host).all(), "the live block of the segment did not keep its bytes"
    DA.free_cuda(other)
    print("child done")
""")


def test_a_pool_block_freed_with_the_pool_switched_off_goes_back_to_its_pool():
    r = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"rc {r.returncode}\n{r.stderr[-3000:]}"
    assert "child done" in r.stdout
    assert "[NBX-CUDA-ERROR]" not in r.stderr, r.stderr[-3000:]
