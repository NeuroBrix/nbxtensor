"""On Metal, `reset_runtime_for_tests` drops the runtime and every fault word bound to it: an armed kernel launched
after the reset binds a fresh word, allocated by the new runtime, and an index out of range is still refused by name.

Before (the Mac, 2026-11-24): the reset cleared the allocator pool, the shape/stride cache, the launcher's and the
driver's caches, but not `faults._FAULT_BUFFERS`. The one-word status buffer an armed kernel binds stayed the one the
old runtime's allocator handed out, so the first armed launch after the reset (an embedding, an index_select)
handed a dead runtime's address to the new one (refused: "lies in no allocation of this Metal runtime"). The
per-kernel status block of the device assert (`metal_dispatch._ASSERT_BLOCKS`) was kept the same way. The engine's suite cured it on its side, clearing the cache by hand
after each reset (its order-leak hand-in); the cache and the runtime have one lifetime, so the reset clears it.

What this test does if the code is wrong:
- the cache cell fails when a fault word armed before the reset is still in `_FAULT_BUFFERS` after it;
- the launch cell fails when an embedding after the reset binds the old word, returns a wrong row, or lets an
  out-of-range id through without a refusal by name (the device assert's status block, also of the dropped
  runtime before the fix, or the fault word).

Run: python -m pytest tests/test_a_runtime_reset_drops_the_fault_words_of_the_old_runtime.py
"""
import sys

import numpy as np
import pytest

if sys.platform != "darwin":
    pytest.skip("Metal only", allow_module_level=True)
pytest.importorskip("triton")

from nbxtensor.backend import faults  # noqa: E402
from nbxtensor.backend import metal_device  # noqa: E402
from nbxtensor.memory.allocator import DeviceAllocator as DA  # noqa: E402
from nbxtensor.ops import index_wrappers as IW  # noqa: E402
from nbxtensor.tensor import NBXTensor  # noqa: E402


def _embedding(ids):
    weight = np.arange(4 * 8, dtype=np.float32).reshape(4, 8)
    out = IW.embedding(NBXTensor.from_numpy(weight), NBXTensor.from_numpy(np.asarray(ids, dtype=np.int64)))
    DA.stream_synchronize(0)
    return weight, out


def test_the_reset_leaves_no_fault_word_of_the_old_runtime():
    old = faults.device_fault_buffer(0)
    old_ptr = old.data_ptr()
    metal_device.reset_runtime_for_tests()
    left = {i: b.data_ptr() for i, b in faults._FAULT_BUFFERS.items()}
    assert not any(b is old for b in faults._FAULT_BUFFERS.values()), (
        f"the fault word 0x{old_ptr:x} of the dropped runtime is still cached after the reset: {left}")


def test_an_armed_launch_after_the_reset_binds_a_fresh_word_and_still_refuses():
    _embedding([0, 1])                                  # arms the channel under the first runtime
    old = faults._FAULT_BUFFERS.get(0)
    metal_device.reset_runtime_for_tests()

    weight, out = _embedding([3, 0, 2])
    fresh = faults._FAULT_BUFFERS.get(0)
    assert fresh is not None and fresh is not old, "the armed launch after the reset did not bind a fresh fault word"
    np.testing.assert_array_equal(out.numpy(), weight[[3, 0, 2]])

    with pytest.raises(RuntimeError, match="index out of range"):
        _, bad = _embedding([1, 9])
        bad.numpy()
        faults.check_device_faults()
