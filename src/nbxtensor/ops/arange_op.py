# Rewritten by NeuroBrix (Copyright NeuroBrix, Apache-2.0); derived from FlagGems (Apache-2.0), original notice kept
# Change: rewritten as a pure Triton kernel with no PyTorch import.
# Original notice: the provenance statement in the docstring below, unchanged, and THIRD_PARTY_NOTICES.md.
"""Arange — pure @triton.jit kernel.

Ported from FlagGems arange_func (Apache-2.0 license).
Generates sequential values: start, start+step, start+2*step, ...
"""

import triton
import nbxtensor.ops._launch_seam  # noqa: F401  this module's launches go through nbxtensor's launcher
import triton.language as tl


from nbxtensor.ops.ew_tile import tile_dependent, tile_independent  # noqa: F401  (ops/ew_tile.py)


@tile_independent
@triton.jit
def arange_kernel(
    output_ptr,
    start,
    step,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Generate sequential values: output[i] = start + i * step."""
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements

    value = start + idx * step
    tl.store(output_ptr + idx, value, mask=mask)
