# Rewritten by NeuroBrix (Copyright NeuroBrix, Apache-2.0); derived from FlagGems (Apache-2.0), original notice kept
# Change: rewritten as a pure Triton kernel with no PyTorch import.
# Original notice: the provenance statement in the docstring below, unchanged, and THIRD_PARTY_NOTICES.md.
"""Exp2 (base-2 exponential) — pure @triton.jit kernel. Logic from FlagGems."""

import triton
import nbxtensor.ops._launch_seam  # noqa: F401  this module's launches go through nbxtensor's launcher
import triton.language as tl

from nbxtensor.ops.ew_tile import tile_dependent, tile_independent  # noqa: F401  (ops/ew_tile.py)


@tile_independent
@triton.jit
def exp2_forward_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements

    x = tl.load(input_ptr + offset, mask=mask)
    out = tl.exp2(x.to(tl.float32))
    tl.store(output_ptr + offset, out, mask=mask)
