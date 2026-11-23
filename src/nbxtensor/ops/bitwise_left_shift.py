# Rewritten by NeuroBrix (Copyright NeuroBrix, Apache-2.0); derived from FlagGems (Apache-2.0), original notice kept
# Change: rewritten as a pure Triton kernel with no PyTorch import.
# Original notice: the provenance statement in the docstring below, unchanged, and THIRD_PARTY_NOTICES.md.
"""Bitwise left shift — pure @triton.jit kernel.

Computes: output = x << y (element-wise).

Both inputs must be integer tensors.

Extracted from FlagGems reference (bitwise_left_shift.py).
"""

import triton
import nbxtensor.ops._launch_seam  # noqa: F401  this module's launches go through nbxtensor's launcher
import triton.language as tl

from nbxtensor.ops.ew_tile import tile_dependent, tile_independent  # noqa: F401  (ops/ew_tile.py)


@tile_independent
@triton.jit
def bitwise_left_shift_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Bitwise left shift: out = x << y."""
    pid = tl.program_id(0).to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask)
    y = tl.load(y_ptr + offset, mask=mask)

    result = x << y
    tl.store(output_ptr + offset, result, mask=mask)

@tile_independent
@triton.jit
def bitwise_left_shift_scalar_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    shift_amount,
    BLOCK_SIZE: tl.constexpr,
):
    """Bitwise left shift by scalar: out = x << shift_amount."""
    pid = tl.program_id(0).to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask)
    result = x << shift_amount
    tl.store(output_ptr + offset, result, mask=mask)
