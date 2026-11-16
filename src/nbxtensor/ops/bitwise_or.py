# Rewritten by NeuroBrix (Copyright NeuroBrix, Apache-2.0); derived from FlagGems (Apache-2.0), original notice kept
# Change: rewritten as a pure Triton kernel with no PyTorch import.
# Original notice: the provenance statement in the docstring below, unchanged, and THIRD_PARTY_NOTICES.md.
"""Bitwise OR kernel — extracted from FlagGems.

Pure Triton. ZERO PyTorch imports.
Logic: result = x | y
"""

import triton
import triton.language as tl

@triton.jit
def bitwise_or_forward_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask)
    y = tl.load(y_ptr + offset, mask=mask)

    result = x | y
    tl.store(output_ptr + offset, result, mask=mask)
