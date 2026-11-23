"""Tanh — pure @triton.jit kernel."""

import triton
import nbxtensor.ops._launch_seam  # noqa: F401  this module's launches go through nbxtensor's launcher
import triton.language as tl

from nbxtensor.ops.ew_tile import tile_dependent, tile_independent  # noqa: F401  (ops/ew_tile.py)


from ._common import sigmoid

@tile_independent
@triton.jit
def tanh_forward_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements

    x = tl.load(input_ptr + offset, mask=mask)
    x_fp32 = x.to(tl.float32)
    out = 2.0 * sigmoid(2.0 * x_fp32) - 1.0
    tl.store(output_ptr + offset, out, mask=mask)
