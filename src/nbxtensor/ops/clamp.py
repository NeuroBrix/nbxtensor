"""Clamp — pure @triton.jit kernel."""

import triton
import nbxtensor.ops._launch_seam  # noqa: F401  this module's launches go through nbxtensor's launcher
import triton.language as tl

from nbxtensor.ops.ew_tile import tile_dependent, tile_independent  # noqa: F401  (ops/ew_tile.py)
from nbxtensor.ops.ew_tile import synthetic  # noqa: E402  the tile proof's synthetic recipe (ops/ew_synthetic.py)
from nbxtensor.ops.ew_synthetic import flat  # noqa: E402


@synthetic(flat("n_elements"))
@tile_independent
@triton.jit
def clamp_forward_kernel(
    input_ptr, output_ptr,
    n_elements,
    min_val, max_val,
    has_min: tl.constexpr,
    has_max: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements

    x = tl.load(input_ptr + offset, mask=mask)
    if has_min:
        x = tl.maximum(min_val, x)
    if has_max:
        x = tl.minimum(max_val, x)
    tl.store(output_ptr + offset, x, mask=mask)
