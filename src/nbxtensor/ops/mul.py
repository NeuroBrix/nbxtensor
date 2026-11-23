"""Multiply — pure @triton.jit kernel."""

import triton
import nbxtensor.ops._launch_seam  # noqa: F401  this module's launches go through nbxtensor's launcher
import triton.language as tl

from nbxtensor.ops.ew_tile import tile_dependent, tile_independent  # noqa: F401  (ops/ew_tile.py)
from nbxtensor.ops.ew_tile import synthetic  # noqa: E402  the tile proof's synthetic recipe (ops/ew_synthetic.py)
from nbxtensor.ops.ew_synthetic import flat  # noqa: E402


@tile_independent
@triton.jit
def mul_forward_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """out = x * y (tensor * tensor)"""
    pid = tl.program_id(0).to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask)
    y = tl.load(y_ptr + offset, mask=mask)
    tl.store(output_ptr + offset, x * y, mask=mask)

@synthetic(flat("n_elements"))
@tile_independent
@triton.jit
def mul_scalar_kernel(
    x_ptr, output_ptr,
    n_elements,
    scalar,
    BLOCK_SIZE: tl.constexpr,
):
    """out = x * scalar"""
    pid = tl.program_id(0).to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask)
    tl.store(output_ptr + offset, x * scalar, mask=mask)


@synthetic(flat("n_elements", s_ptr=1))
@tile_independent
@triton.jit
def mul_scalar_dev_kernel(
    x_ptr, s_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """out = x * s where s is a 0-d DEVICE tensor. Bit-exact mirror
    of mul_scalar_kernel (host float(s) -> f32 arg == load -> f64 ->
    f32). Device-scalar increment 2026-08-15."""
    pid = tl.program_id(0).to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements
    s = tl.load(s_ptr).to(tl.float64).to(tl.float32)
    x = tl.load(x_ptr + offset, mask=mask)
    tl.store(output_ptr + offset, x * s, mask=mask)
