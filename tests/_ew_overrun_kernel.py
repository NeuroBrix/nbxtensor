"""Kernels for the guard's test (`test_an_element_wise_proof_is_a_function_of_the_variant_alone.py`): a flat copy
whose store at ONE tile reaches one element past its extent, and the same copy correct. Imported by the proof's
worker by module name (the test puts this directory on its path)."""
import triton
import triton.language as tl

import nbxtensor.ops._launch_seam  # noqa: F401  launches through nbxtensor's launcher
from nbxtensor.ops.ew_tile import tile_independent


@tile_independent
@triton.jit
def overrun_at_4096_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """y = x; at BLOCK_SIZE 4096 the store also writes element n (one past y's extent)."""
    o = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = o < n_elements
    tl.store(y_ptr + o, tl.load(x_ptr + o, mask=m), mask=m)
    if BLOCK_SIZE == 4096:
        tl.store(y_ptr + n_elements + tl.arange(0, 1), tl.zeros((1,), tl.float32))


@tile_independent
@triton.jit
def masked_copy_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """y = x over n_elements, masked."""
    o = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = o < n_elements
    tl.store(y_ptr + o, tl.load(x_ptr + o, mask=m), mask=m)
