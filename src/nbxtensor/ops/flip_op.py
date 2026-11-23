# Rewritten by NeuroBrix (Copyright NeuroBrix, Apache-2.0); derived from FlagGems (Apache-2.0), original notice kept
# Change: rewritten as a pure Triton kernel with no PyTorch import.
# Original notice: the provenance statement in the docstring below, unchanged, and THIRD_PARTY_NOTICES.md.
"""Flip — pure @triton.jit kernel.

Reverses elements along one dimension. The wrapper computes the
flipped strides and offset; this kernel just copies the data.

Ported from FlagGems flip (Apache-2.0 license).
"""

import triton
import nbxtensor.ops._launch_seam  # noqa: F401  this module's launches go through nbxtensor's launcher
import triton.language as tl

from nbxtensor.ops.ew_tile import synthetic, tile_dependent, tile_independent  # noqa: F401  (ops/ew_tile.py)


@tile_independent
@triton.jit
def flip_1d_kernel(
    src_ptr, dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Reverse a 1D contiguous tensor: dst[i] = src[n_elements - 1 - i]."""
    pid = tl.program_id(0).to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements

    src_idx = n_elements - 1 - offset
    data = tl.load(src_ptr + src_idx, mask=mask)
    tl.store(dst_ptr + offset, data, mask=mask)

def _flip_strided_recipe(S):
    """The synthetic launch of `flip_strided_kernel`: `n_elements` = the ladder's count elements of a contiguous
    [pre, dim_size, post] (pre implicit in the count, its last block possibly partial), dim_size and post
    (`stride_dim`, and `stride_post` the same value as the call site passes) small in their specialization. The
    output `n_elements` elements, the source 1 + the largest mirrored offset a lane below the count reads (a partial
    last block reads its mirror past the count); grid `cdiv(n_elements, tile)`."""
    import numpy as np
    dim, post = S.int("dim_size", 3), S.int("stride_dim", 1)
    off = np.arange(S.n, dtype=np.int64)
    src = (off // (dim * post)) * (dim * post) + (dim - 1 - (off // post) % dim) * post + off % post
    return ({"n_elements": S.n, "dim_size": dim, "stride_dim": post, "stride_post": S.int("stride_post", post),
             "src_ptr": S.buf("src_ptr", int(src.max()) + 1), "dst_ptr": S.buf("dst_ptr", S.n)}, None)


_flip_strided_recipe.count = "n_elements"
_flip_strided_recipe.description = "flip strided: n_elements of [pre, dim_size, post], source = 1 + largest mirrored offset"


@synthetic(_flip_strided_recipe)
@tile_independent
@triton.jit
def flip_strided_kernel(
    src_ptr, dst_ptr,
    n_elements,
    # Dimension being flipped
    dim_size,
    stride_dim,
    stride_post,
    BLOCK_SIZE: tl.constexpr,
):
    """Flip along a single dimension of an N-D contiguous tensor.

    For each element, decomposes the flat index into (pre, dim, post)
    coordinates and reverses the dim coordinate.

    src and dst are both contiguous with the same layout; only the
    dim-axis index is mirrored.
    """
    pid = tl.program_id(0).to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements

    # Decompose flat index
    post_size = stride_dim  # product of all dims after the flip dim
    pre_idx = offset // (dim_size * post_size)
    dim_idx = (offset // post_size) % dim_size
    post_idx = offset % post_size

    # Flip the dim coordinate
    flipped_dim_idx = dim_size - 1 - dim_idx

    src_flat = pre_idx * (dim_size * post_size) + flipped_dim_idx * post_size + post_idx
    data = tl.load(src_ptr + src_flat, mask=mask)
    tl.store(dst_ptr + offset, data, mask=mask)
