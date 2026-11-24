# Rewritten by NeuroBrix (Copyright NeuroBrix, Apache-2.0); derived from FlagGems (Apache-2.0), original notice kept
# Change: rewritten as a pure Triton kernel with no PyTorch import.
# Original notice: the provenance statement in the docstring below, unchanged, and THIRD_PARTY_NOTICES.md.
"""Gather along dim — pure @triton.jit kernel.

Approach from FlagGems: work on flattened index space. For each element in the
index tensor, compute its multi-dim position, load the gather index, then load
from source replacing the gather dim's index with the loaded value.

This static kernel handles any rank by taking pre-computed strides as args.
The wrapper decomposes the gather dim into (outer, gather, inner) strides.
"""

import triton
import nbxtensor.ops._launch_seam  # noqa: F401  this module's launches go through nbxtensor's launcher
import triton.language as tl

from nbxtensor.ops.ew_tile import synthetic, tile_dependent, tile_independent  # noqa: F401  (ops/ew_tile.py)


#: the extent of the gathered operand along the gather dim in a synthetic launch (`_gather_recipe`): every index is
#: drawn in [0, this)
SYNTH_INDEXED_EXTENT = 3


def radix_strides(S, names, extents):
    """Strides for an operand read as (outer, dim, inner) in a synthetic launch (`ops/ew_synthetic.py`): `names` the
    stride parameters outer to inner, `extents` the dims' extents outer to inner (the outermost one unused). Chosen
    from the innermost outward, each the smallest value honouring its parameter's specialization that clears the span
    of the dims inside it, so the positions of the operand never collide (an injective layout: the contiguous one when
    the specialization allows it)."""
    strides, span = [0] * len(names), 0
    for k in reversed(range(len(names))):
        strides[k] = S.int(names[k], span + 1)
        if k:
            span += (int(extents[k]) - 1) * strides[k]
    return strides


def outer_kept_at_zero(S, outer_strides, inner, dim):
    """`inner_size` for a synthetic launch of this family: `inner` as drawn, or, when the variant has one of the
    `outer_strides` i64 (a value past 2^31, `ew_synthetic.Synth.carry`), the smallest in its specialization with
    dim x inner covering the ladder's largest count, so the outer index is 0 on every lane and the wide stride is
    never dereferenced."""
    if not any(S.ty(k) == "i64" for k in outer_strides if k in S.spec):
        return inner
    return S.int("inner_size", max(int(inner), -(-int(S.most) // int(dim))))


def flat_reach(n, dim, inner, s_outer, s_dim, s_inner):
    """1 + the largest offset `s_outer * o + s_dim * d + s_inner * i` over the flat positions 0 .. n - 1 decomposed as
    the kernels of this family decompose them (i = off % inner, d = off // inner % dim, o = off // inner // dim):
    the extent an operand addressed so must have, no more."""
    import numpy as np
    off = np.arange(int(n), dtype=np.int64)
    i, t = off % inner, off // inner
    return int((t // dim * s_outer + t % dim * s_dim + i * s_inner).max()) + 1


def _gather_recipe(S):
    """The synthetic launch of `gather_kernel` (`ops/ew_synthetic.py`): N = the ladder's count positions of the index
    and of the output, read as (outer, dim_size, inner_size) with dim_size and inner_size small and honouring their
    specialization (outer = cdiv(N, dim_size * inner_size), the last outer slab partial, as N's own tail);
    `index` and `out` laid out over (outer, dim_size, inner_size), `inp` over (outer, SYNTH_INDEXED_EXTENT,
    inner_size), each through its stride parameters (`radix_strides`: no two positions of the output collide, so the
    result is the same at every tile). Each buffer's extent is 1 + the largest offset the kernel forms over its N
    positions (`flat_reach`), the input's with the largest index an in-range index can take; every index drawn in
    [0, SYNTH_INDEXED_EXTENT). The grid is the call site's: cdiv(N, tile) on one axis."""
    n, k = S.n, SYNTH_INDEXED_EXTENT
    inner = S.int("inner_size", 3)
    dim = S.int("dim_size", 2)
    inner = outer_kept_at_zero(S, ["idx_stride_outer", "out_stride_outer", "inp_stride_outer"], inner, dim)
    outer = -(-n // (dim * inner))
    io, idd, ii = radix_strides(S, ["idx_stride_outer", "idx_stride_dim", "idx_stride_inner"], [outer, dim, inner])
    oo, od, oi = radix_strides(S, ["out_stride_outer", "out_stride_dim", "out_stride_inner"], [outer, dim, inner])
    po, pk, pi = radix_strides(S, ["inp_stride_outer", "inp_dim_stride", "inp_stride_inner"], [outer, k, inner])
    args = {"N": n, "outer_size": S.int("outer_size", outer), "dim_size": dim, "inner_size": inner,
            "idx_stride_outer": io, "idx_stride_dim": idd, "idx_stride_inner": ii,
            "out_stride_outer": oo, "out_stride_dim": od, "out_stride_inner": oi,
            "inp_stride_outer": po, "inp_dim_stride": pk, "inp_stride_inner": pi}
    args["index"] = S.index("index", flat_reach(n, dim, inner, io, idd, ii), k)
    args["out"] = S.buf("out", flat_reach(n, dim, inner, oo, od, oi))
    args["inp"] = S.buf("inp", flat_reach(n, dim, inner, po, 0, pi) + (k - 1) * pk)
    return args, None


_gather_recipe.count = "N"
_gather_recipe.description = "gather(N over (outer, dim_size, inner_size), indices < 3)"


@synthetic(_gather_recipe)
@tile_independent
@triton.jit
def gather_kernel(
    inp, index, out,
    N,
    inp_dim_stride,
    idx_stride_outer, idx_stride_dim, idx_stride_inner,
    inp_stride_outer, inp_stride_inner,
    out_stride_outer, out_stride_dim, out_stride_inner,
    outer_size, dim_size, inner_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Gather kernel — works for any tensor rank via pre-computed strides.

    The wrapper reshapes the problem into (outer, dim, inner) where:
    - outer = product of dims before gather dim
    - dim = gather dim size (from index tensor)
    - inner = product of dims after gather dim
    """
    pid = tl.program_id(0).to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < N

    # Decompose flat offset into (outer, dim, inner)
    inner_idx = offset % inner_size
    temp = offset // inner_size
    dim_idx = temp % dim_size
    outer_idx = temp // dim_size

    # Load the gather index value
    idx_offset = outer_idx * idx_stride_outer + dim_idx * idx_stride_dim + inner_idx * idx_stride_inner
    cur_index = tl.load(index + idx_offset, mask=mask, other=0)

    # Load from input: same outer/inner position, but use cur_index for the gather dim
    inp_offset = outer_idx * inp_stride_outer + cur_index * inp_dim_stride + inner_idx * inp_stride_inner
    val = tl.load(inp + inp_offset, mask=mask, other=0.0)

    # Store to output
    out_offset = outer_idx * out_stride_outer + dim_idx * out_stride_dim + inner_idx * out_stride_inner
    tl.store(out + out_offset, val, mask=mask)
