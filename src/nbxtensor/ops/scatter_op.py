# Rewritten by NeuroBrix (Copyright NeuroBrix, Apache-2.0); derived from FlagGems (Apache-2.0), original notice kept
# Change: rewritten as a pure Triton kernel with no PyTorch import.
# Original notice: the provenance statement in the docstring below, unchanged, and THIRD_PARTY_NOTICES.md.
"""Scatter along dim — pure @triton.jit kernel.

Approach from FlagGems: work on flattened index space. For each element,
decompose into (outer, dim, inner), load the scatter index, then store
(or atomic_add) into the destination at the indexed position.

scatter: dst[outer][index[outer][dim][inner]][inner] = src[outer][dim][inner]
scatter_add: dst[...] += src[...]
"""

import triton
import nbxtensor.ops._launch_seam  # noqa: F401  this module's launches go through nbxtensor's launcher
import triton.language as tl

from nbxtensor.ops.ew_tile import synthetic, tile_dependent, tile_independent  # noqa: F401  (ops/ew_tile.py)



def collapsed_strides(t, dim, index, op="scatter", bounded_on_dim=False):
    """The (outer, dim, inner) strides of the operand `t` as the kernels of this file (and gather's) address it: the
    walk is the INDEX's own positions, its flat position decomposed into outer (every dim before `dim`, flattened over
    the index's extents), dim, and inner (every dim after, flattened over the index's extents), and `t` is reached
    through its own strides at the index's coordinates. aten's contract: index.size(d) <= t.size(d) on every d other
    than `dim` (on `dim` too when `bounded_on_dim`, scatter's src). A collapsed group of dims is addressed by one stride
    exactly when `t`'s offset over it is linear in the flattened index position: an index dim of extent 1 always sits
    at 0 and costs nothing, so the group's stride is `t`'s stride on its innermost dim of index extent > 1, and every
    other such dim must step `t` by that stride times the index extents inside it. On a contiguous `t` that means: the
    same extent as the index on every dim after the group's outermost dim of index extent > 1, up to its innermost one.
    A layout breaking it (for example dim 2, index (2, 2, 1, 2) of (2, 3, 3, 2)) cannot be walked by the merged
    arithmetic and is refused by name, never addressed wrongly."""
    n = index.ndim
    shape, ishape = tuple(t.shape), tuple(index.shape)
    if t.ndim != n or any(ishape[d] > shape[d] for d in range(n) if d != dim or bounded_on_dim):
        raise RuntimeError(
            f"ZERO FALLBACK: {op} along dim {dim} reaches a tensor of shape {shape} through an index of shape {ishape}:"
            f" the index must have the tensor's rank and no larger an extent on any dim"
            f"{'' if bounded_on_dim else f' other than {dim}'}")

    def merged(dims):
        """The one stride walking `dims` (flattened over the index's extents) through `t`, or None when none does."""
        live = [d for d in dims if ishape[d] > 1]
        if not live:
            return 0
        step = t.stride(live[-1])
        for d in live:
            inside = 1
            for e in range(d + 1, live[-1] + 1):
                inside *= ishape[e]
            if t.stride(d) != step * inside:
                return None
        return step

    outer, inner = merged(range(dim)), merged(range(dim + 1, n))
    if outer is None or inner is None:
        group = "before" if outer is None else "after"
        raise RuntimeError(
            f"ZERO FALLBACK: {op} along dim {dim} cannot walk a tensor of shape {shape} (strides "
            f"{tuple(t.stride(d) for d in range(n))}) through an index of shape {ishape}: on the dims {group} {dim} "
            f"the tensor's extent differs from the index's after the group's outermost dim of index extent > 1, so no "
            f"single (outer, dim, inner) stride addresses it exactly")
    return outer, t.stride(dim), inner


def _scatter_recipe(S):
    """The synthetic launch of the scatter kernels of this file — one signature: `scatter_kernel`, `scatter_add_kernel`,
    `scatter_reduce_amax_kernel`, `scatter_reduce_amin_kernel` (`ops/ew_synthetic.py`): N = the ladder's count
    positions of the index and of the source, read as (outer, dim_size, inner_size) with dim_size and inner_size small
    and honouring their specialization (outer = cdiv(N, dim_size * inner_size), the last outer slab partial, as N's
    own tail); `index` and `src` laid out over (outer, dim_size, inner_size), `out` over (outer,
    SYNTH_INDEXED_EXTENT, inner_size), each through its stride parameters (`gather_op.radix_strides`). Each buffer's
    extent is 1 + the largest offset the kernel forms over its N positions (`gather_op.flat_reach`), the output's
    with the largest index an in-range index can take; every index drawn in [0, SYNTH_INDEXED_EXTENT). The grid is
    the call site's: cdiv(N, tile) on one axis."""
    from nbxtensor.ops.gather_op import SYNTH_INDEXED_EXTENT, flat_reach, radix_strides
    n, k = S.n, SYNTH_INDEXED_EXTENT
    inner = S.int("inner_size", 3)
    dim = S.int("dim_size", 2)
    outer = -(-n // (dim * inner))
    io, idd, ii = radix_strides(S, ["idx_stride_outer", "idx_stride_dim", "idx_stride_inner"], [outer, dim, inner])
    so, sd, si = radix_strides(S, ["src_stride_outer", "src_stride_dim", "src_stride_inner"], [outer, dim, inner])
    oo, ok, oi = radix_strides(S, ["out_stride_outer", "out_dim_stride", "out_stride_inner"], [outer, k, inner])
    args = {"N": n, "outer_size": S.int("outer_size", outer), "dim_size": dim, "inner_size": inner,
            "idx_stride_outer": io, "idx_stride_dim": idd, "idx_stride_inner": ii,
            "src_stride_outer": so, "src_stride_dim": sd, "src_stride_inner": si,
            "out_stride_outer": oo, "out_dim_stride": ok, "out_stride_inner": oi}
    args["index"] = S.index("index", flat_reach(n, dim, inner, io, idd, ii), k)
    args["src"] = S.buf("src", flat_reach(n, dim, inner, so, sd, si))
    args["out"] = S.buf("out", flat_reach(n, dim, inner, oo, 0, oi) + (k - 1) * ok)
    return args, None


_scatter_recipe.count = "N"
_scatter_recipe.description = "scatter(N over (outer, dim_size, inner_size), indices < 3)"


@synthetic(_scatter_recipe)
@tile_dependent("duplicate indices write one address: the last writer wins, an order")
@triton.jit
def scatter_kernel(
    src, index, out,
    N,
    out_dim_stride,
    src_stride_outer, src_stride_dim, src_stride_inner,
    idx_stride_outer, idx_stride_dim, idx_stride_inner,
    out_stride_outer, out_stride_inner,
    outer_size, dim_size, inner_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Scatter: out[outer][index_val][inner] = src[outer][dim][inner]."""
    pid = tl.program_id(0).to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < N

    inner_idx = offset % inner_size
    temp = offset // inner_size
    dim_idx = temp % dim_size
    outer_idx = temp // dim_size

    # Load scatter index
    idx_offset = outer_idx * idx_stride_outer + dim_idx * idx_stride_dim + inner_idx * idx_stride_inner
    cur_index = tl.load(index + idx_offset, mask=mask, other=0)

    # Load source value
    src_offset = outer_idx * src_stride_outer + dim_idx * src_stride_dim + inner_idx * src_stride_inner
    val = tl.load(src + src_offset, mask=mask, other=0.0)

    # Store to output at scattered position
    out_offset = outer_idx * out_stride_outer + cur_index * out_dim_stride + inner_idx * out_stride_inner
    tl.store(out + out_offset, val, mask=mask)

@synthetic(_scatter_recipe)
@tile_dependent("two lanes may add into one address (an atomic accumulate)")
@triton.jit
def scatter_add_kernel(
    src, index, out,
    N,
    out_dim_stride,
    src_stride_outer, src_stride_dim, src_stride_inner,
    idx_stride_outer, idx_stride_dim, idx_stride_inner,
    out_stride_outer, out_stride_inner,
    outer_size, dim_size, inner_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Scatter add: out[outer][index_val][inner] += src[outer][dim][inner]."""
    pid = tl.program_id(0).to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < N

    inner_idx = offset % inner_size
    temp = offset // inner_size
    dim_idx = temp % dim_size
    outer_idx = temp // dim_size

    idx_offset = outer_idx * idx_stride_outer + dim_idx * idx_stride_dim + inner_idx * idx_stride_inner
    cur_index = tl.load(index + idx_offset, mask=mask, other=0)

    src_offset = outer_idx * src_stride_outer + dim_idx * src_stride_dim + inner_idx * src_stride_inner
    val = tl.load(src + src_offset, mask=mask, other=0.0)

    out_offset = outer_idx * out_stride_outer + cur_index * out_dim_stride + inner_idx * out_stride_inner
    tl.atomic_add(out + out_offset, val, mask=mask)

# ═══════════════════════════════════════════════════════════════════════════════
# scatter_reduce kernels — amax, amin using Triton native atomics
# ═══════════════════════════════════════════════════════════════════════════════

@synthetic(_scatter_recipe)
@tile_dependent("two lanes may reduce into one address (an atomic max)")
@triton.jit
def scatter_reduce_amax_kernel(
    src, index, out,
    N,
    out_dim_stride,
    src_stride_outer, src_stride_dim, src_stride_inner,
    idx_stride_outer, idx_stride_dim, idx_stride_inner,
    out_stride_outer, out_stride_inner,
    outer_size, dim_size, inner_size,
    BLOCK_SIZE: tl.constexpr,
):
    """scatter_reduce with reduce='amax': atomic max."""
    pid = tl.program_id(0).to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < N

    inner_idx = offset % inner_size
    temp = offset // inner_size
    dim_idx = temp % dim_size
    outer_idx = temp // dim_size

    idx_offset = outer_idx * idx_stride_outer + dim_idx * idx_stride_dim + inner_idx * idx_stride_inner
    cur_index = tl.load(index + idx_offset, mask=mask, other=0)

    src_offset = outer_idx * src_stride_outer + dim_idx * src_stride_dim + inner_idx * src_stride_inner
    val = tl.load(src + src_offset, mask=mask, other=0.0)

    out_offset = outer_idx * out_stride_outer + cur_index * out_dim_stride + inner_idx * out_stride_inner
    tl.atomic_max(out + out_offset, val, mask=mask)

@synthetic(_scatter_recipe)
@tile_dependent("two lanes may reduce into one address (an atomic min)")
@triton.jit
def scatter_reduce_amin_kernel(
    src, index, out,
    N,
    out_dim_stride,
    src_stride_outer, src_stride_dim, src_stride_inner,
    idx_stride_outer, idx_stride_dim, idx_stride_inner,
    out_stride_outer, out_stride_inner,
    outer_size, dim_size, inner_size,
    BLOCK_SIZE: tl.constexpr,
):
    """scatter_reduce with reduce='amin': atomic min."""
    pid = tl.program_id(0).to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < N

    inner_idx = offset % inner_size
    temp = offset // inner_size
    dim_idx = temp % dim_size
    outer_idx = temp // dim_size

    idx_offset = outer_idx * idx_stride_outer + dim_idx * idx_stride_dim + inner_idx * idx_stride_inner
    cur_index = tl.load(index + idx_offset, mask=mask, other=0)

    src_offset = outer_idx * src_stride_outer + dim_idx * src_stride_dim + inner_idx * src_stride_inner
    val = tl.load(src + src_offset, mask=mask, other=0.0)

    out_offset = outer_idx * out_stride_outer + cur_index * out_dim_stride + inner_idx * out_stride_inner
    tl.atomic_min(out + out_offset, val, mask=mask)
