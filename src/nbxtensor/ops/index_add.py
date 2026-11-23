"""Index add along dim — deterministic, pure @triton.jit.

out[..., index[i], ...] += alpha * src[..., i, ...]

The previous implementation used `tl.atomic_add`, whose inter-block
float accumulation order is non-deterministic. When several source
positions map to the same destination (MoE expert aggregation: many
routed rows -> one token), repeated runs produced different fp32
roundings -> a different greedy argmax on Qwen3-30B::triton's unfused
MoE path (workstream P-TRITON-MOE-DETERMINISM sub-workstream 2).

State of the art consulted (R16, 2026-05-19):
  - Triton #7402: tl.atomic_add return value is incorrect across
    threads (only thread 0 correct) — open upstream bug.
  - Triton #4717: tl.atomic_add is slow (layout conversions).
  - flagos-ai/FlagGems ops/index_add.py still uses
    tl.atomic_add(sem='relaxed') (pins triton==3.3.0, no bit-exact
    test) — no deterministic upstream to vendor.
  - PyTorch index_add_/scatter_add_ are non-deterministic on CUDA via
    atomicAdd; the deterministic path is sort + sequential segmented
    reduce.

Strategy — output-owner gather (deterministic by construction):
invert the scatter. Each output cell is owned by exactly ONE program
that sums its contributing source rows in a fixed ascending source-
position order as a SEQUENTIAL fp32 fold. No atomics, no inter-block
races, no sort dependency. The ascending-j sequential fold is exactly
the order torch's deterministic CUDA index_add_ uses (verified
bit-exact fp32 for bucket sizes up to 4096). A pairwise/tree reduce
(`tl.sum`) would round differently and break bit-exactness, so the
j accumulation is a scalar loop; the feature (inner) dim is vectorised.

Out-of-range source indices never equal a valid destination d in
[0, inp_shape_dim) so they are naturally dropped (mirrors the old
masked-store semantics — no assert/crash). Model-agnostic, no autotune.
"""

import triton
import nbxtensor.ops._launch_seam  # noqa: F401  this module's launches go through nbxtensor's launcher
import triton.language as tl


from nbxtensor.ops.ew_tile import synthetic, tile_dependent, tile_independent  # noqa: F401  (ops/ew_tile.py)


#: the outer extent of a synthetic launch of `index_add_gather_kernel` (not an argument of the kernel: the grid's)
SYNTH_OUTER = 2


def _index_add_recipe(S):
    """The synthetic launch of `index_add_gather_kernel` (`ops/ew_synthetic.py`): inner_size = the ladder's count,
    SYNTH_OUTER outer slabs, dim_size source rows and inp_shape_dim destination rows (small, honouring their
    specialization). `x` and `out` are the contiguous (outer, inp_shape_dim, inner_size) the kernel addresses,
    outer * inp_shape_dim * inner_size elements; `index` holds dim_size destinations in [0, inp_shape_dim) (every
    source row is then read by the program of its destination); `src` is laid out over (outer, dim_size, inner_size)
    through its stride parameters (`gather_op.radix_strides`), its extent 1 + the largest offset the kernel forms.
    The grid is the call site's: outer * inp_shape_dim x cdiv(inner_size, tile) on one axis."""
    from nbxtensor.ops.elementwise_launch import per_tile
    from nbxtensor.ops.gather_op import radix_strides
    inner, outer = S.n, SYNTH_OUTER
    dim = S.int("dim_size", 3)
    rows = S.int("inp_shape_dim", 2)
    so, sd, si = radix_strides(S, ["src_stride_outer", "src_stride_dim", "src_stride_inner"], [outer, dim, inner])
    args = {"inner_size": inner, "dim_size": dim, "inp_shape_dim": rows,
            "src_stride_outer": so, "src_stride_dim": sd, "src_stride_inner": si,
            "x": S.buf("x", outer * rows * inner), "out": S.buf("out", outer * rows * inner),
            "index": S.index("index", dim, rows),
            "src": S.buf("src", (outer - 1) * so + (dim - 1) * sd + (inner - 1) * si + 1)}
    return args, (per_tile(inner, times=outer * rows),)


_index_add_recipe.count = "inner_size"
_index_add_recipe.description = "index_add(inner_size, outer 2 x inp_shape_dim rows, dim_size sources)"


@synthetic(_index_add_recipe)
@tile_independent
@triton.jit
def index_add_gather_kernel(
    x, src, index, out,
    dim_size, inp_shape_dim, inner_size, alpha,
    src_stride_outer, src_stride_dim, src_stride_inner,
    INNER_BLOCK: tl.constexpr,
):
    """One program == one (outer, dest, inner-tile) output cell.

    out[o, d, i] = x[o, d, i]
                 + alpha * sum_{j : index[j] == d} src[o, j, i]
    with the sum taken in ascending-j order as a sequential fp32 fold
    (bit-exact vs torch deterministic index_add_).
    """
    n_inner_tiles = (inner_size + INNER_BLOCK - 1) // INNER_BLOCK
    pid = tl.program_id(0).to(tl.int64)
    tile = pid % n_inner_tiles
    tmp = pid // n_inner_tiles
    d = tmp % inp_shape_dim
    o = tmp // inp_shape_dim

    inner = tile * INNER_BLOCK + tl.arange(0, INNER_BLOCK)
    inner_mask = inner < inner_size

    out_base = o * (inp_shape_dim * inner_size) + d * inner_size + inner
    base = tl.load(x + out_base, mask=inner_mask, other=0.0).to(tl.float32)

    # torch's deterministic CUDA index_add_ computes the per-destination
    # segment sum SEPARATELY (0-initialised, ascending-j sequential
    # fold) and adds it to `self` ONCE at the end — NOT folding into
    # self. Verified by discriminating probe (base=1, src=[1e8,-1e8]:
    # torch -> 1.0 == base + Σ, not 0.0 == fold-into-base).
    seg = tl.zeros([INNER_BLOCK], dtype=tl.float32)
    src_o = o * src_stride_outer + inner * src_stride_inner
    # Scan source positions in ascending j (deterministic order).
    # Scalar loop on purpose: a vectorised tree reduce would round
    # differently and break bit-exactness vs torch.
    for j in range(0, dim_size):
        dest = tl.load(index + j).to(tl.int64)
        if dest == d:
            sv = tl.load(src + src_o + j * src_stride_dim,
                         mask=inner_mask, other=0.0).to(tl.float32)
            seg += (alpha * sv).to(tl.float32)
            # NOTE alpha != 1: a residual 1-ULP fp32 gap vs torch
            # remains (Triton codegen of the scaled term — bitcast /
            # fma / reassoc barriers were tried, all no-effect). It is
            # NOT an ordering bug: numpy fp32 sequential fold == torch
            # bit-exact, and this kernel reproduces that for alpha == 1
            # bit-exact. alpha != 1 never occurs on a transformer path
            # (Qwen3-30B-A3B graph: 6144/6144 index_add use alpha=1).

    # Triton implicitly casts the fp32 result to out's element type on
    # store (same convention as the other elementwise kernels).
    tl.store(out + out_base, base + seg, mask=inner_mask)
