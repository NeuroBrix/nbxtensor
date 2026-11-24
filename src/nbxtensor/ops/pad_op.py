# Rewritten by NeuroBrix (Copyright NeuroBrix, Apache-2.0); derived from FlagGems (Apache-2.0), original notice kept
# Change: rewritten as a pure Triton kernel with no PyTorch import.
# Original notice: the provenance statement in the docstring below, unchanged, and THIRD_PARTY_NOTICES.md.
"""Constant padding — pure @triton.jit kernel.

Handles constant_pad_nd for the common 4D case (N,C,H,W) and generic N-D
via a flat-index approach. For each output element, compute the corresponding
input coordinate. If in bounds, load from input; otherwise, store pad_value.

Logic derived from FlagGems pad.py codegen approach.
"""

import triton
import nbxtensor.ops._launch_seam  # noqa: F401  this module's launches go through nbxtensor's launcher
import triton.language as tl

from nbxtensor.ops.ew_tile import synthetic, tile_dependent, tile_independent  # noqa: F401  (ops/ew_tile.py)
from nbxtensor.ops.ew_synthetic import reach  # noqa: E402  helpers of the synthetic recipes


# ---------------------------------------------------------------------------
# synthetic launches of the tile proof (ops/ew_synthetic.py)
#
# Every pad kernel walks `total` output elements, written contiguous (the output's extent is the count); the
# outermost index (rows, batch) is implicit in the count, so the dimensions are drawn small in each parameter's own
# specialization and the count decides how many rows the launch covers (its last row may be partial: a masked tail).
# The input's extent is 1 + the largest offset a lane below the count loads, from the op's index map: what the
# launch legitimately reads, no more (a masked lane loading past it is a fault).
# ---------------------------------------------------------------------------
def _pad_before(S, name):
    """A leading pad in `name`'s specialization: 0 where the variant has it divisible by 16 (an unpadded side, the
    common case), else the smallest value it allows."""
    return S.carry(name, 0) if S.attr(name) == "D" else S.int(name, 2)


def _cover(S, dims, n):
    """`dims` ([name, value], outermost first) grown, outermost growable one first, in its own specialization, so
    their product covers `n`: the launch's count fits in one row when the row count is the constant 1."""
    for i, (name, _v) in enumerate(dims):
        if isinstance(S.attr(name), tuple):
            continue
        rest = 1
        for j, (_n, v) in enumerate(dims):
            rest *= v if j != i else 1
        dims[i][1] = S.int(name, max(dims[i][1], -(-n // rest)))
        break
    return dims


def _pad_nd_dims(S):
    """The five-dim launch shared by the constant and replication pads: dims 1-4 small in their specialization (an
    extent, a leading pad, the output extent = pad + extent + one element of right pad), the outer dim 0 as many
    rows as the count needs (dims 1-4 cover the count when the variant has `out0` the constant 1); the input strides
    a non-overlapping layout of the input extents (a stride of an extent-1 dim divisible by 16 is 0, as the wrapper
    passes for the dims a tensor lacks)."""
    n = S.n
    ins, pbs, outs, xs = [0] * 5, [0] * 5, [0] * 5, [0] * 5
    for k in range(1, 5):
        ins[k], pbs[k] = S.int(f"in{k}", 3), _pad_before(S, f"pb{k}")
        outs[k] = S.int(f"out{k}", ins[k] + pbs[k] + 1)
    if S.attr("out0") == ("constexpr", 1) or S.ty("xs0") == "i64":      # an i64 xs0 (past 2^31) is never dereferenced
        dims = _cover(S, [[f"out{k}", outs[k]] for k in range(1, 5)], n)
        outs[1:] = [v for _n, v in dims]
    plane = outs[1] * outs[2] * outs[3] * outs[4]
    rows = -(-n // plane)
    pbs[0] = _pad_before(S, "pb0")
    ins[0] = S.int("in0", max(rows - pbs[0], 2))
    outs[0] = S.int("out0", max(rows, ins[0] + pbs[0]))
    span = 1
    for k in range(4, -1, -1):
        name = f"xs{k}"
        xs[k] = 0 if (ins[k] == 1 and S.attr(name) == "D") else S.int(name, span)
        span += (ins[k] - 1) * xs[k]
    args = {"total": n}
    for k in range(5):
        args.update({f"in{k}": ins[k], f"out{k}": outs[k], f"xs{k}": xs[k], f"pb{k}": pbs[k]})
    return args, ins, pbs, outs, xs


def _pad_nd_coords(n, outs):
    """Each output lane's five coordinates (the outer one unbounded, as the kernel decomposes)."""
    import numpy as np
    t, o = np.arange(n, dtype=np.int64), [None] * 5
    for k in (4, 3, 2, 1):
        o[k], t = t % outs[k], t // outs[k]
    o[0] = t
    return o


def _pad_2d_recipe(S):
    """The synthetic launch of `constant_pad_2d_kernel`: `total` = the ladder's count, planes of `out_h` x `out_w`
    (top / left pads + the input plane + one row and column of right pad), `channels_times_batch` planes enough to
    hold the count (the plane covers it when the variant has them the constant 1); the stride arguments, which the
    kernel does not read (it assumes contiguous), the contiguous ones in their specialization. The output `total`
    elements, the input 1 + its largest loaded offset; grid `cdiv(total, tile)`."""
    import numpy as np
    n = S.n
    in_h, in_w = S.int("in_h", 3), S.int("in_w", 3)
    pad_top, pad_left = _pad_before(S, "pad_top"), _pad_before(S, "pad_left")
    dims = [["out_h", S.int("out_h", in_h + pad_top + 1)], ["out_w", S.int("out_w", in_w + pad_left + 1)]]
    if S.attr("channels_times_batch") == ("constexpr", 1):
        _cover(S, dims, n)
    out_h, out_w = dims[0][1], dims[1][1]
    planes = S.int("channels_times_batch", -(-n // (out_h * out_w)))
    args = {"total": n, "in_h": in_h, "in_w": in_w, "out_h": out_h, "out_w": out_w,
            "channels_times_batch": planes, "pad_left": pad_left, "pad_top": pad_top}
    for pre, h, w in (("x_stride_", in_h, in_w), ("out_stride_", out_h, out_w)):
        args[pre + "w"] = S.int(pre + "w", 1)
        args[pre + "h"] = S.int(pre + "h", w)
        args[pre + "c"] = S.int(pre + "c", h * w)
        args[pre + "n"] = S.int(pre + "n", planes * h * w)
    off = np.arange(n, dtype=np.int64)
    ow, t = off % out_w, off // out_w
    oh, nc = t % out_h, t // out_h
    ih, iw = oh - pad_top, ow - pad_left
    live = (ih >= 0) & (ih < in_h) & (iw >= 0) & (iw < in_w) & (nc < planes)
    ext = reach((nc * in_h * in_w + ih * in_w + iw)[live])
    args.update(x_ptr=S.buf("x_ptr", ext), output_ptr=S.buf("output_ptr", n))
    return args, None


_pad_2d_recipe.count = "total"
_pad_2d_recipe.description = "pad 2d: total planes of out_h x out_w, input = 1 + largest loaded offset"


@synthetic(_pad_2d_recipe)
@triton.jit
def constant_pad_2d_kernel(
    x_ptr, output_ptr,
    total,
    in_h, in_w, out_h, out_w,
    channels_times_batch,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    pad_left, pad_top,
    pad_value,
    BLOCK_SIZE: tl.constexpr,
    WIDE: tl.constexpr = True,
):
    """Constant pad for 4D tensors (N,C,H,W). Most common case."""
    if WIDE:
        offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    else:
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 32-bit below 2^30 (`_index_wide`): Apple emulates 64-bit % and //
    mask = offsets < total

    # Decompose flat index into (n*c, h, w)
    ow = offsets % out_w
    temp = offsets // out_w
    oh = temp % out_h
    nc = temp // out_h

    # Map output coords to input coords
    iw = ow - pad_left
    ih = oh - pad_top

    # Check if within input bounds
    in_bounds = (ih >= 0) & (ih < in_h) & (iw >= 0) & (iw < in_w) & (nc < channels_times_batch)

    # Flat stride approach: assumes contiguous input (wrapper ensures contiguous)
    inp_offset = nc * in_h * in_w + ih * in_w + iw

    val = tl.load(x_ptr + inp_offset, mask=mask & in_bounds, other=pad_value)
    out_offset = nc * out_h * out_w + oh * out_w + ow
    tl.store(output_ptr + out_offset, val, mask=mask)

def _pad_1d_recipe(S):
    """The synthetic launch of `constant_pad_1d_kernel`: `total` = the ladder's count, rows of `out_size` = left pad
    + `in_size` + a right pad, `batch_size` rows enough to hold the count (one row when the variant has it the
    constant 1); the output `total` elements, the input 1 + its largest loaded offset; grid `cdiv(total, tile)`."""
    import numpy as np
    n = S.n
    in_size, pad_left = S.int("in_size", 3), _pad_before(S, "pad_left")
    dims = [["out_size", S.int("out_size", in_size + pad_left + 1)]]
    if S.attr("batch_size") == ("constexpr", 1):
        _cover(S, dims, n)
    out_size = dims[0][1]
    batch = S.int("batch_size", -(-n // out_size))
    off = np.arange(n, dtype=np.int64)
    b, i = off // out_size, off % out_size - pad_left
    live = (i >= 0) & (i < in_size) & (b < batch)
    ext = reach((b * in_size + i)[live])
    return ({"total": n, "in_size": in_size, "out_size": out_size, "batch_size": batch, "pad_left": pad_left,
             "x_ptr": S.buf("x_ptr", ext), "output_ptr": S.buf("output_ptr", n)}, None)


_pad_1d_recipe.count = "total"
_pad_1d_recipe.description = "pad 1d: total rows of out_size, input = 1 + largest loaded offset"


@synthetic(_pad_1d_recipe)
@tile_independent
@triton.jit
def constant_pad_1d_kernel(
    x_ptr, output_ptr,
    total,
    in_size, out_size,
    batch_size,
    pad_left,
    pad_value,
    BLOCK_SIZE: tl.constexpr,
    WIDE: tl.constexpr = True,
):
    """Constant pad for 2D/3D tensors along last dim."""
    if WIDE:
        offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    else:
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 32-bit below 2^30 (`_index_wide`): Apple emulates 64-bit % and //
    mask = offsets < total

    out_idx = offsets % out_size
    batch_idx = offsets // out_size

    in_idx = out_idx - pad_left
    in_bounds = (in_idx >= 0) & (in_idx < in_size) & (batch_idx < batch_size)

    inp_offset = batch_idx * in_size + in_idx
    val = tl.load(x_ptr + inp_offset, mask=mask & in_bounds, other=pad_value)

    out_offset = batch_idx * out_size + out_idx
    tl.store(output_ptr + out_offset, val, mask=mask)


def _pad_nd_recipe(S):
    """The synthetic launch of `constant_pad_nd_kernel` (`_pad_nd_dims`): `total` = the ladder's count; the output
    `total` elements (written contiguous), the input 1 + the largest offset a lane inside the input loads through its
    strides; grid `cdiv(total, tile)`."""
    import numpy as np
    args, ins, pbs, outs, xs = _pad_nd_dims(S)
    o = _pad_nd_coords(S.n, outs)
    i = [o[k] - pbs[k] for k in range(5)]
    live = np.ones(S.n, dtype=bool)
    for k in range(5):
        live &= (i[k] >= 0) & (i[k] < ins[k])
    ext = reach((sum(i[k] * xs[k] for k in range(5)))[live])
    args.update(x_ptr=S.buf("x_ptr", ext), output_ptr=S.buf("output_ptr", S.n))
    return args, None


_pad_nd_recipe.count = "total"
_pad_nd_recipe.description = "constant pad nd: 5 dims, rows from total, strided input = 1 + largest loaded offset"


@synthetic(_pad_nd_recipe)
@tile_independent
@triton.jit
def constant_pad_nd_kernel(
    x_ptr, output_ptr,
    total,
    in0, in1, in2, in3, in4,
    out0, out1, out2, out3, out4,
    xs0, xs1, xs2, xs3, xs4,
    pb0, pb1, pb2, pb3, pb4,
    pad_value,
    BLOCK_SIZE: tl.constexpr,
    WIDE: tl.constexpr = True,
):
    """Constant pad of up to 5 dims in ONE pass: each output element maps to its input element (any dim padded,
    a middle one included) or to `pad_value`. Dims a tensor lacks are given extent 1 and pad 0 by the wrapper, so one
    kernel serves 1-D to 5-D. The input is read through its strides (xs*); the output is written contiguous."""
    if WIDE:
        offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    else:
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 32-bit below 2^30 (`_index_wide`): Apple emulates 64-bit % and //
    mask = offsets < total
    o4 = offsets % out4
    t = offsets // out4
    o3 = t % out3
    t = t // out3
    o2 = t % out2
    t = t // out2
    o1 = t % out1
    o0 = t // out1
    i0 = o0 - pb0
    i1 = o1 - pb1
    i2 = o2 - pb2
    i3 = o3 - pb3
    i4 = o4 - pb4
    inside = ((i0 >= 0) & (i0 < in0) & (i1 >= 0) & (i1 < in1) & (i2 >= 0) & (i2 < in2)
              & (i3 >= 0) & (i3 < in3) & (i4 >= 0) & (i4 < in4))
    src = i0 * xs0 + i1 * xs1 + i2 * xs2 + i3 * xs3 + i4 * xs4
    val = tl.load(x_ptr + src, mask=mask & inside, other=pad_value)
    tl.store(output_ptr + offsets, val, mask=mask)


def _replication_nd_recipe(S):
    """The synthetic launch of `replication_pad_nd_kernel` (`_pad_nd_dims`): `total` = the ladder's count; the output
    `total` elements, the input 1 + the largest offset of the clamped positions every lane below the count loads;
    grid `cdiv(total, tile)`."""
    import numpy as np
    args, ins, pbs, outs, xs = _pad_nd_dims(S)
    o = _pad_nd_coords(S.n, outs)
    src = sum(np.clip(o[k] - pbs[k], 0, ins[k] - 1) * xs[k] for k in range(5))
    ext = reach(src)
    args.update(x_ptr=S.buf("x_ptr", ext), output_ptr=S.buf("output_ptr", S.n))
    return args, None


_replication_nd_recipe.count = "total"
_replication_nd_recipe.description = "replication pad nd: 5 dims, rows from total, input = 1 + largest clamped offset"


@synthetic(_replication_nd_recipe)
@tile_independent
@triton.jit
def replication_pad_nd_kernel(
    x_ptr, output_ptr,
    total,
    in0, in1, in2, in3, in4,
    out0, out1, out2, out3, out4,
    xs0, xs1, xs2, xs3, xs4,
    pb0, pb1, pb2, pb3, pb4,
    BLOCK_SIZE: tl.constexpr,
    WIDE: tl.constexpr = True,
):
    """Replication pad of up to 5 dims in ONE pass (PyTorch `replicate`): each output element reads the input element
    at its position clamped into the input, so a padded border repeats the edge. The twin of `constant_pad_nd_kernel`,
    the same index map with a clamp where that one masks; the input is read through its strides, the output written
    contiguous."""
    if WIDE:
        offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    else:
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 32-bit below 2^30 (`_index_wide`)
    mask = offsets < total
    o4 = offsets % out4
    t = offsets // out4
    o3 = t % out3
    t = t // out3
    o2 = t % out2
    t = t // out2
    o1 = t % out1
    o0 = t // out1
    i0 = tl.minimum(tl.maximum(o0 - pb0, 0), in0 - 1)
    i1 = tl.minimum(tl.maximum(o1 - pb1, 0), in1 - 1)
    i2 = tl.minimum(tl.maximum(o2 - pb2, 0), in2 - 1)
    i3 = tl.minimum(tl.maximum(o3 - pb3, 0), in3 - 1)
    i4 = tl.minimum(tl.maximum(o4 - pb4, 0), in4 - 1)
    src = i0 * xs0 + i1 * xs1 + i2 * xs2 + i3 * xs3 + i4 * xs4
    val = tl.load(x_ptr + src, mask=mask)
    tl.store(output_ptr + offsets, val, mask=mask)
