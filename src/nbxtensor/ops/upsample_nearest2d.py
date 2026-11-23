# Rewritten by NeuroBrix (Copyright NeuroBrix, Apache-2.0); derived from FlagGems (Apache-2.0), original notice kept
# Change: rewritten as a pure Triton kernel with no PyTorch import.
# Original notice: the provenance statement in the docstring below, unchanged, and THIRD_PARTY_NOTICES.md.
"""Upsample nearest 2D — pure @triton.jit kernel.

Extracted from FlagGems (FlagOpen/FlagGems) upsample_nearest2d.py.
Stripped FlagGems-specific decorators (@libentry, runtime heuristics).
"""

import triton
import nbxtensor.ops._launch_seam  # noqa: F401  this module's launches go through nbxtensor's launcher
import triton.language as tl


from nbxtensor.ops.ew_tile import synthetic, tile_dependent, tile_independent  # noqa: F401  (ops/ew_tile.py)


def _nearest_index(out_extent, in_extent):
    """The largest input index the kernel's nearest map reaches over [0, out_extent): min(trunc(o * fp32(in / out)),
    in - 1), in fp32 as the kernel computes it (the scale passed as the call site passes it, in / out)."""
    import numpy as np
    scale = np.float32(in_extent / out_extent)
    o = np.arange(out_extent, dtype=np.int64).astype(np.float32)
    return int(np.minimum((o * scale).astype(np.int32), in_extent - 1).max()), float(in_extent / out_extent)


def _upsample_nearest2d_recipe(S):
    """The synthetic launch of `upsample_nearest2d_kernel`, whose tiled axis is the output plane OH*OW (no count
    parameter; the lanes past the plane wrap inside it, unmasked): the ladder walks OW's specialization, OW the
    smallest in it with OH*OW at least the ladder's count, OH = 2*IH, IW about OW/2 (an upsample), N, C small (N*C
    above 4: the channel loop strides across programs), the input read through strides of a non-overlapping layout;
    the scales IH/OH, IW/OW as the call site computes them. The output N*C*OH*OW elements (contiguous), the input 1
    + the largest offset the nearest map reaches; grid (cdiv(OH*OW, tile), cdiv(N*C, 4)) as the call site."""
    from nbxtensor.ops.elementwise_launch import per_tile
    N, C, IH = S.int("N", 1), S.int("C", 5), S.int("IH", 2)
    OH = S.int("OH", 2 * IH)
    OW = S.int("OW", -(-S.n // OH))
    IW = S.int("IW", max(2, OW // 2))
    s_w = S.int("s_w", 1)
    s_h = S.int("s_h", (IW - 1) * s_w + 1)
    plane = (IH - 1) * s_h + (IW - 1) * s_w + 1
    s_c = S.int("s_c", plane)
    s_n = S.int("s_n", (C - 1) * s_c + plane)
    ih, rh = _nearest_index(OH, IH)
    iw, rw = _nearest_index(OW, IW)
    ext = (N - 1) * s_n + (C - 1) * s_c + ih * s_h + iw * s_w + 1
    return ({"N": N, "C": C, "OH": OH, "OW": OW, "IH": IH, "IW": IW, "s_n": s_n, "s_c": s_c, "s_h": s_h, "s_w": s_w,
             "reciprocal_scale_h": rh, "reciprocal_scale_w": rw,
             "ptr_o": S.buf("ptr_o", N * C * OH * OW), "ptr_i": S.buf("ptr_i", ext)},
            (per_tile(OH * OW), -(-(N * C) // 4)))


_upsample_nearest2d_recipe.count = "OW"
_upsample_nearest2d_recipe.description = "upsample nearest 2d: plane OH*OW ~ ladder (OW's spec), grid (per_tile(OH*OW), cdiv(NC,4))"


@synthetic(_upsample_nearest2d_recipe)
@tile_independent
@triton.jit
def upsample_nearest2d_kernel(
    ptr_o,
    ptr_i,
    N,
    C,
    OH,
    OW,
    IH,
    IW,
    s_n,
    s_c,
    s_h,
    s_w,
    reciprocal_scale_h,
    reciprocal_scale_w,
    BLOCK_SIZE: tl.constexpr,
):
    NC = N * C
    nc_stride = tl.num_programs(axis=1)
    nc_iter = (tl.program_id(axis=1).to(tl.int64)).to(tl.int32)   # loop bounds are 32-bit counts: the Metal lowering refuses a 64-bit scf.for bound (addressing stays 64-bit through the program ids)
    pid = tl.program_id(axis=0).to(tl.int64)
    idx = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    ow = idx % OW
    oh = idx // OW % OH

    ih = tl.minimum((oh * reciprocal_scale_h).to(tl.int32), IH - 1)
    iw = tl.minimum((ow * reciprocal_scale_w).to(tl.int32), IW - 1)

    # The input is read through its strides: a first-frame branch feeds this op `x[:, :, 0]` of a
    # 5-D activation, a view whose channel stride spans every frame — flat indexing read another
    # channel's frames there (CogVideoX / Open-Sora frame 0). The output is fresh and contiguous.
    offset_o = (nc_iter * OH + oh) * OW + ow
    in_hw = ih.to(tl.int64) * s_h + iw.to(tl.int64) * s_w
    dst_index_stride = nc_stride * OH * OW
    while nc_iter < NC:
        n = nc_iter // C
        c = nc_iter % C
        data = tl.load(ptr_i + n.to(tl.int64) * s_n + c.to(tl.int64) * s_c + in_hw)
        tl.store(ptr_o + offset_o, data)
        ptr_o += dst_index_stride
        nc_iter += nc_stride
