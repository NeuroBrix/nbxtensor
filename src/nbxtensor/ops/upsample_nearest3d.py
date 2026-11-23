"""Upsample nearest 3D — pure @triton.jit kernel.

3D nearest neighbor upsampling: [N,C,D,H,W] -> [N,C,D*s_d,H*s_h,W*s_w].
Adapted from our upsample_nearest2d.py by adding the depth dimension.
"""

import triton
import nbxtensor.ops._launch_seam  # noqa: F401  this module's launches go through nbxtensor's launcher
import triton.language as tl


from nbxtensor.ops.ew_tile import synthetic, tile_dependent, tile_independent  # noqa: F401  (ops/ew_tile.py)


def _upsample_nearest3d_recipe(S):
    """The synthetic launch of `upsample_nearest3d_kernel`, whose tiled axis is the output volume OD*OH*OW (no count
    parameter; the lanes past the volume wrap inside it, unmasked): the ladder walks OW's specialization, OW the
    smallest in it with OD*OH*OW at least the ladder's count, OD = 2*ID and OH = 2*IH below 16 (else equal: a
    divisible-by-16 depth or height kept at its smallest), IW about OW/2, N, C small (N*C above 4: the channel loop
    strides across programs); the scales ID/OD, IH/OH, IW/OW as the call site computes them. A contiguous input (the
    call site's) of 1 + the largest offset the nearest map reaches, the output N*C*OD*OH*OW elements; grid
    (cdiv(OD*OH*OW, tile), cdiv(N*C, 4)) as the call site."""
    from nbxtensor.ops.elementwise_launch import per_tile
    from nbxtensor.ops.upsample_nearest2d import _nearest_index
    N, C, ID, IH = S.int("N", 1), S.int("C", 5), S.int("ID", 2), S.int("IH", 2)
    OD = S.int("OD", 2 * ID if ID < 16 else ID)
    OH = S.int("OH", 2 * IH if IH < 16 else IH)
    OW = S.int("OW", -(-S.n // (OD * OH)))
    IW = S.int("IW", max(2, OW // 2))
    id_, rd = _nearest_index(OD, ID)
    ih, rh = _nearest_index(OH, IH)
    iw, rw = _nearest_index(OW, IW)
    ext = (((N * C - 1) * ID + id_) * IH + ih) * IW + iw + 1
    return ({"N": N, "C": C, "OD": OD, "OH": OH, "OW": OW, "ID": ID, "IH": IH, "IW": IW,
             "reciprocal_scale_d": rd, "reciprocal_scale_h": rh, "reciprocal_scale_w": rw,
             "ptr_o": S.buf("ptr_o", N * C * OD * OH * OW), "ptr_i": S.buf("ptr_i", ext)},
            (per_tile(OD * OH * OW), -(-(N * C) // 4)))


_upsample_nearest3d_recipe.count = "OW"
_upsample_nearest3d_recipe.description = "upsample nearest 3d: volume OD*OH*OW ~ ladder (OW's spec), grid (per_tile, cdiv(NC,4))"


@synthetic(_upsample_nearest3d_recipe)
@tile_independent
@triton.jit
def upsample_nearest3d_kernel(
    ptr_o,
    ptr_i,
    N,
    C,
    OD,
    OH,
    OW,
    ID,
    IH,
    IW,
    reciprocal_scale_d,
    reciprocal_scale_h,
    reciprocal_scale_w,
    BLOCK_SIZE: tl.constexpr,
):
    """Nearest-neighbor 3D upsample.

    Grid: (cdiv(OD*OH*OW, BLOCK_SIZE), cdiv(N*C, 4))
    Each program handles BLOCK_SIZE output pixels across one (n,c) slice,
    iterating over (n,c) in strides of nc_stride.
    """
    NC = N * C
    nc_stride = tl.num_programs(axis=1)
    nc_iter = (tl.program_id(axis=1).to(tl.int64)).to(tl.int32)   # loop bounds are 32-bit counts: the Metal lowering refuses a 64-bit scf.for bound (addressing stays 64-bit through the program ids)
    pid = tl.program_id(axis=0).to(tl.int64)
    idx = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)

    # Decompose flat index -> (od, oh, ow)
    ow = idx % OW
    oh = (idx // OW) % OH
    od = (idx // (OW * OH)) % OD

    # Map output coords to input coords (nearest neighbor)
    id_ = tl.minimum((od * reciprocal_scale_d).to(tl.int32), ID - 1)
    ih = tl.minimum((oh * reciprocal_scale_h).to(tl.int32), IH - 1)
    iw = tl.minimum((ow * reciprocal_scale_w).to(tl.int32), IW - 1)

    # Compute offsets for contiguous [N, C, D, H, W] layout
    offset_o = ((nc_iter * OD + od) * OH + oh) * OW + ow
    offset_i = ((nc_iter * ID + id_) * IH + ih) * IW + iw
    src_index_stride = nc_stride * ID * IH * IW
    dst_index_stride = nc_stride * OD * OH * OW

    while nc_iter < NC:
        data = tl.load(ptr_i + offset_i)
        tl.store(ptr_o + offset_o, data)
        ptr_i += src_index_stride
        ptr_o += dst_index_stride
        nc_iter += nc_stride
