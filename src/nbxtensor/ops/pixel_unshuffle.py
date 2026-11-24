"""Pixel unshuffle (inverse of pixel shuffle) — pure @triton.jit kernel.

Rearranges: [N, C, H*r, W*r] -> [N, C*r*r, H, W]

For each output element (n, oc, oh, ow):
    c = oc // (r * r)
    sub_h = (oc % (r * r)) // r
    sub_w = (oc % (r * r)) % r
    ih = oh * r + sub_h
    iw = ow * r + sub_w
    output[n, oc, oh, ow] = input[n, c, ih, iw]
"""

import triton
import nbxtensor.ops._launch_seam  # noqa: F401  this module's launches go through nbxtensor's launcher
import triton.language as tl

from nbxtensor.ops.ew_tile import synthetic, tile_dependent, tile_independent  # noqa: F401  (ops/ew_tile.py)


def _unshuffle_dims(v):
    """The pixel unshuffle's sizes from (C, H, W, r): contiguous [N, C, H*r, W*r] in, [N, C*r*r, H, W] out."""
    c, h, w, r = v["C"], v["H"], v["W"], v["r"]
    ih, iw, co = h * r, w * r, c * r * r
    return {"C": c, "IH": ih, "IW": iw, "r": r, "C_out": co, "H": h, "W": w,
            "stride_in_n": c * ih * iw, "stride_in_c": ih * iw, "stride_in_h": iw, "stride_in_w": 1,
            "stride_out_n": co * h * w, "stride_out_c": h * w, "stride_out_h": w, "stride_out_w": 1}


def _pixel_unshuffle_recipe(S):
    """The synthetic launch of `pixel_unshuffle_kernel`: `n_elements` = the ladder's count output elements of a
    contiguous [N, C*r*r, H, W] (N implicit in the count, its last image possibly partial), read from a contiguous
    [N, C, H*r, W*r]; (C, H, W, r) the smallest whose every derived size (IH, IW, C_out, the eight strides)
    specialises as the variant (`ew_synthetic.solve_dims`). The output `n_elements` elements, the input 1 + the
    largest offset a lane below the count reads (the op's index map); grid `cdiv(n_elements, tile)`."""
    import numpy as np
    from nbxtensor.ops.ew_synthetic import solve_dims
    d = solve_dims(S, ["C", "H", "W", "r"], _unshuffle_dims,
                   batch=(("stride_in_n", "stride_out_n"), lambda v: v["C_out"] * v["H"] * v["W"]))
    idx = np.arange(S.n, dtype=np.int64)
    ow, t = idx % d["W"], idx // d["W"]
    oh, t = t % d["H"], t // d["H"]
    oc, n = t % d["C_out"], t // d["C_out"]
    r = d["r"]
    sub = oc % (r * r)
    src = (n * d["stride_in_n"] + (oc // (r * r)) * d["stride_in_c"] + (oh * r + sub // r) * d["stride_in_h"]
           + (ow * r + sub % r) * d["stride_in_w"])
    dst = n * d["stride_out_n"] + oc * d["stride_out_c"] + oh * d["stride_out_h"] + ow * d["stride_out_w"]
    return ({**d, "n_elements": S.n, "input_ptr": S.buf("input_ptr", int(src.max()) + 1),
             "output_ptr": S.buf("output_ptr", int(dst.max()) + 1)}, None)


_pixel_unshuffle_recipe.count = "n_elements"
_pixel_unshuffle_recipe.description = "pixel unshuffle: n_elements of contiguous NCHW out, input = 1 + largest read"


@synthetic(_pixel_unshuffle_recipe)
@tile_independent
@triton.jit
def pixel_unshuffle_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    # Input shape: [N, C, IH, IW] where IH = H*r, IW = W*r
    C,
    IH,
    IW,
    r,
    # Output shape: [N, C*r*r, H, W]
    C_out,  # = C * r * r
    H,      # = IH // r
    W,      # = IW // r
    # Input strides (contiguous NCHW)
    stride_in_n,
    stride_in_c,
    stride_in_h,
    stride_in_w,
    # Output strides (contiguous NCHW)
    stride_out_n,
    stride_out_c,
    stride_out_h,
    stride_out_w,
    BLOCK_SIZE: tl.constexpr,
):
    """Pixel unshuffle: [N, C, H*r, W*r] -> [N, C*r*r, H, W].

    Each thread handles one output element.
    """
    pid = tl.program_id(0).to(tl.int64)
    idx = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = idx < n_elements

    # Decompose flat output index -> (n, oc, oh, ow)
    ow = idx % W
    tmp = idx // W
    oh = tmp % H
    tmp2 = tmp // H
    oc = tmp2 % C_out
    n = tmp2 // C_out

    # Map output coords to input coords
    c = oc // (r * r)
    sub_idx = oc % (r * r)
    sub_h = sub_idx // r
    sub_w = sub_idx % r
    ih = oh * r + sub_h
    iw = ow * r + sub_w

    # Load from input, store to output
    in_offset = n * stride_in_n + c * stride_in_c + ih * stride_in_h + iw * stride_in_w
    out_offset = n * stride_out_n + oc * stride_out_c + oh * stride_out_h + ow * stride_out_w

    val = tl.load(input_ptr + in_offset, mask=mask)
    tl.store(output_ptr + out_offset, val, mask=mask)
