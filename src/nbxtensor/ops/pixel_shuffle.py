"""Pixel shuffle (sub-pixel convolution) — pure @triton.jit kernel.

Rearranges: [N, C*r*r, H, W] -> [N, C, H*r, W*r]

For each output element (n, c, oh, ow):
    ih = oh // r
    iw = ow // r
    ic = c * (r * r) + (oh % r) * r + (ow % r)
    output[n, c, oh, ow] = input[n, ic, ih, iw]
"""

import triton
import nbxtensor.ops._launch_seam  # noqa: F401  this module's launches go through nbxtensor's launcher
import triton.language as tl

from nbxtensor.ops.ew_tile import synthetic, tile_dependent, tile_independent  # noqa: F401  (ops/ew_tile.py)
from nbxtensor.ops.ew_synthetic import reach, solve_dims  # noqa: E402  helpers of the synthetic recipes


# ---------------------------------------------------------------------------
# synthetic launches of the tile proof (ops/ew_synthetic.py)
# ---------------------------------------------------------------------------
def _shuffle_dims(v):
    """The pixel shuffle's sizes from (C_out, H, W, r): contiguous [N, C_out*r*r, H, W] in, [N, C_out, H*r, W*r] out."""
    c, h, w, r = v["C_out"], v["H"], v["W"], v["r"]
    oh, ow = h * r, w * r
    return {"C_out": c, "H": h, "W": w, "r": r, "OH": oh, "OW": ow,
            "stride_in_n": c * r * r * h * w, "stride_in_c": h * w, "stride_in_h": w, "stride_in_w": 1,
            "stride_out_n": c * oh * ow, "stride_out_c": oh * ow, "stride_out_h": ow, "stride_out_w": 1}


def _pixel_shuffle_recipe(S):
    """The synthetic launch of `pixel_shuffle_kernel`: `n_elements` = the ladder's count output elements of a
    contiguous [N, C_out, H*r, W*r] (N implicit in the count, its last image possibly partial), read from a
    contiguous [N, C_out*r*r, H, W]; (C_out, H, W, r) the smallest whose every derived size (OH, OW, the eight
    strides) specialises as the variant (`ew_synthetic.solve_dims`). The output `n_elements` elements, the input 1 + the
    largest offset a lane below the count reads (the op's index map); grid `cdiv(n_elements, tile)`."""
    import numpy as np
    d = solve_dims(S, ["C_out", "H", "W", "r"], _shuffle_dims)
    idx = np.arange(S.n, dtype=np.int64)
    ow, t = idx % d["OW"], idx // d["OW"]
    oh, t = t % d["OH"], t // d["OH"]
    c, n = t % d["C_out"], t // d["C_out"]
    r = d["r"]
    ic = c * r * r + (oh % r) * r + (ow % r)
    src = n * d["stride_in_n"] + ic * d["stride_in_c"] + (oh // r) * d["stride_in_h"] + (ow // r) * d["stride_in_w"]
    dst = n * d["stride_out_n"] + c * d["stride_out_c"] + oh * d["stride_out_h"] + ow * d["stride_out_w"]
    return ({**d, "n_elements": S.n, "input_ptr": S.buf("input_ptr", reach(src)),
             "output_ptr": S.buf("output_ptr", reach(dst))}, None)


_pixel_shuffle_recipe.count = "n_elements"
_pixel_shuffle_recipe.description = "pixel shuffle: n_elements of contiguous NCHW out, input = 1 + largest read offset"


@synthetic(_pixel_shuffle_recipe)
@tile_independent
@triton.jit
def pixel_shuffle_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    # Input shape: [N, C_in, H, W] where C_in = C_out * r * r
    C_out,
    H,
    W,
    r,
    # Output shape: [N, C_out, H*r, W*r]
    OH,  # = H * r
    OW,  # = W * r
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
    """Pixel shuffle: [N, C*r*r, H, W] -> [N, C, H*r, W*r].

    Each thread handles one output element.
    """
    pid = tl.program_id(0).to(tl.int64)
    idx = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = idx < n_elements

    # Decompose flat output index -> (n, c, oh, ow)
    ow = idx % OW
    tmp = idx // OW
    oh = tmp % OH
    tmp2 = tmp // OH
    c = tmp2 % C_out
    n = tmp2 // C_out

    # Map output coords to input coords
    ih = oh // r
    iw = ow // r
    # Sub-pixel position within the r x r block
    sub_h = oh % r
    sub_w = ow % r
    ic = c * (r * r) + sub_h * r + sub_w

    # Load from input, store to output
    in_offset = n * stride_in_n + ic * stride_in_c + ih * stride_in_h + iw * stride_in_w
    out_offset = n * stride_out_n + c * stride_out_c + oh * stride_out_h + ow * stride_out_w

    val = tl.load(input_ptr + in_offset, mask=mask)
    tl.store(output_ptr + out_offset, val, mask=mask)


def _bcast_dims(v):
    """The broadcast-aware shuffle's sizes from (C_out, OH, OW, r, bcast): the view (N, C_pre, bcast, H, W) of a
    contiguous (N, C_pre, H, W) expanded on dim 2 (stride 0), C_pre = C_out*r*r / bcast; a contiguous output."""
    c, oh, ow, r, b = v["C_out"], v["OH"], v["OW"], v["r"], v["bcast"]
    if oh % r or ow % r or (c * r * r) % b:
        return None
    h, w, c_pre = oh // r, ow // r, c * r * r // b
    return {"C_out": c, "OH": oh, "OW": ow, "r": r, "bcast": b,
            "stride_v_n": c_pre * h * w, "stride_v_c": h * w, "stride_v_b": 0, "stride_v_h": w, "stride_v_w": 1,
            "stride_out_n": c * oh * ow, "stride_out_c": oh * ow, "stride_out_h": ow, "stride_out_w": 1}


def _pixel_shuffle_bcast_recipe(S):
    """The synthetic launch of `pixel_shuffle_broadcast_aware_kernel`: `n_elements` = the ladder's count output
    elements of a contiguous [N, C_out, OH, OW] (N implicit in the count), read through the stride-0 broadcast view
    the call site hands (`_bcast_dims`), (C_out, OH, OW, r, bcast) the smallest whose derived sizes specialise as the
    variant (`ew_synthetic.solve_dims`). The output `n_elements` elements, the input (the pre-expand tensor) 1 + the largest
    offset a lane below the count reads; grid `cdiv(n_elements, tile)`."""
    import numpy as np
    d = solve_dims(S, ["C_out", "OH", "OW", "r", "bcast"], _bcast_dims)
    idx = np.arange(S.n, dtype=np.int64)
    ow, t = idx % d["OW"], idx // d["OW"]
    oh, t = t % d["OH"], t // d["OH"]
    c, n = t % d["C_out"], t // d["C_out"]
    r, b = d["r"], d["bcast"]
    icv = c * r * r + (oh % r) * r + (ow % r)
    src = (n * d["stride_v_n"] + (icv // b) * d["stride_v_c"] + (icv % b) * d["stride_v_b"]
           + (oh // r) * d["stride_v_h"] + (ow // r) * d["stride_v_w"])
    dst = n * d["stride_out_n"] + c * d["stride_out_c"] + oh * d["stride_out_h"] + ow * d["stride_out_w"]
    return ({**d, "n_elements": S.n, "bcast_view_ptr": S.buf("bcast_view_ptr", reach(src)),
             "output_ptr": S.buf("output_ptr", reach(dst))}, None)


_pixel_shuffle_bcast_recipe.count = "n_elements"
_pixel_shuffle_bcast_recipe.description = "pixel shuffle through a stride-0 view: n_elements out, input = 1 + largest read"


@synthetic(_pixel_shuffle_bcast_recipe)
@tile_independent
@triton.jit
def pixel_shuffle_broadcast_aware_kernel(
    bcast_view_ptr,  # 5D NBX expand-output view (N, C_pre, B, H, W) with stride_b == 0
    output_ptr,      # 4D contiguous output (N, C_out, OH, OW)
    n_elements,
    C_out,           # output channels = (C_pre * B) // (r*r)
    OH,              # = H * r
    OW,              # = W * r
    r,               # pixel_shuffle upscale factor
    bcast,           # broadcast factor (= B, the size on the stride-0 dim)
    # 5D bcast_view strides (NBX expand sets stride_b = 0).
    stride_v_n,
    stride_v_c,
    stride_v_b,
    stride_v_h,
    stride_v_w,
    # 4D output strides (contiguous NCHW).
    stride_out_n,
    stride_out_c,
    stride_out_h,
    stride_out_w,
    BLOCK_SIZE: tl.constexpr,
):
    """Pixel shuffle that reads through an unmaterialized NBX broadcast view.

    Input layout: (N, C_pre, B, H, W) where the original DAG pattern is
    `unsqueeze(dim=2) -> expand(dim=2, B) -> clone -> view -> pixel_shuffle`.
    NBX expand sets stride_b=0 (kernels/nbx_tensor.py:1581), so the 5D view
    aliases the pre-expand tensor with no materialization. We read through
    that view directly and skip the clone's 8 GiB copy.

    The post-clone-post-view shape consumed by a normal pixel_shuffle is
    (N, C_pre*B, H, W); here the kernel does the equivalent decomposition
    inline:
      ic_view = c*r*r + (oh%r)*r + (ow%r)         in [0, C_pre*B)
      c_pre   = ic_view // bcast                   in [0, C_pre)
      b       = ic_view %  bcast                   in [0, B)
      ih      = oh // r
      iw      = ow // r
    Read offset uses all 5D strides; with stride_v_b == 0, the `b` index
    contributes nothing and the load aliases naturally.
    """
    pid = tl.program_id(0).to(tl.int64)
    # Cast to int64 to support tensors with >= 2^31 elements
    # (e.g. Sana 4Kpx VAE 1*128*4096*4096 = 2^31 — at INT32_MAX
    # boundary; signed int32 multiplication of strides overflows
    # silently and corrupts offsets, producing garbage output).
    idx = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = idx < n_elements

    ow = idx % OW
    tmp = idx // OW
    oh = tmp % OH
    tmp2 = tmp // OH
    c = tmp2 % C_out
    n = tmp2 // C_out

    sub_h = oh % r
    sub_w = ow % r
    ic_view = c * (r * r) + sub_h * r + sub_w
    c_pre = ic_view // bcast
    b = ic_view % bcast

    ih = oh // r
    iw = ow // r

    in_offset = (n * stride_v_n + c_pre * stride_v_c + b * stride_v_b
                 + ih * stride_v_h + iw * stride_v_w)
    out_offset = (n * stride_out_n + c * stride_out_c
                  + oh * stride_out_h + ow * stride_out_w)

    val = tl.load(bcast_view_ptr + in_offset, mask=mask)
    tl.store(output_ptr + out_offset, val, mask=mask)
