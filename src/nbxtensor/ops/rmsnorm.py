# Rewritten by NeuroBrix (Copyright NeuroBrix, Apache-2.0); derived from attorch (MIT), original notice kept
# Change: rewritten as a pure Triton kernel with no PyTorch import, its launch configurations certified through nbxtensor's own autotune.
# Original notice: the provenance statement in the docstring below, unchanged, and THIRD_PARTY_NOTICES.md.
"""RMSNorm — pure @triton.jit kernel. Ported from attorch (MIT)."""

import triton
import triton.language as tl
from triton import next_power_of_2

from nbxtensor.ops._configs import nbx_autotune, row_program_configs, prune_row_programs
from nbxtensor.ops.fixed_order import row_dot_fixed_order


# rows per program and warps: the certified configuration of the key (workstream 1), the key the derived census forms
# (`launch_keys.rms_norm_launches`)
@nbx_autotune(configs=row_program_configs(), key=['BATCH_BUCKET', 'FEAT_BUCKET', 'scale_by_weight', 'SATURATE_F16'],
              prune_configs_by={'early_config_prune': prune_row_programs}, cache_results=True)
@triton.heuristics({
    'BLOCK_SIZE_FEAT': lambda args: next_power_of_2(args['feat_dim']),
})
@triton.jit
def rms_norm_forward_kernel(
    input_ptr, weight_ptr,
    output_ptr,
    batch_dim, feat_dim, BATCH_BUCKET, FEAT_BUCKET,
    input_batch_stride, input_feat_stride,
    output_batch_stride, output_feat_stride,
    eps,
    scale_by_weight: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_FEAT: tl.constexpr,
    SATURATE_F16: tl.constexpr = False,   # the protected fp16 store: finite values clamp to ±65504
):
    """RMS normalization forward.

    input: [batch_dim, feat_dim]
    weight: [feat_dim] (optional)
    output: [batch_dim, feat_dim]
    """
    batch_pid = tl.program_id(0).to(tl.int64)

    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    feat_offset = tl.arange(0, BLOCK_SIZE_FEAT)

    batch_mask = batch_offset < batch_dim
    feat_mask = feat_offset < feat_dim

    input_ptr += (input_batch_stride * batch_offset[:, None] +
                  input_feat_stride * feat_offset[None, :])
    output_ptr += (output_batch_stride * batch_offset[:, None] +
                   output_feat_stride * feat_offset[None, :])

    inp = tl.load(input_ptr,
                  mask=batch_mask[:, None] & feat_mask[None, :]).to(tl.float32)
    # the row's sum of squares in one fixed order: the warps a certified or swept setting picks cannot change it
    # (`tl.cast`, not `.to`: a width of 1 reaches the kernel as a plain int, the compiler specializes the value 1)
    # the division correctly rounded (`tl.div_rn`): `/` is a reciprocal and a multiply the compiler may fuse into the add
    inv_rms = tl.rsqrt(tl.div_rn(row_dot_fixed_order(inp, inp, BLOCK_SIZE_BATCH, BLOCK_SIZE_FEAT), tl.cast(feat_dim, tl.float32)) + eps)
    output = inp * inv_rms[:, None]

    if scale_by_weight:
        weight = tl.load(weight_ptr + feat_offset, mask=feat_mask)
        output *= weight

    if SATURATE_F16:
        # The store narrows to fp16: the same protected conversion the dtype
        # engine's cast copy applied when this kernel stored fp32 and the wrap
        # cast the result (copy_kernel's expression) — an fp16 stream with
        # outliers past 65504 (VibeVoice's Qwen2, 2026-09-08) clamps instead
        # of overflowing to inf.
        is_finite = (output == output) & (output != float("inf")) & (output != float("-inf"))
        clamped = tl.minimum(tl.maximum(output, -65504.0), 65504.0)
        output = tl.where(is_finite, clamped, output)
    tl.store(output_ptr, output,
             mask=batch_mask[:, None] & feat_mask[None, :])
