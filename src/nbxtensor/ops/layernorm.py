# Rewritten by NeuroBrix (Copyright NeuroBrix, Apache-2.0); derived from attorch (MIT), original notice kept
# Change: rewritten as a pure Triton kernel with no PyTorch import, its launch configurations certified through nbxtensor's own autotune.
# Original notice: the provenance statement in the docstring below, unchanged, and THIRD_PARTY_NOTICES.md.
"""LayerNorm — pure @triton.jit kernel. Ported from attorch (MIT)."""

import triton
import nbxtensor.ops._launch_seam  # noqa: F401  this module's launches go through nbxtensor's launcher
import triton.language as tl
from triton import next_power_of_2

from nbxtensor.ops._configs import nbx_autotune, row_program_configs, prune_row_programs
from nbxtensor.ops.fixed_order import row_sum_fixed_order, row_dot_fixed_order


# rows per program and warps: the certified configuration of the key (workstream 1), the key the derived census forms
# (`launch_keys.layer_norm_launches`)
@nbx_autotune(configs=row_program_configs(), key=['BATCH_BUCKET', 'FEAT_BUCKET', 'scale_by_weight', 'add_bias', 'save_stats'],
              prune_configs_by={'early_config_prune': prune_row_programs}, cache_results=True)
@triton.heuristics({
    'BLOCK_SIZE_FEAT': lambda args: next_power_of_2(args['feat_dim']),
})
@triton.jit
def layer_norm_forward_kernel(
    input_ptr, weight_ptr, bias_ptr,
    mean_ptr, inv_std_ptr, output_ptr,
    batch_dim, feat_dim, BATCH_BUCKET, FEAT_BUCKET,
    input_batch_stride, input_feat_stride,
    output_batch_stride, output_feat_stride,
    eps,
    scale_by_weight: tl.constexpr,
    add_bias: tl.constexpr,
    save_stats: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_FEAT: tl.constexpr,
):
    """Layer normalization forward.

    input: [batch_dim, feat_dim]
    weight: [feat_dim] (optional)
    bias: [feat_dim] (optional)
    output: [batch_dim, feat_dim]
    mean, inv_std: [batch_dim] (optional, for backward)
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
    # the row's two sums in one fixed order: the warps a certified or swept setting picks cannot change them
    # divisions correctly rounded (`tl.div_rn`): `/` lowers to a reciprocal and a multiply, which the compiler then
    # fused into the subtraction or the add after it on some launches and not others (a [3, 33] row on 2 rows x
    # 8 warps gave other output bits with the saved stats equal, 2026-11-18)
    mean = tl.div_rn(row_sum_fixed_order(inp, BLOCK_SIZE_BATCH, BLOCK_SIZE_FEAT), tl.cast(feat_dim, tl.float32))
    diff = tl.where(feat_mask[None, :], inp - mean[:, None], 0)
    inv_std = tl.rsqrt(tl.div_rn(row_dot_fixed_order(diff, diff, BLOCK_SIZE_BATCH, BLOCK_SIZE_FEAT), tl.cast(feat_dim, tl.float32)) + eps)

    if save_stats:
        tl.store(mean_ptr + batch_offset, mean, mask=batch_mask)
        tl.store(inv_std_ptr + batch_offset, inv_std, mask=batch_mask)

    output = diff * inv_std[:, None]
    if scale_by_weight:
        weight = tl.load(weight_ptr + feat_offset, mask=feat_mask)
        if add_bias:
            # one explicit fused multiply-add: written as a multiply then an add, the compiler fused them on some
            # launches and not others (a [3, 33] row: 2 rows x 8 warps gave other bits, 2026-11-18)
            bias = tl.load(bias_ptr + feat_offset, mask=feat_mask)
            output = tl.fma(output, weight, bias)
        else:
            output *= weight

    tl.store(output_ptr, output,
             mask=batch_mask[:, None] & feat_mask[None, :])
