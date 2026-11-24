# Element-wise kernels with integer arguments Triton types by value width

The census of `tests/test_a_synthetic_launch_carries_each_integer_at_the_variants_own_width.py`: every class-tiled
kernel (launched with `ew_meta`, or declaring a synthetic recipe), its integer arguments, and what the tile proof
(`ops/ew_synthetic.py`) does when the variant has one of them **i64**.

## How the width is chosen

- Triton's binder, and the library's own (`backend/launcher.specialize_arg`, the same rule): an `int` argument is
  **i32** when -2^31 <= v < 2^31, **i64** when it lies outside that range and inside int64, u64 above; 'D' when
  divisible by 16; the constant **1** (a constexpr) when it equals 1. A float is fp32, a bool u1.
- **No wrapper chooses a width explicitly.** No class-tiled kernel's call site casts an integer argument, and none
  of these kernels declares `do_not_specialize`: the width follows the value of the launch. The only integer that
  declares its width is `ew_synthetic.Int64` (a small count of a wide variant, the proof's own).
- So an argument becomes i64 exactly when the tensor behind it does: a count, an extent, a stride or an offset of
  2^31 elements or more.

## How the proof carries them (`Synth.carry`)

- A **count** (the ladder's, and the extents a recipe names in `counts`: `cat_copy_kernel_4`'s other totals) stays
  small, 64-bit by declaration (`Int64`): the same binary met in kilobytes.
- Every other **i64** integer is carried past 2^31: its small value + 2^31 (the same divisibility by 16, never 1).
  An i32 one keeps its small value, a 'D' one a multiple of 16, a constant 1 stays 1.
- A wide value is harmless where the kernel never dereferences it: a batch stride whose index the ladder keeps at 0
  (`solve_dims(..., batch=...)`: one image covers the ladder's largest count). Where the kernel does dereference it,
  the buffer the recipe sizes from it passes `SMALL` (2^24 elements) and the variant is **refused by name**
  (`WideNotProvable`: "its wide form cannot be proven on small buffers").

## The census

Columns: the integers whose i64 form, alone, is **reproduced** on small buffers (the count included); those whose i64
form is **refused by name** (the kernel dereferences the wide value); those whose i64 form alone contradicts the
recipe's own geometry (a larger size of the same launch would be wide too: the census proves or refuses them only
together with every integer at least as large, which is the launch a real tensor makes). Scalars such as `alpha`,
`scalar`, `value` are fp32 at their call sites when the caller passes a float; they are listed because an int
passed there is an integer argument. Value scalars are never addresses: their wide form is reproduced.

| kernel | i64 reproduced | i64 refused by name | only with the larger sizes |
|---|---|---|---|
| `_token_bitmask_kernel` | vocab | words | - |
| `adaptive_avg_pool2d_kernel` | OH, OW, n_elements, stride_c, stride_n | IH, IW, stride_h, stride_w | - |
| `add_bias_broadcast_kernel` | feat_dim, n_elements | - | - |
| `add_forward_kernel` | alpha, n_elements | - | - |
| `add_scalar_dev_kernel` | alpha, n_elements | - | - |
| `add_scalar_kernel` | n_elements, scalar | - | - |
| `addcdiv_forward_kernel` | n_elements, value | - | - |
| `addcmul_forward_kernel` | n_elements, value | - | - |
| `arange_kernel` | n_elements, start, step | - | - |
| `bit_reverse_rows_kernel` | total | log2n, n | - |
| `bitwise_left_shift_scalar_kernel` | n_elements, shift_amount | - | - |
| `bitwise_right_shift_scalar_kernel` | n_elements, shift_amount | - | - |
| `cat_copy_kernel_4` | total_elements_a, total_elements_b, total_elements_c, total_elements_d | dim_offset_b, dim_offset_c, dim_offset_d, dim_prod_post, dim_size_in_a, dim_size_in_b, dim_size_in_c, dim_size_out | dim_offset_a, dim_size_in_d |
| `celu_forward_kernel` | alpha, n_elements | - | - |
| `clamp_forward_kernel` | max_val, min_val, n_elements | - | - |
| `clamp_min_forward_kernel` | min_val, n_elements | - | - |
| `col2im1d_kernel` | K, L_in, L_out, pad, stride | C_out | - |
| `constant_pad_1d_kernel` | batch_size, out_size, total | - | in_size, pad_left |
| `constant_pad_2d_kernel` | out_stride_c, out_stride_h, out_stride_n, out_stride_w, total, x_stride_c, x_stride_h, x_stride_n, x_stride_w | - | channels_times_batch, in_h, in_w, out_h, out_w, pad_left, pad_top |
| `constant_pad_nd_kernel` | out0, out1, out2, out3, out4, total, xs0 | xs4 | in0-in4, pb0-pb4, xs1, xs2, xs3 |
| `conv2d_bias_inplace_kernel` | HW, n | C | - |
| `dft_rfft_matrix_kernel` | N | N_bins | - |
| `div_scalar_kernel` | n_elements, scalar | - | - |
| `elu_forward_kernel` | alpha, n_elements | - | - |
| `eq_scalar_kernel` | n_elements, scalar | - | - |
| `fft_stage_rows_kernel` | pairs | n, stage | - |
| `fill_kernel` | n_elements, value | - | - |
| `flip_strided_kernel` | n_elements, stride_post | dim_size, stride_dim | - |
| `gather_kernel` | N, idx_stride_outer, inp_stride_outer, out_stride_outer, outer_size | idx_stride_inner, inner_size, inp_dim_stride, inp_stride_inner, out_stride_inner | dim_size, idx_stride_dim, out_stride_dim |
| `ge_scalar_kernel` | n_elements, scalar | - | - |
| `gt_scalar_kernel` | n_elements, scalar | - | - |
| `idft_c2r_matrix_kernel` | N | N_bins | n_even |
| `index_add_gather_kernel` | inner_size | dim_size, inp_shape_dim, src_stride_dim, src_stride_inner, src_stride_outer | - |
| `index_put_kernel` | N | R, T | - |
| `index_select_mid_kernel` | inner | N, index_len, outer | - |
| `le_scalar_kernel` | n_elements, scalar | - | - |
| `leaky_relu_forward_kernel` | n_elements, negative_slope | - | - |
| `lerp_scalar_head_kernel` | n_elements, weight | - | - |
| `lerp_scalar_tail_kernel` | n_elements, weight | - | - |
| `lt_scalar_kernel` | n_elements, scalar | - | - |
| `masked_fill_forward_kernel` | fill_value, n_elements | - | - |
| `mul_scalar_kernel` | n_elements, scalar | - | - |
| `nan_to_num_forward_kernel` | n_elements, nan_val, neginf_val, posinf_val | - | - |
| `ne_scalar_kernel` | n_elements, scalar | - | - |
| `pixel_shuffle_broadcast_aware_kernel` | C_out, OH, OW, bcast, n_elements, stride_out_n, stride_v_b, stride_v_n | r, stride_out_c, stride_out_h, stride_v_c, stride_v_h | - |
| `pixel_shuffle_kernel` | C_out, H, OH, OW, W, n_elements, stride_in_n, stride_out_n | r, stride_in_c, stride_in_h, stride_out_c, stride_out_h | - |
| `pixel_unshuffle_kernel` | C, C_out, H, IH, IW, W, n_elements, stride_in_n, stride_out_n | r, stride_in_c, stride_in_h, stride_out_c, stride_out_h | - |
| `pow_forward_kernel` | exponent, n_elements | - | - |
| `rand_kernel`, `randn_kernel` | n_elements, seed | - | - |
| `remainder_scalar_kernel` | divisor, n_elements | - | - |
| `replication_pad_nd_kernel` | out0, out1, out2, out3, out4, total, xs0 | xs3, xs4 | in0-in4, pb0-pb4, xs1, xs2 |
| `rsub_forward_kernel` | n_elements, scalar | - | - |
| `scale_kernel` | n_elements, scale | - | - |
| `scatter_kernel`, `scatter_add_kernel`, `scatter_reduce_amax_kernel`, `scatter_reduce_amin_kernel` | N, idx_stride_outer, out_stride_outer, outer_size, src_stride_outer | idx_stride_inner, inner_size, out_dim_stride, out_stride_inner, src_stride_inner | dim_size, idx_stride_dim, src_stride_dim |
| `silu_mul_split_kernel` | N | M, stride_gm, stride_gn, stride_om, stride_on, stride_um, stride_un | - |
| `softplus_forward_kernel` | beta, n_elements, threshold | - | - |
| `sub_forward_kernel` | alpha, n_elements | - | - |
| `threshold_forward_kernel` | n_elements, threshold_val, value | - | - |
| `upsample_nearest2d_kernel` | OW | C, IH, IW, N, OH, s_c, s_h, s_n, s_w | - |
| `upsample_nearest3d_kernel` | OW | C, ID, IH, IW, N, OD, OH | - |

The kernels of the shape-and-stride family (`*_strided_nd_kernel`, `strided_copy*`, `strided_scatter_kernel`) and
the one-count kernels (`abs`, `exp`, `where`, the comparisons, ...) carry their only integer, the count: their wide
form is the ladder's 64-bit count, always reproduced. Their shapes and strides are buffers, not arguments.

## Which shipped graphs reach the wide form

Scan: `wide_scan.py` (scratchpad `scan/wide_scan.py`) over `Super-NeuroBrix-Cache/*/components/*/graph.json`, each
graph's symbols evaluated at the model's run size (topology shapes, else defaults' height / width): the tensors of
2^31 elements, or of an offset past 2^31, and the ops that touch them. Sizes are an estimate: the VAE inherits the
loop's batch (2 under guidance) and Sana's topology latent (120 x 128, a 3840 x 4096 image); at 4096 x 4096 and batch
1 Sana's last pixel shuffle writes [1, 128, 4096, 4096] = 2^31 exactly. Five components, all video / 4K VAE decodes:

| model / component | wide shape classes (run size) | class-tiled kernels with an i64 integer argument |
|---|---|---|
| Sana_1600M_4Kpx_BF16 / vae | [N, 128-512, 3840-4096, 4096] activations; the expand view [N, 256, 2, H, W] | `pixel_shuffle_broadcast_aware_kernel` (stride_out_n, the count: THE RED, reproduced), the element-wise ops' counts, `upsample_nearest2d_kernel` (none: its plane, strides and batch stay below 2^31) |
| CogVideoX-2b, CogVideoX-5b-I2V / vae | [2, 128-256, 61-127, 240-480, 360-720] | `constant_pad_nd_kernel` (xs0, total: reproduced), `cat_copy_kernel_4` (an input's total: reproduced as a count), the element-wise ops' counts |
| Allegro / vae, mochi-1-preview / vae | permuted / sliced views with an offset past 2^31 | strided copies (count; strides in buffers) |

No shipped graph reaches a form the census refuses by name. Every refused form needs a single extent, row length,
inner stride or bias width past 2^31 elements.
