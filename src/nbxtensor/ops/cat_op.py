# Rewritten by NeuroBrix (Copyright NeuroBrix, Apache-2.0); derived from FlagGems (Apache-2.0), original notice kept
# Change: rewritten as a pure Triton kernel with no PyTorch import.
# Original notice: the provenance statement in the docstring below, unchanged, and THIRD_PARTY_NOTICES.md.
"""Concatenation — pure @triton.jit kernel.

Ported from FlagGems cat_copy_func_kernel_4 (Apache-2.0 license).
Copies up to 4 input tensors per launch into the correct position
in the output tensor along the cat dimension.
"""

import triton
import nbxtensor.ops._launch_seam  # noqa: F401  this module's launches go through nbxtensor's launcher
import triton.language as tl


from nbxtensor.ops.ew_tile import synthetic, tile_dependent, tile_independent  # noqa: F401  (ops/ew_tile.py)


def _cat_recipe(S):
    """The synthetic launch of `cat_copy_kernel_4` (`ops/ew_synthetic.py`): four inputs, every one live (the binary
    does not know how many of the four slots its call site filled), input a of total_elements_a = the ladder's count
    elements, each other one of about as many, every total honouring its own specialization. Each input i is read
    as (pre, dim_size_in_i, dim_prod_post) (small, honouring their specialization; its last pre slab partial when its
    total is not a multiple) and is EXACTLY its total long; the inputs occupy consecutive slots of the output's cat
    dim (dim_offset_i the first slot free, 0 where the variant has a 'D' offset and nothing precedes it), so no two
    writes collide; dim_size_out covers the last. The output's extent is 1 + the largest output index any input's
    last element maps to (the map is increasing in the input's index). The grid is the call site's: cdiv(the
    largest total, tile) x the inputs."""
    from nbxtensor.ops.elementwise_launch import per_tile
    slots = "abcd"
    post = S.int("dim_prod_post", 3)
    sizes = {c: S.int(f"dim_size_in_{c}", 2) for c in slots}
    offsets, free = {}, 0
    for c in slots:
        name = f"dim_offset_{c}"
        offsets[c] = 0 if free == 0 and S.attr(name) == "D" else S.int(name, free)
        free = offsets[c] + sizes[c]
    out_dim = S.int("dim_size_out", free)
    totals = {c: (S.n if c == "a" else S.int(f"total_elements_{c}", S.n)) for c in slots}

    def last(c):
        """The output index input c's last element is copied to."""
        i = totals[c] - 1
        return (i // (sizes[c] * post)) * out_dim * post + ((i // post) % sizes[c] + offsets[c]) * post + i % post

    args = {"dim_prod_post": post, "dim_size_out": out_dim,
            "out_ptr": S.buf("out_ptr", max(last(c) for c in slots) + 1)}
    for c in slots:
        args.update({f"dim_size_in_{c}": sizes[c], f"dim_offset_{c}": offsets[c],
                     f"total_elements_{c}": totals[c], f"in_ptr_{c}": S.buf(f"in_ptr_{c}", totals[c])})
    return args, (per_tile(max(totals.values())), len(slots))


_cat_recipe.count = "total_elements_a"
_cat_recipe.counts = ("total_elements_b", "total_elements_c", "total_elements_d")   # each input's own count
_cat_recipe.description = "cat4(total_elements_a, four live inputs over (pre, dim_size_in, dim_prod_post))"


@synthetic(_cat_recipe)
@tile_independent
@triton.jit
def cat_copy_kernel_4(
    out_ptr,
    in_ptr_a,
    in_ptr_b,
    in_ptr_c,
    in_ptr_d,
    dim_size_in_a,
    dim_size_in_b,
    dim_size_in_c,
    dim_size_in_d,
    dim_size_out,
    dim_prod_post,
    dim_offset_a,
    dim_offset_b,
    dim_offset_c,
    dim_offset_d,
    total_elements_a,
    total_elements_b,
    total_elements_c,
    total_elements_d,
    BLOCK_X: tl.constexpr,
):
    """Copy up to 4 input tensors into their correct positions in the output.

    pid_x indexes blocks within a single input tensor.
    pid_y selects which of the 4 input tensors this program handles.

    The index remapping converts flat input indices to flat output indices
    by splitting into (pre_dim, cat_dim, post_dim) components and applying
    the cat dimension offset.
    """
    pid_x = tl.program_id(0).to(tl.int64)
    pid_y = tl.program_id(1).to(tl.int64)

    # Select the correct input tensor and its metadata
    if pid_y == 0:
        in_ptr = in_ptr_a
    elif pid_y == 1:
        in_ptr = in_ptr_b
    elif pid_y == 2:
        in_ptr = in_ptr_c
    else:
        in_ptr = in_ptr_d

    # The per-input scalars are selected OUTSIDE the pointer branches, by tl.where,
    # which unifies their types: Triton specialises a Python int as int32 when it
    # fits, int64 when it does not, and as a constexpr when it equals 1 — so one
    # input past 2^31 elements beside a small one made the if/elif branches
    # disagree (`Mismatched type for total_elements`, Open-Sora-v2 VAE decode,
    # 2026-09-25), and a `.to(tl.int64)` inside the branches broke on the
    # constexpr form ('int' object has no attribute 'to', Flex.1-alpha).
    dim_size_in = tl.where(pid_y == 0, dim_size_in_a, tl.where(pid_y == 1, dim_size_in_b, tl.where(pid_y == 2, dim_size_in_c, dim_size_in_d)))
    dim_offset = tl.where(pid_y == 0, dim_offset_a, tl.where(pid_y == 1, dim_offset_b, tl.where(pid_y == 2, dim_offset_c, dim_offset_d)))
    total_elements = tl.where(pid_y == 0, total_elements_a, tl.where(pid_y == 1, total_elements_b, tl.where(pid_y == 2, total_elements_c, total_elements_d)))

    block_start = pid_x * BLOCK_X
    offsets = tl.arange(0, BLOCK_X)
    mask = block_start + offsets < total_elements

    idx = block_start + offsets

    # Decompose flat index into (pre, dim, post) coordinates
    pre_idx = idx // (dim_size_in * dim_prod_post)
    dim_idx = (idx // dim_prod_post) % dim_size_in
    post_idx = idx % dim_prod_post

    # Compute output index with the cat dimension offset applied
    out_idx = (
        pre_idx * dim_size_out * dim_prod_post
        + (dim_idx + dim_offset) * dim_prod_post
        + post_idx
    )

    data = tl.load(in_ptr + idx, mask=mask)
    tl.store(out_ptr + out_idx, data, mask=mask)
