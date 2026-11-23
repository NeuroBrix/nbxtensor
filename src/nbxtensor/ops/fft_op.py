"""FFT — pure @triton.jit Cooley-Tukey radix-2 kernels.

Implements forward FFT, rfft, and irfft as pure Triton kernels.
Based on FlagGems PR #1243 (Cooley-Tukey butterfly decomposition).

Algorithm (every row of a [rows, N] batch at once — one launch per step, not per row):
  1. Bit-reversal permutation (reorder input)
  2. log2(N) butterfly stages with twiddle factors
  3. For rfft: take first N//2+1 complex outputs
  4. For irfft: reconstruct full spectrum from half, inverse FFT, scale by 1/N

Constraint: Input size must be power of 2 (standard Cooley-Tukey).
For non-power-of-2: caller must zero-pad to next power of 2.
"""

import math

import triton
import nbxtensor.ops._launch_seam  # noqa: F401  this module's launches go through nbxtensor's launcher
import triton.language as tl


from nbxtensor.ops.ew_tile import synthetic, tile_dependent, tile_independent  # noqa: F401  (ops/ew_tile.py)
from nbxtensor.ops.ew_synthetic import holds  # noqa: E402  helpers of the synthetic recipes
from nbxtensor.ops.ew_synthetic import flat


def _as_int(S, name, v):
    """`v` as parameter `name` carries it (64-bit for an i64 parameter)."""
    from nbxtensor.ops.ew_synthetic import width
    return width(S.ty(name), int(v))


def _pow2_length(S, low: int, **attrs):
    """The smallest FFT length N = 2^k, k >= `low`, whose integers specialise as the variant's: `attrs` maps a
    parameter to a function of k giving its value (n -> N, log2n -> k)."""
    from nbxtensor.ops.ew_synthetic import NotSynthesisable
    for k in range(low, 21):
        if all(holds(S.attr(name), f(k)) for name, f in attrs.items()):
            return k
    raise NotSynthesisable(f"{S.kernel.__name__}: no power-of-two length up to 2^20 specialises as the variant "
                           f"({ {name: S.attr(name) for name in attrs} })")


def _whole_rows(S, name, unit: int) -> int:
    """The whole number of rows x `unit` lanes (a row's lanes are `unit`) nearest the ladder's count (the lower on a
    tie, at least one row) that specialises as the variant's `name`: a rung just below or just past a tile stays on
    its side of it."""
    from nbxtensor.ops.ew_synthetic import NotSynthesisable
    n = int(S.n)
    fits = [r * unit for r in range(max(1, n // unit - 64), n // unit + 66) if holds(S.attr(name), r * unit)]
    if not fits:
        raise NotSynthesisable(f"{S.kernel.__name__}: no whole number of rows of {unit} lanes specialises {name} as "
                               f"the variant ({S.attr(name)!r})")
    return min(fits, key=lambda v: (abs(v - n), v))


def _bit_reverse_recipe(S):
    """The synthetic launch of `bit_reverse_rows_kernel` (`ops/ew_synthetic.py`), as `_triton_fft_rows` launches
    it: rows of a power-of-two length N (the smallest whose n and log2n specialise as the variant's), `total` =
    rows x N lanes, rows the nearest to the ladder's count with `total` in its own specialization (`_whole_rows`).
    Every lane reads its element of the four [rows, N] buffers (`offs < total`) and writes row x N + bit_reverse(i)
    < total: each buffer is exactly `total` elements. Grid cdiv(total, tile), the call site's `_1d_grid(total)`."""
    from nbxtensor.ops.elementwise_launch import per_tile
    k = _pow2_length(S, 0, n=lambda k: 1 << k, log2n=lambda k: k)
    N = 1 << k
    total = _whole_rows(S, "total", N)
    args = {"n": _as_int(S, "n", N), "total": _as_int(S, "total", total), "log2n": _as_int(S, "log2n", k)}
    for p in ("real_in", "imag_in", "real_out", "imag_out"):
        args[p] = S.buf(p, total)
    return args, (per_tile(total),)


_bit_reverse_recipe.count = "total"
_bit_reverse_recipe.description = "fft rows: total = rows x N (N a power of 2), four buffers of total, per_tile(total)"


@synthetic(_bit_reverse_recipe)
@tile_independent
@triton.jit
def bit_reverse_rows_kernel(
    real_in, imag_in, real_out, imag_out,
    n, total, log2n,
    BLOCK_SIZE: tl.constexpr,
):
    """Bit-reversal permutation of EVERY row of a [rows, n] pair in one launch:
    row r's element i lands at row r, position bit_reverse(i).

    One element per lane, `total` = rows * n lanes. The rows used to be launched one
    at a time from the host: an STFT of a spoken sentence is thousands of rows, and
    its FFT cost thousands of launches per stage (py-spy on chatterbox's census walk,
    2026-09-28: every sample inside that per-row loop).
    """
    pid = tl.program_id(0).to(tl.int64)
    offs = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit: rows * n passes 2^31 on a long speech
    m = offs < total
    row = offs // n
    idx = offs % n

    # `rev` is an explicit int64 from the start: a bare 0 is int32, and the loop
    # re-assigns an int64 into it — Triton refuses a loop-carried type change
    # (found 2026-09-18 by chatterbox's `aten::stft`).
    rev = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    t = idx
    for _ in range(0, log2n):
        rev = (rev << 1) | (t & 1)
        t = t >> 1

    val_real = tl.load(real_in + offs, mask=m, other=0.0)
    val_imag = tl.load(imag_in + offs, mask=m, other=0.0)
    tl.store(real_out + row * n + rev, val_real, mask=m)
    tl.store(imag_out + row * n + rev, val_imag, mask=m)


def _fft_stage_recipe(S):
    """The synthetic launch of `fft_stage_rows_kernel` (`ops/ew_synthetic.py`), as `_triton_fft_rows` launches each
    stage: `stage` the smallest value of the variant's specialization (1, 2 or 16), the length N = 2^k the smallest
    with k >= stage (a stage's half block 2^(stage - 1) lies in a row) whose n specialises as the variant's, `pairs` =
    rows x N / 2 lanes, rows the nearest to the ladder's count with `pairs` in its own specialization. A lane
    below `pairs` addresses row x N + its pair's two positions < N: the two buffers are exactly rows x N = 2 x pairs
    elements. Grid cdiv(pairs, tile), the call site's `_1d_grid(pairs)`."""
    from nbxtensor.ops.elementwise_launch import per_tile
    stage = S.int("stage", 2)
    k = _pow2_length(S, max(1, int(stage)), n=lambda k: 1 << k)
    N = 1 << k
    pairs = _whole_rows(S, "pairs", N // 2)
    args = {"n": _as_int(S, "n", N), "pairs": _as_int(S, "pairs", pairs), "stage": stage,
            "real_ptr": S.buf("real_ptr", 2 * pairs), "imag_ptr": S.buf("imag_ptr", 2 * pairs)}
    return args, (per_tile(pairs),)


_fft_stage_recipe.count = "pairs"
_fft_stage_recipe.description = "fft stage: pairs = rows x N/2 (N a power of two >= 2^stage), 2 buffers of 2 x pairs"


@synthetic(_fft_stage_recipe)
@tile_independent
@triton.jit
def fft_stage_rows_kernel(
    real_ptr, imag_ptr,
    n, pairs, stage,
    INVERSE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One butterfly stage of the Cooley-Tukey FFT on EVERY row of a [rows, n] pair,
    in place, one butterfly pair per lane (`pairs` = rows * n // 2 lanes).

    The forward twiddle is e^(-i*pi*k/half_block); INVERSE takes its conjugate. The
    arithmetic of each pair is the one-row kernel's, written the same way, so a row
    computed in a batch is bit-identical to the row computed alone.
    """
    PI = math.pi
    pid = tl.program_id(0).to(tl.int64)
    offs = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    half_n = n // 2
    row = offs // half_n
    tid = offs % half_n
    half_block = 1 << (stage - 1)

    butterfly_group = tid // half_block
    pos_in_group = tid % half_block
    first_local = butterfly_group * half_block * 2 + pos_in_group
    second_local = first_local + half_block
    # A lane beyond the last pair, or a pair whose second element lies beyond its row,
    # reads and writes nothing (no early exit — the portable form is the mask).
    m = (offs < pairs) & (second_local < n)
    first_idx = row * n + first_local
    second_idx = row * n + second_local

    a_real = tl.load(real_ptr + first_idx, mask=m, other=0.0)
    a_imag = tl.load(imag_ptr + first_idx, mask=m, other=0.0)
    b_real = tl.load(real_ptr + second_idx, mask=m, other=0.0)
    b_imag = tl.load(imag_ptr + second_idx, mask=m, other=0.0)

    angle = PI * pos_in_group / half_block
    if INVERSE:
        w_real = tl.cos(angle)
        w_imag = tl.sin(angle)
    else:
        w_real = tl.cos(-angle)
        w_imag = tl.sin(-angle)

    tw_real = b_real * w_real - b_imag * w_imag
    tw_imag = b_real * w_imag + b_imag * w_real

    tl.store(real_ptr + first_idx, a_real + tw_real, mask=m)
    tl.store(imag_ptr + first_idx, a_imag + tw_imag, mask=m)
    tl.store(real_ptr + second_idx, a_real - tw_real, mask=m)
    tl.store(imag_ptr + second_idx, a_imag - tw_imag, mask=m)


# The synthetic launch of `scale_kernel`: its one runtime integer `n_elements` is the count (the generic recipe would
# find it alone), declared so the signature's float `scale` is not read as a second size: both buffers of the count,
# `scale` a drawn float, grid cdiv(n_elements, tile) as the call site's `_1d_grid(total)`.
@synthetic(flat("n_elements"))
@tile_independent
@triton.jit
def scale_kernel(
    real_ptr, imag_ptr,
    n_elements, scale,
    BLOCK_SIZE: tl.constexpr,
):
    """Scale all elements by 1/N after inverse FFT."""
    pid = tl.program_id(0).to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements

    r = tl.load(real_ptr + offset, mask=mask)
    i = tl.load(imag_ptr + offset, mask=mask)
    tl.store(real_ptr + offset, r * scale, mask=mask)
    tl.store(imag_ptr + offset, i * scale, mask=mask)
