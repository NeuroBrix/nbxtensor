"""DFT / iDFT basis matrices — pure @triton.jit kernels.

Builds the real cos/sin basis matrices for a Discrete Fourier Transform computed
by matrix multiplication. This is the standard small-N / non-power-of-2 path
(the DFT is an N×N matrix of complex exponentials applied to the signal) — used
when the radix-2 butterfly cannot run (n_fft not a power of 2, e.g. an iSTFT head
with n_fft=20). Model-agnostic: any non-pow2 FFT length routes here.

Layout (matches the rfft convention): frequency bins k in [0, N//2+1), samples
n in [0, N). The matrices are [N_bins, N], applied as `frames @ mat.T`.
"""

import triton
import nbxtensor.ops._launch_seam  # noqa: F401  this module's launches go through nbxtensor's launcher
import triton.language as tl


from nbxtensor.ops.ew_tile import synthetic, tile_dependent, tile_independent  # noqa: F401  (ops/ew_tile.py)


def _basis_launch(S):
    """The synthetic launch of a DFT basis kernel (`ops/ew_synthetic.py`), as `spectral_wrappers` launches it: an
    [N_bins, N] pair of matrices, N the ladder's count (the row the lanes walk: k = offs // N, n = offs % N), N_bins
    the fewest bins of the variant's specialization (1, 2 or 16: the matrices stay kilobytes to a few hundred, the
    tails are the rows'). Every lane below N_bins x N writes its own element of the two matrices: each is exactly
    N_bins x N elements. Grid cdiv(N_bins x N, tile), the call site's `_1d_grid(total)`."""
    from nbxtensor.ops.elementwise_launch import per_tile
    bins = S.int("N_bins", 2)
    total = int(bins) * int(S.n)
    ptrs = [p for p in S.names if S.ty(p).startswith("*")]
    args = {"N_bins": bins, "N": S.n}
    for p in ptrs:
        args[p] = S.buf(p, total)
    return args, (per_tile(total),)


_basis_launch.count = "N"
_basis_launch.description = "dft basis: N = count, N_bins 1/2/16, two [N_bins, N] matrices, grid per_tile(N_bins x N)"


def _idft_basis_launch(S):
    """The synthetic launch of `idft_c2r_matrix_kernel`: the DFT basis launch (`_basis_launch`) and the Nyquist flag
    `n_even` as the call site passes it, 1 (the variant's constant 1) or 0 (the variant's 'D')."""
    from nbxtensor.ops.ew_synthetic import NotSynthesisable
    args, axes = _basis_launch(S)
    a = S.attr("n_even")
    if a == "D":
        args["n_even"] = 0
    elif isinstance(a, tuple):
        args["n_even"] = a[1]
    else:
        raise NotSynthesisable(f"idft_c2r_matrix_kernel: n_even specialised {a!r}: the call site passes 0 or 1")
    return args, axes


_idft_basis_launch.count = "N"
_idft_basis_launch.description = "idft basis: as dft basis, n_even 0 / 1 as the variant"


@synthetic(_basis_launch)
@tile_independent
@triton.jit
def dft_rfft_matrix_kernel(
    cos_ptr,      # (N_bins, N) — cos(2π k n / N)
    nsin_ptr,     # (N_bins, N) — -sin(2π k n / N)
    N_bins, N,
    two_pi_over_N,
    BLOCK_SIZE: tl.constexpr,
):
    """Forward rfft basis: X[k] = Σ_n x[n] (cos - i sin)(2π k n / N).

    cos[k,n] = cos(2π k n / N), nsin[k,n] = -sin(2π k n / N) so that
    X_real = x @ cos.T and X_imag = x @ nsin.T.
    """
    pid = tl.program_id(0).to(tl.int64)
    offs = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    total = N_bins * N
    mask = offs < total
    k = offs // N
    n = offs % N
    angle = (k * n).to(tl.float32) * two_pi_over_N
    tl.store(cos_ptr + offs, tl.cos(angle), mask=mask)
    tl.store(nsin_ptr + offs, -tl.sin(angle), mask=mask)


@synthetic(_idft_basis_launch)
@tile_independent
@triton.jit
def idft_c2r_matrix_kernel(
    cos_ptr,      # (N_bins, N) — weight_k cos(2π k n / N) / N
    sin_ptr,      # (N_bins, N) — -weight_k sin(2π k n / N) / N
    N_bins, N,
    two_pi_over_N,
    inv_N,
    n_even,       # 1 if N is even (Nyquist bin present), else 0
    BLOCK_SIZE: tl.constexpr,
):
    """Inverse rfft (irfft) basis, exploiting Hermitian symmetry.

    x[n] = (1/N) Σ_k weight_k [Re(X[k]) cos(2π k n / N) - Im(X[k]) sin(2π k n / N)]
    weight_k = 1 for k=0 and (N even) k=N/2, else 2. So
    real_out = X_real @ cos + X_imag @ sin.
    """
    pid = tl.program_id(0).to(tl.int64)
    offs = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    total = N_bins * N
    mask = offs < total
    k = offs // N
    n = offs % N
    is_edge = (k == 0) | ((n_even == 1) & (k == N_bins - 1))
    weight = tl.where(is_edge, 1.0, 2.0) * inv_N
    angle = (k * n).to(tl.float32) * two_pi_over_N
    tl.store(cos_ptr + offs, weight * tl.cos(angle), mask=mask)
    tl.store(sin_ptr + offs, -weight * tl.sin(angle), mask=mask)
