"""The gated delta rule over a sequence, CHUNKED: the same recurrence as `ops.gated_delta_rule` (its serial scan), solved
`GATED_DELTA_RULE_CHUNK` tokens at a time with matrix products, the state carried from chunk to chunk float32.

PORTED FROM flash-linear-attention (fla), MIT License, Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
(https://github.com/fla-org/flash-linear-attention; the license text is in THIRD_PARTY_NOTICES.md). Source: the PyPI
release fla-core 0.6.0 (the wheel carries no upstream commit; the files are pinned by their sha256):
    fla/ops/gated_delta_rule/chunk_fwd.py   09ab23a57ee31dcda63785ad7e5bd0ff803efebab93ff69c0d05b103b5ae24f6
    fla/ops/gated_delta_rule/wy_fast.py     3689cf7aa45c6a21e66074edea280febd5ed893dc1ae5da9ee23aa9a57a672ae
    fla/ops/common/chunk_delta_h.py         ac9d7c0b841e8bab354ac1e6336b289460709fa5cfa3d466e0d87577b3e5d651
    fla/ops/common/chunk_o.py               1f49bbccafec99563bd8a00d10ac70651d8ced353c3ace96199fe61b94a286a2
    fla/ops/gated_delta_rule/chunk.py       d5334e3c70ad775b349fd0abd8f01703dfc82373521ac927b3bbd5ec76d961a6
Nothing of fla is imported: its forward algebra is re-written here on the library's tensors, strides and launches.
What changed against the source: (1) one prepare kernel fuses fla's chunk-local cumsum of g, its fused
k k^T + triangular solve (`chunk_gated_delta_rule_fwd_kkt_solve_kernel`, the four 16-row sub-blocks, their forward
substitution and their block merge, kept as written) and `recompute_w_u_fwd_kernel`, so the solved (I + A)^-1 never
leaves the chip; (2) the q and k L2 norm is inside the kernels (the Gram matrix's diagonal gives k's norms; the state
and output kernels sum q*q and k*k over their own tiles), never a normalised copy; (3) every product, the chunk states
and the corrected values are float32 at every input dtype (fla stores its chunk states and its inverse at the input
dtype and lets float32 products run as tf32): the deviation bound of the serial scan holds; (4) the gate is in natural
units (`exp`), not log2; (5) q, k, v, g and beta are read through their own strides, q and k un-expanded (Hk heads);
(6) the state comes in and goes back [B, Hv, dk, dv] float32 like the serial scan's, so the two forms hand it to each
other; (7) no variable-length packing, no context parallelism, no backward.

THE ALGEBRA (per batch row b, value head h, chunk c of positions t0..t0+C-1; gc the cumulative sum of g inside the
chunk; k and q normalised and q scaled as the serial scan does them; S the state at the chunk's start):
    A   = strictly-lower( beta_i * k_i . k_j * exp(gc_i - gc_j) )          [C, C]
    W   = (I + A)^-1 diag(beta * exp(gc)) K,   U = (I + A)^-1 diag(beta) V    (the WY form of the chunk's writes)
    V'  = U - W S                                                         (the corrected values)
    O   = diag(exp(gc)) Q S + tril( Q K^T * exp(gc_i - gc_j) ) V'
    S  <- exp(gc_last) S + K^T diag(exp(gc_last - gc)) V'
Three kernels: `gated_delta_rule_chunk_prepare_kernel` (one program per chunk and value head: gc, W, U), then
`gated_delta_rule_chunk_state_kernel` (one program per value head and dv tile: the only serial loop, over chunks;
writes each chunk's starting state and V' over U), then `gated_delta_rule_chunk_output_kernel` (one program per chunk,
value head and dv tile: O). The tiles and warps are the certified configuration of each kernel's key (dk, dv),
never tuned at run time; the length is symbolic (a loop bound and a grid extent, never a key).

Which form runs a call is not decided here: `ops.gated_delta_rule.gated_delta_rule_op` reads it from the census
(`gated_delta_rule_census.json`, beside this file). Its float64 reference is `nbxtensor.ops.oracles.fp64.
delta_rule_oracle`, the same as the serial scan's.
"""
from __future__ import annotations

from typing import Dict, Tuple

import triton
import nbxtensor.ops._launch_seam  # noqa: F401  this module's launches go through nbxtensor's launcher
import triton.language as tl

from nbxtensor.backend.runtime import set_device
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.ops._configs import nbx_autotune
from nbxtensor.ops.gated_delta_rule import gated_delta_rule_dtypes, gated_delta_rule_shapes
from nbxtensor.tensor import NBXTensor

#: The tokens solved together, and the sub-block of the fused triangular solve: the solve is written for exactly four
#: sub-blocks (fla's fused path), so the chunk is four times the sub-block.
GATED_DELTA_RULE_SUB_BLOCK = 16
GATED_DELTA_RULE_CHUNK = 4 * GATED_DELTA_RULE_SUB_BLOCK

_PREPARE_CONFIGS = [triton.Config({'BK': bk, 'BV': bv}, num_warps=w, num_stages=1)
                    for bk in (32, 64) for bv in (32, 64) for w in (1, 2, 4)]
_STATE_CONFIGS = [triton.Config({'BV': bv}, num_warps=w, num_stages=s)
                  for bv in (16, 32, 64) for w in (2, 4, 8) for s in (1, 2)]
_OUTPUT_CONFIGS = [triton.Config({'BK': bk, 'BV': bv}, num_warps=w, num_stages=s)
                   for bk in (32, 64) for bv in (32, 64) for w in (4, 8) for s in (1, 2)]


@triton.jit
def _forward_substitution(a, t_left, o_i, BC: tl.constexpr):
    """(I + a)^-1 of a strictly-lower [BC, BC] block, its first `t_left` rows real (fla's in-register solve)."""
    ai = -a
    for i in range(2, min(BC, t_left)):
        row = tl.sum(tl.where((o_i == i)[:, None], -a, 0.), 0)
        row = tl.where(o_i < i, row, 0.)
        row = row + tl.sum(row[:, None] * ai, 0)
        ai = tl.where((o_i == i)[:, None], row, ai)
    return ai + (o_i[:, None] == o_i[None, :]).to(tl.float32)


@triton.jit
def _ieee(a, b):
    return tl.dot(a, b, input_precision="ieee")


@nbx_autotune(configs=_PREPARE_CONFIGS, key=['DK', 'DV'], cache_results=True)
@triton.jit
def gated_delta_rule_chunk_prepare_kernel(
    k_ptr, v_ptr, g_ptr, beta_ptr,            # [B, T, Hk, DK], [B, T, Hv, DV], [B, T, Hv] x2: own strides
    w_ptr, u_ptr, gc_ptr,                     # [B, T, Hv, DK], [B, T, Hv, DV], [B, T, Hv] float32, contiguous
    T, DK, DV, HV, GROUPS,
    sk_b, sk_t, sk_h, sk_d,
    sv_b, sv_t, sv_h, sv_d,
    sg_b, sg_t, sg_h,
    sb_b, sb_t, sb_h,
    eps,
    QK_L2NORM: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    b = (i_bh // HV).to(tl.int64)
    h = (i_bh % HV).to(tl.int64)
    kh = h // GROUPS
    o_i = tl.arange(0, BC)
    t0 = i_t * BT
    t1, t2, t3 = t0 + BC, t0 + 2 * BC, t0 + 3 * BC
    r0, r1, r2, r3 = t0 + o_i, t1 + o_i, t2 + o_i, t3 + o_i
    m0, m1, m2, m3 = r0 < T, r1 < T, r2 < T, r3 < T

    # the gate, summed inside the chunk, and beta
    gb = g_ptr + b * sg_b + h * sg_h
    g0 = tl.load(gb + r0.to(tl.int64) * sg_t, mask=m0, other=0.).to(tl.float32)
    g1 = tl.load(gb + r1.to(tl.int64) * sg_t, mask=m1, other=0.).to(tl.float32)
    g2 = tl.load(gb + r2.to(tl.int64) * sg_t, mask=m2, other=0.).to(tl.float32)
    g3 = tl.load(gb + r3.to(tl.int64) * sg_t, mask=m3, other=0.).to(tl.float32)
    c0 = tl.cumsum(g0, 0)
    c1 = tl.cumsum(g1, 0) + tl.sum(g0, 0)
    c2 = tl.cumsum(g2, 0) + tl.sum(g0, 0) + tl.sum(g1, 0)
    c3 = tl.cumsum(g3, 0) + tl.sum(g0, 0) + tl.sum(g1, 0) + tl.sum(g2, 0)
    gcb = gc_ptr + (b * T) * HV + h
    tl.store(gcb + r0.to(tl.int64) * HV, c0, mask=m0)
    tl.store(gcb + r1.to(tl.int64) * HV, c1, mask=m1)
    tl.store(gcb + r2.to(tl.int64) * HV, c2, mask=m2)
    tl.store(gcb + r3.to(tl.int64) * HV, c3, mask=m3)
    bb = beta_ptr + b * sb_b + h * sb_h
    b0 = tl.load(bb + r0.to(tl.int64) * sb_t, mask=m0, other=0.).to(tl.float32)
    b1 = tl.load(bb + r1.to(tl.int64) * sb_t, mask=m1, other=0.).to(tl.float32)
    b2 = tl.load(bb + r2.to(tl.int64) * sb_t, mask=m2, other=0.).to(tl.float32)
    b3 = tl.load(bb + r3.to(tl.int64) * sb_t, mask=m3, other=0.).to(tl.float32)

    # the ten lower blocks of the Gram matrix k k^T
    a00 = tl.zeros([BC, BC], dtype=tl.float32)
    a11 = tl.zeros([BC, BC], dtype=tl.float32)
    a22 = tl.zeros([BC, BC], dtype=tl.float32)
    a33 = tl.zeros([BC, BC], dtype=tl.float32)
    a10 = tl.zeros([BC, BC], dtype=tl.float32)
    a20 = tl.zeros([BC, BC], dtype=tl.float32)
    a21 = tl.zeros([BC, BC], dtype=tl.float32)
    a30 = tl.zeros([BC, BC], dtype=tl.float32)
    a31 = tl.zeros([BC, BC], dtype=tl.float32)
    a32 = tl.zeros([BC, BC], dtype=tl.float32)
    kb = k_ptr + b * sk_b + kh * sk_h
    for i_k in range(0, tl.cdiv(DK, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        mk = o_k < DK
        pk = kb + o_k[None, :].to(tl.int64) * sk_d
        k0 = tl.load(pk + r0[:, None].to(tl.int64) * sk_t, mask=m0[:, None] & mk[None, :], other=0.).to(tl.float32)
        k1 = tl.load(pk + r1[:, None].to(tl.int64) * sk_t, mask=m1[:, None] & mk[None, :], other=0.).to(tl.float32)
        k2 = tl.load(pk + r2[:, None].to(tl.int64) * sk_t, mask=m2[:, None] & mk[None, :], other=0.).to(tl.float32)
        k3 = tl.load(pk + r3[:, None].to(tl.int64) * sk_t, mask=m3[:, None] & mk[None, :], other=0.).to(tl.float32)
        a00 = tl.dot(k0, tl.trans(k0), a00, input_precision="ieee")
        a11 = tl.dot(k1, tl.trans(k1), a11, input_precision="ieee")
        a22 = tl.dot(k2, tl.trans(k2), a22, input_precision="ieee")
        a33 = tl.dot(k3, tl.trans(k3), a33, input_precision="ieee")
        a10 = tl.dot(k1, tl.trans(k0), a10, input_precision="ieee")
        a20 = tl.dot(k2, tl.trans(k0), a20, input_precision="ieee")
        a21 = tl.dot(k2, tl.trans(k1), a21, input_precision="ieee")
        a30 = tl.dot(k3, tl.trans(k0), a30, input_precision="ieee")
        a31 = tl.dot(k3, tl.trans(k1), a31, input_precision="ieee")
        a32 = tl.dot(k3, tl.trans(k2), a32, input_precision="ieee")

    # k's norms, read on the Gram diagonal (rows and columns)
    m_I = o_i[:, None] == o_i[None, :]
    if QK_L2NORM:
        n0 = 1.0 / tl.sqrt(tl.sum(tl.where(m_I, a00, 0.), 1) + eps)
        n1 = 1.0 / tl.sqrt(tl.sum(tl.where(m_I, a11, 0.), 1) + eps)
        n2 = 1.0 / tl.sqrt(tl.sum(tl.where(m_I, a22, 0.), 1) + eps)
        n3 = 1.0 / tl.sqrt(tl.sum(tl.where(m_I, a33, 0.), 1) + eps)
        n0c = 1.0 / tl.sqrt(tl.sum(tl.where(m_I, a00, 0.), 0) + eps)
        n1c = 1.0 / tl.sqrt(tl.sum(tl.where(m_I, a11, 0.), 0) + eps)
        n2c = 1.0 / tl.sqrt(tl.sum(tl.where(m_I, a22, 0.), 0) + eps)
        a00 = a00 * n0[:, None] * n0c[None, :]
        a11 = a11 * n1[:, None] * n1c[None, :]
        a22 = a22 * n2[:, None] * n2c[None, :]
        a33 = a33 * n3[:, None] * (1.0 / tl.sqrt(tl.sum(tl.where(m_I, a33, 0.), 0) + eps))[None, :]
        a10 = a10 * n1[:, None] * n0c[None, :]
        a20 = a20 * n2[:, None] * n0c[None, :]
        a21 = a21 * n2[:, None] * n1c[None, :]
        a30 = a30 * n3[:, None] * n0c[None, :]
        a31 = a31 * n3[:, None] * n1c[None, :]
        a32 = a32 * n3[:, None] * n2c[None, :]
    else:
        n0 = tl.full([BC], 1.0, tl.float32)
        n1 = tl.full([BC], 1.0, tl.float32)
        n2 = tl.full([BC], 1.0, tl.float32)
        n3 = tl.full([BC], 1.0, tl.float32)

    # the gate and beta: A = strictly-lower(beta_i k_i.k_j exp(gc_i - gc_j)), a padded row or column zero
    m_d = o_i[:, None] > o_i[None, :]
    a00 *= tl.where(m_d & m0[:, None] & m0[None, :], tl.exp(c0[:, None] - c0[None, :]), 0.) * b0[:, None]
    a11 *= tl.where(m_d & m1[:, None] & m1[None, :], tl.exp(c1[:, None] - c1[None, :]), 0.) * b1[:, None]
    a22 *= tl.where(m_d & m2[:, None] & m2[None, :], tl.exp(c2[:, None] - c2[None, :]), 0.) * b2[:, None]
    a33 *= tl.where(m_d & m3[:, None] & m3[None, :], tl.exp(c3[:, None] - c3[None, :]), 0.) * b3[:, None]
    a10 *= tl.where(m1[:, None] & m0[None, :], tl.exp(c1[:, None] - c0[None, :]), 0.) * b1[:, None]
    a20 *= tl.where(m2[:, None] & m0[None, :], tl.exp(c2[:, None] - c0[None, :]), 0.) * b2[:, None]
    a21 *= tl.where(m2[:, None] & m1[None, :], tl.exp(c2[:, None] - c1[None, :]), 0.) * b2[:, None]
    a30 *= tl.where(m3[:, None] & m0[None, :], tl.exp(c3[:, None] - c0[None, :]), 0.) * b3[:, None]
    a31 *= tl.where(m3[:, None] & m1[None, :], tl.exp(c3[:, None] - c1[None, :]), 0.) * b3[:, None]
    a32 *= tl.where(m3[:, None] & m2[None, :], tl.exp(c3[:, None] - c2[None, :]), 0.) * b3[:, None]

    # (I + A)^-1: the diagonal blocks by forward substitution, then the block merge
    i00 = _forward_substitution(a00, T - t0, o_i, BC)
    i11 = _forward_substitution(a11, T - t1, o_i, BC)
    i22 = _forward_substitution(a22, T - t2, o_i, BC)
    i33 = _forward_substitution(a33, T - t3, o_i, BC)
    i10 = -_ieee(_ieee(i11, a10), i00)
    i21 = -_ieee(_ieee(i22, a21), i11)
    i32 = -_ieee(_ieee(i33, a32), i22)
    i20 = -_ieee(i22, _ieee(a20, i00) + _ieee(a21, i10))
    i31 = -_ieee(i33, _ieee(a31, i11) + _ieee(a32, i21))
    i30 = -_ieee(i33, _ieee(a30, i00) + _ieee(a31, i10) + _ieee(a32, i20))

    # U = (I + A)^-1 diag(beta) V
    vb = v_ptr + b * sv_b + h * sv_h
    ub = u_ptr + (b * T * HV + h) * DV
    for i_v in range(0, tl.cdiv(DV, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        mv = o_v < DV
        pv = vb + o_v[None, :].to(tl.int64) * sv_d
        x0 = tl.load(pv + r0[:, None].to(tl.int64) * sv_t, mask=m0[:, None] & mv[None, :], other=0.).to(tl.float32)
        x1 = tl.load(pv + r1[:, None].to(tl.int64) * sv_t, mask=m1[:, None] & mv[None, :], other=0.).to(tl.float32)
        x2 = tl.load(pv + r2[:, None].to(tl.int64) * sv_t, mask=m2[:, None] & mv[None, :], other=0.).to(tl.float32)
        x3 = tl.load(pv + r3[:, None].to(tl.int64) * sv_t, mask=m3[:, None] & mv[None, :], other=0.).to(tl.float32)
        x0, x1, x2, x3 = x0 * b0[:, None], x1 * b1[:, None], x2 * b2[:, None], x3 * b3[:, None]
        pu = ub + o_v[None, :]
        tl.store(pu + r0[:, None].to(tl.int64) * (HV * DV), _ieee(i00, x0), mask=m0[:, None] & mv[None, :])
        tl.store(pu + r1[:, None].to(tl.int64) * (HV * DV), _ieee(i10, x0) + _ieee(i11, x1),
                 mask=m1[:, None] & mv[None, :])
        tl.store(pu + r2[:, None].to(tl.int64) * (HV * DV), _ieee(i20, x0) + _ieee(i21, x1) + _ieee(i22, x2),
                 mask=m2[:, None] & mv[None, :])
        tl.store(pu + r3[:, None].to(tl.int64) * (HV * DV),
                 _ieee(i30, x0) + _ieee(i31, x1) + _ieee(i32, x2) + _ieee(i33, x3), mask=m3[:, None] & mv[None, :])

    # W = (I + A)^-1 diag(beta exp(gc)) K (K normalised)
    s0, s1, s2, s3 = b0 * tl.exp(c0) * n0, b1 * tl.exp(c1) * n1, b2 * tl.exp(c2) * n2, b3 * tl.exp(c3) * n3
    wb = w_ptr + (b * T * HV + h) * DK
    for i_k in range(0, tl.cdiv(DK, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        mk = o_k < DK
        pk = kb + o_k[None, :].to(tl.int64) * sk_d
        x0 = tl.load(pk + r0[:, None].to(tl.int64) * sk_t, mask=m0[:, None] & mk[None, :], other=0.).to(tl.float32)
        x1 = tl.load(pk + r1[:, None].to(tl.int64) * sk_t, mask=m1[:, None] & mk[None, :], other=0.).to(tl.float32)
        x2 = tl.load(pk + r2[:, None].to(tl.int64) * sk_t, mask=m2[:, None] & mk[None, :], other=0.).to(tl.float32)
        x3 = tl.load(pk + r3[:, None].to(tl.int64) * sk_t, mask=m3[:, None] & mk[None, :], other=0.).to(tl.float32)
        x0, x1, x2, x3 = x0 * s0[:, None], x1 * s1[:, None], x2 * s2[:, None], x3 * s3[:, None]
        pw = wb + o_k[None, :]
        tl.store(pw + r0[:, None].to(tl.int64) * (HV * DK), _ieee(i00, x0), mask=m0[:, None] & mk[None, :])
        tl.store(pw + r1[:, None].to(tl.int64) * (HV * DK), _ieee(i10, x0) + _ieee(i11, x1),
                 mask=m1[:, None] & mk[None, :])
        tl.store(pw + r2[:, None].to(tl.int64) * (HV * DK), _ieee(i20, x0) + _ieee(i21, x1) + _ieee(i22, x2),
                 mask=m2[:, None] & mk[None, :])
        tl.store(pw + r3[:, None].to(tl.int64) * (HV * DK),
                 _ieee(i30, x0) + _ieee(i31, x1) + _ieee(i32, x2) + _ieee(i33, x3), mask=m3[:, None] & mk[None, :])


# The state kernel writes V' over U, its own input: tuning benchmarks every config on the same buffers, so U is restored
# before each one (without it every config after the first read a V' as its U, and the cold first call returned another
# sum: 2.96e+02 against a 4e-06 bound, Mac M4 Pro 2026-11-19).
@nbx_autotune(configs=_STATE_CONFIGS, key=['DK', 'DV'], cache_results=True, restore_value=['u_ptr'])
@triton.jit
def gated_delta_rule_chunk_state_kernel(
    k_ptr,                                    # [B, T, Hk, DK]: own strides
    w_ptr, u_ptr, gc_ptr,                     # the prepare kernel's W, U (V' written over it) and gc, float32
    hs_ptr,                                   # [B, NT, Hv, DK, DV] float32: each chunk's starting state
    s_in_ptr,                                 # [B, Hv, DK, DV] float32 (read only when HAS_STATE)
    s_out_ptr,                                # [B, Hv, DK, DV] float32, contiguous
    T, DK, DV, HV, GROUPS,
    sk_b, sk_t, sk_h, sk_d,
    ss_b, ss_h, ss_k, ss_v,
    eps,
    HAS_STATE: tl.constexpr,
    QK_L2NORM: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_K: tl.constexpr,                    # DK rounded up to a power of two: the whole key axis in one tile
    BV: tl.constexpr,
):
    i_v = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    b = (i_bh // HV).to(tl.int64)
    h = (i_bh % HV).to(tl.int64)
    kh = h // GROUPS
    o_k = tl.arange(0, BLOCK_K)
    o_v = i_v * BV + tl.arange(0, BV)
    mk, mv = o_k < DK, o_v < DV
    m_s = mk[:, None] & mv[None, :]
    if HAS_STATE:
        s = tl.load(s_in_ptr + b * ss_b + h * ss_h + o_k[:, None] * ss_k + o_v[None, :] * ss_v,
                    mask=m_s, other=0.).to(tl.float32)
    else:
        s = tl.zeros([BLOCK_K, BV], dtype=tl.float32)
    NT = tl.cdiv(T, BT)
    kb = k_ptr + b * sk_b + kh * sk_h
    wb = w_ptr + (b * T * HV + h) * DK
    ub = u_ptr + (b * T * HV + h) * DV
    gb = gc_ptr + b * T * HV + h
    for i_t in range(0, NT):                  # a 32-bit bound: the Metal lowering refuses a 64-bit scf.for bound
        tl.store(hs_ptr + (((b * NT + i_t) * HV + h) * DK + o_k[:, None]) * DV + o_v[None, :], s, mask=m_s)
        o_t = i_t * BT + tl.arange(0, BT)
        mt = o_t < T
        rows = o_t.to(tl.int64)
        w = tl.load(wb + rows[:, None] * (HV * DK) + o_k[None, :], mask=mt[:, None] & mk[None, :], other=0.)
        pu = ub + rows[:, None] * (HV * DV) + o_v[None, :]
        vn = tl.load(pu, mask=mt[:, None] & mv[None, :], other=0.) - tl.dot(w, s, input_precision="ieee")
        tl.store(pu, vn, mask=mt[:, None] & mv[None, :])        # V' over U: the output kernel reads it
        last = (tl.minimum((i_t + 1) * BT, T) - 1).to(tl.int64)
        gl = tl.load(gb + last * HV)
        gc = tl.load(gb + rows * HV, mask=mt, other=0.)
        vn = vn * tl.where(mt, tl.exp(gl - gc), 0.)[:, None]
        kt = tl.load(kb + o_k[:, None].to(tl.int64) * sk_d + rows[None, :] * sk_t, mask=mk[:, None] & mt[None, :],
                     other=0.).to(tl.float32)
        if QK_L2NORM:
            kt = kt * (1.0 / tl.sqrt(tl.sum(kt * kt, 0) + eps))[None, :]
        s = tl.dot(kt, vn, s * tl.exp(gl), input_precision="ieee")
    tl.store(s_out_ptr + ((b * HV + h) * DK + o_k[:, None]) * DV + o_v[None, :], s, mask=m_s)


@nbx_autotune(configs=_OUTPUT_CONFIGS, key=['DK', 'DV'], cache_results=True)
@triton.jit
def gated_delta_rule_chunk_output_kernel(
    q_ptr, k_ptr,                             # [B, T, Hk, DK] x2: own strides
    u_ptr, gc_ptr, hs_ptr,                    # V', gc and the chunk states, float32, contiguous
    o_ptr,                                    # [B, T, Hv, DV] at v's dtype
    T, DK, DV, HV, GROUPS,
    sq_b, sq_t, sq_h, sq_d,
    sk_b, sk_t, sk_h, sk_d,
    so_b, so_t, so_h, so_d,
    eps, scale,
    QK_L2NORM: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_v = tl.program_id(0).to(tl.int64)
    i_t = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    b = (i_bh // HV).to(tl.int64)
    h = (i_bh % HV).to(tl.int64)
    kh = h // GROUPS
    NT = tl.cdiv(T, BT)
    o_t = i_t * BT + tl.arange(0, BT)
    mt = o_t < T
    rows = o_t.to(tl.int64)
    o_v = i_v * BV + tl.arange(0, BV)
    mv = o_v < DV
    qb = q_ptr + b * sq_b + kh * sq_h
    kb = k_ptr + b * sk_b + kh * sk_h
    hb = hs_ptr + ((b * NT + i_t) * HV + h) * DK * DV
    acc = tl.zeros([BT, BV], dtype=tl.float32)
    a = tl.zeros([BT, BT], dtype=tl.float32)
    qq = tl.zeros([BT], dtype=tl.float32)
    kk = tl.zeros([BT], dtype=tl.float32)
    for i_k in range(0, tl.cdiv(DK, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        mk = o_k < DK
        q = tl.load(qb + rows[:, None] * sq_t + o_k[None, :].to(tl.int64) * sq_d, mask=mt[:, None] & mk[None, :],
                    other=0.).to(tl.float32)
        kt = tl.load(kb + o_k[:, None].to(tl.int64) * sk_d + rows[None, :] * sk_t, mask=mk[:, None] & mt[None, :],
                     other=0.).to(tl.float32)
        st = tl.load(hb + o_k[:, None] * DV + o_v[None, :], mask=mk[:, None] & mv[None, :], other=0.)
        acc = tl.dot(q, st, acc, input_precision="ieee")
        a = tl.dot(q, kt, a, input_precision="ieee")
        if QK_L2NORM:
            qq += tl.sum(q * q, 1)
            kk += tl.sum(kt * kt, 0)
    if QK_L2NORM:
        rq = scale / tl.sqrt(qq + eps)
        rk = 1.0 / tl.sqrt(kk + eps)
    else:
        rq = tl.full([BT], 1.0, tl.float32) * scale
        rk = tl.full([BT], 1.0, tl.float32)
    gc = tl.load(gc_ptr + (b * T + rows) * HV + h, mask=mt, other=0.)
    acc = acc * (rq * tl.exp(gc))[:, None]
    causal = (o_t[:, None] >= o_t[None, :]) & mt[:, None] & mt[None, :]
    a = a * rq[:, None] * rk[None, :] * tl.where(causal, tl.exp(gc[:, None] - gc[None, :]), 0.)
    vn = tl.load(u_ptr + ((b * T + rows[:, None]) * HV + h) * DV + o_v[None, :], mask=mt[:, None] & mv[None, :],
                 other=0.)
    out = tl.dot(a, vn, acc, input_precision="ieee")
    tl.store(o_ptr + b * so_b + rows[:, None] * so_t + h * so_h + o_v[None, :].to(tl.int64) * so_d,
             out.to(o_ptr.dtype.element_ty), mask=mt[:, None] & mv[None, :])


def gated_delta_rule_chunked_workspace(B: int, L: int, Hv: int, dk: int, dv: int) -> Dict[str, Tuple[int, ...]]:
    """The float32 buffers the chunked form allocates beside its two outputs, by name: W [B, L, Hv, dk], U (then V')
    [B, L, Hv, dv], gc [B, L, Hv] and the chunk states [B, ceil(L / chunk), Hv, dk, dv]. One answer for the wrapper,
    its cost and its tests."""
    chunks = triton.cdiv(int(L), GATED_DELTA_RULE_CHUNK)
    return {"w": (B, L, Hv, dk), "u": (B, L, Hv, dv), "gc": (B, L, Hv), "states": (B, chunks, Hv, dk, dv)}


def gated_delta_rule_chunked_tuning_copies(B: int, L: int, Hv: int, dk: int, dv: int) -> Dict[str, Tuple[int, ...]]:
    """The float32 buffers a call that tunes holds beside the workspace, by the state kernel's argument they copy: the
    state kernel overwrites U with V' and declares `restore_value=['u_ptr']`, so the sweep clones U before each
    configuration it measures and copies it back after (one copy alive at a time; 134 MB at B=1, L=8192, Hv=32,
    dv=128). A call that finds its setting tuned makes none. One answer for the cost and its tests."""
    return {"u_ptr": (B, L, Hv, dv)}


def gated_delta_rule_chunked_launch(q, k, v, g, beta, state, out, state_out, ws, *, qk_l2norm: bool, eps: float,
                                    scale: float) -> None:
    """The three launches over buffers already formed (shapes and dtypes checked by the caller): q, k, v, g, beta and
    `state` (None: zeros) read through their own strides, `out` and `state_out` written, `ws` the workspace of
    `gated_delta_rule_chunked_workspace` by name (contiguous float32)."""
    B, L, Hk, dk = q.shape
    Hv, dv = v.shape[2], v.shape[3]
    groups = Hv // Hk
    w, u, gc, states = ws["w"], ws["u"], ws["gc"], ws["states"]
    nt = triton.cdiv(L, GATED_DELTA_RULE_CHUNK)
    l2, eps, scale = bool(qk_l2norm), float(eps), float(scale)
    gated_delta_rule_chunk_prepare_kernel[(nt, B * Hv)](
        k, v, g, beta, w, u, gc, L, dk, dv, Hv, groups,
        *k.stride(), *v.stride(), *g.stride(), *beta.stride(), eps,
        QK_L2NORM=l2, BT=GATED_DELTA_RULE_CHUNK, BC=GATED_DELTA_RULE_SUB_BLOCK)
    state_in = state_out if state is None else state
    gated_delta_rule_chunk_state_kernel[lambda META: (triton.cdiv(dv, META['BV']), B * Hv)](
        k, w, u, gc, states, state_in, state_out, L, dk, dv, Hv, groups,
        *k.stride(), *state_in.stride(), eps,
        HAS_STATE=state is not None, QK_L2NORM=l2, BT=GATED_DELTA_RULE_CHUNK,
        BLOCK_K=max(triton.next_power_of_2(dk), GATED_DELTA_RULE_SUB_BLOCK))
    gated_delta_rule_chunk_output_kernel[lambda META: (triton.cdiv(dv, META['BV']), nt, B * Hv)](
        q, k, u, gc, states, out, L, dk, dv, Hv, groups,
        *q.stride(), *k.stride(), *out.stride(), eps, scale,
        QK_L2NORM=l2, BT=GATED_DELTA_RULE_CHUNK)


def gated_delta_rule_chunked_wrapper(q, k, v, g, beta, state=None, *, qk_l2norm: bool, eps: float, scale: float,
                                     compute_dtype):
    """The gated delta rule, chunked, over q, k [B, L, Hk, dk], v [B, L, Hv, dv], g, beta [B, L, Hv] from `state`
    [B, Hv, dk, dv] float32 (None: zeros) -> (o [B, L, Hv, dv] at v's dtype, state_out [B, Hv, dk, dv] float32, a new
    tensor): the contract of `ops.gated_delta_rule.gated_delta_rule_wrapper` (the same shape and dtype checks, the
    same refusals by name), the state handed to and from the serial scan unchanged. Beside its two outputs it allocates
    the float32 workspace `gated_delta_rule_chunked_workspace` names."""
    B, L, Hk, Hv, dk, dv = gated_delta_rule_shapes(q, k, v, g, beta, state)
    gated_delta_rule_dtypes(*(None if t is None else t.nbx_dtype for t in (q, k, v, g, beta, state)), compute_dtype)
    out = NBXTensor.empty((B, L, Hv, dv), dtype=v.nbx_dtype, device=v.device)
    state_out = NBXTensor.empty((B, Hv, dk, dv), dtype=NBXDtype.float32, device=v.device)
    ws = {name: NBXTensor.empty(shape, dtype=NBXDtype.float32, device=v.device)
          for name, shape in gated_delta_rule_chunked_workspace(B, L, Hv, dk, dv).items()}
    set_device(v)
    gated_delta_rule_chunked_launch(q, k, v, g, beta, state, out, state_out, ws, qk_l2norm=qk_l2norm, eps=eps,
                                    scale=scale)
    return out, state_out


gated_delta_rule_chunked_wrapper.self_manages_dtype = True
