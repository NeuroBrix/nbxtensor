"""RMS norm, layer norm, the vector GEMV and the paged decode attention give the same bits under EVERY launch
candidate in the kernel's own run-time list (its configs through its own prune): the list a user's autotune sweeps
for a key with no certificate, and a user's certification proves. The certified directory gives a 16 GB and a 32 GB
card different settings for one key (on a V100, 2026-11-18: the gemv chunk width and the paged decode tile differed,
and each alone turned a served conversation's fifth answer), and a run-time sweep picks by timing.

The GEMV sums K in fixed sub-chunks whatever its chunk width; the paged decode steps its online softmax by one fixed
tile whatever BLOCK_N a certificate names (its run-time list is the warps only; every tile the certificates name is
launched here); every sum inside is pairwise (`ops.fixed_order`).

A card test: on a host with no device it skips by name.

A pairwise sum of products is not enough by itself: the compiler fuses a multiply into the add it feeds, on the
side the launch's layout chooses, so the products' rounding still follows the warps (measured 2026-11-18: gemv rows
per program and RMS norm warps changed bits with the pairwise sum alone). `ops.fixed_order.row_dot_fixed_order`
writes the first level as one explicit fused multiply-add per pair. Likewise `/` lowers to a reciprocal and a
multiply that the compiler fused into the next add on some launches (a layer norm's [3, 33] row on 2 rows x 8 warps):
the norms divide with `tl.div_rn`.

What it catches: a reduction in these kernels whose order depends on any launch setting - red."""
import itertools
import numpy as np
import pytest
import triton


@pytest.fixture(autouse=True)
def _accelerator():
    from nbxtensor.memory.allocator import DeviceAllocator
    if DeviceAllocator.device_count() == 0:          # only a machine without an accelerator skips; any other error fails
        pytest.skip("no accelerator on this machine")


def _dev(a):
    from nbxtensor.tensor import NBXTensor
    return NBXTensor.from_numpy(np.ascontiguousarray(a)).to(device="cuda:0")


def _same_within_groups(outs, group_of, what):
    groups = {}
    for cfg, out in outs.items():
        groups.setdefault(group_of(cfg), []).append((cfg, out))
    compared = 0
    for g, members in groups.items():
        first_cfg, first = members[0]
        other = [c for c, o in members[1:] if not np.array_equal(o.view(np.uint32), first.view(np.uint32))]
        compared += len(members) - 1
        assert not other, f"{what} group {g}: {len(other)} of {len(members)} candidates give other bits than {first_cfg}: {other}"
    assert compared > 0, f"{what}: no two candidates in one group, nothing compared"


RMS_SHAPES = [(64, 4096), (7, 2880), (320, 512), (3, 33), (5, 1)]   # a feature width of 1: the single product


@pytest.mark.parametrize("rows,feat", RMS_SHAPES, ids=[f"{r}x{f}" for r, f in RMS_SHAPES])
def test_rms_norm_every_candidate_gives_the_same_bits(rows, feat):
    from nbxtensor.ops import rmsnorm as R
    from nbxtensor.ops._configs import row_program_configs, prune_row_programs
    from nbxtensor.ops.launch_keys import bucket_of, softmax_feat_bucket
    from nbxtensor.tensor import NBXTensor
    rng = np.random.default_rng(rows * 31 + feat)
    a = (rng.standard_normal((rows, feat)) * 4.0).astype(np.float32)
    w = rng.standard_normal(feat).astype(np.float32)
    x, wt = _dev(a), _dev(w)
    y = NBXTensor.empty_like(x)
    kernel = R.rms_norm_forward_kernel.fn
    outs = {}
    for c in prune_row_programs(row_program_configs(), {"feat_dim": feat, "batch_dim": rows}):
        b, nw = c.kwargs["BLOCK_SIZE_BATCH"], c.num_warps
        kernel[(triton.cdiv(rows, b),)](x, wt, y, rows, feat, bucket_of("M", rows), softmax_feat_bucket(feat),
                                        x.stride(0), x.stride(1), y.stride(0), y.stride(1), 1e-6,
                                        scale_by_weight=True, BLOCK_SIZE_BATCH=b,
                                        BLOCK_SIZE_FEAT=triton.next_power_of_2(feat), num_warps=nw)
        outs[(b, nw)] = y.numpy().copy()
    _same_within_groups(outs, lambda cfg: "all", f"rms_norm {rows}x{feat}")
    ref = a / np.sqrt((a.astype(np.float64) ** 2).mean(axis=1, keepdims=True) + 1e-6) * w
    assert float(np.abs(next(iter(outs.values())) - ref).max()) <= 1e-4


GEMV_SHAPES = [(1024, 2048), (2880, 2880), (512, 4096)]


@pytest.mark.parametrize("n,k", GEMV_SHAPES, ids=[f"{n}x{k}" for n, k in GEMV_SHAPES])
def test_gemv_vec_every_candidate_gives_the_same_bits(n, k):
    from nbxtensor.ops import gemv_vec as G
    from nbxtensor.ops._configs import gemv_vec_configs, prune_gemv_vec
    from nbxtensor.ops.launch_keys import bucket_of
    from nbxtensor.tensor import NBXTensor
    rng = np.random.default_rng(n * 7 + k)
    A = rng.standard_normal((n, k)).astype(np.float16)
    B = rng.standard_normal(k).astype(np.float32)
    a, b = _dev(A), _dev(B)
    c = NBXTensor.from_numpy(np.zeros(n, np.float32)).to(device="cuda:0")
    kernel = G.gemv_vec_kernel.fn
    outs = {}
    for cfg in prune_gemv_vec(gemv_vec_configs(), {"N": n, "K": k}):
        bn, bk = cfg.kwargs["BLOCK_N"], cfg.kwargs["BLOCK_K"]
        kernel[(triton.cdiv(n, bn),)](a, b, c, n, k, bucket_of("N", n), bucket_of("K", k),
                                      a.stride(0), a.stride(1), b.stride(0), c.stride(0),
                                      BLOCK_N=bn, BLOCK_K=bk, num_warps=cfg.num_warps, num_stages=cfg.num_stages)
        outs[(bn, bk, cfg.num_warps, cfg.num_stages)] = c.numpy().copy()
    _same_within_groups(outs, lambda cfg: "all", f"gemv_vec {n}x{k}")
    ref = A.astype(np.float64) @ B.astype(np.float64)
    assert float(np.abs(next(iter(outs.values())) - ref).max()) <= 1e-4 * max(1.0, float(np.abs(ref).max()))


@pytest.mark.parametrize("bk, says", [(64, "at least K_SUB"), (192, "whole multiple of K_SUB")],
                         ids=["below-K_SUB", "not-a-multiple"])
def test_gemv_vec_refuses_a_chunk_that_is_not_whole_sub_chunks(bk, says):
    """A chunk narrower than the fixed sub-chunk ran zero sub-chunks and stored 0 for every row, without a word; a
    chunk of 1.5 sub-chunks dropped the half. Such a candidate must be refused when it is compiled."""
    from triton.compiler.errors import CompilationError
    from nbxtensor.ops import gemv_vec as G
    from nbxtensor.ops.launch_keys import bucket_of
    from nbxtensor.tensor import NBXTensor
    n, k = 64, 512
    a = _dev(np.ones((n, k), np.float16))
    b = _dev(np.ones(k, np.float32))
    c = NBXTensor.from_numpy(np.full(n, -1.0, np.float32)).to(device="cuda:0")
    with pytest.raises(CompilationError, match=says):
        G.gemv_vec_kernel.fn[(triton.cdiv(n, 8),)](a, b, c, n, k, bucket_of("N", n), bucket_of("K", k),
                                                   a.stride(0), a.stride(1), b.stride(0), c.stride(0),
                                                   BLOCK_N=8, BLOCK_K=bk, num_warps=4, num_stages=1)


PAGED = [(4, 1, 128, 64, 300), (8, 2, 64, 64, 1000)]   # heads, kv heads, head dim, page tokens, positions


@pytest.mark.parametrize("hq,hkv,d,pt,t", PAGED, ids=[f"h{p[0]}k{p[1]}d{p[2]}p{p[3]}t{p[4]}" for p in PAGED])
def test_paged_decode_every_candidate_gives_the_same_bits(hq, hkv, d, pt, t):
    from nbxtensor.ops import paged_decode_attn as P
    from nbxtensor.tensor import NBXTensor
    rng = np.random.default_rng(hq * 101 + t)
    pages = -(-t // pt)
    q = rng.standard_normal((hq, d)).astype(np.float32)
    kp = rng.standard_normal((pages, hkv, pt, d)).astype(np.float16)
    vp = rng.standard_normal((pages, hkv, pt, d)).astype(np.float16)
    order = rng.permutation(pages).astype(np.int32)
    bt = order.reshape(1, pages)
    sl = np.array([t], np.int32)
    split = 2
    q2, k2, v2, bt2, sl2 = _dev(q), _dev(kp), _dev(vp), _dev(bt), _dev(sl)
    op = NBXTensor.from_numpy(np.zeros((hq, split, d), np.float32)).to(device="cuda:0")
    mp = NBXTensor.from_numpy(np.zeros((hq, split), np.float32)).to(device="cuda:0")
    lp = NBXTensor.from_numpy(np.zeros((hq, split), np.float32)).to(device="cuda:0")
    kernel = P.paged_decode_attn_split_kernel.fn
    outs = {}
    from nbxtensor.ops.tiling_sizes import PAGED_DECODE_TILE_CANDIDATES
    # the run-time list (warps) AND every tile a certificate in the directory may name
    for cfg in [triton.Config({'BLOCK_N': bn}, num_warps=w) for bn in PAGED_DECODE_TILE_CANDIDATES for w in (2, 4, 8)]:
        bn = cfg.kwargs["BLOCK_N"]
        if pt % bn:
            continue
        kernel[(hq, split)](q2, k2, v2, bt2, sl2, op, mp, lp, 1, pt, split, 1.0 / np.sqrt(d), hq,
                            q2.stride(0), k2.stride(0), k2.stride(1), k2.stride(2),
                            v2.stride(0), v2.stride(1), v2.stride(2), bt2.stride(0),
                            op.stride(0), op.stride(1), mp.stride(0),
                            GQA_GROUPS=hq // hkv, BLOCK_D=triton.next_power_of_2(d), D=d, D_V=d,
                            SPLIT_ALIGN=P.PAGED_DECODE_SPLIT_ALIGN, BLOCK_N=bn, num_warps=cfg.num_warps, num_stages=1)
        outs[(bn, cfg.num_warps)] = np.concatenate([op.numpy().ravel(), mp.numpy().ravel(), lp.numpy().ravel()])
    _same_within_groups(outs, lambda cfg: "all", f"paged decode h{hq} d{d} t{t}")


@pytest.mark.parametrize("rows,feat", RMS_SHAPES, ids=[f"{r}x{f}" for r, f in RMS_SHAPES])
def test_layer_norm_every_candidate_gives_the_same_bits(rows, feat):
    from nbxtensor.ops import layernorm as L
    from nbxtensor.ops._configs import row_program_configs, prune_row_programs
    from nbxtensor.ops.launch_keys import bucket_of, softmax_feat_bucket
    from nbxtensor.tensor import NBXTensor
    rng = np.random.default_rng(rows * 37 + feat)
    a = (rng.standard_normal((rows, feat)) * 4.0 + 1.5).astype(np.float32)
    w = rng.standard_normal(feat).astype(np.float32)
    bb = rng.standard_normal(feat).astype(np.float32)
    x, wt, bt = _dev(a), _dev(w), _dev(bb)
    y = NBXTensor.empty_like(x)
    # as the one launch in the library (`norm_wrappers`) makes it: the stats saved. Measured 2026-11-18: with
    # save_stats=False a [3, 33] row on 1 warp gave other bits in the OUTPUT (the stats equal), a path nothing launches.
    st = NBXTensor.from_numpy(np.zeros(rows, np.float32)).to(device="cuda:0")
    st2 = NBXTensor.from_numpy(np.zeros(rows, np.float32)).to(device="cuda:0")
    kernel = L.layer_norm_forward_kernel.fn
    outs = {}
    for c in prune_row_programs(row_program_configs(), {"feat_dim": feat, "batch_dim": rows}):
        b, nw = c.kwargs["BLOCK_SIZE_BATCH"], c.num_warps
        kernel[(triton.cdiv(rows, b),)](x, wt, bt, st, st2, y, rows, feat, bucket_of("M", rows), softmax_feat_bucket(feat),
                                        x.stride(0), x.stride(1), y.stride(0), y.stride(1), 1e-5,
                                        scale_by_weight=True, add_bias=True, save_stats=True, BLOCK_SIZE_BATCH=b,
                                        BLOCK_SIZE_FEAT=triton.next_power_of_2(feat), num_warps=nw)
        outs[(b, nw)] = np.concatenate([y.numpy().ravel(), st.numpy(), st2.numpy()])
    _same_within_groups(outs, lambda cfg: "all", f"layer_norm {rows}x{feat}")
    d = a.astype(np.float64)
    ref = (d - d.mean(axis=1, keepdims=True)) / np.sqrt(d.var(axis=1, keepdims=True) + 1e-5) * w + bb
    assert float(np.abs(next(iter(outs.values()))[:rows * feat].reshape(rows, feat) - ref).max()) <= 1e-4
