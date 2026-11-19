"""Matmul, addmm and baddbmm give the same bits under EVERY launch candidate the tuner may choose (tile,
K-chunk width, warps, stages), in float16 and float32. A key the certified directory does not hold is chosen by a
run-time sweep, by timing; candidates a few per cent apart swap places from one run to the next, so if the candidates
differed in bits, one release would give different answers on two fresh installs. Measured on a V100, 2026-11-19:
every candidate of matmul, addmm, baddbmm and conv2d gave the same bits on twelve shapes (K is one sequential loop in one program, the
K-chunk only cuts it, and no candidate splits K). This test keeps that fact guarded for the three matmul kernels.
conv2d is not guarded here: on Metal one of its tiles (BHW128 x OUTF128, 4 warps) still gives other bits than the
other candidates, and conv2d follows on its own branch.

Not covered here: the matrix unit a certificate may route a call to instead of these kernels; it gives other bits
and is chosen by the certificate, never by a run-time sweep.

A card test: on a host with no device it skips by name.

What it catches: a candidate whose sum over K is taken in another order (a split K, a reordered chunk) - red."""
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


def _data(seed, dtype, *shape):
    return (np.random.default_rng(seed).standard_normal(shape) * 0.05).astype(dtype)


def _every_candidate(configs, launch, out):
    """The bits of each candidate, launched as itself behind the tuner; a candidate the card cannot hold (too much
    shared memory) is one the tuner discards too, every other error fails."""
    from triton.runtime.errors import OutOfResources
    bits = {}
    for c in configs:
        kw = dict(c.kwargs)
        try:
            launch(kw, c.num_warps, c.num_stages)
        except OutOfResources:
            continue
        o = out()
        bits[(tuple(sorted(kw.items())), c.num_warps, c.num_stages)] = o.view(np.uint16 if o.dtype == np.float16 else np.uint32).copy()
    assert len(bits) > 1, "fewer than two candidates ran, nothing compared"
    first = next(iter(bits))
    other = [k for k, v in bits.items() if not np.array_equal(v, bits[first])]
    assert not other, f"{len(other)} of {len(bits)} candidates give other bits than {first}: {other[:4]}"
    return out()


def _empty_like_dtype(shape, a):
    from nbxtensor.tensor import NBXTensor
    return NBXTensor.empty(shape, device="cuda:0", dtype=a.nbx_dtype)


MM_SHAPES = [(16, 2048, 2048, np.float16), (512, 1024, 1024, np.float16), (64, 512, 512, np.float32)]


@pytest.mark.parametrize("bias", [False, True], ids=["matmul", "addmm"])
@pytest.mark.parametrize("M,K,N,dt", MM_SHAPES, ids=[f"{m}x{k}x{n}-{np.dtype(t).name}" for m, k, n, t in MM_SHAPES])
def test_matmul_and_addmm(M, K, N, dt, bias):
    from nbxtensor.ops import matmul as MM
    from nbxtensor.ops.launch_keys import bucket_of
    an, bn, cn = _data(1, dt, M, K), _data(2, dt, K, N), _data(3, dt, N)
    a, b, bb = _dev(an), _dev(bn), _dev(cn)
    c = _empty_like_dtype((M, N), a)
    mb = bucket_of("M", M)

    def launch(kw, nw, ns):
        grid = (triton.cdiv(M, kw["BLOCK_M"]) * triton.cdiv(N, kw["BLOCK_N"]),)
        if bias:
            MM.addmm_kernel.fn[grid](a, b, bb, c, M, N, K, mb, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                     c.stride(0), c.stride(1), 1.0, 1.0, **kw, num_warps=nw, num_stages=ns)
        else:
            MM.matmul_kernel.fn[grid](a, b, c, M, N, K, mb, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                      c.stride(0), c.stride(1), **kw, num_warps=nw, num_stages=ns)
    y = _every_candidate(MM._MATMUL_AUTOTUNE_CONFIGS, launch, lambda: c.numpy())
    ref = an.astype(np.float64) @ bn.astype(np.float64) + (cn.astype(np.float64) if bias else 0.0)
    err = float(np.abs(y.astype(np.float64) - ref).max())
    assert err <= (2e-2 if dt == np.float16 else 1e-4), f"max |error| vs float64 {err:.3e}"


BMM_SHAPES = [(4, 128, 64, 128, np.float16), (8, 256, 128, 256, np.float16), (2, 128, 64, 128, np.float32)]


@pytest.mark.parametrize("B,M,K,N,dt", BMM_SHAPES, ids=[f"{b}x{m}x{k}x{n}-{np.dtype(t).name}" for b, m, k, n, t in BMM_SHAPES])
def test_baddbmm(B, M, K, N, dt):
    from nbxtensor.ops import baddbmm_op as BB, matmul as MM
    from nbxtensor.ops.launch_keys import bucket_of
    an, bn, cn = _data(4, dt, B, M, K), _data(5, dt, B, K, N), _data(6, dt, B, M, N)
    a, b, bias = _dev(an), _dev(bn), _dev(cn)
    c = _empty_like_dtype((B, M, N), a)

    def launch(kw, nw, ns):
        grid = (triton.cdiv(M, kw["BLOCK_M"]) * triton.cdiv(N, kw["BLOCK_N"]), 1, B)
        BB.baddbmm_kernel.fn[grid](a, b, c, bias, 1.0, 1.0, M, N, K, bucket_of("M", M), bucket_of("N", N), bucket_of("K", K),
                                   a.stride(0), a.stride(1), a.stride(2), b.stride(0), b.stride(1), b.stride(2),
                                   c.stride(0), c.stride(1), c.stride(2), bias.stride(0), bias.stride(1), bias.stride(2),
                                   HAS_BIAS=True, **kw, num_warps=nw, num_stages=ns)
    y = _every_candidate(MM._MATMUL_AUTOTUNE_CONFIGS, launch, lambda: c.numpy())
    ref = np.matmul(an.astype(np.float64), bn.astype(np.float64)) + cn.astype(np.float64)
    err = float(np.abs(y.astype(np.float64) - ref).max())
    assert err <= (2e-2 if dt == np.float16 else 1e-4), f"max |error| vs float64 {err:.3e}"

