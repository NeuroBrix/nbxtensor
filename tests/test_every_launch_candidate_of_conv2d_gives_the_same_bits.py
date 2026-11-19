"""Conv2d gives the same bits under EVERY launch candidate the tuner may choose (tile, input-feature chunk, warps,
stages, orientation), in float16 and float32. A key the certified directory does not hold is chosen by a run-time
sweep, by timing; if two candidates differed in bits, one release would give different answers on two fresh installs.
conv2d sums its input features in fixed 16-wide sub-chunks into one accumulator whatever the candidate (Metal,
2026-11-19: before it, a 3x3 fp16 conv gave other bits at BLOCK_SIZE_INF 32 / 4 warps than at 16 / 2 warps).

A card test: on a host with no device it skips by name.

The candidates are those the tuner may choose ON THIS DEVICE (`conv2d.conv2d_candidates`): a tile whose float32
accumulator is past the device's threadgroup pool is lowered in passes on Metal and sums in another order, so the
list leaves it out by the profile's pool (Mac, 2026-11-19: 16 and 32 KB tiles gave the reference bits, 64 and 128 KB
tiles other bits; 34 of 36 candidates kept on Apple, 36 of 36 on a V100).

What it catches: a candidate whose sum over the input features is taken in another order - red."""
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


CONV_SHAPES = [(1, 64, 32, 32, 64, np.float16), (2, 32, 16, 16, 48, np.float16), (1, 32, 16, 16, 32, np.float32)]


@pytest.mark.parametrize("N_,C,H,W,O,dt", CONV_SHAPES, ids=[f"N{n}C{c}H{h}W{w}O{o}-{np.dtype(t).name}" for n, c, h, w, o, t in CONV_SHAPES])
def test_conv2d_3x3(N_, C, H, W, O, dt):
    from nbxtensor.ops import conv2d as CV
    from nbxtensor.ops.conv_wrappers import _conv_width_key
    xn, wn = _data(7, dt, N_, C, H, W), _data(8, dt, O, C, 3, 3)
    x, w = _dev(xn), _dev(wn)
    y = _empty_like_dtype((N_, O, H, W), x)
    iwk, owk = _conv_width_key(H, H, W, W, 3, 1, 1, 1)

    def launch(kw, nw, ns):
        grid = (triton.cdiv(N_ * H * W, kw["BLOCK_SIZE_BHW"]), triton.cdiv(O, kw["BLOCK_SIZE_OUTF"]), 1)
        CV.conv2d_forward_kernel.fn[grid](x, w, y, N_, C, H, W, O, H, W, iwk, owk, *x.stride(), *w.stride(), *y.stride(),
                                          kernel_height=3, kernel_width=3, stride_height=1, stride_width=1,
                                          padding_height=1, padding_width=1, dilation_height=1, dilation_width=1,
                                          groups=1, fp16=dt == np.float16, WIDE_OFFSETS=False, **kw,
                                          num_warps=nw, num_stages=ns)
    out = _every_candidate(CV.conv2d_candidates(CV._CONV2D_AUTOTUNE), launch, lambda: y.numpy())
    xp = np.pad(xn.astype(np.float64), ((0, 0), (0, 0), (1, 1), (1, 1)))
    ref = np.zeros((N_, O, H, W))
    for i in range(3):
        for j in range(3):
            ref += np.einsum("nchw,oc->nohw", xp[:, :, i:i + H, j:j + W], wn[:, :, i, j].astype(np.float64))
    err = float(np.abs(out.astype(np.float64) - ref).max())
    assert err <= (2e-2 if dt == np.float16 else 1e-4), f"max |error| vs float64 {err:.3e}"
