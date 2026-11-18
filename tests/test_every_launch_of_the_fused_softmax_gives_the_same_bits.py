"""The fused softmax gives the same bits under EVERY launch candidate the tuner may choose (rows per program x warps),
for softmax and log-softmax, at row lengths that are and are not a power of two. A key the certified directory does
not hold is chosen by a run-time sweep, by timing; candidates a few per cent apart swap places from one run to the
next, so if the candidates differ in bits, one release gives different answers on two fresh installs. Measured on a
V100, 2026-11-18: of 24 candidates of a [320, 512] softmax, 13 gave other bits than 1 row x 4 warps (the row sum
reduced in the order of the launch's layout), and a served conversation's fifth answer changed with the warps a
fresh sweep chose (3 of 6 fresh homes on card 0, published 0.6.16).

A card test: on a host with no device it skips by name.

What it catches: a row reduction whose order depends on the launch configuration - red."""
import numpy as np
import pytest
import triton


SHAPES = [(320, 512), (384, 256), (320, 500), (7, 64), (64, 2048), (3, 33)]


@pytest.fixture(autouse=True)
def _accelerator():
    from nbxtensor.memory.allocator import DeviceAllocator
    if DeviceAllocator.device_count() == 0:          # only a machine without an accelerator skips; any other error fails
        pytest.skip("no accelerator on this machine")


def _candidates(rows, feat):
    """The candidates the tuner sweeps at this shape: the kernel's own list through its own prune."""
    from nbxtensor.ops import softmax as S
    return S._prune_softmax(S._softmax_configs(), {"feat_dim": feat, "batch_dim": rows})


@pytest.mark.parametrize("log", [False, True], ids=["softmax", "log_softmax"])
@pytest.mark.parametrize("rows,feat", SHAPES, ids=[f"{r}x{f}" for r, f in SHAPES])
def test_every_candidate_gives_the_bits_of_every_other(rows, feat, log):
    from nbxtensor.ops import softmax as S
    from nbxtensor.ops.launch_keys import bucket_of, softmax_feat_bucket
    from nbxtensor.tensor import NBXTensor
    a = (np.random.default_rng(rows * 7919 + feat).standard_normal((rows, feat)) * 8.0).astype(np.float32)
    x = NBXTensor.from_numpy(a).to(device="cuda:0")
    y = NBXTensor.empty_like(x)
    kernel = S.softmax_forward_kernel.fn                # behind the tuner: each candidate launched as itself
    outs = {}
    for c in _candidates(rows, feat):
        b, w = c.kwargs["BLOCK_SIZE_BATCH"], c.num_warps
        kernel[(triton.cdiv(rows, b),)](x, y, rows, feat, bucket_of("M", rows), softmax_feat_bucket(feat),
                                        x.stride(0), x.stride(1), y.stride(0), y.stride(1), log=log,
                                        BLOCK_SIZE_BATCH=b, num_warps=w)
        outs[(b, w)] = y.numpy().copy()
    assert len(outs) > 1, f"{rows}x{feat}: fewer than two candidates, nothing compared"
    first = next(iter(outs))
    other = sorted(k for k, v in outs.items() if not np.array_equal(v.view(np.uint32), outs[first].view(np.uint32)))
    assert not other, (f"{rows}x{feat} log={log}: {len(other)} of {len(outs)} candidates give other bits than "
                       f"{first} rows x warps: {other}")
    d = a.astype(np.float64) - a.max(axis=1, keepdims=True)
    ref = d - np.log(np.exp(d).sum(axis=1, keepdims=True)) if log else np.exp(d) / np.exp(d).sum(axis=1, keepdims=True)
    err = float(np.abs(outs[first].astype(np.float64) - ref).max())
    assert err <= (1e-5 if log else 1e-6), f"{rows}x{feat} log={log}: max |error| vs float64 {err:.3e}"
