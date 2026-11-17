"""`moe_route` on a device over a prefill: ONE launch over hidden [T, H] (T = 2, 23, 64) and [B, T, H] (2 x 23) gives,
row by row, the bytes of the same row routed ALONE (T = 1): the indices AND the weights, tolerance 0 - each row is its
own program running the same instructions whatever B and T, only its row address changes. Then the one launch is the
float64 oracle under the device test's own judgement (`_moe_route_compare.compare`, the bounds of
`test_one_token_of_the_moe_route_on_a_device_is_its_oracle`), at the hidden row's float32, float16 and bfloat16
(bfloat16 where the device has it natively), with a sigmoid and a bias on grouped top-k and a softmax on plain top-k.

A card test: on a host with no device it skips by name. It runs on a V100 card job and on the Mac (Metal), the same
file; Triton's interpreter runs the same comparison on any host
(`test_the_moe_route_over_tokens_is_its_plain_reference_on_the_interpreter`).

What it catches on a card: a row read at the wrong batch or token stride, a row written to another row's slot, a row
of the prefill computed differently from the token step - red."""
import math

import numpy as np
import pytest

from _moe_route_compare import compare
from _moe_route_draw import draw
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.ops.oracles import fp64 as O

WEIGHTS, CHOICE = 4e-6, 1e-6                     # the device bounds of the one-token device test, unchanged
CASES = {
    "sigmoid-bias-grouped": (64, 2048, 0.0, dict(score_fn="sigmoid", group_score="top2_sum", renorm_eps=1e-20,
                                                  top_k=4, n_group=8, topk_group=4, fill=-math.inf, scale=1.8)),
    "softmax-topk": (160, 1000, None, dict(score_fn="softmax", group_score="max", renorm_eps=None, top_k=6,
                                           n_group=1, topk_group=1, fill=-math.inf, scale=1.0)),
}
SHAPES = {"T2": (None, 2), "T23": (None, 23), "T64": (None, 64), "B2T23": (2, 23)}
DTYPES = [NBXDtype.float32, NBXDtype.float16, NBXDtype.bfloat16]


@pytest.fixture(autouse=True)
def _accelerator():
    from nbxtensor.memory.allocator import DeviceAllocator
    if DeviceAllocator.device_count() == 0:          # only a machine without an accelerator skips; any other error fails
        pytest.skip("no accelerator on this machine")


def _to_device(a, dt):
    from nbxtensor.tensor import NBXTensor
    return NBXTensor.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device="cuda:0").to(dt)


@pytest.mark.parametrize("dt", DTYPES, ids=["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("name", list(CASES))
def test_one_launch_over_the_prefill_is_each_token_alone_and_the_oracle(dt, shape, name):
    from nbxtensor.ops.moe_route import moe_route
    from nbxtensor.ops.wrappers import has_native_bf16
    if dt == NBXDtype.bfloat16 and not has_native_bf16():
        pytest.skip("bfloat16: this device has no native bfloat16")
    E, H, bias_mean, a = CASES[name]
    B, T = SHAPES[shape]
    rows = (B or 1) * T
    x, w, b = draw(rows, E, H, 53, bias_mean)
    dx, dw = _to_device(x, dt), _to_device(w, dt)
    db = None if b is None else _to_device(b, NBXDtype.float32)
    hidden = dx if B is None else dx.reshape((B, T, H))
    i, v = moe_route(hidden, dw, choice_bias=db, **a)
    assert tuple(i.shape) == tuple(v.shape) == tuple(hidden.shape)[:-1] + (a["top_k"],)
    idx = O.host_values(i).reshape(rows, -1).astype(np.int64)
    wt = O.host_values(v).reshape(rows, -1)
    for r in range(rows):
        i1, v1 = moe_route(dx[r:r + 1], dw, choice_bias=db, **a)
        assert np.array_equal(O.host_values(i1)[0], idx[r]), f"{name} {shape} row {r}: indices differ from alone"
        assert O.host_values(v1)[0].tobytes() == wt[r].tobytes(), f"{name} {shape} row {r}: weights differ from alone"
    xr = O.host_values(dx.to(NBXDtype.float32)).astype(np.float64)
    wr = O.host_values(dw.to(NBXDtype.float32)).astype(np.float64)
    ref = O.moe_route_oracle(xr, wr, b, **a)()
    worst, ties = compare(idx, wt.astype(np.float64), ref, a, name, xr, wr, WEIGHTS, CHOICE)
    print(f"{name} {shape} {dt.name}: weights {worst:.2e} (bound {WEIGHTS:.0e}); near-ties {len(ties)} of {rows}")
    assert worst <= WEIGHTS
