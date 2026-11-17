"""`moe_route` over T tokens in ONE launch (the token step T = 1 and the prefill, hidden [T, H] and [B, T, H]) against
the routing written plainly in the test (`_moe_route_compare.plain_route`, token by token in float64), on ANY host:
Triton's interpreter runs the kernel through the op's own launch on torch CPU buffers, in a child process started with
TRITON_INTERPRET=1 (`_moe_route_interpreter_run.py`), pinned to one of its configurations, the configuration changing
from case to case so each hidden tile is read.

The cases: every gate (softmax or sigmoid, without or with a bias added before the selection) x every selection
(plain top-k; grouped top-k ranked by the top-2 sum with a fill of -inf; grouped by the maximum with a fill of 0) at
T = 1, 2 and 23 and at [B 2, T 23], float32 and bfloat16 (the router weight at the same dtype, the bias float32: the
library's range brick keeps float32 operands float32, nothing is narrowed); k = E; exact ties (rows of zeros: every
logit 0, lower id first, the groups too; a negative bias under a fill of 0: every kept choice below the fill). H 100 is
not a multiple of any hidden tile.

Two judgements per case. (1) Against the plain reference through `_moe_route_compare.compare`: the indices as a set
(an exact tie must give the reference set; a different set only on a near-tie inside the band, its weights then judged
on the kernel's own set), the weights paired by expert id within WEIGHTS of the token's largest weight. (2) Each row of
the one launch against the same row launched ALONE (T = 1): indices AND weights byte-equal, tolerance 0, because each
row is its own program running the same instructions whatever B and T (only its row address changes).

THE BOUNDS, measured on this interpreter (2026-11-17, the 102 cases below, the measuring run in this lot's proof
folder): worst weight deviation relative to the token's largest weight 1.45e-07 at float32 and 3.89e-07 at bfloat16
(the reference reads the same bfloat16-rounded operands, so what remains is the float32 accumulation order), no
near-tie. WEIGHTS 1e-6 is 2.6 times the worst (room for the tile order of the other configurations); CHOICE 1e-6, the
choice values' bound of the device test's host emulation, gives the near-tie band (two of them). The device bounds are the device test's
(`test_one_token_of_the_moe_route_on_a_device_is_its_oracle`); its prefill case is the same byte-equality on a card.

Needs torch (the interpreter's buffers; the library never imports it): without it the test skips by name. What it
catches: a row read at the wrong batch or token stride, a row written to another row's slot, the token step and the
prefill computing a row differently, a gate, a bias, a group score, a fill, a normalisation or a scale wrong for any
pair, a tie not resolved lower id first, k = E dropping an expert - red."""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _moe_route_compare import compare, plain_route
from _moe_route_draw import draw
from nbxtensor.dtype.convert import bf16_bits_to_f32, f32_to_bf16_bits

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
WEIGHTS, CHOICE = 1e-6, 1e-6
E, H = 32, 100

GATES = {"softmax": ("softmax", None), "softmax-bias": ("softmax", 0.0),
         "sigmoid": ("sigmoid", None), "sigmoid-bias": ("sigmoid", 0.0)}
SELECTIONS = {
    "topk": dict(top_k=4, n_group=1, topk_group=1, group_score="top2_sum", fill=-math.inf, renorm_eps=1e-20,
                 scale=2.5),
    "grouped-top2-inf": dict(top_k=4, n_group=4, topk_group=2, group_score="top2_sum", fill=-math.inf,
                             renorm_eps=1e-20, scale=1.8),
    "grouped-max-zero": dict(top_k=6, n_group=8, topk_group=3, group_score="max", fill=0.0, renorm_eps=None,
                             scale=1.0),
}
SHAPES = {"T1": (None, 1), "T2": (None, 2), "T23": (None, 23), "B2T23": (2, 23)}


def _case(dtype, B, T, gate, sel, cfg, seed, e=E, zero_rows=(), bias_mean=None, **over):
    score_fn, bias = GATES[gate]
    attrs = dict(SELECTIONS[sel], score_fn=score_fn, **over)
    return dict(E=e, H=H, B=B, T=T, dtype=dtype, seed=seed, cfg=cfg, zero_rows=list(zero_rows),
                bias_mean=bias if bias_mean is None else bias_mean, attrs=attrs)


CASES = {}
_i = 0
for _dt in ("float32", "bfloat16"):
    for _g in GATES:
        for _s in SELECTIONS:
            for _shape, (_B, _T) in SHAPES.items():
                _zero = (0, (_B or 1) * _T - 1) if _T > 2 else ()      # rows of zeros: an exact tie in every case
                CASES[f"{_dt}-{_g}-{_s}-{_shape}"] = _case(_dt, _B, _T, _g, _s, _i, 100 + _i, zero_rows=_zero)
                _i += 1
    for _g in ("softmax", "sigmoid-bias"):
        CASES[f"{_dt}-{_g}-k-equals-E-T23"] = _case(_dt, None, 23, _g, "topk", _i, 100 + _i, e=16, top_k=16)
        _i += 1
    # every kept choice value below the fill of 0: each token routes to filled experts, lower id first
    CASES[f"{_dt}-sigmoid-bias-below-fill-T23"] = _case(_dt, None, 23, "sigmoid-bias", "grouped-max-zero", _i,
                                                        100 + _i, bias_mean=-1.5)
    _i += 1


def _read_as(a, dtype):
    """The numbers the kernel reads from a float32 draw stored at `dtype` (round to nearest even)."""
    if dtype == "bfloat16":
        return bf16_bits_to_f32(f32_to_bf16_bits(np.ascontiguousarray(a, np.float32)))
    return np.asarray(a, np.float32)


@pytest.fixture(scope="module")
def results():
    try:
        import torch  # noqa: F401  (the interpreter's buffers; torch is never imported by the library)
    except ImportError:
        pytest.skip("needs torch for Triton's interpreter buffers: torch is absent on this host")
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join([str(SRC), str(HERE), os.environ.get("PYTHONPATH", "")]))
    run = subprocess.run([sys.executable, str(HERE / "_moe_route_interpreter_run.py"), json.dumps(list(CASES.values()))],
                         env=env, capture_output=True, text=True, timeout=1800)
    assert run.returncode == 0, f"the interpreter run failed:\n{run.stderr[-3000:]}"
    rows = [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]
    assert len(rows) == len(CASES), f"{len(rows)} results for {len(CASES)} cases:\n{run.stderr[-3000:]}"
    return dict(zip(CASES, rows))


@pytest.mark.parametrize("name", list(CASES))
def test_one_launch_over_the_tokens_is_the_plain_reference_and_each_row_alone(results, name):
    r, c = results[name], CASES[name]
    a, rows = c["attrs"], (c["B"] or 1) * c["T"]
    x, w, b = draw(rows, c["E"], c["H"], c["seed"], c["bias_mean"])
    x[c["zero_rows"]] = 0.0
    xr, wr = _read_as(x, c["dtype"]).astype(np.float64), _read_as(w, c["dtype"]).astype(np.float64)
    ref = plain_route(xr, wr, b, a)
    idx, wt = np.asarray(r["idx"], np.int64), np.asarray(r["wt"])
    assert idx.shape == wt.shape == (rows, a["top_k"])
    for t in c["zero_rows"]:
        if c["bias_mean"] is None:                     # every logit 0: the exact tie, lower id first
            assert sorted(idx[t].tolist()) == sorted(ref[0][t].tolist()), f"{name} row {t}: tie not lower id first"
    worst, ties = compare(idx, wt, ref, a, name, xr, wr, WEIGHTS, CHOICE)
    print(f"{name}: weights {worst:.2e} (bound {WEIGHTS:.0e}); near-ties {len(ties)} of {rows}")
    assert worst <= WEIGHTS
    assert all(r["alone_same_bytes"]), (f"{name}: rows {[i for i, s in enumerate(r['alone_same_bytes']) if not s]} "
                                        f"differ from the same rows launched alone")


def test_the_cases_route_somewhere_filled_and_exercise_every_pair():
    """The matrix holds what it claims: every gate x selection pair at every shape and both dtypes, and the
    below-the-fill case really routes its tokens to filled experts in the reference."""
    for dt in ("float32", "bfloat16"):
        for g in GATES:
            for s in SELECTIONS:
                for shape in SHAPES:
                    assert f"{dt}-{g}-{s}-{shape}" in CASES
    c = CASES["float32-sigmoid-bias-below-fill-T23"]
    x, w, b = draw(c["T"], c["E"], c["H"], c["seed"], c["bias_mean"])
    ri, _, rc, _ = plain_route(x, w, b, c["attrs"])
    assert (rc[np.arange(c["T"])[:, None], ri] == 0.0).all(), "the draw must route every token to filled experts"
