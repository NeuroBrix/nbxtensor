"""Run by `test_the_moe_route_over_tokens_is_its_plain_reference_on_the_interpreter` in a child process started with
TRITON_INTERPRET=1 (Triton's interpreter must be chosen before Triton is imported, so never in the test's process):
the routing kernel launched through the op's own launch (`moe_route._launch`) on torch CPU buffers, pinned to ONE of
its configurations (the interpreter has no timer to tune with; the case names which).

argv[1] is a JSON list of cases; one JSON line per case goes to stdout: the indices and weights of the ONE launch over
every row, and whether each row launched ALONE gives the same bytes. A case: E, H, B (null: hidden [T, H]), T, dtype
("float32" | "float16" | "bfloat16"), seed, bias_mean (null: no bias), zero_rows (rows of hidden set to zero: every
logit 0, an exact tie), cfg, attrs (the op's attributes).
"""
import json
import os
import sys

if os.environ.get("TRITON_INTERPRET") != "1":
    raise SystemExit("_moe_route_interpreter_run.py: run me with TRITON_INTERPRET=1 (the test does)")

import numpy as np
import torch

from _moe_route_draw import draw
from nbxtensor.ops import moe_route as R

TORCH = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
TUNER = R.moe_route_kernel


class Pinned:
    """The tuned kernel launched at one of its configurations."""

    def __init__(self, tuner, cfg):
        self.fn, self.cfg = tuner.fn, cfg

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            return self.fn[grid](*args, **dict(kwargs, **self.cfg.kwargs))
        return launch


def route(hidden, w, b, case):
    """One launch over hidden's rows: (indices [rows, k], weights [rows, k]) as host arrays."""
    a, E, H = case["attrs"], case["E"], case["H"]
    B, T = (1,) + tuple(hidden.shape[:1]) if hidden.dim() == 2 else tuple(hidden.shape[:2])
    idx = torch.empty(B * T, a["top_k"], dtype=torch.int64)
    wt = torch.empty(B * T, a["top_k"], dtype=torch.float32)
    R._launch(hidden, w, b, idx, wt, B, T, H, E, **a)
    return idx.numpy(), wt.numpy()


def run(case):
    R.moe_route_kernel = Pinned(TUNER, TUNER.configs[case["cfg"] % len(TUNER.configs)])
    B, T, E, H, dt = case["B"], case["T"], case["E"], case["H"], TORCH[case["dtype"]]
    x, w, b = draw((B or 1) * T, E, H, case["seed"], case["bias_mean"])
    x[case["zero_rows"]] = 0.0
    hidden = torch.from_numpy(x).to(dt)
    hidden = hidden if B is None else hidden.reshape(B, T, H)
    wd = torch.from_numpy(w).to(dt)
    bd = None if b is None else torch.from_numpy(b)
    idx, wt = route(hidden, wd, bd, case)
    rows = hidden.reshape(-1, H)
    alone = [route(rows[r:r + 1], wd, bd, case) for r in range(rows.shape[0])]
    same = [bool(np.array_equal(i[0], idx[r]) and wt[r].tobytes() == v[0].tobytes())
            for r, (i, v) in enumerate(alone)]
    return {"idx": idx.tolist(), "wt": wt.astype(np.float64).tolist(), "alone_same_bytes": same}


for c in json.loads(sys.argv[1]):
    print(json.dumps(run(c)), flush=True)
