"""Run by `test_the_chunked_gated_delta_rule_is_its_oracle_on_the_interpreter` in a child process started with
TRITON_INTERPRET=1 (Triton's interpreter must be chosen before Triton is imported, so never in the test's process):
the three chunked kernels launched on torch CPU buffers, each pinned to ONE of its configurations (the interpreter has
no timer to tune with, and a pinned configuration is the one the case names), against the float64 oracle.

argv[1] is a JSON list of cases; one JSON line per case goes to stdout: the case and its deviations. A case:
    B, L, Hk, Hv, dk, dv, dtype ("float32" | "float16"), seed, nonzero (a non-zero start state), cfg (the index of the
    configuration each kernel is pinned to, modulo its count), cut (null, or the length after which the chunked form
    hands its state to a second chunked call), oracle_cut (null, or the length up to which the oracle runs, its state
    rounded to float32 then handed to the chunked form: the state a serial scan hands over), decay_max (optional, 0.1 when absent:
    the bound of the decay the draw uses).
"""
import json
import os
import sys

if os.environ.get("TRITON_INTERPRET") != "1":
    raise SystemExit("_chunked_interpreter_run.py: run me with TRITON_INTERPRET=1 (the test does)")

import numpy as np
import torch

from _delta_rule_draw import draw
from nbxtensor.ops import gated_delta_rule_chunked as C
from nbxtensor.ops.oracles import fp64 as O

KERNELS = ("gated_delta_rule_chunk_prepare_kernel", "gated_delta_rule_chunk_state_kernel",
           "gated_delta_rule_chunk_output_kernel")
TORCH = {"float32": torch.float32, "float16": torch.float16}
NUMPY = {"float32": np.float32, "float16": np.float16}


class Pinned:
    """A tuned kernel launched at one of its configurations, its grid formed from that configuration's tiles."""

    def __init__(self, tuner, cfg):
        self.fn, self.cfg = tuner.fn, cfg

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            meta = dict(kwargs, **self.cfg.kwargs)
            return self.fn[grid(meta) if callable(grid) else grid](*args, **meta)
        return launch


TUNERS = {n: getattr(C, n) for n in KERNELS}


def pin(index):
    for n, tuner in TUNERS.items():
        setattr(C, n, Pinned(tuner, tuner.configs[index % len(tuner.configs)]))


def chunked(q, k, v, g, beta, s, dt, attrs):
    B, L, Hk, dk = q.shape
    Hv, dv = v.shape[2:]
    t = lambda a, d: torch.from_numpy(np.ascontiguousarray(a.astype(d)))
    out = torch.empty(B, L, Hv, dv, dtype=TORCH[dt])
    so = torch.empty(B, Hv, dk, dv, dtype=torch.float32)
    ws = {n: torch.empty(sh, dtype=torch.float32) for n, sh in C.gated_delta_rule_chunked_workspace(B, L, Hv, dk, dv).items()}
    C.gated_delta_rule_chunked_launch(t(q, NUMPY[dt]), t(k, NUMPY[dt]), t(v, NUMPY[dt]), t(g, np.float32),
                                      t(beta, NUMPY[dt]), None if s is None else t(s, np.float32), out, so, ws, **attrs)
    return out.float().numpy(), so.numpy()


def run(case):
    B, L, Hk, Hv, dk, dv, dt = (case[n] for n in ("B", "L", "Hk", "Hv", "dk", "dv", "dtype"))
    attrs = dict(qk_l2norm=True, eps=1e-6, scale=dk ** -0.5)
    pin(case["cfg"])
    q, k, v, g, beta, s = draw(B, L, Hk, Hv, dk, dv, case["seed"], case["nonzero"], decay_max=case.get("decay_max", 0.1))
    # the oracle reads what the kernels read: q, k, v and beta at the case's dtype, g and the state float32
    q, k, v, beta = (a.astype(NUMPY[dt]).astype(np.float32) for a in (q, k, v, beta))
    ref_o, ref_s = O.delta_rule_oracle(q, k, v, g, beta, s, **attrs)()
    cut, ocut = case.get("cut"), case.get("oracle_cut")
    if ocut is not None:                        # the oracle runs [0, ocut), the chunked form from its float32 state
        _, mid = O.delta_rule_oracle(*(a[:, :ocut] for a in (q, k, v, g, beta)), s, **attrs)()
        out, so = chunked(*(a[:, ocut:] for a in (q, k, v, g, beta)), np.asarray(mid, np.float32), dt, attrs)
        ref_o = np.asarray(ref_o)[:, ocut:]
    elif cut is not None:                       # chunked over [0, cut), then chunked over [cut, L) from its state
        o1, s1 = chunked(*(a[:, :cut] for a in (q, k, v, g, beta)), s, dt, attrs)
        o2, so = chunked(*(a[:, cut:] for a in (q, k, v, g, beta)), s1, dt, attrs)
        out = np.concatenate([o1, o2], 1)
    else:
        out, so = chunked(q, k, v, g, beta, s, dt, attrs)
    return dict(case, out=float(O.oracle_deviation(out, ref_o)), state=float(O.oracle_deviation(so, ref_s)),
                shape=list(out.shape))


if __name__ == "__main__":
    for c in json.loads(sys.argv[1]):
        print(json.dumps(run(c)), flush=True)
