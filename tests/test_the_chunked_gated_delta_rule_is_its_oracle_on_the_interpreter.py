"""The chunked gated delta rule's three kernels (`ops.gated_delta_rule_chunked`) against the float64 oracle
(`delta_rule_oracle`), on ANY host: Triton's interpreter runs them on torch CPU buffers, in a child process started
with TRITON_INTERPRET=1 (`_chunked_interpreter_run.py`; the interpreter must be chosen before Triton is imported), each
kernel pinned to one of its configurations, the configuration changing from case to case so the tiles of each are
read. Inputs at float32 and float16, L = 1, 23, 63, 64, 65, 130 and 200 (one token, under a chunk, both sides of the
64-token chunk, several chunks with a ragged tail), B = 1 from zeros and B = 2 from a non-zero state, Hv = Hk and
Hv = 3 Hk (key heads read un-expanded), and the state handed over: chunked to chunked at a cut, and from the state the
serial recurrence hands (the oracle's own, rounded to float32) to the chunked form.

THE BOUNDS, stated from what was measured (2026-11-16, 130 cases on this interpreter: L 1..200, 4 draws a length and
a shape, dk 32 and 64, both hand-overs; the measuring script and its output are in this lot's proof folder): worst
out 4.2e-07, state 5.1e-07 at float32; out 4.3e-04 at float16 (its output's rounding, 2^-11 = 4.9e-04, dominates),
state 5.1e-07. The bound below is twice the float32 worst rounded up, 1e-06, plus the output's rounding at float16;
the state is float32 at every input dtype. A card's own fused multiply-adds and exp round differently: the device test
(`test_the_chunked_gated_delta_rule_on_a_device_is_its_oracle`) carries the device's bound.

Needs torch (the interpreter's buffers): without it the test skips by name. What it catches: a chunk's triangular
solve or its decay wrong, a ragged last chunk read or written past L, the state not carried from chunk to chunk or
from a call to the next, a value head reading the wrong key head, an output stored at the wrong dtype - red."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
OUT_F32, STATE = 1e-6, 1e-6
ROUNDING = {"float32": 0.0, "float16": 2.0 ** -11}


def _case(dtype, L, B, Hk, Hv, nonzero, cfg, dk=32, dv=24, cut=None, oracle_cut=None):
    return dict(B=B, L=L, Hk=Hk, Hv=Hv, dk=dk, dv=dv, dtype=dtype, seed=31 * L + B + cfg, nonzero=nonzero, cfg=cfg,
                cut=cut, oracle_cut=oracle_cut)


CASES = {}
for dt in ("float32", "float16"):
    for i, L in enumerate((1, 23, 63, 64, 65, 130, 200)):
        CASES[f"{dt}-L{L}-B1-Hv1Hk-zeros"] = _case(dt, L, 1, 2, 2, False, cfg=i)
        CASES[f"{dt}-L{L}-B2-Hv3Hk-state"] = _case(dt, L, 2, 1, 3, True, cfg=i + 7)
    CASES[f"{dt}-chunked-to-chunked-at-65"] = _case(dt, 130, 2, 1, 2, True, cfg=3, cut=65)
    CASES[f"{dt}-chunked-to-chunked-at-23"] = _case(dt, 130, 2, 1, 2, True, cfg=4, cut=23)
    CASES[f"{dt}-serial-state-to-chunked-at-23"] = _case(dt, 130, 2, 1, 2, True, cfg=5, oracle_cut=23)
CASES["float32-L200-dk64-dv64"] = _case("float32", 200, 2, 2, 4, True, cfg=11, dk=64, dv=64)


@pytest.fixture(scope="module")
def results():
    try:
        import torch  # noqa: F401  (the interpreter's buffers; torch is never imported by the library)
    except ImportError:
        pytest.skip("needs torch for Triton's interpreter buffers: torch is absent on this host")
    env = dict(os.environ, TRITON_INTERPRET="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join([str(SRC), str(HERE), os.environ.get("PYTHONPATH", "")]))
    run = subprocess.run([sys.executable, str(HERE / "_chunked_interpreter_run.py"), json.dumps(list(CASES.values()))],
                         env=env, capture_output=True, text=True, timeout=1800)
    assert run.returncode == 0, f"the interpreter run failed:\n{run.stderr[-3000:]}"
    rows = [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]
    assert len(rows) == len(CASES), f"{len(rows)} results for {len(CASES)} cases:\n{run.stderr[-3000:]}"
    return dict(zip(CASES, rows))


@pytest.mark.parametrize("name", list(CASES))
def test_the_chunked_form_is_the_oracle(results, name):
    r, c = results[name], CASES[name]
    start = c["oracle_cut"] or 0
    assert r["shape"] == [c["B"], c["L"] - start, c["Hv"], c["dv"]]
    print(f"{name} (configuration {c['cfg']}): out {r['out']:.2e} (bound {OUT_F32 + ROUNDING[c['dtype']]:.1e}) "
          f"state {r['state']:.2e} (bound {STATE:.1e})")
    assert r["out"] <= OUT_F32 + ROUNDING[c["dtype"]]
    assert r["state"] <= STATE
