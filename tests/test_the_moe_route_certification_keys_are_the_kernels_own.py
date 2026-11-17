"""The keys a card certifies for the routing kernel are the kernel's own: `tools/moe_route_certification_keys.py` writes
one '<qualname>::<key>' line a census cell, each formed by `launch_keys.moe_route_launches`, readable by the
certifier's keys file reader (qualname, '::', a tuple literal); `--dtypes` keeps the cells of the dtypes named and a
dtype no cell has is refused with exit 2. The harness's synthesis of a census key (`autotune_harness.synthesize`) draws
one token row at the key's dtypes and its oracle is the routing's own weights [1, top_k] (the plain reference of the
same drawn numbers), its output the kernel's weights argument. No launch: host only.

What it catches: a key written by hand that drifts from the kernel's, a cell dropped or doubled, a filter that keeps
everything, a synthesis measuring the indices or a weights oracle at other attributes than those it runs - red."""
import ast
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from _moe_route_compare import plain_route
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.ops import autotune_harness as AH
from nbxtensor.ops import launch_keys as LK
from nbxtensor.ops.oracles import fp64 as F

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("moe_route_keys", ROOT / "tools" / "moe_route_certification_keys.py")
KT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(KT)
CENSUS = json.loads(KT.CENSUS.read_text())


def test_one_line_a_cell_each_the_kernels_own_key(tmp_path):
    out = tmp_path / "keys.txt"
    assert KT.main(["--out", str(out)]) == 0
    lines = out.read_text().splitlines()
    assert len(lines) == len(CENSUS["cells"]) == len(set(lines))
    for line, c in zip(lines, CENSUS["cells"]):
        qual, sep, key = line.partition("::")
        assert sep and qual == LK.MOE_ROUTE
        want = LK.moe_route_launches(c["E"], c["H"], c["top_k"], NBXDtype[c["hidden"]], NBXDtype[c["router_weight"]])
        assert [(qual, ast.literal_eval(key))] == want


def test_the_dtypes_filter_keeps_only_the_cells_named_and_refuses_an_unknown_one(tmp_path):
    out = tmp_path / "keys.txt"
    assert KT.main(["--out", str(out), "--dtypes", "float32,float16"]) == 0
    keys = [ast.literal_eval(l.partition("::")[2]) for l in out.read_text().splitlines()]
    assert keys and {k[3] for k in keys} == {"fp32", "fp16"}
    assert len(keys) == sum(c["hidden"] in ("float32", "float16") for c in CENSUS["cells"])
    assert KT.main(["--out", str(out), "--dtypes", "float64"]) == 2


@pytest.mark.parametrize("dt", ["fp32", "fp16", "bf16"])
def test_the_harness_synthesis_of_a_key_is_the_routing_weights_of_its_draw(dt):
    key = LK.moe_route_launches(64, 100, 4, AH._nbx(dt), AH._nbx(dt))[0][1]
    made = AH.synthesize(LK.MOE_ROUTE, None, key, np.random.default_rng(3))
    assert made is not None
    run, oracle, out_name = made
    assert callable(run) and out_name == "wt_ptr"
    ref = oracle()
    assert ref.shape == (1, 4)
    a = AH._MOE_ROUTE_CERTIFIED_AT
    assert a["score_fn"] == "sigmoid" and a["renorm_eps"] is not None and a["fill"] == -math.inf
    # the same draw again: the oracle is the plain routing of those numbers at the certified attributes
    rng = np.random.default_rng(3)
    x, w, b = (F.f64(AH.draw(rng, s, d)) for s, d in (((1, 100), dt), ((64, 100), dt), ((64,), "fp32")))
    _, plain_w, _, _ = plain_route(x, w, b, dict(a, top_k=4))
    assert np.allclose(ref, plain_w, rtol=1e-12, atol=0)
