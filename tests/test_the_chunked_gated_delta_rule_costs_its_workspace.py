"""`_cost.gated_delta_rule_chunked`, the cost of `gated_delta_rule_chunked_wrapper`, on any host: its two returned
buffers (the output at v's dtype, the state out float32), then the float32 workspace the wrapper allocates, by the
names and shapes `gated_delta_rule_chunked_workspace` gives (one answer for the wrapper, its cost and its tests), the
chunk states counted whole for a ragged last chunk; the bytes each of its three launches reads and writes; its flops
per chunk; and every refusal of the wrapper, by name. What it catches: a workspace buffer forgotten or sized at the
input dtype, the chunk count floored, a state in counted when none is given, a cost that accepts what the wrapper
refuses - red."""
import pytest

from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.ops import _cost as C
from nbxtensor.ops.gated_delta_rule_chunked import GATED_DELTA_RULE_CHUNK, gated_delta_rule_chunked_workspace

F32 = NBXDtype.float32


def _specs(B=2, L=65, Hk=2, Hv=6, dk=128, dv=96, dt="float16", state=True):
    s = C.Spec.contiguous
    return [s((B, L, Hk, dk), dt), s((B, L, Hk, dk), dt), s((B, L, Hv, dv), dt), s((B, L, Hv), "float32"),
            s((B, L, Hv), dt), s((B, Hv, dk, dv), "float32") if state else None]


def _n(shape):
    n = 1
    for d in shape:
        n *= d
    return n


@pytest.mark.parametrize("dt", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("state", [True, False], ids=["state", "zeros"])
@pytest.mark.parametrize("L", [1, 64, 65, 200])
def test_the_cost_is_the_outputs_then_the_float32_workspace_its_helper_names(dt, state, L):
    B, Hk, Hv, dk, dv = 2, 2, 6, 128, 96
    specs = _specs(B, L, Hk, Hv, dk, dv, dt, state)
    c = C.gated_delta_rule_chunked(*specs, qk_l2norm=True, compute_dtype=F32)
    size = {"float16": 2, "bfloat16": 2, "float32": 4}[dt]
    ws = {n: 4 * _n(sh) for n, sh in gated_delta_rule_chunked_workspace(B, L, Hv, dk, dv).items()}
    chunks = -(-L // GATED_DELTA_RULE_CHUNK)
    assert ws["states"] == 4 * B * chunks * Hv * dk * dv
    out, so = B * L * Hv * dv * size, B * Hv * dk * dv * 4
    assert [(b.name, b.nbytes) for b in c.outputs] == [("output", out), ("state out", so)]
    assert [(b.name, b.nbytes) for b in c.workspace] == [(f"workspace {n}", b) for n, b in ws.items()]
    assert c.peak_bytes == out + so + sum(ws.values())
    q, k, v, g, beta, s = (0 if x is None else x.nbytes for x in specs)
    assert c.bytes_moved == (k + v + g + beta + ws["w"] + ws["u"] + ws["gc"]
                             + k + ws["w"] + ws["u"] + ws["gc"] + s + ws["states"] + ws["u"] + so
                             + q + k + ws["u"] + ws["gc"] + ws["states"] + out)
    Cc = GATED_DELTA_RULE_CHUNK
    gram, solve, wu = 2 * Cc * Cc * dk, 2 * Cc ** 3 // 3, 2 * Cc * Cc * (dk + dv)
    corrected, write, decay, output = 2 * Cc * dk * dv, 2 * Cc * dk * dv, dk * dv, 2 * Cc * dk * dv + 2 * Cc * Cc * (dk + dv)
    per_chunk = gram + solve + wu + corrected + write + decay + output
    assert c.flops == B * Hv * (chunks * per_chunk + L * 2 * (3 * dk + 1))


def test_the_cost_grows_by_the_chunk_not_by_the_token_in_its_workspace():
    at = lambda L: C.gated_delta_rule_chunked(*_specs(L=L), qk_l2norm=False, compute_dtype=F32)
    states = lambda L: dict((b.name, b.nbytes) for b in at(L).workspace)["workspace states"]
    assert states(1) == states(64) < states(65) == states(128) < states(129)
    assert at(64).flops == at(1).flops < at(65).flops                       # no norms: products by the whole chunk


@pytest.mark.parametrize("at,dt,kw,match", [
    (None, None, dict(compute_dtype=NBXDtype.float16), "compute_dtype float16 is refused"),
    (5, "float16", {}, "state is float16; it must be float32"),
    (3, "bfloat16", {}, "g is bfloat16; it must be float32"),
    (0, "float64", {}, "q is float64; it must be one of"),
])
def test_the_cost_refuses_by_name_what_the_wrapper_refuses(at, dt, kw, match):
    specs = _specs()
    if at is not None:
        specs[at] = C.Spec.contiguous(specs[at].shape, dt)
    with pytest.raises(ValueError, match=match):
        C.gated_delta_rule_chunked(*specs, **dict(dict(qk_l2norm=True, compute_dtype=F32), **kw))


def test_the_cost_refuses_a_state_of_the_wrong_shape_and_a_sequence_of_no_token_by_name():
    specs = _specs()
    specs[5] = C.Spec.contiguous((2, 6, 96, 128), "float32")
    with pytest.raises(ValueError, match="state has shape \\(2, 6, 96, 128\\); .* ask \\(2, 6, 128, 96\\)"):
        C.gated_delta_rule_chunked(*specs, qk_l2norm=True, compute_dtype=F32)
    with pytest.raises(ValueError, match="a sequence of length 0: the delta rule runs over at least 1 token"):
        C.gated_delta_rule_chunked(*_specs(L=0), qk_l2norm=True, compute_dtype=F32)
