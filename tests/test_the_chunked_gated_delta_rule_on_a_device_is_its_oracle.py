"""`gated_delta_rule_chunked_wrapper` on a device: its output and its state against the float64 oracle
(`delta_rule_oracle`) at L = 1, 23, 63, 64, 65, 130 and 200 (one token, under a chunk, both sides of the 64-token
chunk, several chunks and a ragged tail), from a non-zero state at every length and from zeros at L = 23 and 65,
Hv / Hk = 1 and 3 read un-expanded, B = 1 and 2, inputs at float32, float16 and bfloat16 (bfloat16 where the device has
it natively), q, k and v handed as views of one projection (strided, never copied). Beside it, the state handed over
between the two forms of the op: a prompt run chunked, then token steps by the serial scan, then chunked again (and the
reverse) is one pass of the oracle; and the chunked form's workspace is the one `gated_delta_rule_chunked_workspace`
names.

A card test: on a host with no device it skips by name (the fixture below asks the one door; `_no_device.py`).

THE BOUNDS, stated from what was measured (2026-11-16, this lot's proof folder): the kernels on Triton's interpreter
(float32 numpy arithmetic in the kernels' own order), worst over 130 draws at dk 32 and 64 (out 4.2e-07, state 5.1e-07;
`test_the_chunked_gated_delta_rule_is_its_oracle_on_the_interpreter` carries those) and over 12 draws at dk = dv = 128,
the dimensions below, decay bound 0.1 and 1.0: out 1.3e-06, state 2.1e-06. The device's bound is the serial scan's on
a device, 4e-06 out and state, about twice the interpreter's worst at these dimensions (the card's fused multiply-adds
and its own exp and sqrt round differently from numpy). An output stored at float16 or bfloat16 adds at most its
rounding, 2^-11 and 2^-8; the state is float32 at every input dtype. Each test prints its measured deviation: the
card's log is the measurement of these bounds.

What it catches: a chunk's solve or decay wrong on the card's own dot, a ragged last chunk, a stride misread, the
state not carried or narrowed, a value head on the wrong key head, a state one form hands back that the other cannot
start from - red."""
import numpy as np
import pytest

from _delta_rule_draw import draw
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.ops.oracles import fp64 as O

F32 = NBXDtype.float32
SCALE = 128 ** -0.5
ATTRS = dict(qk_l2norm=True, eps=1e-6, scale=SCALE)
OUT_F32, STATE = 4e-6, 4e-6
ROUNDING = {NBXDtype.float32: 0.0, NBXDtype.float16: 2.0 ** -11, NBXDtype.bfloat16: 2.0 ** -8}
CASES = ([(L, B, hk, hv, 1) for L in (1, 23, 63, 64, 65, 130, 200) for B, hk, hv in ((1, 4, 4), (2, 2, 6))] +
         [(L, B, hk, hv, 0) for L in (23, 65) for B, hk, hv in ((1, 4, 4), (2, 2, 6))])
IDS = [f"L{L}-B{B}-Hv{hv}-Hk{hk}-{'state' if nz else 'zeros'}" for L, B, hk, hv, nz in CASES]


@pytest.fixture(autouse=True)
def _accelerator():
    from nbxtensor.memory.allocator import DeviceAllocator
    if DeviceAllocator.device_count() == 0:          # only a machine without an accelerator skips; any other error fails
        pytest.skip("no accelerator on this machine")


def _to_device(a, dt):
    from nbxtensor.tensor import NBXTensor
    return NBXTensor.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device="cuda:0").to(dt)


def _host(t):
    return O.host_values(t.to(F32)).astype(np.float64)


def _rounded(a, dt):
    """The numbers a float32 draw carries once stored at `dt` (the oracle reads what the kernel reads)."""
    return _host(_to_device(a, dt))


def _operands(q, k, v, g, beta, s, dt):
    """q, k, v as views of ONE projection [B, L, (2 Hk dk + Hv dv)] at `dt` (strided, un-expanded), beta at `dt`, g and
    the state float32."""
    B, L, hk, dk = q.shape
    hv, dv = v.shape[2:]
    proj = _to_device(np.concatenate([q.reshape(B, L, -1), k.reshape(B, L, -1), v.reshape(B, L, -1)], -1), dt)
    n = hk * dk
    return (proj[:, :, :n].unflatten(2, (hk, dk)), proj[:, :, n:2 * n].unflatten(2, (hk, dk)),
            proj[:, :, 2 * n:].unflatten(2, (hv, dv)), _to_device(g, F32), _to_device(beta, dt),
            None if s is None else _to_device(s, F32))


def _chunked(ops):
    from nbxtensor.ops.gated_delta_rule_chunked import gated_delta_rule_chunked_wrapper
    return gated_delta_rule_chunked_wrapper(*ops, compute_dtype=F32, **ATTRS)


def _serial(ops):
    from nbxtensor.ops.gated_delta_rule import gated_delta_rule_wrapper
    return gated_delta_rule_wrapper(*ops, compute_dtype=F32, **ATTRS)


@pytest.mark.parametrize("dt", [NBXDtype.float32, NBXDtype.float16, NBXDtype.bfloat16], ids=["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("L,B,hk,hv,nonzero", CASES, ids=IDS)
def test_the_chunked_form_over_a_length_is_the_oracle(dt, L, B, hk, hv, nonzero):
    from nbxtensor.ops.wrappers import has_native_bf16
    if dt == NBXDtype.bfloat16 and not has_native_bf16():
        pytest.skip("bfloat16: this device has no native bfloat16")
    q, k, v, g, beta, s = draw(B, L, hk, hv, 128, 128, 600 + L + B, bool(nonzero), decay_max=0.1)
    out, state = _chunked(_operands(q, k, v, g, beta, s, dt))
    assert out.nbx_dtype == dt and tuple(out.shape) == (B, L, hv, 128)
    assert state.nbx_dtype == F32 and tuple(state.shape) == (B, hv, 128, 128)
    ref_o, ref_s = O.delta_rule_oracle(_rounded(q, dt), _rounded(k, dt), _rounded(v, dt), g, _rounded(beta, dt), s,
                                       **ATTRS)()
    dev_o, dev_s = O.oracle_deviation(_host(out), ref_o), O.oracle_deviation(_host(state), ref_s)
    print(f"chunked {dt.name} L={L} B={B} Hv/Hk={hv // hk} {'state' if nonzero else 'zeros'}: "
          f"out {dev_o:.2e} (bound {OUT_F32 + ROUNDING[dt]:.1e}) state {dev_s:.2e} (bound {STATE:.1e})")
    assert dev_o <= OUT_F32 + ROUNDING[dt]
    assert dev_s <= STATE


@pytest.mark.parametrize("plan", [
    [("chunked", 100), ("serial", 1), ("serial", 1), ("serial", 1), ("chunked", 47)],
    [("serial", 23), ("chunked", 65), ("serial", 1), ("chunked", 61)],
], ids=["chunked-tokens-chunked", "serial-chunked-token-chunked"])
def test_the_state_passes_between_the_two_forms_as_one_pass(plan):
    L = sum(n for _, n in plan)
    q, k, v, g, beta, s = draw(2, L, 2, 6, 128, 128, 700 + L, True, decay_max=0.1)
    form = {"chunked": _chunked, "serial": _serial}
    state, outs, a = _to_device(s, F32), [], 0
    for name, n in plan:
        ops = _operands(*(x[:, a:a + n] for x in (q, k, v, g, beta)), None, F32)
        o, state = form[name](ops[:5] + (state,))
        outs.append(_host(o))
        a += n
    ref_o, ref_s = O.delta_rule_oracle(q, k, v, g, beta, s, **ATTRS)()
    dev_o, dev_s = O.oracle_deviation(np.concatenate(outs, 1), ref_o), O.oracle_deviation(_host(state), ref_s)
    print(f"{' -> '.join(f'{n} {c}' for n, c in plan)}: out {dev_o:.2e} state {dev_s:.2e}")
    assert dev_o <= OUT_F32 and dev_s <= STATE


def test_the_chunked_form_allocates_the_workspace_its_helper_names(monkeypatch):
    from nbxtensor.ops import gated_delta_rule_chunked as CH
    seen = []
    real = CH.gated_delta_rule_chunked_launch

    def launch(*args, **kw):
        seen.append({n: (tuple(t.shape), t.nbx_dtype) for n, t in args[8].items()})
        return real(*args, **kw)

    monkeypatch.setattr(CH, "gated_delta_rule_chunked_launch", launch)
    q, k, v, g, beta, s = draw(2, 65, 2, 6, 128, 64, 800, True, decay_max=0.1)
    _chunked(_operands(q, k, v, g, beta, s, NBXDtype.float16))
    assert seen == [{n: (sh, F32) for n, sh in CH.gated_delta_rule_chunked_workspace(2, 65, 6, 128, 64).items()}]
    assert seen[0]["states"][0] == (2, 2, 6, 128, 64)          # ceil(65 / 64) chunk states
