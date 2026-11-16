"""Measure, on a named card, the gated delta rule's two forms in the library (the chunked form,
`ops.gated_delta_rule_chunked.gated_delta_rule_chunked_wrapper`, and the serial scan,
`ops.gated_delta_rule.gated_delta_rule_wrapper`) and the flash-linear-attention library's `chunk_gated_delta_rule` (the
reference to beat), at every cell of the op's census (`src/nbxtensor/ops/gated_delta_rule_census.json`: its batches,
heads, head dims, dtypes and lengths; nothing of the grid is written here), and propose the census cells.

THE VERDICT RULE, fixed before any run (task 547, 2026-11-16):
    PASS iff, at EVERY measured point (cell x length), the library's chunked form is
      (a) never slower than fla: slower means the chunked form's first quartile of wall time is above fla's third
          quartile (the two distributions do not overlap and ours is the slower one); and
      (b) within the bound: its output and its state against the float64 oracle (`delta_rule_oracle`, on the first and
          the last value head of batch row 0, the delta rule being independent per value head) at most 4e-6 (output
          plus its storage rounding, 2^-11 at float16, 2^-8 at bfloat16; state 4e-6), the bound of the device tests.
    FAIL otherwise, with every point that fails named. A point fla cannot run (an error it raises) is reported with
    its error and counts as a FAIL of the measurement, never as a pass.
THE PROPOSED CELL of each census cell: `chunked_from_length` = the shortest measured length L0 such that at L0 and at
every longer measured length the chunked form is within the bound and faster than the serial scan (its third quartile
below the serial scan's first quartile, or the serial scan out of device memory); null when there is none. Its
`proof` names the card, the commit, the date and the result file. The proposal is written in the result file; the
census itself is changed by hand from it, never by this script.

fla runs in a CHILD process (`--fla-child`), the only process that imports torch: the library is measured in a process
where torch is absent (the library's Triton branch never imports it). fla is the reference only: nothing of it is
imported by the library.

Run (on the card's host, the library installed or `src` on PYTHONPATH, fla and torch in the same python for the child):
    python tools/measure_the_chunked_gated_delta_rule.py --card "<the card's name>" --device cuda:0 \
        --warmup 3 --reps 10 --out <result.json>
Optional: --cells i,j (census cell indexes; default every cell), --lengths a,b (a subset of each cell's lengths).
Exit 0 on PASS, 1 on FAIL, 2 when the run could not be made (a named error).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CENSUS = ROOT / "src" / "nbxtensor" / "ops" / "gated_delta_rule_census.json"
OUT_BOUND, STATE_BOUND = 4e-6, 4e-6
ROUNDING = {"float32": 0.0, "float16": 2.0 ** -11, "bfloat16": 2.0 ** -8}
EPS = 1e-6


def draw(B, L, Hk, Hv, dk, dv, seed):
    """The operands, float32 on the host, as the library's tests draw them (q, k, v normal; g = -A softplus(a) <= 0;
    beta in (0, 1); a non-zero start state)."""
    r = np.random.default_rng(seed)
    q = r.standard_normal((B, L, Hk, dk), dtype=np.float32)
    k = r.standard_normal((B, L, Hk, dk), dtype=np.float32)
    v = r.standard_normal((B, L, Hv, dv), dtype=np.float32)
    A = r.uniform(0.01, 0.1, Hv).astype(np.float32)
    g = (-A * np.log1p(np.exp(r.standard_normal((B, L, Hv), dtype=np.float32)))).astype(np.float32)
    beta = (1.0 / (1.0 + np.exp(-r.standard_normal((B, L, Hv), dtype=np.float32)))).astype(np.float32)
    s = (0.5 * r.standard_normal((B, Hv, dk, dv), dtype=np.float32)).astype(np.float32)
    return q, k, v, g, beta, s


def quartiles(ms):
    q1, q2, q3 = np.percentile(np.asarray(ms, np.float64), [25, 50, 75])
    return {"q1": float(q1), "median": float(q2), "q3": float(q3), "ms": [float(x) for x in ms]}


def timed(fn, sync, warmup, reps):
    out = None
    for _ in range(warmup):
        out = fn()
    sync()
    ms = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        out = fn()
        sync()
        ms.append((time.perf_counter() - t0) * 1e3)
    return out, quartiles(ms)


# --- fla, in its own process (the only torch here) -------------------------------------------------------------------
def fla_child(spec):
    import torch                                                   # the reference's own runtime, this process only
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    import fla
    dt = getattr(torch, spec["dtype"])
    dev = torch.device(spec["device"])
    q, k, v, g, beta, s = draw(*(spec[n] for n in ("B", "L", "Hk", "Hv", "dk", "dv", "seed")))
    t = lambda a, d: torch.from_numpy(a).to(dev).to(d)
    args = (t(q, dt), t(k, dt), t(v, dt), t(g, torch.float32), t(beta, dt))
    s0 = t(s, torch.float32)
    run = lambda: chunk_gated_delta_rule(*args, scale=spec["dk"] ** -0.5, initial_state=s0, output_final_state=True,
                                         use_qk_l2norm_in_kernel=True)
    (o, so), times = timed(run, torch.cuda.synchronize, spec["warmup"], spec["reps"])
    heads = spec["heads"]
    np.savez(spec["arrays"], **{f"out{h}": o[:1, :, h:h + 1].float().cpu().numpy() for h in heads},
             **{f"state{h}": so[:1, h:h + 1].float().cpu().numpy() for h in heads})
    return {"fla_version": getattr(fla, "__version__", None), "torch_version": torch.__version__,
            "device_name": torch.cuda.get_device_name(dev), "time": times}


def run_fla(spec):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--fla-child", json.dumps(spec)],
                           env=env, capture_output=True, text=True)
    if child.returncode != 0:
        return {"error": child.stderr.strip().splitlines()[-1] if child.stderr.strip() else f"exit {child.returncode}"}
    return json.loads(child.stdout.strip().splitlines()[-1])


# --- the library ---------------------------------------------------------------------------------------------------
def oracle_deviation_on_heads(q, k, v, g, beta, s, out_of, state_of, heads, groups, dt):
    """The output's and the state's deviation from the float64 oracle on value heads `heads` of batch row 0
    (`out_of(h)` [1, L, 1, dv] and `state_of(h)` [1, 1, dk, dv] the measured ones), the oracle reading the inputs
    rounded as the kernels read them."""
    from nbxtensor.ops.oracles import fp64 as O
    rnd = rounder(dt)
    worst_o = worst_s = 0.0
    for h in heads:
        kh = h // groups
        ref_o, ref_s = O.delta_rule_oracle(rnd(q[:1, :, kh:kh + 1]), rnd(k[:1, :, kh:kh + 1]), rnd(v[:1, :, h:h + 1]),
                                           g[:1, :, h:h + 1], rnd(beta[:1, :, h:h + 1]), s[:1, h:h + 1],
                                           qk_l2norm=True, eps=EPS, scale=q.shape[-1] ** -0.5)()
        worst_o = max(worst_o, O.oracle_deviation(out_of(h), ref_o))
        worst_s = max(worst_s, O.oracle_deviation(state_of(h), ref_s))
    return worst_o, worst_s


def rounder(dt):
    """float32 -> the numbers it carries once stored at `dt` (bfloat16 by the library's own host conversion)."""
    if dt == "float32":
        return lambda a: a.astype(np.float64)
    if dt == "float16":
        return lambda a: a.astype(np.float16).astype(np.float64)
    if dt == "bfloat16":
        from nbxtensor.dtype.convert import bf16_bits_to_f32, f32_to_bf16_bits
        return lambda a: bf16_bits_to_f32(f32_to_bf16_bits(np.ascontiguousarray(a, np.float32))).astype(np.float64)
    raise ValueError(f"measure: dtype {dt!r} is not one the gated delta rule takes (float32, float16, bfloat16)")


def measure_ours(form, ops, warmup, reps):
    from nbxtensor import device
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.memory.allocator import DeviceOOMError
    from nbxtensor.ops.gated_delta_rule import gated_delta_rule_wrapper
    from nbxtensor.ops.gated_delta_rule_chunked import gated_delta_rule_chunked_wrapper
    fn = {"chunked": gated_delta_rule_chunked_wrapper, "serial": gated_delta_rule_wrapper}[form]
    attrs = dict(qk_l2norm=True, eps=EPS, scale=ops[0].shape[-1] ** -0.5, compute_dtype=NBXDtype.float32)
    try:
        (o, so), times = timed(lambda: fn(*ops, **attrs), device.synchronize, warmup, reps)
    except DeviceOOMError as e:                     # recorded by name in the result: the serial scan's known limit
        return None, None, {"error": f"DeviceOOMError: {e}"}
    return o, so, times


def point(cell, L, args):
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.ops.oracles.fp64 import host_values
    from nbxtensor.tensor import NBXTensor
    B, Hk, Hv, dk, dv, dt = (cell[n] for n in ("B", "Hk", "Hv", "dk", "dv", "dtype"))
    seed = 547 + L + 7 * B
    q, k, v, g, beta, s = draw(B, L, Hk, Hv, dk, dv, seed)
    ndt = NBXDtype[dt]
    dev = lambda a, d: NBXTensor.from_numpy(np.ascontiguousarray(a)).to(device=args.device).to(d)
    ops = (dev(q, ndt), dev(k, ndt), dev(v, ndt), dev(g, NBXDtype.float32), dev(beta, ndt), dev(s, NBXDtype.float32))
    heads = sorted({0, Hv - 1})
    res = {"L": L}
    host = lambda t: host_values(t.to(NBXDtype.float32)).astype(np.float64)
    for form in ("chunked", "serial"):
        o, so, times = measure_ours(form, ops, args.warmup, args.reps)
        res[form] = {"time": times}
        if o is not None:
            ho, hs = host(o), host(so)
            dev_o, dev_s = oracle_deviation_on_heads(q, k, v, g, beta, s, lambda h: ho[:1, :, h:h + 1],
                                                     lambda h: hs[:1, h:h + 1], heads, Hv // Hk, dt)
            res[form].update(out_deviation=dev_o, state_deviation=dev_s,
                             within=bool(dev_o <= OUT_BOUND + ROUNDING[dt] and dev_s <= STATE_BOUND))
        del o, so
    del ops
    arrays = Path(args.out).with_suffix(f".fla_{B}_{L}_{dt}.npz")
    spec = dict(B=B, L=L, Hk=Hk, Hv=Hv, dk=dk, dv=dv, dtype=dt, seed=seed, device=args.device, warmup=args.warmup,
                reps=args.reps, heads=heads, arrays=str(arrays))
    fla = run_fla(spec)
    if "error" not in fla:
        with np.load(arrays) as z:
            fo, fs = {h: z[f"out{h}"] for h in heads}, {h: z[f"state{h}"] for h in heads}
        arrays.unlink()
        dev_o, dev_s = oracle_deviation_on_heads(q, k, v, g, beta, s, fo.get, fs.get, heads, Hv // Hk, dt)
        fla.update(out_deviation=dev_o, state_deviation=dev_s)
    res["fla"] = fla
    ch, fl = res["chunked"], res["fla"]
    failures = []
    if "error" in ch["time"]:
        failures.append(f"the chunked form failed: {ch['time']['error']}")
    elif not ch["within"]:
        failures.append(f"the chunked form is outside the bound (out {ch['out_deviation']:.2e}, "
                        f"state {ch['state_deviation']:.2e})")
    if "error" in fl:
        failures.append(f"fla could not run: {fl['error']}")
    elif "error" not in ch["time"] and ch["time"]["q1"] > fl["time"]["q3"]:
        failures.append(f"slower than fla: chunked q1 {ch['time']['q1']:.3f} ms > fla q3 {fl['time']['q3']:.3f} ms")
    res["failures"] = failures
    sr = res["serial"]["time"]
    res["chunked_beats_serial"] = bool("error" not in ch["time"] and ch["within"] and
                                       ("error" in sr or ch["time"]["q3"] < sr["q1"]))
    return res


def proposed_cell(cell, points, proof):
    start = None
    for p in sorted(points, key=lambda p: p["L"], reverse=True):     # from the longest down: the run of wins
        if not p["chunked_beats_serial"]:
            break
        start = p["L"]
    return dict(cell, chunked_from_length=start if start is not None and start > 1 else None, proof=proof)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--card", help="the card's name as the result will cite it (required)")
    ap.add_argument("--device", help="the library's device string of the card, e.g. cuda:0 (required)")
    ap.add_argument("--warmup", type=int, help="untimed calls before the timed ones (required)")
    ap.add_argument("--reps", type=int, help="timed calls a point (required, at least 4: quartiles)")
    ap.add_argument("--out", help="the result file (JSON) to write (required)")
    ap.add_argument("--cells", help="census cell indexes, comma-separated (default: every cell)")
    ap.add_argument("--lengths", help="a subset of the cells' lengths, comma-separated (default: every length)")
    ap.add_argument("--fla-child", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.fla_child:
        print(json.dumps(fla_child(json.loads(a.fla_child))))
        return 0
    missing = [f"--{n}" for n in ("card", "device", "warmup", "reps", "out") if getattr(a, n) in (None, "")]
    if missing:
        print(f"measure: missing {', '.join(missing)}: the result names its card, device, repetitions and file",
              file=sys.stderr)
        return 2
    if a.reps < 4 or a.warmup < 1:
        print("measure: --reps must be at least 4 (quartiles) and --warmup at least 1 (the first call compiles)",
              file=sys.stderr)
        return 2
    doc = json.loads(CENSUS.read_text())
    cells = doc["cells"]
    chosen = range(len(cells)) if not a.cells else [int(i) for i in a.cells.split(",")]
    only = None if not a.lengths else {int(x) for x in a.lengths.split(",")}
    from nbxtensor.ops.wrappers import has_native_bf16
    no_bf16 = [i for i in chosen if cells[i]["dtype"] == "bfloat16"]
    if no_bf16 and not has_native_bf16():
        print(f"measure: census cells {no_bf16} are bfloat16 and this card has no native bfloat16: pass --cells "
              f"with the others ({[i for i in chosen if i not in no_bf16]})", file=sys.stderr)
        return 2
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if not commit:
        print(f"measure: {ROOT} is not a git checkout: the result must name the commit it measured", file=sys.stderr)
        return 2
    when = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    proof = {"card": a.card, "commit": commit, "date": when, "result": Path(a.out).name}
    result = {"what": "the gated delta rule: the library's chunked form, its serial scan and fla's chunk form",
              "card": a.card, "device": a.device, "commit": commit, "date": when, "warmup": a.warmup, "reps": a.reps,
              "rule": __doc__[__doc__.index("THE VERDICT RULE"):__doc__.index("THE PROPOSED CELL")].strip(),
              "cells": [], "proposed_cells": []}
    for i in chosen:
        cell = {n: cells[i][n] for n in ("B", "Hk", "Hv", "dk", "dv", "dtype")}
        points = []
        result["cells"].append({"index": i, "cell": cell, "points": points})
        for L in cells[i]["lengths"]:
            if only is not None and L not in only:
                continue
            p = point(cell, L, a)
            points.append(p)
            print(f"cell {i} {cell} L={L}: chunked {p['chunked']['time'].get('median', p['chunked']['time'])} ms, "
                  f"serial {p['serial']['time'].get('median', p['serial']['time'])} ms, "
                  f"fla {p['fla'].get('time', {}).get('median', p['fla'].get('error'))} ms"
                  f"{'  FAIL: ' + '; '.join(p['failures']) if p['failures'] else ''}", flush=True)
            Path(a.out).write_text(json.dumps(dict(result, partial=True), indent=1))
        result["proposed_cells"].append(proposed_cell(dict(cells[i]), points, proof))
    fails = [f"cell {c['index']} L={p['L']}: {f}" for c in result["cells"] for p in c["points"] for f in p["failures"]]
    result["verdict"] = "PASS" if not fails else "FAIL"
    result["failures"] = fails
    Path(a.out).write_text(json.dumps(result, indent=1))
    print(f"VERDICT {result['verdict']}" + "".join(f"\n  {f}" for f in fails))
    print(f"proposed census cells: {Path(a.out)} (key proposed_cells)")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
