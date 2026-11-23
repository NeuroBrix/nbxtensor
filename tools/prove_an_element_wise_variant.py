"""Prove element-wise variants on the synthetic ladder alone — no model, no launch of a caller (`ops/ew_proof.py`).

Each case is ONE variant, its specialization written out below so its id matches a proof table's record:

  mul_scalar_strided_nd  `binary_strided.mul_scalar_strided_nd_kernel` (*fp32 D, *fp32 D, i32 D, fp32, *i64 D,
                         *i64 D, tile, constexpr NDIM 3), debug False — variant 6be93ab046480f3a; the red cell of a
                         V100: (4096, 2, 3) FAULTS and (256, 2, 3) PASSES (the register's records).
  add_strided_nd         `binary_strided.add_strided_nd_kernel` (*fp16 D, *fp16 D, *fp16 D, i32 D, fp32, *i64 D,
                         *i64 D, *i64 D, tile, constexpr NDIM 4), debug False — variant 6859212e3972bf7a; the register
                         says 2 of its 15 tiles faulted in Flex.1-alpha's launch without naming them: its red cell is
                         SOME tile faulting and (256, 2, 3) PASSING (the passing tile derived here, the smallest).

The proof runs every candidate of the element-wise class on the variant's synthetic launches (`ops/ew_synthetic.py`:
the ladder of counts with masked tails, contiguous and permuted layouts), every buffer against a guard
(`memory.guarded`), in the proof's worker process, and prints the record: the tiles that pass, every fault with its
text, the ladder, the guard and the synthetic bytes.

    PYTHONPATH=$PWD/src python tools/prove_an_element_wise_variant.py --red
    PYTHONPATH=$PWD/src python tools/prove_an_element_wise_variant.py --case add_strided_nd

`--red`: both cases; exit 0 only when, for EACH, its known-bad tile faults with a DEVICE fault and its known-good
tile passes (`verdict`): "RED FOUND". Otherwise "NOT FOUND" (the bad tile passed, or the good one did not) or
"COULD NOT RUN" (the proof was refused: it reached no verdict), exit 1. A fault in a record is always a device fault
(`ops/ew_proof_child.run_candidates`: anything else refuses the proof). `--case NAME`: one case, exit 0 when its
variant is proven.
"""
import argparse
import json
import sys

CASES = {
    "mul_scalar_strided_nd": ("mul_scalar_strided_nd_kernel",
                              ["*fp32", "*fp32", ("i32", "D"), ("fp32", None), "*i64", "*i64"], 3, "6be93ab046480f3a"),
    "add_strided_nd": ("add_strided_nd_kernel",
                       ["*fp16", "*fp16", "*fp16", ("i32", "D"), ("fp32", None), "*i64", "*i64", "*i64"], 4,
                       "6859212e3972bf7a"),
}
#: each case's red cell: (the tile that must fault, or None for "some tile"; the tile that must pass)
RED = {"mul_scalar_strided_nd": ((4096, 2, 3), (256, 2, 3)), "add_strided_nd": (None, (256, 2, 3))}


def verdict(rec: dict, bad, good) -> str:
    """The red cell's verdict on a variant's record: 'COULD NOT RUN' when the proof was refused before any verdict
    on a tile (unkept: unguarded, no recipe, a worker that could not run), 'RED FOUND' when the bad tile (or, None,
    some tile) faulted AND the good tile passed, 'NOT FOUND' otherwise."""
    if rec.get("unkept"):
        return "COULD NOT RUN"
    faulted = {tuple(f["tile"]) for f in rec.get("faults", [])}
    passed = {tuple(t) for t in rec.get("passed", [])}
    bad_seen = bool(faulted) if bad is None else tuple(bad) in faulted
    return "RED FOUND" if bad_seen and tuple(good) in passed else "NOT FOUND"


def prove_case(name: str) -> dict:
    """Prove one case's variant and print its record; the record."""
    from nbxtensor.ops import binary_strided, ew_proof
    from nbxtensor.ops.elementwise_launch import class_candidates
    from nbxtensor.ops.ew_tile import EwTile
    kname, params, ndim, want_id = CASES[name]
    kernel = getattr(binary_strided, kname)
    tile = EwTile(max(c[0] for c in class_candidates()), 0, "tile")
    spec = [(p, "D") if isinstance(p, str) else p for p in params] + [("constexpr", tile), ("constexpr", ndim)]
    options = {"num_warps": 2, "num_stages": 3, "debug": False}
    vid, _t = ew_proof.variant_of(kernel, spec, options, {"tile": tile})
    key = ew_proof.stamp_key({"v": vid})
    print(f"== {name}: variant {key} (the register's {want_id}: {'the same' if key == want_id else 'NOT THE SAME'})"
          f"\n   {vid}", flush=True)
    rec = ew_proof.prove(kernel, spec, options, {"tile": tile}, vid)
    print(json.dumps({k: rec.get(k) for k in ("kernel", "method", "guard", "recipe", "ladder", "synthetic_bytes",
                                                 "refused", "passed", "dissent", "seconds")}))
    for f in rec["faults"]:
        print(f"   FAULT at {tuple(f['tile'])}: {str(f['fault']).strip().splitlines()[-1][:300]}")
    return rec


def main(argv=None) -> int:
    """Prove the asked case(s), print the records, answer by the exit code."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--red", action="store_true", help="both cases, each expecting its red cell")
    ap.add_argument("--case", choices=sorted(CASES), default="mul_scalar_strided_nd")
    args = ap.parse_args(argv)
    if args.red:
        verdicts = []
        for name in ("mul_scalar_strided_nd", "add_strided_nd"):
            bad, good = RED[name]
            v = verdict(prove_case(name), bad, good)
            verdicts.append(v)
            print(f"{v}: {name}: " + (f"{bad} must fault" if bad else "some tile must fault")
                  + f" with a device fault and {good} must pass")
        return 0 if all(v == "RED FOUND" for v in verdicts) else 1
    rec = prove_case(args.case)
    return 0 if not rec["refused"] else 1


if __name__ == "__main__":
    sys.exit(main())
