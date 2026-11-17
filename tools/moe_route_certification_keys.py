#!/usr/bin/env python3
# Copyright NeuroBrix, Apache-2.0
"""The keys of the routing kernel a card certifies, written from the op's census (`src/nbxtensor/ops/
moe_route_census.json`, its `cells`) as the certifier's `--keys` file: one '<kernel qualname>::<key repr>' line a
cell, the key formed by the library's own `launch_keys.moe_route_launches` (nothing of the key is written here).

    python tools/moe_route_certification_keys.py --out moe_route_keys.txt [--dtypes float32,float16]

`--dtypes` keeps only the cells whose hidden dtype is listed (a card without native bfloat16). Exit 0 with the file
written, 2 on a census that does not hold or a dtype it does not name (a named error)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CENSUS = ROOT / "src" / "nbxtensor" / "ops" / "moe_route_census.json"
FORMAT = "nbx-moe-route-census/1"


def key_lines(census: dict, dtypes=None) -> list:
    """The '<qualname>::<key repr>' lines of the census cells (of `dtypes` only, when given)."""
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.ops.launch_keys import moe_route_launches
    if census.get("format") != FORMAT:
        raise ValueError(f"the census has format {census.get('format')!r}; this tool reads {FORMAT!r}")
    known = {c["hidden"] for c in census["cells"]}
    if dtypes is not None and set(dtypes) - known:
        raise ValueError(f"--dtypes {sorted(set(dtypes) - known)}: no census cell at that dtype (cells: {sorted(known)})")
    lines = []
    for c in census["cells"]:
        if dtypes is not None and c["hidden"] not in dtypes:
            continue
        for qual, key in moe_route_launches(c["E"], c["H"], c["top_k"], NBXDtype[c["hidden"]],
                                            NBXDtype[c["router_weight"]]):
            lines.append(f"{qual}::{key!r}")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="the keys file to write")
    ap.add_argument("--dtypes", default=None, help="comma-separated hidden dtypes to keep (default: every cell)")
    a = ap.parse_args(argv)
    try:
        lines = key_lines(json.loads(CENSUS.read_text()), None if a.dtypes is None else a.dtypes.split(","))
    except (ValueError, KeyError, OSError) as e:
        print(f"moe_route_certification_keys: {e}", file=sys.stderr)
        return 2
    Path(a.out).write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} keys of the routing kernel -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
