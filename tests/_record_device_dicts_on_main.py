"""Record the device dict every listed compute capability resolves to, through the library's own resolution
(`device_files.for_arch` -> `_configs.vendor_profile_for_arch` -> `_profiles_for`, its exact / major-base /
`memory.by_capability` / prefix path), and the shared-memory budget the same resolution returns.

Run once against library main 50597afcc7a3 (a detached worktree), never against the tree under test:

    PYTHONNOUSERSITE=1 python tests/_record_device_dicts_on_main.py <main-worktree>/src \
        tests/_device_dicts_on_main_50597afcc7a3.json

The fixture it writes is what `test_every_class_that_runs_today_resolves_the_dict_it_resolved_on_main.py` compares
the tree under test against. The library's half only: no application directory is given.
"""
from __future__ import annotations

import json
import sys

MAIN = "50597afcc7a3"


def listed_capabilities() -> list:
    """Every capability a device file of the library lists: its `compute_capability` and the minors its
    `memory.by_capability` documents, sorted, each once."""
    from nbxtensor.backend import device_files
    caps = set()
    for vendor, name, _path in device_files.names():
        cfg = device_files.read(vendor, name)
        declared = str(cfg.get("compute_capability", "")).strip().lower()
        if declared:
            caps.add(declared)
        for minor in ((cfg.get("memory") or {}).get("by_capability") or {}):
            caps.add(str(minor).strip().lower())
    return sorted(caps)


def normalized(value):
    """A value as the JSON fixture holds it (dict keys as strings), so a live dict compares to a recorded one."""
    return json.loads(json.dumps(value, sort_keys=True))


def resolve(capability: str) -> dict:
    """What one capability resolves to through the library's own path: `profile` (the dict in force, None when
    nothing matched) and `budget` (the shared-memory budget `_profiles_for` returns, None when refused)."""
    from nbxtensor.backend import device_files
    from nbxtensor.ops import _configs
    device_files.forget()                                  # a capability that matches nothing must not inherit
    budget = _configs._profiles_for(capability)
    profile = dict(_configs._ACTIVE_PROFILE) or None
    return {"profile": normalized(profile), "budget": budget}


def record() -> dict:
    """The fixture: {capability: resolve(capability)} for every capability the library lists."""
    return {"main": MAIN, "capabilities": {c: resolve(c) for c in listed_capabilities()}}


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    sys.path.insert(0, src)
    import nbxtensor
    if not nbxtensor.__file__.startswith(src.rstrip("/")):
        raise SystemExit(f"nbxtensor imported from {nbxtensor.__file__}, not from {src}")
    with open(out, "w") as f:
        json.dump(record(), f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"{out}: {len(record()['capabilities'])} capabilities")
