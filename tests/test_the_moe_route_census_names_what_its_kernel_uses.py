"""The routing brick's op census (`src/nbxtensor/ops/moe_route_census.json`) holds against the sources, with no Triton
and no device: the Triton primitives and dtypes it lists are exactly those the kernel's source calls (read with `ast`),
every backend the library runs on (cuda, metal, cpu) has its lowering and its certification status, the certification
commands of the two host cards name the keys tool, their profile and the certified-only device tests, every cell names
a geometry and two model dtypes with no proof yet, and the certification harness registers a synthesis for the kernel.

What it catches: a primitive added to the kernel (a tl.dot, an atomic) the census does not list, a primitive listed
the kernel no longer uses, a backend dropped from the census, a certification command naming no card profile or a
keys file the tool does not write, a cell certified without a proof, the harness losing the kernel's synthesis - red."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OPS = ROOT / "src" / "nbxtensor" / "ops"
CENSUS = json.loads((OPS / "moe_route_census.json").read_text())
DTYPES = ("float32", "float16", "bfloat16")


def _kernel_tl_names():
    tree = ast.parse((OPS / "moe_route.py").read_text())
    kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "moe_route_kernel")
    calls, names = set(), set()
    for n in ast.walk(kernel):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "tl":
            names.add(n.attr)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) \
                and n.func.value.id == "tl":
            calls.add(n.func.attr)
    return calls, names - calls - {"constexpr"}


def test_the_census_lists_exactly_the_primitives_and_dtypes_of_the_kernel_source():
    calls, dtypes = _kernel_tl_names()
    assert CENSUS["format"] == "nbx-moe-route-census/1"
    assert CENSUS["kernel"] == "nbxtensor.ops.moe_route.moe_route_kernel"
    assert set(CENSUS["primitives"]) == calls, f"kernel calls {sorted(calls)}, census {sorted(CENSUS['primitives'])}"
    assert set(CENSUS["dtypes"]) == dtypes
    assert "dot" not in calls and not any(c.startswith("atomic") for c in calls)


def test_every_backend_has_its_lowering_and_its_certification_status():
    assert set(CENSUS["backends"]) == {"cuda", "metal", "cpu"}
    for name, b in CENSUS["backends"].items():
        assert b["lowering"] and b["certification"], name
    assert CENSUS["backends"]["cuda"]["certification"].startswith("NEEDS CARD")
    assert CENSUS["backends"]["metal"]["certification"].startswith("NEEDS CARD")


def test_the_two_host_certifications_are_commands_a_card_runs():
    cert = CENSUS["certification"]
    for host, profile, vendor in (("dell", "volta", "nvidia"), ("mac", "apple_m4_pro", "apple")):
        cmd = cert[host]
        assert "python tools/moe_route_certification_keys.py --out moe_route_keys.txt" in cmd
        assert f"neurobrix autotune certify --profile {profile} --vendor {vendor} --keys moe_route_keys.txt" in cmd
        assert "NBX_AUTOTUNE_CERTIFIED_ONLY=1" in cmd and "test_one_token_of_the_moe_route_on_a_device" in cmd
    assert (ROOT / "tools" / "moe_route_certification_keys.py").is_file()


def test_every_cell_is_a_geometry_at_model_dtypes_with_no_proof_yet():
    cells = CENSUS["cells"]
    assert cells
    for c in cells:
        assert set(c) == {"E", "H", "top_k", "hidden", "router_weight", "from", "proof"}
        assert 1 <= c["top_k"] <= c["E"] <= 256 and c["H"] >= 1
        assert c["hidden"] in DTYPES and c["router_weight"] in DTYPES
        assert c["proof"] is None, "a cell is certified only by a card's run, its proof written from it"
    assert len({(c["E"], c["H"], c["top_k"], c["hidden"]) for c in cells}) == len(cells)


def test_the_certification_harness_registers_the_kernels_synthesis():
    tree = ast.parse((OPS / "autotune_harness.py").read_text())
    fams = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "_FAMILIES" for t in n.targets))
    pairs = {k.value: v.id for k, v in zip(fams.keys, fams.values) if isinstance(k, ast.Constant)}
    assert pairs.get("moe_route_kernel") == "_moe_route"
