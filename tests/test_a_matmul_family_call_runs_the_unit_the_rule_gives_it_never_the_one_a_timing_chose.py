"""A matmul-family call runs ONE unit by rule: the matrix unit where the profile declares one for the operands and the
call's rows fill its smallest tile, the FMA kernel everywhere else. A certificate (a timing, made once on one card) can
no longer move it: an entry that names the other unit is refused by name, and with no certificate, with the switch off,
or in a confirmation run the answer is the same.

(49) of the 1121 campaign: before, `_gemm_route` took the unit only where the certificate measured it faster (a near tie
flipped between two certifications), and a fused MoE took the FMA kernel with no certificate where a GEMM took the unit.

The routing is read on the Volta profile file (the one profile that declares a unit) and on an Apple profile (none), with
the certificate's reading stubbed: no card, no kernel launch. Run: python -m pytest tests/<this file>"""
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip("triton")

from nbxtensor.ops.linalg_wrappers import NBXDtype  # noqa: E402
from nbxtensor.ops import _configs as CF  # noqa: E402
from nbxtensor.ops import autotune_certified as CERT  # noqa: E402
from nbxtensor.ops import linalg_wrappers as LW  # noqa: E402
from nbxtensor.ops import matmul as MM  # noqa: E402
from nbxtensor.ops import moe_wrappers as MW  # noqa: E402
from nbxtensor.ops.fused_moe import fused_moe_kernel  # noqa: E402

DEVICES = Path(CF.__file__).resolve().parents[1] / "backend" / "devices"
FIELDS = CERT._UNIT_TILE_FIELDS


def _profile(monkeypatch, name):
    cfg = yaml.safe_load((DEVICES / name).read_text())
    monkeypatch.setattr(CF, "active_vendor_profile", lambda: dict(cfg))
    return cfg


def _operand(dtype):
    return SimpleNamespace(nbx_dtype=dtype)


def _tile(row):
    return {f: int(row[f]) for f in FIELDS}


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    forget = getattr(CERT, "reset_refused_unit_entries", lambda: None)
    forget()
    monkeypatch.setattr(CERT, "certifying", lambda on=None: False)
    from nbxtensor.ops.launch_observer import launch_observer
    monkeypatch.setattr(launch_observer(), "active", lambda: False, raising=False)
    yield
    forget()


def _route(monkeypatch, reading, a=NBXDtype.float16, b=NBXDtype.float16, M=256, N=512, tuner=MM.matmul_kernel):
    """`_gemm_route`'s tile for an M x N GEMM of these operands when the certificate reads as `reading`."""
    monkeypatch.setattr(CERT, "unit_choice", lambda *_a, **_k: reading)
    unit_tile = LW._gemm_unit_tile(_operand(a), _operand(b), M, N)
    return unit_tile, LW._gemm_route(tuner, (), {}, unit_tile)


@pytest.mark.parametrize("dtype,rows_of", [(NBXDtype.float16, "mm"), (NBXDtype.float32, "fp32_split")])
def test_the_unit_runs_with_no_certificate_with_one_naming_the_fma_kernel_and_with_one_naming_a_tile(monkeypatch, dtype, rows_of):
    cfg = _profile(monkeypatch, "nvidia/volta.yml")
    rows = cfg["matrix_unit"]["mm"] if rows_of == "mm" else cfg["matrix_unit"]["fp32_split"]["mm"]
    for reading in (None, ("fma", None), *[("unit", _tile(r)) for r in rows]):
        unit_tile, tile = _route(monkeypatch, reading, dtype, dtype)
        assert unit_tile is not None, f"{dtype} {reading}: the rule gave the call to the FMA kernel"
        assert tile is not None, f"{dtype}: a certificate reading {reading} moved a unit call to the FMA kernel"
        assert _tile(tile) in [_tile(r) for r in rows]


def test_a_certificate_naming_the_fma_kernel_is_refused_by_name_and_the_rules_tile_runs(monkeypatch, caplog):
    cfg = _profile(monkeypatch, "nvidia/volta.yml")
    unit_tile, tile = _route(monkeypatch, ("fma", None))
    assert _tile(tile) == _tile(unit_tile)
    refused = CERT.refused_unit_entries()
    assert len(refused) == 1
    (kernel, _key), why = next(iter(refused.items()))
    assert "matmul_kernel" in kernel and "FMA kernel" in why
    # said once per entry, not once per launch
    caplog.clear()
    with caplog.at_level("WARNING"):
        _route(monkeypatch, ("fma", None))
    assert not [r for r in caplog.records if "REFUSED" in r.getMessage()]


def test_a_certificate_naming_a_tile_the_profile_does_not_declare_is_refused_by_name(monkeypatch):
    _profile(monkeypatch, "nvidia/volta.yml")
    bogus = {f: 48 for f in FIELDS}
    unit_tile, tile = _route(monkeypatch, ("unit", bogus))
    assert _tile(tile) == _tile(unit_tile)
    assert "does not declare" in next(iter(CERT.refused_unit_entries().values()))


def test_the_certificates_tile_is_taken_when_it_is_one_the_profile_declares(monkeypatch):
    cfg = _profile(monkeypatch, "nvidia/volta.yml")
    last = _tile(cfg["matrix_unit"]["mm"][-1])
    _unit, tile = _route(monkeypatch, ("unit", last))
    assert _tile(tile) == last and not CERT.refused_unit_entries()


def test_the_real_reading_of_an_entry_that_carries_no_unit_is_refused_by_name(monkeypatch):
    """The whole chain, `unit_choice` included: the directory's entry stubbed, the reading and the rule are the library's."""
    _profile(monkeypatch, "nvidia/volta.yml")
    monkeypatch.setattr(CERT, "lookup", lambda *a, **k: {"config": {"kwargs": {}}})       # an FMA-chosen entry
    monkeypatch.setattr(CERT, "executing_memory_gb", lambda *a, **k: 16)
    CERT._UNIT_CHOICES.clear()
    unit_tile = LW._gemm_unit_tile(_operand(NBXDtype.float16), _operand(NBXDtype.float16), 256, 512)
    tile = LW._gemm_route(MM.matmul_kernel, (), {}, unit_tile)
    assert tile is not None and CERT.refused_unit_entries()
    CERT._UNIT_CHOICES.clear()


def test_a_gemm_whose_rows_do_not_fill_the_unit_runs_the_fma_kernel_whatever_the_certificate_says(monkeypatch):
    cfg = _profile(monkeypatch, "nvidia/volta.yml")
    smallest = min(int(r["block_m"]) for r in cfg["matrix_unit"]["mm"])
    for reading in (None, ("fma", None), ("unit", _tile(cfg["matrix_unit"]["mm"][0]))):
        unit_tile, tile = _route(monkeypatch, reading, M=smallest - 1)
        assert unit_tile is None and tile is None
    assert not CERT.refused_unit_entries()


@pytest.mark.parametrize("profile", sorted(str(p.relative_to(DEVICES)) for p in (DEVICES / "apple").glob("*.yml")))
def test_a_card_with_no_matrix_unit_runs_one_path_the_fma_kernel_whatever_the_certificate_says(monkeypatch, profile):
    _profile(monkeypatch, profile)
    for dtype in (NBXDtype.float16, NBXDtype.float32):
        for reading in (None, ("fma", None), ("unit", {f: 64 for f in FIELDS})):
            unit_tile, tile = _route(monkeypatch, reading, dtype, dtype)
            assert unit_tile is None and tile is None


def _moe(monkeypatch, reading, dtype=NBXDtype.float16):
    monkeypatch.setattr(MW, "_fused_moe_launch", lambda *a, **k: (None, (), {}))
    monkeypatch.setattr(CERT, "unit_choice", lambda *_a, **_k: reading)
    return MW.fused_moe_unit_tile(_operand(dtype), *([None] * 11))


def test_a_fused_moe_runs_the_unit_with_no_certificate_as_a_gemm_does_and_refuses_an_entry_naming_the_fma_kernel(monkeypatch):
    cfg = _profile(monkeypatch, "nvidia/volta.yml")
    rows = [_tile(r) for r in cfg["matrix_unit"]["mm"]]
    assert _moe(monkeypatch, None) == rows[0]                  # before: None, the FMA kernel
    assert not CERT.refused_unit_entries()
    assert _moe(monkeypatch, ("fma", None)) == rows[0]
    assert CERT.refused_unit_entries()
    assert _moe(monkeypatch, ("unit", rows[-1])) == rows[-1]


def test_a_fused_moe_on_experts_the_unit_does_not_carry_or_on_a_card_without_one_runs_the_fma_kernel(monkeypatch):
    _profile(monkeypatch, "nvidia/volta.yml")
    assert _moe(monkeypatch, ("unit", {f: 64 for f in FIELDS}), NBXDtype.float32) is None
    _profile(monkeypatch, "apple/apple_m4_pro.yml")
    assert _moe(monkeypatch, None) is None


def test_the_candidates_a_certification_measures_are_only_the_ruled_units_tiles(monkeypatch):
    cfg = _profile(monkeypatch, "nvidia/volta.yml")
    declared = {tuple(_tile(r)[f] for f in FIELDS) for r in cfg["matrix_unit"]["mm"]}
    monkeypatch.setattr(MW, "_fused_moe_launch", lambda *a, **k: (None, (), {}))
    got = {tuple(t[f] for f in FIELDS) for t in MW.fused_moe_unit_tiles(_operand(NBXDtype.float16))}
    assert got == declared
    assert MW.fused_moe_unit_tiles(_operand(NBXDtype.float32)) == []
    _profile(monkeypatch, "apple/apple_m4_pro.yml")
    assert MW.fused_moe_unit_tiles(_operand(NBXDtype.float16)) == []
