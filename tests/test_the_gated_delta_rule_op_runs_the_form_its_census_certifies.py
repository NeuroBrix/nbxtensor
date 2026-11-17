"""`gated_delta_rule_op` (dispatch name "gated_delta_rule") runs the form `gated_delta_rule_path` answers from the census
beside the kernel (`gated_delta_rule_census.json`), on any host (no launch: the two wrappers are stood in for by
recorders). One token runs the serial scan whatever the cell; a longer call runs the serial scan while its cell's
`chunked_from_length` is null (the shipped census: no cell is certified until a card measures it), and the chunked form
from the cell's start length once a cell is filled; dimensions in no cell are refused by name, and a census that does
not hold (its format, a missing dimension, a start length of one token) is refused by name. The launch keys follow the
same answer: one serial launch, or the chunked form's three.

What it catches: a token step sent to the chunked form, a null cell read as certified, a start length read as "longer
than" instead of "from", a cell matched on fewer dimensions than it names, an uncensused call run silently, a census
of the wrong format read, the launch keys of one form announced while the other runs - red."""
import json

import pytest

from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.ops import gated_delta_rule as G
from nbxtensor.ops import gated_delta_rule_chunked as CH
from nbxtensor.ops import launch_keys as LK
from nbxtensor.tensor import NBXTensor

F16, BF16, F32 = NBXDtype.float16, NBXDtype.bfloat16, NBXDtype.float32
ATTRS = dict(qk_l2norm=True, eps=1e-6, scale=128 ** -0.5, compute_dtype=F32)
CELL = dict(B=1, Hk=16, Hv=48, dk=128, dv=128)          # the dimensions of a shipped cell


@pytest.fixture
def census(tmp_path, monkeypatch):
    """A census file of the test's own, read in place of the shipped one (the reader's cache emptied before and
    after)."""
    path = tmp_path / "gated_delta_rule_census.json"

    def write(cells, fmt=G._GATED_DELTA_RULE_CENSUS_FORMAT):
        path.write_text(json.dumps({"format": fmt, "cells": cells}))
        G.gated_delta_rule_census.cache_clear()

    monkeypatch.setattr(G, "_GATED_DELTA_RULE_CENSUS", path)
    G.gated_delta_rule_census.cache_clear()
    yield write
    G.gated_delta_rule_census.cache_clear()


def _cell(start, dtype="float32", **dims):
    return dict(CELL, dtype=dtype, chunked_from_length=start, proof=None, **dims)


def test_the_shipped_census_holds_and_certifies_no_cell_yet():
    G.gated_delta_rule_census.cache_clear()
    doc = G._gated_delta_rule_census()
    assert doc["format"] == G._GATED_DELTA_RULE_CENSUS_FORMAT
    dims = {(c["B"], c["Hk"], c["Hv"], c["dk"], c["dv"], c["dtype"]) for c in doc["cells"]}
    assert dims == {(b, 16, 48, 128, 128, dt) for b in (1, 4) for dt in ("float32", "float16", "bfloat16")}
    assert all(c["chunked_from_length"] is None and c["proof"] is None for c in doc["cells"])
    for b in (1, 4):
        for dt in (F32, F16, BF16):
            for L in (1, 2, 512, 16384):
                assert G.gated_delta_rule_path(b, L, 16, 48, 128, 128, dt) == "serial"


def test_one_token_runs_the_serial_scan_whatever_the_cell(census):
    census([_cell(2)])
    assert G.gated_delta_rule_path(1, 1, 16, 48, 128, 128, F32) == "serial"
    assert G.gated_delta_rule_path(1, 2, 16, 48, 128, 128, F32) == "chunked"
    assert G.gated_delta_rule_path(1, 1, 3, 5, 7, 9, F32) == "serial"        # one token needs no cell


def test_a_null_cell_runs_the_serial_scan_at_every_length(census):
    census([_cell(None)])
    assert {G.gated_delta_rule_path(1, L, 16, 48, 128, 128, F32) for L in (2, 64, 65, 16384)} == {"serial"}


def test_a_filled_cell_runs_the_chunked_form_from_its_start_length(census):
    census([_cell(1024), _cell(None, dtype="float16")])
    assert G.gated_delta_rule_path(1, 1023, 16, 48, 128, 128, F32) == "serial"
    assert G.gated_delta_rule_path(1, 1024, 16, 48, 128, 128, F32) == "chunked"
    assert G.gated_delta_rule_path(1, 16384, 16, 48, 128, 128, F32) == "chunked"
    assert G.gated_delta_rule_path(1, 16384, 16, 48, 128, 128, F16) == "serial"     # its own cell, null


@pytest.mark.parametrize("dims", [dict(B=2), dict(Hk=8), dict(Hv=32), dict(dk=64), dict(dv=64), dict(dtype=BF16)],
                         ids=["B", "Hk", "Hv", "dk", "dv", "dtype"])
def test_a_call_in_no_cell_is_refused_by_name(census, dims):
    census([_cell(2)])
    call = dict(CELL, dtype=F32, L=64)
    call.update(dims)
    with pytest.raises(ValueError, match="no census cell for .*measure_the_chunked_gated_delta_rule"):
        G.gated_delta_rule_path(call["B"], call["L"], call["Hk"], call["Hv"], call["dk"], call["dv"], call["dtype"])


@pytest.mark.parametrize("doc,match", [
    (dict(fmt="nbx-gated-delta-rule-census/0", cells=[]), "has format"),
    (dict(cells=[{k: v for k, v in _cell(None).items() if k != "dv"}]), r"malformed: missing \['dv'\]"),
    (dict(cells=[{k: v for k, v in _cell(None).items() if k != "chunked_from_length"}]), "malformed: missing"),
    (dict(cells=[_cell(1)]), "malformed.*one token always runs the serial scan"),
    (dict(cells=[_cell("512")]), "malformed"),
    (dict(cells=[_cell(512.0)]), "malformed"),
], ids=["format", "missing-dimension", "missing-start", "start-one-token", "start-text", "start-float"])
def test_a_census_that_does_not_hold_is_refused_by_name(census, doc, match):
    census(doc["cells"], **({"fmt": doc["fmt"]} if "fmt" in doc else {}))
    with pytest.raises(RuntimeError, match=match):
        G.gated_delta_rule_path(1, 64, 16, 48, 128, 128, F32)


def test_an_unreadable_census_is_refused_by_name(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "_GATED_DELTA_RULE_CENSUS", tmp_path / "absent.json")
    G.gated_delta_rule_census.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="cannot be read.*reinstall the library"):
            G.gated_delta_rule_path(1, 64, 16, 48, 128, 128, F32)
    finally:
        G.gated_delta_rule_census.cache_clear()


def _ops(B, L, dt=F32, Hk=16, Hv=48, dk=128, dv=128):
    t = lambda shape, d=dt: NBXTensor.empty_cpu(tuple(shape), dtype=d)
    return (t((B, L, Hk, dk)), t((B, L, Hk, dk)), t((B, L, Hv, dv)), t((B, L, Hv), F32), t((B, L, Hv)),
            t((B, Hv, dk, dv), F32))


@pytest.mark.parametrize("L,want", [(1, "serial"), (1023, "serial"), (1024, "chunked"), (4096, "chunked")])
def test_the_op_runs_the_form_the_path_answers_and_hands_the_state_through(census, monkeypatch, L, want):
    census([_cell(1024)])
    ran = []
    monkeypatch.setattr(G, "gated_delta_rule_wrapper", lambda *a, **kw: ran.append(("serial", a, kw)) or "s")
    monkeypatch.setattr(CH, "gated_delta_rule_chunked_wrapper", lambda *a, **kw: ran.append(("chunked", a, kw)) or "c")
    ops = _ops(1, L)
    assert G.gated_delta_rule_op(*ops, **ATTRS) == want[0]
    [(form, args, kw)] = ran
    assert form == want and all(a is b for a, b in zip(args, ops)) and kw == ATTRS


def test_the_op_refuses_an_uncensused_call_before_any_form_runs(census, monkeypatch):
    census([_cell(1024)])
    ran = []
    monkeypatch.setattr(G, "gated_delta_rule_wrapper", lambda *a, **kw: ran.append("serial"))
    monkeypatch.setattr(CH, "gated_delta_rule_chunked_wrapper", lambda *a, **kw: ran.append("chunked"))
    with pytest.raises(ValueError, match="no census cell"):
        G.gated_delta_rule_op(*_ops(2, 64), **ATTRS)
    assert ran == []


@pytest.mark.parametrize("L,want", [(1, "serial"), (1023, "serial"), (1024, "chunked")])
def test_the_launch_keys_are_those_of_the_form_that_runs(census, L, want):
    census([_cell(1024, dtype="float16")])
    keys = LK.gated_delta_rule_op_launches(1, L, 16, 48, 128, 128, F16, F16, F16, F32, F16)
    serial = LK.gated_delta_rule_launches(128, 128, F16, F16, F16, F32, F16)
    chunked = LK.gated_delta_rule_chunked_launches(128, 128, F16, F16, F16, F32, F16)
    assert keys == (serial if want == "serial" else chunked)
    assert [name for name, _ in chunked] == list(LK.GATED_DELTA_RULE_CHUNKED)
    assert all(key[:2] == (128, 128) for _, key in chunked)        # the length is never a key
    assert chunked == LK.gated_delta_rule_chunked_launches(128, 128, F16, F16, F16, F32, F16)
    with pytest.raises(ValueError, match="no census cell"):
        LK.gated_delta_rule_op_launches(4, L if L > 1 else 2, 16, 48, 128, 128, F16, F16, F16, F32, F16)


def test_the_chunked_kernels_are_named_where_the_launch_keys_name_them():
    for qualified in LK.GATED_DELTA_RULE_CHUNKED:
        module, name = qualified.rsplit(".", 1)
        assert module == CH.__name__ and hasattr(CH, name)
