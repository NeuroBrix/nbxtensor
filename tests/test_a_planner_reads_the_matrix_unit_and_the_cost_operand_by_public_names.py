"""A planner (NeuroBrix's Prism) reads the device's matrix unit and describes a cost's operands through PUBLIC names:
`nbxtensor.ops.matrix_units.matrix_unit` and `nbxtensor.ops.operand.Spec`, the very objects the library itself uses.

What this test would catch if the code were wrong: a public name that is a copy (two functions, two classes, drifting
apart), a library caller still bound to a private second definition, a test's replacement of the one decision point
that the tile choices or the launch keys do not see, or a `Spec` the cost declarations do not accept as their own —
RED."""
import importlib

import pytest

from nbxtensor.ops import matrix_units
from nbxtensor.ops.operand import Spec

WRAPPERS = ("wrappers", "attention_wrappers", "conv_wrappers", "elementwise_wrappers", "index_wrappers",
            "linalg_wrappers", "loss_wrappers", "moe_wrappers", "norm_wrappers", "random_wrappers",
            "reduction_wrappers", "shape_wrappers", "spectral_wrappers")


def test_the_public_names_are_the_objects_the_library_uses():
    from nbxtensor.ops import _cost
    assert _cost.Spec is Spec
    assert "matrix_unit" not in vars(importlib.import_module("nbxtensor.ops._configs")), \
        "a second matrix_unit in ops._configs: one job, one function"
    for name in WRAPPERS:
        assert importlib.import_module(f"nbxtensor.ops.{name}")._matrix_unit is matrix_units.matrix_unit, name


def test_the_tile_choices_and_the_launch_keys_read_the_one_decision_point(monkeypatch):
    from nbxtensor.ops import _configs, launch_keys
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    unit = {"shape": "m8n8k4", "operand_dtype": "float16", "accumulator_dtype": "float32",
            "mm": [{"block_m": 64, "block_n": 64, "block_k": 16, "warps_m": 2, "warps_n": 2, "group_m": 8}],
            "flash": [{"block_m": 64, "block_n": 32, "warps_m": 4, "head_dim_le": 128}]}
    monkeypatch.setattr(matrix_units, "matrix_unit", lambda vendor_profile=None: unit)
    assert _configs.matrix_unit_mm_tile(128, 128)["block_m"] == 64
    assert _configs.matrix_unit_flash_tile(64)["block_n"] == 32
    assert launch_keys.matrix_unit_operands(NBXDtype.float16, NBXDtype.float16, 128) is not None
    monkeypatch.setattr(matrix_units, "matrix_unit", lambda vendor_profile=None: {})
    assert _configs.matrix_unit_mm_tile(128, 128) is None and _configs.matrix_unit_flash_tile(64) is None
    assert launch_keys.matrix_unit_operands(NBXDtype.float16, NBXDtype.float16, 128) is None


def test_a_profile_handed_in_is_read_and_checked_by_the_public_name():
    assert matrix_units.matrix_unit({}) == {}
    with pytest.raises(ValueError, match="has no kernel"):
        matrix_units.matrix_unit({"matrix_unit": {"shape": "m16n8k16"}})


def test_the_public_operand_prices_a_cost_declaration():
    from nbxtensor.ops._cost import elementwise_unary
    x = Spec.contiguous((4, 8), "float16")
    assert x.strides == (8, 1) and x.nbytes == 64
    assert elementwise_unary(x).outputs[0].nbytes == 64
