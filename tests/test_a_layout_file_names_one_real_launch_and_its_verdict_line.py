"""`tools/prove_an_element_wise_variant.py --layout`, card-less: the Flex layout file parses and writes out the launch
the library's add wrapper makes (each operand's own strides in its stride array, every extent holding what its shape
and strides address, exactly), it specialises as the register's variant 6859212e3972bf7a, and the verdict line."""
import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FLEX = ROOT / "tools" / "layouts" / "flex_t5_position_bias_add.json"
CANDS = [(b, w, 3) for b in (256, 512, 1024, 2048, 4096) for w in (2, 4, 8)]


def _tool():
    """The tool loaded as a module (it is no package)."""
    spec = importlib.util.spec_from_file_location("prove_an_element_wise_variant",
                                                  ROOT / "tools" / "prove_an_element_wise_variant.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_flex_layout_parses_as_the_add_wrapper_launches_it():
    """The Flex file parses; its stride arrays are each operand's own strides; each extent is what its view reaches."""
    lay = _tool().read_layout(str(FLEX))
    p = lay["pointers"]
    assert lay["kernel"] == {"module": "nbxtensor.ops.binary_strided", "name": "add_strided_nd_kernel"}
    assert lay["count"] == {"name": "n_elements", "value": 1 << 24} and lay["scalars"] == {"alpha": 1.0}
    assert p["shape_ptr"]["values"] == [1, 64, 512, 512] == p["x_ptr"]["shape"] == p["y_ptr"]["shape"]
    assert p["x_stride_ptr"]["values"] == p["x_ptr"]["strides"] == [16777216, 262144, 512, 1]
    assert p["y_stride_ptr"]["values"] == p["y_ptr"]["strides"] == [16777216, 1, 32768, 64]
    for name in ("x_ptr", "y_ptr", "output_ptr"):          # each storage extent is exactly what the view reaches
        reach = 1 + sum((d - 1) * s for d, s in zip(p[name]["shape"], p[name]["strides"]))
        assert p[name]["extent"] == reach == 1 << 24 and p[name]["dtype"] == "*fp16"


def test_a_layout_whose_extent_is_short_of_its_strides_is_refused(tmp_path):
    """An extent one element short of what the strides address is refused, naming the pointer."""
    lay = json.loads(FLEX.read_text())
    lay["pointers"]["y_ptr"]["extent"] = (1 << 24) - 1
    bad = tmp_path / "short.json"
    bad.write_text(json.dumps(lay))
    with pytest.raises(ValueError, match="y_ptr"):
        _tool().read_layout(str(bad))


def test_the_flex_launch_specialises_as_the_registers_variant():
    """The launch the file writes out binds as variant 6859212e3972bf7a, three 2^24 buffers and three arrays of 4."""
    from nbxtensor.ops import ew_proof
    mod = _tool()
    kernel, spec, options, launch = mod.layout_launch(mod.read_layout(str(FLEX)), CANDS)
    tile = next(v for _t, v in spec if ew_proof._is_tile(v))
    vid, _tile = ew_proof.variant_of(kernel, spec, dict(options, num_warps=2, num_stages=3), {"tile": tile})
    assert ew_proof.stamp_key({"v": vid}) == "6859212e3972bf7a"
    assert [b.elements for b in launch.bufs()] == [1 << 24] * 3 + [4] * 3 and launch.count == 1 << 24


def test_the_layout_verdict_line():
    """The verdict line: faulting tiles named, CLEAN only when all 15 pass, else COULD NOT RUN."""
    v = _tool().layout_verdict
    assert v({c: "pass" for c in CANDS}, 15) == "LAYOUT CLEAN 15/15"
    two = {c: ("device fault" if c in ((4096, 2, 3), (2048, 8, 3)) else "pass") for c in CANDS}
    assert v(two, 15) == "LAYOUT FAULTS AT (2048, 8, 3), (4096, 2, 3)"
    mixed = {**two, (256, 2, 3): "could not run"}
    assert v(mixed, 15) == "LAYOUT FAULTS AT (2048, 8, 3), (4096, 2, 3) (1 could not run)"
    assert v({**{c: "pass" for c in CANDS}, (256, 2, 3): "could not run"}, 15) == "COULD NOT RUN"
    assert v({}, 15) == "COULD NOT RUN" and v({c: "pass" for c in CANDS[:14]}, 15) == "COULD NOT RUN"
