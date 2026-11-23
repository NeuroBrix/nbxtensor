"""A one-row conv key holds its W bucket's top, where the matrix unit may take the conv and the key's FMA kernel never
runs; the harness draws it at the widest width of the bucket whose launch forms the key (`conv_width_representative`,
as `gemm_row_representative` does for M). qa 1244: the bucket 65..80 keys in_w 80, M = 74 goes to the unit, in_w 65..69
(M 59..63) launches `conv2d_forward_kernel` under the same key — 8 certifications never proven there.

Injection, seen red: `conv_widths` returning the key's own width (the top)."""
from __future__ import annotations

from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.ops import autotune_harness as H
from nbxtensor.ops import launch_keys as LK


def _volta(monkeypatch):
    import pathlib
    import yaml
    import nbxtensor
    from nbxtensor.ops import _configs
    path = pathlib.Path(nbxtensor.__file__).parent / "backend" / "devices" / "nvidia" / "volta.yml"
    profile = yaml.safe_load(path.read_text())
    monkeypatch.setattr(_configs, "active_vendor_profile", lambda: profile)


def _key(in_w, F16=NBXDtype.float16):
    (launch,) = LK.conv2d_launches(1, 768, 1, in_w, 768, 1, 7, 1, 1, 0, 0, 1, 1, 1, F16, F16, None, 1 << 40)
    return launch


def test_a_one_row_conv_key_is_drawn_at_a_width_whose_launch_forms_it(monkeypatch):
    _volta(monkeypatch)
    top = int(LK.bucket_of("W", 69))
    assert LK.conv2d_launches(1, 768, 1, top, 768, 1, 7, 1, 1, 0, 0, 1, 1, 1,
                              NBXDtype.float16, NBXDtype.float16, None, 1 << 40) == []     # the unit takes the top
    name, key = _key(69)                                                                    # the FMA kernel's key
    iw, ow = H.conv_widths(key, "fp16", "fp16")
    assert iw < top and int(LK.bucket_of("W", iw)) == top
    assert _key(iw) == (name, key)                                                          # drawn where it is formed
    assert ow == iw - 6


def test_a_two_d_conv_key_keeps_its_own_extents():
    key = (1, 64, 32, 32, 64, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, True, "fp16", "fp16", "fp16")
    assert H.conv_widths(key, "fp16", "fp16") == (32, 32)
