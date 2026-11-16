"""`io.safetensors.fp16_range_scan`: whether the floating weights of safetensors files fit float16, scanned from the
stored bytes shard by shard (no fp32 copy of the model); FP16_MAX is the dtype table's, exported once."""
import json
import struct
import tracemalloc

import numpy as np

import nbxtensor.dtype.itemsize as itemsize
from nbxtensor.dtype.convert import f32_to_bf16_bits
from nbxtensor.io.safetensors import FP16_MAX, fp16_range_scan


def _write(path, tensors):
    header, blobs, at = {}, [], 0
    for key, dtype, shape, data in tensors:
        header[key] = {"dtype": dtype, "shape": list(shape), "data_offsets": [at, at + len(data)]}
        blobs.append(data)
        at += len(data)
    h = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))
    return path


def _bf16(a):
    a = np.asarray(a, np.float32)
    return "BF16", a.shape, f32_to_bf16_bits(a).tobytes()


def test_in_range_bf16_and_fp16_shards_fit(tmp_path):
    k, s, d = _bf16([1.0, -65000.0, np.nan])
    a = _write(tmp_path / "a.safetensors", [("w", k, s, d)])
    b = _write(tmp_path / "b.safetensors", [("h", "F16", (2,), np.array([6e4, -6e4], np.float16).tobytes())])
    r = fp16_range_scan([a, b])
    assert r.fits and r.scanned == 5 and r.key is None


def test_a_value_above_or_below_the_range_is_named_with_its_tensor(tmp_path):
    k, s, d = _bf16([1.0])
    ok = _write(tmp_path / "a.safetensors", [("ok", k, s, d)])
    k, s, d = _bf16([2.0, -7e4])
    bad = _write(tmp_path / "b.safetensors", [("ok2", "F32", (1,), np.float32([1]).tobytes()), ("neg", k, s, d)])
    r = fp16_range_scan([ok, bad])
    assert not r.fits and r.key == "neg" and r.path == str(bad) and r.abs_max > FP16_MAX
    k, s, d = _bf16([7e4])
    up = _write(tmp_path / "c.safetensors", [("pos", k, s, d)])
    assert fp16_range_scan([up]).key == "pos"


def test_an_infinity_is_outside_the_range_and_a_file_without_floats_scans_zero(tmp_path):
    inf = _write(tmp_path / "i.safetensors", [("x", "F32", (1,), np.float32([np.inf]).tobytes())])
    assert not fp16_range_scan([inf]).fits
    ints = _write(tmp_path / "n.safetensors", [("i", "I64", (1,), np.int64([10**12]).tobytes())])
    r = fp16_range_scan([ints])
    assert r.fits and r.scanned == 0


def test_the_scan_never_widens_the_model_peak_memory_is_a_few_steps(tmp_path):
    n = 8 << 20                                    # 16 MB of bf16 (a 32 MB fp32 copy would show)
    p = _write(tmp_path / "big.safetensors", [("w", "BF16", (n,), np.zeros(n, np.uint16).tobytes())])
    tracemalloc.start()
    r = fp16_range_scan([p])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert r.fits and r.scanned == n and peak < 8 << 20


def test_the_constant_is_the_dtype_tables_and_exported_once():
    import nbxtensor.io.safetensors as st
    assert FP16_MAX == itemsize.finite_max("float16") == 65504.0
    assert st.FP16_MAX is itemsize.FP16_MAX
