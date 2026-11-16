"""`io.safetensors.stack_shards` / `concat_shards`: ONE tensor built from several stored tensors, bit-equal to the
in-memory join, its peak memory the result plus one source, every disagreement refused by name."""
import json
import struct
import tracemalloc

import numpy as np
import pytest

from nbxtensor.dtype.convert import f32_to_bf16_bits
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.io.safetensors import (ShardConcat, ShardRef, ShardsRefused, ShardStack, concat_shards, stack_shards)

CODES = {"F16": (np.float16, NBXDtype.float16), "F32": (np.float32, NBXDtype.float32),
         "BF16": (np.uint16, NBXDtype.bfloat16)}


def _write(path, tensors):
    header, blobs, at = {}, [], 0
    for name, code, arr in tensors:
        data = arr.tobytes()
        header[name] = {"dtype": code, "shape": list(arr.shape), "data_offsets": [at, at + len(data)]}
        blobs.append(data)
        at += len(data)
    h = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))
    return str(path)


def _draw(code, shape, seed):
    a = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    return f32_to_bf16_bits(a) if code == "BF16" else a.astype(CODES[code][0])


def _bits(t):
    return np.ctypeslib.as_array((__import__("ctypes").c_uint8 * t.nbytes()).from_address(t.data_ptr())).copy()


def _set(tmp_path, code, n, shape):
    arrs = [_draw(code, shape, i) for i in range(n)]
    path = _write(tmp_path / "s.safetensors", [(f"t{i}", code, a) for i, a in enumerate(arrs)])
    return path, arrs


@pytest.mark.parametrize("code", ["F16", "BF16", "F32"])
def test_a_stack_and_a_concat_equal_the_in_memory_join_byte_for_byte(tmp_path, code):
    path, arrs = _set(tmp_path, code, 4, (3, 5))
    refs = [ShardRef(path, f"t{i}") for i in range(4)]
    for axis in (0, 1, 2, -1):
        got = stack_shards(refs, axis)
        assert got.nbx_dtype == CODES[code][1] and tuple(got.shape) == np.stack(arrs, axis).shape
        assert _bits(got).tobytes() == np.ascontiguousarray(np.stack(arrs, axis)).tobytes()
    for axis in (0, 1, -1):
        got = concat_shards(refs, axis)
        assert _bits(got).tobytes() == np.ascontiguousarray(np.concatenate(arrs, axis)).tobytes()


def test_the_order_of_the_sources_is_the_order_of_the_result(tmp_path):
    path, arrs = _set(tmp_path, "F32", 3, (2, 2))
    got = stack_shards([ShardRef(path, "t2"), ShardRef(path, "t0"), ShardRef(path, "t1")], 0)
    assert _bits(got).tobytes() == np.stack([arrs[2], arrs[0], arrs[1]]).tobytes()


def test_a_set_of_sixty_four_sources_is_stacked(tmp_path):
    path, arrs = _set(tmp_path, "BF16", 64, (4, 6))
    got = stack_shards([ShardRef(path, f"t{i}") for i in range(64)], 0)
    assert tuple(got.shape) == (64, 4, 6) and _bits(got).tobytes() == np.stack(arrs).tobytes()


def test_a_concat_per_expert_then_a_stack_is_one_nested_declaration(tmp_path):
    a = [_draw("F16", (3, 4), i) for i in range(8)]
    b = [_draw("F16", (5, 4), 100 + i) for i in range(8)]
    path = _write(tmp_path / "e.safetensors", [(f"a{i}", "F16", a[i]) for i in range(8)]
                  + [(f"b{i}", "F16", b[i]) for i in range(8)])
    decl = [ShardConcat((ShardRef(path, f"a{i}"), ShardRef(path, f"b{i}")), 0) for i in range(8)]
    got = stack_shards(decl, 0)
    want = np.stack([np.concatenate([a[i], b[i]], 0) for i in range(8)])
    assert tuple(got.shape) == (8, 8, 4) and _bits(got).tobytes() == want.tobytes()


def test_the_peak_memory_is_the_result_plus_one_source_not_the_sum_of_the_sources(tmp_path):
    n, shape = 32, (256, 256)                         # a 256 KiB source, an 8 MiB result
    path, _ = _set(tmp_path, "F32", n, shape)
    refs = [ShardRef(path, f"t{i}") for i in range(n)]
    one, result = 256 * 256 * 4, n * 256 * 256 * 4
    tracemalloc.start()
    got = stack_shards(refs, 0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert got.nbytes() == result
    assert peak <= result + one + (1 << 20), f"peak {peak} bytes over result {result} + one source {one}"


def test_sources_that_disagree_are_refused_naming_both(tmp_path):
    p = _write(tmp_path / "m.safetensors", [("a", "F16", _draw("F16", (2, 3), 0)), ("b", "F32", _draw("F32", (2, 3), 1)),
                                            ("c", "F16", _draw("F16", (2, 4), 2))])
    with pytest.raises(ShardsRefused, match=r"m.safetensors:a.*m.safetensors:b.*dtype"):
        stack_shards([ShardRef(p, "a"), ShardRef(p, "b")], 0)
    with pytest.raises(ShardsRefused, match=r"m.safetensors:a.*m.safetensors:c"):
        stack_shards([ShardRef(p, "a"), ShardRef(p, "c")], 0)
    with pytest.raises(ShardsRefused, match="m.safetensors:a.*m.safetensors:c"):
        concat_shards([ShardRef(p, "a"), ShardRef(p, "c")], 0)
    assert tuple(concat_shards([ShardRef(p, "a"), ShardRef(p, "c")], 1).shape) == (2, 7)


def test_a_missing_tensor_a_bad_axis_and_a_conversion_are_refused_by_name(tmp_path):
    p = _write(tmp_path / "m.safetensors", [("a", "F16", _draw("F16", (2, 3), 0))])
    with pytest.raises(ShardsRefused, match="holds no tensor named 'zz'"):
        stack_shards([ShardRef(p, "zz")], 0)
    with pytest.raises(ShardsRefused, match="axis 5"):
        concat_shards([ShardRef(p, "a")], 5)
    with pytest.raises(ShardsRefused, match="never converts"):
        stack_shards([ShardRef(p, "a")], 0, out_dtype=NBXDtype.float32)
    with pytest.raises(ShardsRefused, match="non-empty"):
        stack_shards([], 0)
