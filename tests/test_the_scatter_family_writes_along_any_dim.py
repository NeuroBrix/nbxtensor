"""scatter, scatter_add, scatter_reduce (sum, amax, amin) and index_add write along ANY dim, not only dim 0 or 1: the
result equals a numpy reference built element by element, on the layouts the shipped graphs reach (dim 2 of
[T, heads, 128] for the MoE models, dim 1 of [230, 128] for Qwen3-VL) and on small 2-D, 3-D and 4-D tensors along
every dim. The four wrappers once passed stride(0) as the stride of the collapsed outer dims: right with one outer
dim, wrong values and reads and writes past `src` and `out` with two or more (dim 2 of a 3-D tensor)."""
import zlib

import numpy as np
import pytest

from nbxtensor.tensor import NBXTensor

#: (shape, dim): the shipped layouts first, then every dim of small 2-D, 3-D and 4-D tensors
LAYOUTS = [((23, 8, 128), 2), ((230, 8, 128), 2), ((230, 6, 128), 2), ((230, 128), 1),
           ((7, 9), 0), ((7, 9), 1),
           ((4, 5, 6), 0), ((4, 5, 6), 1), ((4, 5, 6), 2),
           ((3, 4, 5, 6), 1), ((3, 4, 5, 6), 2), ((3, 4, 5, 6), 3)]
IDS = [f"{'x'.join(map(str, s))}-dim{d}" for s, d in LAYOUTS]


def _accelerator():
    """Skip only on a machine with no accelerator at all; any other error fails the test."""
    from nbxtensor.memory.allocator import DeviceAllocator
    if DeviceAllocator.device_count() == 0:
        pytest.skip("no accelerator on this machine")


def _coords(index, dim):
    """The full coordinate tuple of every element of `index` with the coordinate along `dim` replaced by its value."""
    c = list(np.indices(index.shape))
    c[dim] = index
    return tuple(c)


def _unique_index(rng, shape, dim):
    """An int64 index of `shape` holding a permutation of 0 .. shape[dim] - 1 along `dim` on every line: no two
    elements write one place, so a plain scatter has one answer."""
    lines = rng.random(shape).argsort(axis=dim)
    return lines.astype(np.int64)


def _nbx(a, transposed):
    """The array on the device: contiguous, or (transposed) a non-contiguous view of a transposed buffer."""
    if not transposed or a.ndim < 2:
        return NBXTensor.from_numpy(np.ascontiguousarray(a))
    t = NBXTensor.from_numpy(np.ascontiguousarray(np.swapaxes(a, 0, -1)))
    v = t.transpose(0, a.ndim - 1)
    assert tuple(v.shape) == a.shape and not v.is_contiguous()
    return v


@pytest.mark.parametrize("transposed", [False, True], ids=["contiguous", "transposed-src-index"])
@pytest.mark.parametrize("shape,dim", LAYOUTS, ids=IDS)
def test_scatter_writes_every_element_where_its_index_says(shape, dim, transposed):
    _accelerator()
    from nbxtensor.ops.index_wrappers import scatter_wrapper
    rng = np.random.default_rng(zlib.crc32(repr((shape, dim)).encode()))
    x = rng.standard_normal(shape).astype(np.float32)
    src = rng.standard_normal(shape).astype(np.float32)
    index = _unique_index(rng, shape, dim)
    ref = x.copy()
    ref[_coords(index, dim)] = src
    got = scatter_wrapper(NBXTensor.from_numpy(x), dim, _nbx(index, transposed), _nbx(src, transposed)).numpy()
    assert np.array_equal(got, ref)


@pytest.mark.parametrize("op", ["scatter_add", "sum", "amax", "amin"])
@pytest.mark.parametrize("shape,dim", LAYOUTS, ids=IDS)
def test_scatter_add_and_scatter_reduce_accumulate_where_the_index_says(shape, dim, op):
    _accelerator()
    from nbxtensor.ops.index_wrappers import scatter_add_wrapper
    from nbxtensor.ops.reduction_wrappers import scatter_reduce_wrapper
    rng = np.random.default_rng(zlib.crc32(repr((shape, dim, op)).encode()))
    out_shape = tuple(s + 3 if d == dim else s for d, s in enumerate(shape))   # a longer destination along dim
    x = rng.standard_normal(out_shape).astype(np.float32)
    src = rng.standard_normal(shape).astype(np.float32)
    index = rng.integers(0, out_shape[dim], size=shape, dtype=np.int64)   # duplicates: several adds into one place
    ref = x.copy()
    {"scatter_add": np.add, "sum": np.add, "amax": np.maximum, "amin": np.minimum}[op].at(ref, _coords(index, dim), src)
    args = (NBXTensor.from_numpy(x), dim, NBXTensor.from_numpy(index), NBXTensor.from_numpy(src))
    got = (scatter_add_wrapper(*args) if op == "scatter_add" else scatter_reduce_wrapper(*args, op)).numpy()
    if op in ("amax", "amin"):
        assert np.array_equal(got, ref)
    else:
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape,dim", LAYOUTS, ids=IDS)
def test_index_add_adds_every_source_slice_into_its_destination(shape, dim):
    _accelerator()
    from nbxtensor.ops.index_wrappers import index_add_wrapper
    rng = np.random.default_rng(zlib.crc32(repr((shape, dim, "index_add")).encode()))
    x_shape = tuple(s + 2 if d == dim else s for d, s in enumerate(shape))
    x = rng.standard_normal(x_shape).astype(np.float32)
    source = rng.standard_normal(shape).astype(np.float32)
    index = rng.integers(0, x_shape[dim], size=shape[dim], dtype=np.int64)
    ref = x.copy()
    for j, dest in enumerate(index):
        sl_out = [slice(None)] * len(shape)
        sl_src = [slice(None)] * len(shape)
        sl_out[dim], sl_src[dim] = int(dest), j
        ref[tuple(sl_out)] += source[tuple(sl_src)]
    got = index_add_wrapper(NBXTensor.from_numpy(x), dim, NBXTensor.from_numpy(index),
                            NBXTensor.from_numpy(source)).numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)
