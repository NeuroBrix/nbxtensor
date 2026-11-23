"""scatter, scatter_add and scatter_reduce (sum, amax, amin) take an index smaller than `src` and `out` on the dims
other than `dim` (aten's contract: index.size(d) <= src.size(d) and <= out.size(d)): the index walks its own
positions and each operand is reached through its own strides at the index's coordinates. The merged
(outer, dim, inner) walk addresses such a layout exactly when each operand's offset is linear in the flattened index
position over the dims before `dim` and over the dims after it; `scatter_op.collapsed_strides` addresses every such
layout (an index of extent 1 on a dim where the operand is longer among them: dim 2, index (1, 1, 1, 2) of
(2, 2, 3, 2)) and refuses by name exactly the others. The card-less half checks that boundary against the exact
offsets on random layouts; the device half checks the results against numpy element by element."""
import zlib

import numpy as np
import pytest

from nbxtensor.ops.scatter_op import collapsed_strides
from nbxtensor.tensor import NBXTensor


class _Layout:
    """A card-less stand-in for a contiguous tensor: its shape and its element strides, nothing on a device."""

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self._strides = tuple(s // 8 for s in np.empty(shape, dtype=np.int64).strides)

    def stride(self, d):
        """The element stride of dim `d`, as NBXTensor.stride gives it."""
        return self._strides[d]


def _exact_offsets(t, ishape):
    """The offset of `t` at every index coordinate, in the index's row-major order (what the walk must form)."""
    c = np.indices(ishape).reshape(len(ishape), -1)
    return sum(c[d] * t.stride(d) for d in range(len(ishape)))


def _walk_offsets(ishape, dim, strides):
    """The offset the kernels of scatter_op form at every flat index position from the (outer, dim, inner) strides."""
    inner = int(np.prod(ishape[dim + 1:], dtype=np.int64))
    off = np.arange(int(np.prod(ishape, dtype=np.int64)), dtype=np.int64)
    i, rest = off % inner, off // inner
    return rest // ishape[dim] * strides[0] + rest % ishape[dim] * strides[1] + i * strides[2]


def _walkable(t, ishape, dim):
    """Whether ANY (outer, dim, inner) strides form the exact offsets: the only candidates are the offsets at outer
    position 1 and inner position 1, so try them."""
    exact = _exact_offsets(t, ishape)
    inner = int(np.prod(ishape[dim + 1:], dtype=np.int64))
    s_outer = int(exact[ishape[dim] * inner]) if exact.size > ishape[dim] * inner else 0
    s_inner = int(exact[1]) if inner > 1 else 0
    return np.array_equal(_walk_offsets(ishape, dim, (s_outer, t.stride(dim), s_inner)), exact)


def test_qa_s_layout_is_addressed_and_a_non_linear_one_is_refused_by_name():
    out = _Layout((2, 2, 3, 2))
    index = _Layout((1, 1, 1, 2))
    got = collapsed_strides(out, 2, index)
    assert np.array_equal(_walk_offsets(index.shape, 2, got), _exact_offsets(out, index.shape))
    with pytest.raises(RuntimeError, match="cannot walk"):
        collapsed_strides(_Layout((2, 3, 3, 2)), 2, _Layout((2, 2, 1, 2)))


def test_every_walkable_smaller_index_is_addressed_exactly_and_only_the_others_are_refused():
    rng = np.random.default_rng(1312)
    addressed = refused = 0
    for _ in range(3000):
        n = int(rng.integers(1, 5))
        shape = tuple(int(v) for v in rng.integers(1, 5, size=n))
        ishape = tuple(int(rng.integers(1, s + 1)) for s in shape)
        dim = int(rng.integers(0, n))
        t, index = _Layout(shape), _Layout(ishape)
        if _walkable(t, ishape, dim):
            got = collapsed_strides(t, dim, index)
            assert np.array_equal(_walk_offsets(ishape, dim, got), _exact_offsets(t, ishape)), (shape, ishape, dim)
            addressed += 1
        else:
            with pytest.raises(RuntimeError, match="cannot walk"):
                collapsed_strides(t, dim, index)
            refused += 1
    assert addressed > 1000 and refused > 100


#: (out shape, index shape, dim): qa's layout, then indices smaller than the operands along every dim of 2-D to 4-D
LAYOUTS = [((2, 2, 3, 2), (1, 1, 1, 2), 2),
           ((7, 9), (3, 9), 1), ((7, 9), (7, 4), 0), ((7, 9), (2, 5), 0), ((7, 9), (2, 5), 1),
           ((4, 5, 6), (2, 3, 4), 1), ((4, 5, 6), (3, 2, 6), 0), ((4, 5, 6), (2, 5, 3), 2), ((4, 5, 6), (4, 1, 3), 2),
           ((3, 4, 5, 6), (2, 4, 3, 6), 1), ((3, 4, 5, 6), (1, 2, 5, 4), 2), ((3, 4, 5, 6), (3, 1, 1, 6), 3),
           ((3, 4, 5, 6), (2, 4, 5, 2), 3), ((3, 4, 5, 6), (2, 1, 5, 6), 0)]
IDS = [f"{'x'.join(map(str, o))}-index{'x'.join(map(str, i))}-dim{d}" for o, i, d in LAYOUTS]


def _accelerator():
    """Skip only on a machine with no accelerator at all; any other error fails the test."""
    from nbxtensor.memory.allocator import DeviceAllocator
    if DeviceAllocator.device_count() == 0:
        pytest.skip("no accelerator on this machine")


def _operands(rng, out_shape, ishape, dim, unique):
    """x (out's shape), src (longer than the index on dim 0 and on dim), and an int64 index of `ishape` into out along
    `dim`: distinct along every line when `unique` (a plain scatter then has one answer)."""
    src_shape = tuple(s + 1 if d in (0, dim) else s for d, s in enumerate(ishape))   # longer on dim 0 and dim: walkable
    x = rng.standard_normal(out_shape).astype(np.float32)
    src = rng.standard_normal(src_shape).astype(np.float32)
    if unique:
        wide = tuple(out_shape[dim] if d == dim else s for d, s in enumerate(ishape))
        index = np.take(rng.random(wide).argsort(axis=dim), np.arange(ishape[dim]), axis=dim).astype(np.int64)
    else:
        index = rng.integers(0, out_shape[dim], size=ishape, dtype=np.int64)
    coords = list(np.indices(ishape))
    read = src[tuple(coords)]
    coords[dim] = index
    return x, src, index, tuple(coords), read


@pytest.mark.parametrize("out_shape,ishape,dim", LAYOUTS, ids=IDS)
def test_scatter_writes_through_an_index_smaller_than_src_and_out(out_shape, ishape, dim):
    _accelerator()
    from nbxtensor.ops.index_wrappers import scatter_wrapper
    rng = np.random.default_rng(zlib.crc32(repr((out_shape, ishape, dim)).encode()))
    x, src, index, coords, read = _operands(rng, out_shape, ishape, dim, unique=True)
    ref = x.copy()
    ref[coords] = read
    got = scatter_wrapper(NBXTensor.from_numpy(x), dim, NBXTensor.from_numpy(index), NBXTensor.from_numpy(src))
    assert np.array_equal(got.numpy(), ref)


@pytest.mark.parametrize("op", ["scatter_add", "sum", "amax", "amin"])
@pytest.mark.parametrize("out_shape,ishape,dim", LAYOUTS, ids=IDS)
def test_scatter_add_and_reduce_accumulate_through_an_index_smaller_than_src_and_out(out_shape, ishape, dim, op):
    _accelerator()
    from nbxtensor.ops.index_wrappers import scatter_add_wrapper
    from nbxtensor.ops.reduction_wrappers import scatter_reduce_wrapper
    rng = np.random.default_rng(zlib.crc32(repr((out_shape, ishape, dim, op)).encode()))
    x, src, index, coords, read = _operands(rng, out_shape, ishape, dim, unique=False)
    ref = x.copy()
    {"scatter_add": np.add, "sum": np.add, "amax": np.maximum, "amin": np.minimum}[op].at(ref, coords, read)
    args = (NBXTensor.from_numpy(x), dim, NBXTensor.from_numpy(index), NBXTensor.from_numpy(src))
    got = (scatter_add_wrapper(*args) if op == "scatter_add" else scatter_reduce_wrapper(*args, op)).numpy()
    if op in ("amax", "amin"):
        assert np.array_equal(got, ref)
    else:
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)
