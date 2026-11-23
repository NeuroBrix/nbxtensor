"""gather walks the INDEX's own extents and reads the input through its strides at the index's coordinates: an index
smaller than the input on a dim other than `dim` (aten's contract) reads exactly its positions, along every dim of
2-D to 4-D inputs, against numpy element by element. The wrapper once flattened the walk over the INPUT's outer and
inner extents while walking the index's positions: wrong values and reads past the input. The shipped layouts
(dim 3, index (1, 8, 23, 23) of (1, 8, 23, 45) and (1, 8, 180, 180) of (1, 8, 180, 359); dim 1, index (1, 23) of
(1, 512)) stay right; a layout the merged walk cannot address is refused by name."""
import zlib

import numpy as np
import pytest

from nbxtensor.tensor import NBXTensor

#: (input shape, index shape, dim): the shipped layouts, then indices smaller than the input on non-dim dims (and
#: shorter or longer along dim) along every dim of 2-D to 4-D inputs
LAYOUTS = [((1, 8, 23, 45), (1, 8, 23, 23), 3), ((1, 8, 180, 359), (1, 8, 180, 180), 3), ((1, 512), (1, 23), 1),
           ((7, 9), (3, 9), 1), ((7, 9), (3, 12), 1), ((7, 9), (7, 4), 0), ((7, 9), (2, 5), 0), ((7, 9), (2, 5), 1),
           ((4, 5, 6), (2, 3, 4), 1), ((4, 5, 6), (3, 2, 6), 0), ((4, 5, 6), (2, 5, 3), 2), ((4, 5, 6), (4, 1, 3), 2),
           ((4, 5, 6), (2, 7, 4), 1),
           ((3, 4, 5, 6), (2, 4, 5, 6), 0), ((3, 4, 5, 6), (2, 1, 5, 6), 0), ((3, 4, 5, 6), (2, 4, 3, 6), 1),
           ((3, 4, 5, 6), (1, 2, 5, 4), 2), ((3, 4, 5, 6), (2, 4, 2, 3), 2), ((3, 4, 5, 6), (3, 1, 1, 6), 3),
           ((3, 4, 5, 6), (2, 4, 5, 2), 3), ((2, 2, 3, 2), (1, 1, 1, 2), 2)]
IDS = [f"{'x'.join(map(str, i))}-index{'x'.join(map(str, x))}-dim{d}" for i, x, d in LAYOUTS]


def _accelerator():
    """Skip only on a machine with no accelerator at all; any other error fails the test."""
    from nbxtensor.memory.allocator import DeviceAllocator
    if DeviceAllocator.device_count() == 0:
        pytest.skip("no accelerator on this machine")


@pytest.mark.parametrize("shape,ishape,dim", LAYOUTS, ids=IDS)
def test_gather_reads_every_index_coordinate_of_the_input(shape, ishape, dim):
    _accelerator()
    from nbxtensor.ops.index_wrappers import gather_wrapper
    rng = np.random.default_rng(zlib.crc32(repr((shape, ishape, dim)).encode()))
    x = rng.standard_normal(shape).astype(np.float32)
    index = rng.integers(0, shape[dim], size=ishape, dtype=np.int64)
    coords = list(np.indices(ishape))
    coords[dim] = index
    got = gather_wrapper(NBXTensor.from_numpy(x), dim, NBXTensor.from_numpy(index)).numpy()
    assert got.shape == ishape and np.array_equal(got, x[tuple(coords)])


@pytest.mark.parametrize("shape,ishape,dim", [((4, 5, 6), (3, 2, 4), 0), ((2, 3, 3, 2), (2, 2, 1, 2), 2)],
                         ids=["4x5x6-index3x2x4-dim0", "2x3x3x2-index2x2x1x2-dim2"])
def test_gather_refuses_by_name_a_layout_the_merged_walk_cannot_address(shape, ishape, dim):
    _accelerator()
    from nbxtensor.ops.index_wrappers import gather_wrapper
    x = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    index = np.zeros(ishape, dtype=np.int64)
    with pytest.raises(RuntimeError, match="gather along dim .* cannot walk"):
        gather_wrapper(NBXTensor.from_numpy(x), dim, NBXTensor.from_numpy(index))
