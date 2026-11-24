"""`memory.bind_tied(weights, param, source)`: a weight table's entry `param` made an ALIAS of another component's
loaded tensor — the same tensor, no copy, no allocation, 0 bytes counted for the table — and kept across the
consumer's unload (`release_weights`) and reload (`join_weights` of the released table and the new load); a shape or
dtype that differs is refused by name; the source released while the consumer holds the alias stays alive (the alias is
a reference, as every view keeps its allocation), and a consumer released AND its source released refuses its reload
by name instead of keeping a released component's memory alive behind it.

What this test would do if the code were wrong: a copy (another pointer, bytes allocated), the reload bringing back
the stored copy or nothing, a mismatch bound, the source's memory freed under the alias, a released source kept
alive by a released consumer — RED."""
import gc

import numpy as np
import pytest

from nbxtensor import memory
from nbxtensor.memory.allocator import DeviceAllocator as DA
from nbxtensor.tensor import NBXTensor


@pytest.fixture(autouse=True)
def _device():
    """Skip on a machine without an accelerator only: the tensors are device tensors."""
    if DA.device_count() == 0:          # only a machine without an accelerator skips; any other error fails
        pytest.skip("no accelerator on this machine")


def _t(a):
    """A device tensor holding the numpy array `a` (made contiguous)."""
    return NBXTensor.from_numpy(np.ascontiguousarray(a))


def _source_table():
    """The source component's loaded table: its token table [8, 4] and a norm weight."""
    embed = _t(np.arange(32, dtype=np.float32).reshape(8, 4))
    return {"embed_tokens.weight": embed, "norm.weight": _t(np.ones(4, np.float32))}


def _consumer_load(with_stored_copy=True):
    """What the consumer's loader returns: its own weight, and its stored copy of the tied param when asked."""
    loaded = {"proj.weight": _t(np.full((4, 4), 2.0, np.float32))}
    if with_stored_copy:
        loaded["lm_head.weight"] = _t(np.zeros((8, 4), np.float32))
    return loaded


def test_the_param_becomes_the_source_s_tensor_with_no_allocation_and_no_bytes():
    """The bind puts the source itself at the param: same object and pointer, nothing allocated, 0 bytes."""
    src = _source_table()
    head = _consumer_load()
    source = src["embed_tokens.weight"]
    own = memory.weight_bytes(head) - head["lm_head.weight"].nbytes()
    before = memory.memory_allocated()
    assert memory.bind_tied(head, "lm_head.weight", source) is source
    assert memory.memory_allocated() == before, "the bind allocated"
    assert head["lm_head.weight"] is source and head["lm_head.weight"].data_ptr() == source.data_ptr()
    assert memory.weight_bytes(head) == own, "the alias was counted in the consumer's bytes"
    assert memory.tied_params(head) == frozenset({"lm_head.weight"})
    np.testing.assert_array_equal(head["lm_head.weight"].numpy(), np.arange(32, dtype=np.float32).reshape(8, 4))


def test_the_alias_holds_across_the_consumer_s_unload_and_reload():
    """Released then joined with a new load, twice: the param is the source's tensor each time."""
    src = _source_table()
    source = src["embed_tokens.weight"]
    head = _consumer_load()
    memory.bind_tied(head, "lm_head.weight", source)
    assert memory.release_weights(head) is True and head == {}
    assert memory.tied_params(head) == frozenset({"lm_head.weight"}), "the unload forgot the tie"
    # the reload: the loader skips what is tied (no stored copy read), the library joins the released table
    reloaded = memory.join_weights(head, _consumer_load(with_stored_copy=False))
    assert reloaded["lm_head.weight"] is source and reloaded["lm_head.weight"].data_ptr() == source.data_ptr()
    assert memory.tied_params(reloaded) == frozenset({"lm_head.weight"})
    # a loader that read the stored copy anyway: the alias wins, the copy is not what the table holds
    assert memory.release_weights(reloaded) is True
    again = memory.join_weights(reloaded, _consumer_load(with_stored_copy=True))
    assert again["lm_head.weight"] is source
    # and a second cycle on the new table
    assert memory.release_weights(again) is True
    assert memory.join_weights(again, _consumer_load(False))["lm_head.weight"] is source


def test_a_partial_load_into_a_held_table_keeps_the_alias():
    """A load that adds readers to a held table leaves the tied param the source's, its stored copy unused."""
    src = _source_table()
    source = src["embed_tokens.weight"]
    head = _consumer_load()
    memory.bind_tied(head, "lm_head.weight", source)
    head.update(memory.join_weights(head, {"extra.weight": _t(np.ones(2, np.float32)),
                                           "lm_head.weight": _t(np.zeros((8, 4), np.float32))}))
    assert head["lm_head.weight"] is source and "extra.weight" in head


def test_a_shape_or_dtype_that_differs_is_refused_by_name():
    """A differing shape or dtype, or a source that is no library tensor: refused by name, the table unchanged."""
    src = _source_table()
    head = _consumer_load()
    wrong_shape = _t(np.zeros((4, 8), np.float32))
    with pytest.raises(memory.TiedWeightError, match=r"lm_head\.weight.*shape"):
        memory.bind_tied(head, "lm_head.weight", wrong_shape)
    wrong_dtype = _t(np.zeros((8, 4), np.float16))
    with pytest.raises(memory.TiedWeightError, match=r"lm_head\.weight.*dtype"):
        memory.bind_tied(head, "lm_head.weight", wrong_dtype)
    with pytest.raises(memory.TiedWeightError, match=r"lm_head\.weight.*shape"):
        memory.bind_tied({}, "lm_head.weight", src["embed_tokens.weight"], shape=(4, 8))
    with pytest.raises(memory.TiedWeightError, match=r"lm_head\.weight.*not a library tensor"):
        memory.bind_tied(head, "lm_head.weight", np.zeros((8, 4), np.float32))
    assert head["lm_head.weight"] is not src["embed_tokens.weight"], "a refused bind changed the table"
    assert memory.tied_params(head) == frozenset()


def test_the_source_released_under_a_held_alias_stays_alive_by_reference():
    """The source's table released while the consumer holds the alias: its memory stays allocated and readable."""
    src = _source_table()
    source = src["embed_tokens.weight"]
    ptr = source.data_ptr()
    head = _consumer_load()
    memory.bind_tied(head, "lm_head.weight", source)
    del source
    assert memory.release_weights(src) is True and src == {}
    gc.collect()
    alias = head["lm_head.weight"]
    assert alias.data_ptr() == ptr and ptr in memory.blocks_in_use(alias.device_idx), "the source was freed"
    np.testing.assert_array_equal(alias.numpy(), np.arange(32, dtype=np.float32).reshape(8, 4))


def test_a_released_consumer_does_not_keep_a_released_source_alive_and_its_reload_is_refused_by_name():
    """Consumer released, then source released: the source is freed and the consumer's reload refused by name."""
    src = _source_table()
    source = src["embed_tokens.weight"]
    ptr, dev = source.data_ptr(), source.device_idx
    head = _consumer_load()
    memory.bind_tied(head, "lm_head.weight", source)
    assert memory.release_weights(head) is True
    del source
    assert memory.release_weights(src) is True
    gc.collect()
    assert ptr not in memory.blocks_in_use(dev), "a released consumer kept the released source alive"
    with pytest.raises(memory.TiedWeightError, match=r"lm_head\.weight.*released"):
        memory.join_weights(head, _consumer_load(False))
