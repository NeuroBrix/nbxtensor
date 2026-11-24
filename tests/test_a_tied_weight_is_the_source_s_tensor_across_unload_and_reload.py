"""`memory.bind_tied(weights, param, source_weights, source_key)`: a weight table's entry `param` made an ALIAS of the
tensor another table holds at `source_key` (no copy, no allocation) and kept across the unloads and reloads the engine
really makes (`release_weights`, `join_weights`, and the engine's own bind again at each request).

The sources here are the real thing: a safetensors shard written by the test, read through the library's own
placement (`io.safetensors.place_in_arena`) into a `ComponentArena` sized by `WeightArenas`, the table's "_arenas"
the arena's only reference — exactly the engine's loader. A weight there is an NBXTensor over a sub-allocation that
holds NO reference to its arena, so a tie that kept only the tensor would read freed memory once the source's table
lets go of "_arenas". Each cell checks that fact, not luck: the alias's address lies inside a block the allocator
still holds (`blocks_in_use`), and its values are the shard's after a same-size allocation filled with a sentinel.

The orders replayed are the engine's: the text flow releases the language model at each request's end while the head
stays loaded (triton/flow/autoregressive.py:344, :829-835), then reloads it and binds the ties again; a consumer and
its source both released and reloaded; a tie inside one component; a consumer's dict replaced after its load
(graph_executor.py `_reconcile_weight_keys`).

What this test would do if the code were wrong: the source's arena freed under a loaded alias, an arena kept after
its last alias went, a stale tensor aliased after the source's reload, a refusal where the source can come back, a
replaced dict kept or written, a tie counted twice or not at all — RED."""
import gc
import json
import struct
import weakref

import numpy as np
import pytest

from nbxtensor import memory
from nbxtensor.io import safetensors as sf
from nbxtensor.memory.allocator import DeviceAllocator as DA
from nbxtensor.tensor import NBXTensor

EMBED = np.arange(32, dtype=np.float32).reshape(8, 4)


@pytest.fixture(autouse=True)
def _device():
    """Skip on a machine without an accelerator only: the arenas and tensors are device memory."""
    if DA.device_count() == 0:          # only a machine without an accelerator skips; any other error fails
        pytest.skip("no accelerator on this machine")


def _write_shard(path, arrays):
    """A safetensors file at `path` holding `arrays` ({name: float32 array}), written as the format says."""
    header, blobs, offset = {}, [], 0
    for name, a in arrays.items():
        raw = np.ascontiguousarray(a, dtype=np.float32).tobytes()
        header[name] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    text = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(text)) + text + b"".join(blobs))
    return path


def _load_shard(path):
    """The shard loaded as the engine's loader does: sized by `WeightArenas`, one `ComponentArena` on the current
    device, each tensor placed by the library's `place_in_arena`, the arenas under "_arenas" (their only reference)."""
    with open(path, "rb") as f:
        (hsize,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hsize))
        data = f.read()
    dev = DA.get_device()
    arenas = memory.weight_arenas()
    for info in header.values():
        arenas.reserve(dev, info["data_offsets"][1] - info["data_offsets"][0])
    arenas.allocate()
    table = {}
    for name, info in header.items():
        np_dtype, nbx_dtype, _ = sf.SF_DTYPE_INFO[info["dtype"]]
        start, end = info["data_offsets"]
        table[name] = sf.place_in_arena(data[start:end], tuple(info["shape"]), nbx_dtype, np_dtype, nbx_dtype,
                                        False, arenas.devices[dev], dev, None)
    table["_arenas"] = arenas.devices
    return table


def _lm(tmp_path, tag="lm"):
    """The source component (a language model): its token table and a norm, from its own shard and arena."""
    return _load_shard(_write_shard(tmp_path / f"{tag}.safetensors",
                                    {"embed_tokens.weight": EMBED, "norm.weight": np.ones(4, np.float32)}))


def _head(tmp_path, stored_copy=True, tag="head"):
    """The consumer (a head): its own weight and, when asked, its stored copy of the tied param, from its own shard."""
    arrays = {"proj.weight": np.full((4, 4), 2.0, np.float32)}
    if stored_copy:
        arrays["lm_head.weight"] = np.zeros((8, 4), np.float32)
    return _load_shard(_write_shard(tmp_path / f"{tag}.safetensors", arrays))


def _arena_base(table):
    """The device address of the (one) arena a loaded table's "_arenas" holds."""
    return next(a._base_ptr for k, a in table["_arenas"].items() if k != "_tied")


def _live(ptr):
    """Whether `ptr` lies inside a block the allocator still holds for a tensor (not freed, not parked)."""
    return any(base <= ptr < base + n for base, n in memory.blocks_in_use().items())


def _reads_the_shard(alias):
    """The alias's live memory still holds the shard's values once a same-size allocation is filled with a sentinel
    (freed memory would be handed out again and overwritten)."""
    assert _live(alias.data_ptr()), "the alias points into memory the allocator freed"
    sentinel = NBXTensor.from_numpy(np.full(EMBED.shape, -7.0, np.float32))
    np.testing.assert_array_equal(alias.numpy(), EMBED)
    del sentinel


def test_the_param_becomes_the_source_s_tensor_with_no_copy_and_no_allocation(tmp_path):
    """The bind puts the source's tensor itself at the param: same object and address, nothing allocated."""
    lm, head = _lm(tmp_path), _head(tmp_path)
    before = memory.memory_allocated()
    assert memory.bind_tied(head, "lm_head.weight", lm, "embed_tokens.weight") is lm["embed_tokens.weight"]
    assert memory.memory_allocated() == before, "the bind allocated"
    assert head["lm_head.weight"] is lm["embed_tokens.weight"]
    assert memory.tied_params(head) == frozenset({"lm_head.weight"})
    np.testing.assert_array_equal(head["lm_head.weight"].numpy(), EMBED)


def test_the_source_released_under_a_loaded_alias_keeps_its_arena_until_the_last_alias_goes(tmp_path):
    """Releasing the source releases its NAME: the arena stays while the head holds the alias, and goes with it."""
    lm, head = _lm(tmp_path), _head(tmp_path)
    base = _arena_base(lm)
    memory.bind_tied(head, "lm_head.weight", lm, "embed_tokens.weight")
    assert memory.release_weights(lm) is True and lm == {}
    gc.collect()
    assert _live(base), "the source's arena was freed under a loaded alias"
    _reads_the_shard(head["lm_head.weight"])
    assert memory.release_weights(head) is True
    gc.collect()
    assert not _live(base), "the source's arena outlived its last alias"


def test_the_bytes_are_counted_once_where_they_are(tmp_path):
    """Source loaded: the tie is 0 in the head, its bytes the source's. Source released: the kept arena is counted
    once, in the tables that keep it, however many of them are summed together."""
    lm, head, head2 = _lm(tmp_path), _head(tmp_path), _head(tmp_path, tag="head2")
    lm_own, head_own = memory.weight_bytes(lm), memory.weight_bytes(head) - EMBED.nbytes
    memory.bind_tied(head, "lm_head.weight", lm, "embed_tokens.weight")
    memory.bind_tied(head2, "lm_head.weight", lm, "embed_tokens.weight")
    assert memory.weight_bytes(head) == head_own
    assert memory.weight_bytes(lm, head, head2) == lm_own + 2 * head_own
    kept = next(a for k, a in lm["_arenas"].items() if k != "_tied").capacity
    memory.release_weights(lm)
    assert memory.weight_bytes(head) == head_own + kept
    assert memory.weight_bytes(head, head2) == 2 * head_own + kept, "the kept arena was counted twice"


def test_the_text_flow_s_order_rebinds_the_head_to_the_reloaded_language_model(tmp_path):
    """The LM released at the request's end with the head loaded (autoregressive.py:344), reloaded at the next one
    the engine's way (a new dict), then the twin's per-request bind: the head aliases the NEW tensor and the old
    arena is freed. Several requests in a row keep nothing behind."""
    from nbxtensor.memory import tied
    parked = len(tied._PARKED)
    head = _head(tmp_path)
    lm = _lm(tmp_path)
    memory.bind_tied(head, "lm_head.weight", lm, "embed_tokens.weight")
    for turn in range(3):
        old_base = _arena_base(lm)
        memory.release_weights(lm)                                  # :344, the head stays
        gc.collect()
        assert _live(old_base)
        _reads_the_shard(head["lm_head.weight"])                    # the head may still run on the old one
        lm = _lm(tmp_path, tag=f"lm{turn}")                         # the reload: `self._weights = loaded`
        memory.bind_tied(head, "lm_head.weight", lm, "embed_tokens.weight")   # the twin's bind, every request
        gc.collect()
        assert head["lm_head.weight"] is lm["embed_tokens.weight"], "the head kept the stale tensor"
        assert not _live(old_base), "the old language model's arena outlived the re-bind"
    assert len(tied._PARKED) <= parked, f"released tables pile up: {len(tied._PARKED) - parked} more"


def test_the_source_reloaded_through_join_weights_is_what_the_consumer_reload_aliases(tmp_path):
    """Consumer released, source released, source reloaded (`join_weights`), consumer reloaded: the consumer aliases
    the source's NEW tensor; never refused, since the source came back."""
    lm, head = _lm(tmp_path), _head(tmp_path)
    old_base = _arena_base(lm)
    memory.bind_tied(head, "lm_head.weight", lm, "embed_tokens.weight")
    memory.release_weights(head)
    memory.release_weights(lm)
    gc.collect()
    assert not _live(old_base), "a released consumer kept its released source's arena"
    lm = memory.join_weights(lm, _lm(tmp_path, tag="lm2"))
    head = memory.join_weights(head, _head(tmp_path, stored_copy=False, tag="head2"))
    assert head["lm_head.weight"] is lm["embed_tokens.weight"]
    _reads_the_shard(head["lm_head.weight"])


def test_a_consumer_reloaded_before_its_source_waits_for_it_then_binds(tmp_path):
    """The twin loads the consumer before the source: the consumer's reload leaves the tie waiting (named by
    `tied_params`, no stale alias), and the bind after the source's load completes it."""
    lm, head = _lm(tmp_path), _head(tmp_path)
    memory.bind_tied(head, "lm_head.weight", lm, "embed_tokens.weight")
    memory.release_weights(head)
    memory.release_weights(lm)
    head = memory.join_weights(head, _head(tmp_path, stored_copy=False, tag="head2"))
    assert "lm_head.weight" not in head and memory.tied_params(head) == frozenset({"lm_head.weight"})
    lm = _lm(tmp_path, tag="lm2")
    memory.bind_tied(head, "lm_head.weight", lm, "embed_tokens.weight")
    assert head["lm_head.weight"] is lm["embed_tokens.weight"]


def test_a_source_that_can_never_come_back_is_refused_by_name_at_the_consumer_s_reload(tmp_path):
    """The source's table dropped without a release (its record gone): nothing can bring it back, so the consumer's
    reload refuses the tie by name."""
    lm, head = _lm(tmp_path), _head(tmp_path)
    memory.bind_tied(head, "lm_head.weight", lm, "embed_tokens.weight")
    memory.release_weights(head)
    del lm
    gc.collect()
    with pytest.raises(memory.TiedWeightError, match=r"lm_head\.weight.*never come back"):
        memory.join_weights(head, _head(tmp_path, stored_copy=False, tag="head2"))


def test_a_tie_inside_one_component_aliases_the_new_load_never_the_old_memory(tmp_path):
    """A table tied to itself, released and reloaded: the param is the NEW load's tensor, the old arena freed."""
    t = _load_shard(_write_shard(tmp_path / "one.safetensors", {"embed_tokens.weight": EMBED,
                                                                "lm_head.weight": np.zeros((8, 4), np.float32)}))
    old_base = _arena_base(t)
    memory.bind_tied(t, "lm_head.weight", t, "embed_tokens.weight")
    memory.release_weights(t)
    gc.collect()
    assert not _live(old_base), "the table's own tie kept its released arena"
    new = _load_shard(_write_shard(tmp_path / "two.safetensors", {"embed_tokens.weight": EMBED}))
    t = memory.join_weights(t, new)
    assert t["lm_head.weight"] is t["embed_tokens.weight"]
    _reads_the_shard(t["lm_head.weight"])


def test_a_replaced_dict_is_neither_kept_nor_written(tmp_path):
    """The engine replaces a loaded table's dict (`_reconcile_weight_keys`): the tie's record moves with "_arenas";
    the old dict is not referenced by the library (it is collected) and nothing writes into a dict it kept."""
    class Marker:
        """A weakly referenced value only the old dict holds."""

    lm, head = _lm(tmp_path), _head(tmp_path)
    memory.bind_tied(head, "lm_head.weight", lm, "embed_tokens.weight")
    memory.bind_tied(head, "lm_head.weight", lm, "embed_tokens.weight")      # a second bind changes nothing
    marker = Marker()
    head["marker"] = marker
    probe = weakref.ref(marker)
    renamed = {k: v for k, v in head.items() if k != "marker"}              # the engine's `self._weights = new`
    kept_lm = dict(lm)                                                       # an old source dict the test keeps
    snapshot = {k: id(v) for k, v in kept_lm.items()}
    del head, marker
    gc.collect()
    assert probe() is None, "the library kept the replaced dict"
    assert memory.tied_params(renamed) == frozenset({"lm_head.weight"})
    memory.release_weights(lm)
    lm = memory.join_weights(lm, _lm(tmp_path, tag="lm2"))
    memory.bind_tied(renamed, "lm_head.weight", lm, "embed_tokens.weight")
    assert {k: id(v) for k, v in kept_lm.items()} == snapshot, "the library wrote into a dict it should not hold"
    assert renamed["lm_head.weight"] is lm["embed_tokens.weight"]


def test_a_partial_load_into_a_held_table_keeps_the_alias_and_the_table_s_arenas(tmp_path):
    """A load that adds readers to a held table: the tied param stays the source's, the table's own arenas stay."""
    lm, head = _lm(tmp_path), _head(tmp_path)
    memory.bind_tied(head, "lm_head.weight", lm, "embed_tokens.weight")
    base = _arena_base(head)
    head.update(memory.join_weights(head, {"extra.weight": NBXTensor.from_numpy(np.ones(2, np.float32)),
                                           "lm_head.weight": NBXTensor.from_numpy(np.zeros((8, 4), np.float32))}))
    gc.collect()
    assert head["lm_head.weight"] is lm["embed_tokens.weight"] and "extra.weight" in head
    assert _arena_base(head) == base and _live(base), "the partial load dropped the table's own arena"


def test_a_shape_dtype_or_key_that_differs_is_refused_by_name(tmp_path):
    """A differing shape or dtype, a key the source does not hold, a source that is no library tensor: refused by
    name, the table unchanged."""
    lm, head = _lm(tmp_path), _head(tmp_path)
    stored = head["lm_head.weight"]
    other = {"wide": NBXTensor.from_numpy(np.zeros((4, 8), np.float32)),
             "half": NBXTensor.from_numpy(np.zeros((8, 4), np.float16)),
             "host": np.zeros((8, 4), np.float32)}
    with pytest.raises(memory.TiedWeightError, match=r"lm_head\.weight.*shape"):
        memory.bind_tied(head, "lm_head.weight", other, "wide")
    with pytest.raises(memory.TiedWeightError, match=r"lm_head\.weight.*dtype"):
        memory.bind_tied(head, "lm_head.weight", other, "half")
    with pytest.raises(memory.TiedWeightError, match=r"lm_head\.weight.*not a library tensor"):
        memory.bind_tied(head, "lm_head.weight", other, "host")
    with pytest.raises(memory.TiedWeightError, match=r"lm_head\.weight.*'missing'"):
        memory.bind_tied(head, "lm_head.weight", lm, "missing")
    with pytest.raises(memory.TiedWeightError, match=r"lm_head\.weight.*shape"):
        memory.bind_tied({}, "lm_head.weight", lm, "embed_tokens.weight", shape=(4, 8))
    assert head["lm_head.weight"] is stored and memory.tied_params(head) == frozenset()


def test_a_source_whose_memory_no_arena_of_its_table_holds_is_refused_by_name_unless_it_owns_it(tmp_path):
    """A loader-placed weight in a table without its "_arenas" (it owns nothing, no arena of the table holds it): the
    tie could keep nothing alive, so it is refused by name. A tensor that owns its memory, or a view of one, needs no
    arena: the reference keeps it, and it is accepted."""
    lm = _lm(tmp_path)
    stripped = {"embed_tokens.weight": lm["embed_tokens.weight"]}          # the loader's tensor, its arena not here
    head = _head(tmp_path)
    with pytest.raises(memory.TiedWeightError, match=r"lm_head\.weight.*no arena"):
        memory.bind_tied(head, "lm_head.weight", stripped, "embed_tokens.weight")
    assert memory.tied_params(head) == frozenset()
    owning = NBXTensor.from_numpy(EMBED)
    assert memory.bind_tied(head, "lm_head.weight", {"w": owning}, "w") is owning
    view = owning.reshape(8, 4)
    assert memory.bind_tied(_head(tmp_path, tag="h2"), "lm_head.weight", {"w": view}, "w") is view


def test_a_refused_rejoin_leaves_the_parked_record_for_a_corrected_reload(tmp_path):
    """A released source reloaded from entries whose memory no arena of the load holds: the rejoin is refused by name
    BEFORE anything changes, so the released table's parked record (its ties) is still there, and the corrected
    reload (the loader's table, its "_arenas" with it) rejoins and the consumer's bind aliases the new tensor."""
    from nbxtensor.memory import tied
    lm, head = _lm(tmp_path), _head(tmp_path)
    memory.bind_tied(head, "lm_head.weight", lm, "embed_tokens.weight")
    memory.release_weights(lm)
    rec = tied.record_of(lm)
    assert rec is not None and len(rec.sourced) == 1
    lm2 = _lm(tmp_path, tag="lm2")
    stripped = {"embed_tokens.weight": lm2["embed_tokens.weight"]}         # the loader's tensor, its arena not here
    with pytest.raises(memory.TiedWeightError, match=r"lm_head\.weight.*no arena"):
        memory.join_weights(lm, stripped)
    assert tied._parked_record(lm) is rec, "a refused rejoin deleted the released table's parked record"
    assert len(rec.sourced) == 1
    lm = memory.join_weights(lm, lm2)
    assert tied.record_of(lm) is rec
    memory.bind_tied(head, "lm_head.weight", lm, "embed_tokens.weight")
    assert head["lm_head.weight"] is lm2["embed_tokens.weight"]
    _reads_the_shard(head["lm_head.weight"])
