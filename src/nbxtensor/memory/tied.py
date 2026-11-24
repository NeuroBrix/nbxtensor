"""Tied weights: a weight table's entry made an ALIAS of the tensor another table holds at a key, kept across the
unloads and reloads of both tables.

A weight table is the application's dict {name: tensor} with the library's bookkeeping under "_arenas" (the blocks
its weights point into: a weight placed by the loader is an NBXTensor over a sub-allocation that holds NO reference
to its arena, so "_arenas" is the arena's only owner). `bind_tied(weights, param, source_weights, source_key)` puts
`source_weights[source_key]` itself at `weights[param]` (no copy, no allocation) and records the tie in each table's
bookkeeping, `"_arenas"["_tied"]`: one record per table, which moves with "_arenas" when the application replaces
the dict (a rebuilt dict that keeps its "_arenas", as `_reconcile_weight_keys` does). The library holds NO table
dict while it is loaded: it neither keeps a replaced dict alive nor writes into one.

Lifetime (the lead's ruling, 1371):
  * the alias LOADED keeps the source's arenas (the blocks whose range holds its address) alive: releasing the
    source (`release_weights`) releases its name, not its memory; the arenas go when the last alias goes, after the
    release that drops it synchronised its devices;
  * the consumer released: its tie keeps nothing (no tensor, no arena) and waits for the table's reload;
  * the source reloaded through `join_weights(released_table, loaded)`: its ties learn the new tensor at their key;
    the consumer's reload (`join_weights`) or the application's next `bind_tied` then aliases THAT tensor, never the
    released one. A source reloaded as a new dict is re-bound by the application's `bind_tied` (by table and key);
  * the consumer reloaded while its source is not loaded: the tie waits (`tied_params` names it, the entry is not
    set), and the next `bind_tied` completes it; refused by name only when the source can never come back (its
    record is gone: its table dropped without a release);
  * a table tied to itself: released and reloaded, the param is the new load's tensor at the key.
Counting (`weight_bytes`): a tie is 0 while its source is loaded (its bytes are the source's); once the source is
released, the arenas the alias keeps are counted once, in the tables that keep them, whatever tables are summed.
No torch; the tensors are the library's (NBXTensor)."""
import weakref

KEY = "_tied"                           # the record's place in a table's "_arenas"


class TiedWeightError(ValueError):
    """A tie that cannot be made or restored; the message names the entry and why."""


class _Record:
    """One table's ties: `binds` {param: _Tie} where it is the consumer, `sourced` the ties it is the source of."""
    __slots__ = ("binds", "sourced", "loaded", "__weakref__")

    def __init__(self):
        """A record for a loaded table with no tie yet."""
        self.binds = {}
        self.sourced = weakref.WeakSet()  # held by their consumers' records: a dropped consumer leaves nothing here
        self.loaded = True


class _Tie:
    """One tie: the consumer's `param` is the source's tensor at `key`. `offer` is what the source holds there while
    loaded (tensor, the arenas holding it); `alias` what the consumer's table holds while loaded (the same pair, its
    arenas kept alive by it)."""
    __slots__ = ("param", "key", "source", "offer", "alias", "__weakref__")

    def __init__(self, param, key, source):
        """A tie of `param` to the record `source` at `key`, nothing offered or aliased yet."""
        self.param, self.key = param, key
        self.source = weakref.ref(source, _withdraw(weakref.ref(self)))
        self.offer = self.alias = None


def _withdraw(tie_ref):
    """The callback run when a tie's source record dies (its table dropped without a release): the tie lets go of
    what the source offered, so a dropped source's memory is not kept by a tie that can never alias it again."""
    def withdraw(_):
        """Drop the dead source's offer from the tie, if the tie still lives."""
        tie = tie_ref()
        if tie is not None:
            tie.offer = None
    return withdraw


_PARKED = {}                            # id(released table) -> (that table, its record): an empty dict waiting


def _held_record(weights):
    """The record a LOADED table carries in its bookkeeping (`_arenas["_tied"]`), None without one: where the ties
    live while the table is loaded, so a rebuilt dict that keeps `_arenas` keeps them too."""
    arenas = weights.get("_arenas")
    rec = arenas.get(KEY) if isinstance(arenas, dict) else None
    return rec if isinstance(rec, _Record) else None


def _parked_record(weights):
    """The record kept apart for a RELEASED table, None when this table was not released with ties: looked up by the
    table's id and matched by identity, so another dict reusing that id never takes its ties."""
    entry = _PARKED.get(id(weights))
    return entry[1] if entry is not None and entry[0] is weights else None


def record_of(weights):
    """The ties of a loaded table, or of a released one waiting for its reload; None without any."""
    return _held_record(weights) or _parked_record(weights)


def tied_params(weights) -> frozenset:
    """The entries of `weights` that are another table's tensor, aliased now or due at the next bind or reload."""
    rec = record_of(weights)
    return frozenset(rec.binds) if rec is not None else frozenset()


def _install(weights, rec, arenas=None):
    """Put `rec` in a table's bookkeeping (a new "_arenas" dict: the loader's own is never written), on top of
    `arenas` when given (the bookkeeping to keep), else the table's own."""
    base = weights.get("_arenas") if arenas is None else arenas
    weights["_arenas"] = {**(base or {}), KEY: rec}
    rec.loaded = True


def _record_for(weights):
    """The record of a table being bound: its own, its parked one taken back, or a new one installed."""
    rec = _held_record(weights)
    if rec is None:
        rec = _parked_record(weights)
        if rec is not None:
            del _PARKED[id(weights)]
        else:
            rec = _Record()
        _install(weights, rec)
    return rec


def _arenas_holding(weights, tensor) -> tuple:
    """The arenas of a table's bookkeeping whose block holds `tensor`'s address: what a weight placed by the loader
    needs alive, since it holds no reference to them. None for a tensor that owns its memory (it keeps it)."""
    ptr = tensor.data_ptr()
    found = []
    for name, arena in (weights.get("_arenas") or {}).items():
        base, total = getattr(arena, "_base_ptr", 0), getattr(arena, "_total", 0)
        if name != KEY and base and base <= ptr < base + total:
            found.append(arena)
    return tuple(found)


def _keeper(weights, tensor, param) -> tuple:
    """What a tie of `param` must hold to keep `tensor`'s memory alive: the arenas of the table holding its address,
    else nothing when the tensor (or the allocation it views) owns its memory or keeps an owner as its base. A tensor
    that owns nothing and lies in no arena of its table is refused by name: a reference to it would keep nothing
    alive, and the alias would read freed memory once the table that does hold it lets go."""
    arenas = _arenas_holding(weights, tensor)
    if arenas:
        return arenas
    root = tensor.storage_root()
    if root._owns_data or root._base is not None:
        return ()
    raise TiedWeightError(f"bind_tied: the source of '{param}' owns no memory and no arena of its table holds its "
                          f"address {tensor.data_ptr():#x}: the tie cannot keep it alive. Bind from the table the "
                          f"loader filled (its \"_arenas\" with it), not from a copy of its entries")


def _spec_differs(name, what, mine, theirs):
    """The refusal of a tie whose `what` (shape or dtype) differs, naming the entry and both values: a tie reads the
    source in place, so a different layout would read wrong memory."""
    return TiedWeightError(f"bind_tied: '{name}' {what} {mine} differs from its source's {theirs}; a tie binds a "
                           f"weight of the same {what} only")


def _dtype_of(d):
    """A dtype given by name ("bfloat16") read as the library's NBXDtype, any other value as given: so a caller may
    state the param's dtype as its graph writes it and still be compared with the source's."""
    if isinstance(d, str):
        from nbxtensor.dtype.nbx_dtype import parse_dtype
        return parse_dtype(d)
    return d


def _purge():
    """Released tables whose record ties nothing any more are forgotten (their empty dict and record dropped): what a
    re-bind to a new source dict or a dropped consumer leaves behind does not pile up request after request."""
    for k, (_, rec) in list(_PARKED.items()):
        if not rec.binds and not len(rec.sourced):
            del _PARKED[k]


def bind_tied(weights: dict, param: str, source_weights: dict, source_key: str, *, shape=None, dtype=None):
    """Make `weights[param]` the tensor `source_weights[source_key]` (another table's loaded weight, or this table's
    own): the same tensor, no copy, no allocation; the source's arenas kept alive while the alias is loaded. Refused by
    name (TiedWeightError) when the source holds no such key, when it is not a library tensor, or when its shape or
    dtype differs from the entry already at `param` (the stored copy) or from `shape` / `dtype` when given. Binding
    again (each request, after a reload) re-binds to what the source holds now. Returns the source tensor."""
    from nbxtensor.tensor import NBXTensor
    if source_key not in source_weights:
        raise TiedWeightError(f"bind_tied: the source of '{param}' holds no weight '{source_key}'")
    source = source_weights[source_key]
    if not isinstance(source, NBXTensor):
        raise TiedWeightError(f"bind_tied: the source of '{param}' is not a library tensor ({type(source).__name__})")
    held = weights.get(param)
    if held is not None and held is not source:
        if tuple(getattr(held, "shape", ())) != tuple(source.shape):
            raise _spec_differs(param, "shape", tuple(getattr(held, "shape", ())), tuple(source.shape))
        if getattr(held, "dtype", None) != source.dtype:
            raise _spec_differs(param, "dtype", getattr(held, "dtype", None), source.dtype)
    if shape is not None and tuple(shape) != tuple(source.shape):
        raise _spec_differs(param, "shape", tuple(shape), tuple(source.shape))
    if dtype is not None and _dtype_of(dtype) != source.dtype:
        raise _spec_differs(param, "dtype", _dtype_of(dtype), source.dtype)
    offer = (source, _keeper(source_weights, source, param))
    cons = _record_for(weights)
    src = cons if source_weights is weights else _record_for(source_weights)
    tie = cons.binds.get(param)
    if tie is None or tie.source() is not src or tie.key != source_key:
        old = tie.source() if tie is not None else None
        if old is not None and tie is not None:
            old.sourced.discard(tie)
        tie = cons.binds[param] = _Tie(param, source_key, src)
    for other in list(src.sourced):         # a released copy of this consumer tied the same: superseded by this bind
        if other is not tie and other.key == source_key and other.param == param and other.alias is None:
            src.sourced.discard(other)
            for _, rec in list(_PARKED.values()):
                if rec.binds.get(param) is other:
                    del rec.binds[param]
    src.sourced.add(tie)
    tie.offer = tie.alias = offer
    weights[param] = source
    _purge()
    return source


def _bytes_of(t, seen) -> int:
    """A tensor's bytes, its allocation counted once across the call (`seen` holds the ids already counted)."""
    from nbxtensor.tensor import NBXTensor
    if isinstance(t, NBXTensor):
        root = t.storage_root()
        if id(root) in seen:
            return 0
        seen.add(id(root))
        return root.nbytes()
    return t.nbytes() if callable(getattr(t, "nbytes", None)) else 0


def weight_bytes(*tables) -> int:
    """The bytes the tables hold, each allocation counted once across them: their own weights (the bookkeeping
    "_arenas" is not a weight); a tied entry 0 while its source is loaded (the bytes are the source's), and once the
    source is released the arenas its alias keeps alive, counted once where they are."""
    seen, kept, total = set(), set(), 0
    for weights in tables:
        rec = record_of(weights)
        binds = rec.binds if rec is not None else {}
        for name, t in weights.items():
            if name == "_arenas" or t is None:
                continue
            tie = binds.get(name)
            if tie is None:
                total += _bytes_of(t, seen)
                continue
            src = tie.source()
            if tie.alias is None or (src is not None and src.loaded):
                continue
            tensor, arenas = tie.alias
            if not arenas:
                total += _bytes_of(tensor, seen)
            for arena in arenas:
                if id(arena) not in kept:
                    kept.add(id(arena))
                    total += arena.capacity
    return total


def before_release(weights: dict):
    """`release_weights`'s first step on a library table: its record taken out of its bookkeeping (so the arenas
    the release reads are the table's own), nothing dropped yet. Returns the record, None without one."""
    arenas = weights.get("_arenas")
    rec = arenas.pop(KEY, None) if isinstance(arenas, dict) else None
    return rec if isinstance(rec, _Record) else None


def after_release(weights: dict, rec) -> None:
    """`release_weights`'s last step, once the devices are synchronised and the table cleared: as a consumer, its
    aliases dropped (the source arenas they kept go if they were the last); as a source, its name withdrawn from
    every tie (a loaded consumer keeps its alias and arenas); the record kept for the table's reload."""
    if rec is None:
        return
    for tie in rec.binds.values():
        tie.alias = None
    for tie in list(rec.sourced):
        tie.offer = None
    rec.loaded = False
    _PARKED[id(weights)] = (weights, rec)
    _purge()


def rejoin(held: dict, joined: dict) -> dict:
    """`join_weights`'s last step: `joined` (what `held` is updated with, or replaced by on a reload) with `held`'s
    record in its bookkeeping. As a source, each tie learns the tensor now at its key; as a consumer, each tie whose
    source offers a tensor aliases it (a stored copy the loader read anyway replaced), a tie whose source is not
    loaded waits, and a tie whose source can never come back is refused by name, `held` left as it was."""
    rec = _held_record(held)
    parked = rec is None
    if parked:
        rec = _parked_record(held)
    if rec is None:
        return joined
    gone = sorted(p for p, tie in rec.binds.items() if tie.source() is None)
    if gone:
        raise TiedWeightError(f"bind_tied: the source of {gone} can never come back (its table was dropped without "
                              f"a release); load the source component and bind the tie again")
    if parked:
        del _PARKED[id(held)]
    out = dict(joined)
    _install(out, rec, {**(held.get("_arenas") or {}), **(joined.get("_arenas") or {})})
    view = {**held, **out}
    for tie in list(rec.sourced):
        t = view.get(tie.key)
        if t is not None and tie.key not in rec.binds:
            tie.offer = (t, _keeper(view, t, tie.param))
    for param, tie in rec.binds.items():
        if tie.offer is not None:
            tie.alias = tie.offer
            out[param] = tie.offer[0]
    return out
