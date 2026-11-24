"""Tied weights: a weight table's entry made an ALIAS of another component's loaded tensor, kept across the
consumer's unload and reload.

A weight table is the application's dict {name: tensor} with the library's bookkeeping under "_arenas" (the device
blocks its weights point into). `bind_tied(weights, param, source)` puts `source` itself at `param` (no copy, no
allocation) and records the tie in that bookkeeping, `weights["_arenas"]["_tied"]`: the record holds the source by
reference, as a view holds its allocation, so it is memory bookkeeping like the arenas beside it, and a table's
readers that already skip "_arenas" never meet it.

Lifetime (the library's: an allocation lives while a reference to it does, `NBXTensor._base`, `ComponentArena.__del__`):
  * the consumer unloaded (`release_weights`): its table is cleared like any other, the record kept apart for that
    table (it still references the source, so the reload aliases the same storage);
  * the consumer reloaded (`join_weights(released_table, loaded)`, the library's reload door): every tied entry is
    the source's tensor again, a stored copy the loader read anyway replaced by it; `tied_params` names the entries a
    loader need not read at all;
  * the source unloaded while the consumer holds the alias: the source stays alive by that reference, freed when the
    consumer's table lets go of it;
  * the source unloaded after the consumer was: the record drops the source (a released consumer keeps nothing of a
    released component alive), and the consumer's reload is refused by name: load the source and bind again.
No torch; the tensors are the library's (NBXTensor)."""
import weakref

KEY = "_tied"                           # the record's place in a table's "_arenas"


class TiedWeightError(ValueError):
    """A tie that cannot be made or restored; the message names the entry and why."""


class _Ties:
    """One table's ties: {param: source tensor, or None once the source was released under a released consumer}."""
    __slots__ = ("binds", "orphaned", "__weakref__")

    def __init__(self):
        self.binds = {}
        self.orphaned = set()           # params whose source's table was released while this table held the alias


_LIVE = weakref.WeakSet()               # every record, to tell it when a table holding its source is released
_PARKED = {}                            # id(released table) -> (that table, its record): the ties of an unloaded table


def _held_record(weights):
    arenas = weights.get("_arenas")
    rec = arenas.get(KEY) if isinstance(arenas, dict) else None
    return rec if isinstance(rec, _Ties) else None


def _parked_record(weights):
    entry = _PARKED.get(id(weights))
    return entry[1] if entry is not None and entry[0] is weights else None


def record_of(weights):
    """The ties of a loaded table, or of a released one waiting for its reload; None without any."""
    return _held_record(weights) or _parked_record(weights)


def tied_params(weights) -> frozenset:
    """The entries of `weights` that are another component's tensor (loaded, or due again at the table's reload)."""
    rec = record_of(weights)
    return frozenset(rec.binds) if rec is not None else frozenset()


def _spec_differs(name, what, mine, theirs):
    return TiedWeightError(f"bind_tied: '{name}' {what} {mine} differs from its source's {theirs}; a tie binds a "
                           f"weight of the same {what} only")


def _dtype_of(d):
    if isinstance(d, str):
        from nbxtensor.dtype.nbx_dtype import parse_dtype
        return parse_dtype(d)
    return d


def bind_tied(weights: dict, param: str, source, *, shape=None, dtype=None):
    """Make `weights[param]` the tensor `source` (another component's loaded weight): the same tensor, no copy, no
    allocation, 0 bytes for this table (`weight_bytes`). Refused by name (TiedWeightError) when `source` is not a
    library tensor, or when its shape or dtype differs from the entry already at `param` (the stored copy) or from
    `shape` / `dtype` when given. The tie is kept across the table's `release_weights` and `join_weights` (module
    docstring). Returns `source`."""
    from nbxtensor.tensor import NBXTensor
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
    rec = _held_record(weights)
    if rec is None:
        parked = _PARKED.pop(id(weights), None)
        rec = parked[1] if parked is not None and parked[0] is weights else _Ties()
    rec.binds[param] = source
    rec.orphaned.discard(param)
    weights[param] = source
    arenas = weights.get("_arenas")
    weights["_arenas"] = {**(arenas or {}), KEY: rec}
    _LIVE.add(rec)
    return source


def weight_bytes(weights: dict) -> int:
    """The bytes a table's own weights hold: each allocation counted once (a view by its allocation), the tied entries
    0 (they are their source's), the bookkeeping ("_arenas") not a weight."""
    from nbxtensor.tensor import NBXTensor
    tied = tied_params(weights)
    seen, total = set(), 0
    for name, t in weights.items():
        if name == "_arenas" or name in tied or t is None:
            continue
        if isinstance(t, NBXTensor):
            root = t.storage_root()
            if id(root) not in seen:
                seen.add(id(root))
                total += root.nbytes()
        elif callable(getattr(t, "nbytes", None)):
            total += t.nbytes()
    return total


def before_release(weights: dict):
    """`release_weights`'s first step on a library table: its record taken out of its bookkeeping, and every other
    record whose source is one of this table's tensors told so (a released consumer drops it; a loaded one keeps
    the alias, the source alive by it). Returns the table's own record, None without one."""
    from nbxtensor.tensor import NBXTensor
    arenas = weights.get("_arenas")
    rec = arenas.pop(KEY, None) if isinstance(arenas, dict) else None
    rec = rec if isinstance(rec, _Ties) else None
    own = set(rec.binds) if rec is not None else set()
    roots = set()
    for name, t in weights.items():
        if name not in own and isinstance(t, NBXTensor):
            roots.update((id(t), id(t.storage_root())))
    if roots:
        parked = {id(r) for _, r in _PARKED.values()}
        for other in list(_LIVE):
            if other is rec:
                continue
            for p, s in list(other.binds.items()):
                if s is not None and (id(s) in roots or id(s.storage_root()) in roots):
                    if id(other) in parked:
                        other.binds[p] = None
                    else:
                        other.orphaned.add(p)
    return rec


def after_release(weights: dict, rec) -> None:
    """`release_weights`'s last step: the cleared table's record kept for its reload; a source whose table was
    released meanwhile is forgotten (nothing released stays alive behind a released consumer)."""
    if rec is None:
        return
    for p in rec.orphaned:
        rec.binds[p] = None
    rec.orphaned.clear()
    _PARKED[id(weights)] = (weights, rec)


def rejoin(held: dict, joined: dict) -> dict:
    """`join_weights`'s last step: `joined` (what `held` is updated with, or replaced by on a reload) with every tie of
    `held` restored, the source's tensor at each tied entry (a stored copy the loader read anyway replaced), the
    record in its bookkeeping. A tie whose source was released is refused by name, `held` left as it was."""
    rec = _held_record(held)
    parked = rec is None
    if parked:
        rec = _parked_record(held)
    if rec is None:
        return joined
    gone = sorted(p for p, s in rec.binds.items() if s is None)
    if gone:
        raise TiedWeightError(f"bind_tied: the source of {gone} was released after this table was: the reload "
                              f"cannot alias it; load the source component and bind the tie again")
    if parked:
        del _PARKED[id(held)]
    out = {**joined, **rec.binds}
    out["_arenas"] = {**(joined.get("_arenas") or {}), KEY: rec}
    return out
