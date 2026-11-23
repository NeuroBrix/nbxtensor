"""The synthetic launches an element-wise variant's tile is proven on (`ops/ew_proof.py`): a function of the VARIANT
alone, never of the launch that first met it.

A variant is one binary: the kernel, its specialization (each pointer's dtype and 16-byte alignment, each integer's
width and divisibility by 16 or its constant 1, every constexpr) and its options. Everything a synthetic launch is
made of is drawn from that and from the kernel's signature:

  * THE LADDER of counts, fixed, every one holding masked tails (`ladder`): with s and L the smallest and the largest
    tile of the class, n = 1, s - 1, L - 1, L + 1, 2L + 3 — each kept in the count's own specialization: a count
    specialised divisible by 16 ('D') walks 16, s - 16, L - 16, L + 16, 2L + 48; one specialised not divisible by 16
    walks 2 (1 is Triton's constant 1, another binary), s - 1, L - 1, L + 1, L + 8 (its buffers of 2-byte elements
    and wider end exactly at the guard: no slack), 2L + 3; a count the variant fixed (its
    constant 1) is that count alone. A count the variant has 64-bit (a launch past 2^31 elements) walks the same
    small counts as 64-bit integers (`Int64`): the same wide binary, met in kilobytes.
  * LAYOUTS drawn from the rank: contiguous, and a permuted one (the operand's memory in the reverse dimension order:
    a transposed view, the same extent) — for kernels that read an operand through shape and stride arrays.
  * BUFFERS of each pointer's dtype, of the extent the recipe gives, filled deterministically (`fill_bytes`):
    finite values of the dtype, or indices below a bound, or the shape and stride arrays themselves.
  * SCALARS drawn deterministically from the parameter's name, an integer honouring its specialization.

How a kernel's arguments become sizes — the RECIPE:

  * GENERIC (no declaration): a signature with exactly ONE runtime integer, the count. Every pointer is a buffer of
    the count's elements, every float a drawn scalar, the grid `cdiv(count, tile)` on one axis.
  * DECLARED next to the kernel (`ops/ew_tile.synthetic(recipe)`): `flat(count, ...)` names the count among several
    integers and any pointer whose extent is not the count; `strided_nd(...)` is the shape-and-stride family; any
    other kernel declares its own function of a `Synth` (each argument, the grid's axes).
  * A kernel that is not generic and declares nothing is REFUSED by name ("not provable on synthetic inputs: declare
    ..."), never proven on a launch's data.

Each synthetic launch is checked against the variant: its arguments must specialise exactly as the variant does (the
same binary), or the recipe is refused by name. The cost is kilobytes to megabytes of synthetic buffers, the same
for every launch of the variant.
"""
from __future__ import annotations

import hashlib
import random
import struct
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: the proof method: part of the proof table's stamp (`ew_proof.stamp`), so a table made by another method is not read
METHOD = "synthetic-guarded-2"
UNIT = 16                                  # Triton's divisibility unit ('D')
LAYOUTS = ("contiguous", "permuted")
_INT_TYPES = ("i32", "i64", "u64", "i8", "i16", "u8", "u16", "u32")


class Int64(int):
    """An integer argument declared 64-bit whatever its value (`nbx_int64`, read by the launcher's binder): a small
    count of a variant specialised i64 — the same binary a launch past 2^31 elements compiles."""
    nbx_int64 = True

    def __reduce__(self):
        """Pickles as itself (the proof's worker rebuilds it)."""
        return (Int64, (int(self),))

    def __repr__(self):
        """Prints as the plain int it holds, so a key or a log line reads the same number."""
        return int.__repr__(self)

    __str__ = __repr__


def width(ty: str, v):
    """`v` as the parameter of type `ty` must carry it: an `Int64` for an i64 parameter, else as it is."""
    return Int64(v) if ty == "i64" and isinstance(v, int) and not isinstance(v, bool) else v


class NotSynthesisable(RuntimeError):
    """A variant no synthetic launch can be made for (by name): no recipe, a 64-bit count, a recipe that does not
    specialise as the variant."""


# ---------------------------------------------------------------------------
# the ladder
# ---------------------------------------------------------------------------
def ladder(attr, cands) -> List[int]:
    """The counts a variant's count parameter walks, from its specialization `attr` ('D', '', or ('constexpr', v))
    and the class's candidates (block, warps, stages): every one holds a masked tail for some tile, the first below
    every tile, the last past two of the largest. Each layout meets a count-sized buffer with ZERO slack before its
    guard (`memory.guarded`): every 'D' count does; for a '' count the rung L + 8 does (a multiple of 8, not of 16:
    its bytes are a multiple of 16 for every element of 2 bytes or more)."""
    blocks = sorted({int(c[0]) for c in cands})
    s, big = blocks[0], blocks[-1]
    if isinstance(attr, tuple):
        return [int(attr[1])]
    if attr == "D":
        return [UNIT, s - UNIT, big - UNIT, big + UNIT, 2 * big + 3 * UNIT]
    return [2, s - 1, big - 1, big + 1, big + UNIT // 2, 2 * big + 3]


def fit(attr, near: int) -> int:
    """An integer at or above `near` (at least 2) that specialises as `attr`: a multiple of 16 for 'D', a value that
    is neither 1 nor a multiple of 16 for '', the variant's own constant for ('constexpr', v)."""
    if isinstance(attr, tuple):
        return int(attr[1])
    v = max(2, int(near))
    if attr == "D":
        return -(-v // UNIT) * UNIT
    return v + 1 if v % UNIT == 0 else v


# ---------------------------------------------------------------------------
# a synthetic launch
# ---------------------------------------------------------------------------
class Buf:
    """A synthetic buffer: its parameter, pointer type ('*fp32'), elements, element bytes, 16-byte alignment, and how
    it is filled: ('data',) finite values of the dtype, ('index', high) integers in [0, high), ('values', tuple) the
    values themselves."""
    __slots__ = ("name", "ty", "elements", "elem", "aligned", "fill", "seed")

    def __init__(self, name, ty, elements, elem, aligned, fill, seed):
        """A buffer of `elements` elements of type `ty` for parameter `name`."""
        self.name, self.ty, self.elements, self.elem = name, ty, int(elements), int(elem)
        self.aligned, self.fill, self.seed = bool(aligned), tuple(fill), str(seed)

    @property
    def nbytes(self) -> int:
        """Its bytes (at least one element)."""
        return max(self.elements, 1) * self.elem

    def key(self) -> tuple:
        """What makes two buffers the same buffer (for dropping duplicate launches)."""
        return (self.name, self.ty, self.elements, self.aligned, self.fill)


class Launch:
    """One synthetic launch: {parameter: Buf | value} for every runtime parameter and constexpr, and the grid's
    axes (ints, or `elementwise_launch.per_tile(n)`)."""
    __slots__ = ("args", "axes", "count", "layout")

    def __init__(self, args: Dict[str, Any], axes: tuple, count: int, layout: str):
        """A launch of `args` on `axes`, made for ladder count `count` in `layout`."""
        self.args, self.axes, self.count, self.layout = args, tuple(axes), int(count), layout

    def bufs(self) -> List[Buf]:
        """Its buffers, in parameter order."""
        return [v for v in self.args.values() if isinstance(v, Buf)]

    def key(self) -> tuple:
        """What makes two launches the same launch."""
        return (tuple((k, v.key() if isinstance(v, Buf) else ("v", repr(v))) for k, v in self.args.items()),
                tuple((a.n, a.times) if hasattr(a, "times") else int(a) for a in self.axes))


_DTYPES: Dict[str, Any] = {}


def dtype_of(ty: str):
    """The library dtype whose pointer type Triton's binder spells `ty` ('*fp32' -> float32), and its bytes."""
    if not _DTYPES:
        from nbxtensor.backend.launcher import specialize_arg
        from nbxtensor.dtype.nbx_dtype import NBXDtype, dtype_size
        for d in NBXDtype:
            spelt = specialize_arg(_Fake(0, d, 1, 1), True, True)[0]
            _DTYPES.setdefault(spelt, (d, int(dtype_size(d))))
    hit = _DTYPES.get(ty)
    if hit is None:
        raise NotSynthesisable(f"no library dtype is spelt {ty!r} by the binder: its buffer cannot be made")
    return hit


class _Fake:
    """A pointer argument as the binder reads it (address, dtype, shape, strides, element size), for checking a
    synthetic launch's specialization without a device."""

    def __init__(self, address, dtype, elements, elem):
        """A fake pointer of `elements` elements at `address`."""
        self._a, self.dtype, self.shape, self._e = int(address), dtype, (int(elements),), int(elem)

    def data_ptr(self):
        """The stand-in's device address: the binder reads it for the pointer's alignment attribute ("D")."""
        return self._a

    def stride(self):
        """The stand-in is one contiguous row: stride 1, as the binder reads a flat pointer argument."""
        return (1,)

    def element_size(self):
        """The bytes of one element of the stand-in's dtype, which the binder reads with the address."""
        return self._e


class Synth:
    """What a recipe builds one synthetic launch from: the variant's attribute of each parameter, the ladder count
    `n`, the `layout`, and helpers that honour the specialization (`int`, `buf`, `index`, `values`, `scalar`,
    `shape`, `strided`)."""

    def __init__(self, kernel, specialization, count: int, layout: str):
        """The helper for ladder count `count` in `layout` of the variant `specialization` of `kernel`."""
        from nbxtensor.backend.launcher import _param_table
        self.kernel, self.n, self.layout = kernel, count if isinstance(count, Int64) else int(count), layout
        table = _param_table(kernel)
        self.names = [p[0] for p in table]
        self.declared_constexpr = {p[0] for p in table if p[1]}
        self.spec = dict(zip(self.names, specialization))
        self._qual = f"{getattr(kernel.fn, '__module__', '?')}.{kernel.__name__}"

    # -- reading the variant --------------------------------------------------
    def ty(self, name: str) -> str:
        """The parameter's type in the variant ('*fp32', 'i32', 'fp32', 'constexpr', ...)."""
        return str(self.spec[name][0])

    def attr(self, name: str):
        """The parameter's specialization: 'D' / '' for a specialised pointer or integer, None for an unspecialised
        one, ('constexpr', v) for a value the variant fixed."""
        t, a = self.spec[name]
        return ("constexpr", a) if t == "constexpr" else a

    def seed(self, name: str) -> str:
        """The deterministic seed of a parameter's draw: kernel, parameter, count, layout."""
        return f"{self._qual}:{name}:{self.n}:{self.layout}"

    # -- making arguments ----------------------------------------------------
    def int(self, name: str, near: int) -> int:
        """An integer for `name` at or above `near`, specialised as the variant specialises it (`fit`), 64-bit for
        an i64 parameter."""
        return width(self.ty(name), fit(self.attr(name), near))

    def buf(self, name: str, elements: int) -> Buf:
        """A buffer of `elements` finite values of the parameter's dtype."""
        return self._buf(name, elements, ("data",))

    def index(self, name: str, elements: int, high: int) -> Buf:
        """A buffer of `elements` integers in [0, `high`) of the parameter's dtype (indices that stay in bounds)."""
        return self._buf(name, elements, ("index", max(int(high), 1)))

    def values(self, name: str, vals: Sequence[int]) -> Buf:
        """A buffer holding exactly `vals` (a shape or a stride array)."""
        return self._buf(name, len(vals), ("values", tuple(int(v) for v in vals)))

    def _buf(self, name, elements, fill) -> Buf:
        """A buffer for pointer parameter `name`, aligned as the variant's pointer is."""
        ty = self.ty(name)
        if not ty.startswith("*"):
            raise NotSynthesisable(f"{self.kernel.__name__}: the recipe makes a buffer for {name}, which the variant "
                                   f"types {ty}")
        _d, elem = dtype_of(ty)
        return Buf(name, ty, elements, elem, self.attr(name) == "D", fill, self.seed(name))

    def scalar(self, name: str):
        """A deterministic scalar for `name`: the variant's constant, a float in [0.5, 2), a bool False, or a small
        integer honouring the specialization."""
        a = self.attr(name)
        if isinstance(a, tuple):
            return a[1]
        t = self.ty(name)
        rng = random.Random(self.seed(name))
        if t.startswith("fp") or t.startswith("bf"):
            return round(0.5 + 1.5 * rng.random(), 3)
        if t == "u1":
            return False
        if t in _INT_TYPES:
            if t not in ("i32", "i64"):
                raise NotSynthesisable(f"{self.kernel.__name__}: scalar {name} is specialised {t}: no small value "
                                       f"draws it")
            return width(t, fit(a, 2 + rng.randrange(7)))
        raise NotSynthesisable(f"{self.kernel.__name__}: scalar {name} of type {t} has no draw")

    def shape(self, rank: int, count: Optional[int] = None) -> List[int]:
        """`rank` extents whose product is `count` (default the ladder's): its prime factors dealt from the innermost
        dimension outward, so every dimension above one has an index to walk."""
        n = self.n if count is None else int(count)
        dims = [1] * max(int(rank), 1)
        f, rest, i = 2, n, 0
        factors = []
        while f * f <= rest:
            while rest % f == 0:
                factors.append(f)
                rest //= f
            f += 1
        if rest > 1:
            factors.append(rest)
        for p in sorted(factors, reverse=True):
            dims[len(dims) - 1 - (i % len(dims))] *= p
            i += 1
        return dims

    def strided(self, dims: Sequence[int]) -> Tuple[List[int], int]:
        """(strides, extent in elements) of an operand of logical extents `dims` in this launch's layout:
        contiguous (row-major), or permuted (its memory in the reverse dimension order: a transposed view)."""
        order = list(range(len(dims)))
        if self.layout == "permuted":
            order = order[::-1]
        strides = [0] * len(dims)
        step = 1
        for d in reversed(order):
            strides[d] = step
            step *= int(dims[d])
        extent = 1 + sum((int(d) - 1) * s for d, s in zip(dims, strides))
        return strides, extent


# ---------------------------------------------------------------------------
# recipes
# ---------------------------------------------------------------------------
def _runtime_ints(S: Synth) -> List[str]:
    """The runtime integer parameters of the variant (a constant 1 included: the binder made an integer of 1 so)."""
    out = []
    for name in S.names:
        if name in S.declared_constexpr:
            continue
        t, a = S.spec[name]
        if str(t) in _INT_TYPES or (str(t) == "constexpr" and isinstance(a, int) and not isinstance(a, bool)
                                    and not _is_tile(a)):
            out.append(name)
    return out


def _is_tile(v) -> bool:
    """Whether `v` is an element-wise class tile (`ew_tile.EwTile`)."""
    from nbxtensor.ops.ew_tile import EwTile
    return isinstance(v, EwTile)


def flat(count: str, **extents) -> Callable:
    """The recipe of a flat kernel: `count` is the count, every pointer a buffer of `count` elements but those named
    in `extents` ({parameter: elements, or a function of the Synth}), every other integer a drawn scalar. Names
    what the signature alone does not say: which integer counts."""
    def recipe(S: Synth):
        """A flat launch at the ladder's count."""
        args: Dict[str, Any] = {count: S.n}
        for name in _runtime_ints(S):
            if name != count:
                args[name] = S.scalar(name)
        for name, ext in extents.items():
            args[name] = S.buf(name, ext(S) if callable(ext) else int(ext))
        return args, None
    recipe.count = count
    recipe.description = f"flat({count}{', ' + ', '.join(f'{k}={v}' for k, v in extents.items()) if extents else ''})"
    return recipe


def strided_nd(count: str, shape: str, strided: Dict[str, str], ndim: str = "NDIM") -> Callable:
    """The recipe of the shape-and-stride family: `count` elements of logical extents `shape` (an int64 array of
    `ndim` entries); each operand named in `strided` ({pointer: its stride array}) is read or written through its
    strides in the launch's layout; every other pointer is contiguous over the count."""
    def recipe(S: Synth):
        """A shape-and-stride launch at the ladder's count, in the Synth's layout."""
        rank = int(S.attr(ndim)[1])
        dims = S.shape(rank)
        args: Dict[str, Any] = {count: S.n, shape: S.values(shape, dims)}
        for ptr, st in strided.items():
            strides, extent = S.strided(dims)
            args[st] = S.values(st, strides)
            args[ptr] = S.buf(ptr, extent)
        return args, None
    recipe.count = count
    recipe.layouts = LAYOUTS
    recipe.description = f"strided_nd({count}, {shape}, {strided})"
    return recipe


def generic(S: Synth):
    """The recipe a signature determines: exactly one runtime integer, the count; refused by name otherwise."""
    ints = _runtime_ints(S)
    if len(ints) != 1:
        raise NotSynthesisable(
            f"{S.kernel.__name__} is not provable on synthetic inputs: its signature holds {len(ints)} runtime "
            f"integers ({ints}), so which one counts the elements and how long each pointer is are not determined; "
            f"declare its recipe next to its definition (`ops/ew_tile.synthetic(ew_synthetic.flat(...))`, or a "
            f"function of `ew_synthetic.Synth`)")
    return {ints[0]: S.n}, None


generic.count = None
generic.description = "generic"


def recipe_of(kernel) -> Callable:
    """The kernel's declared recipe (`ew_tile.synthetic`), else the generic one."""
    from nbxtensor.ops.ew_tile import declared_recipe
    return declared_recipe(kernel) or generic


def _count_param(kernel, specialization, recipe) -> Optional[str]:
    """The count parameter the ladder walks: the recipe's own, or the generic one."""
    if getattr(recipe, "count", None):
        return recipe.count
    S = Synth(kernel, specialization, 0, LAYOUTS[0])
    ints = _runtime_ints(S)
    if len(ints) != 1:
        generic(S)                                  # raises its named refusal
    return ints[0]


# ---------------------------------------------------------------------------
# helpers of declared recipes
# ---------------------------------------------------------------------------
_DIMS_MEMO: dict = {}


def walk(attr):
    """Small values in a specialization, smallest first: the constant itself, multiples of 16 ('D'), values neither
    1 nor a multiple of 16 (''), any small size for a dimension that is no parameter (None)."""
    if isinstance(attr, tuple):
        return [int(attr[1])]
    if attr == "D":
        return [16, 32, 48, 64]
    if attr == "":
        return [v for v in range(2, 20) if v % 16]
    return list(range(1, 17)) + [32, 48, 64]


def holds(attr, v) -> bool:
    """Whether integer `v` specialises as `attr` (Triton's binder: the constant 1, 'D' divisible by 16, '' neither)."""
    if isinstance(attr, tuple):
        return v == attr[1]
    if attr == "D":
        return v % 16 == 0
    if attr == "":
        return v != 1 and v % 16 != 0
    return True


def solve_dims(S, free, derive):
    """Dimensions for a synthetic launch whose sizes are tied by products (an output extent = an input extent x a
    factor, a contiguous stride = a product of extents): the first assignment of the `free` names (each walking small
    values of its own specialization, `walk`; a name that is no parameter walks any small size) for which
    `derive(values)` -> {parameter: value} (None to skip the assignment) gives every parameter it names the
    variant's own specialization. Memoised per variant (the dimensions do not depend on the ladder's count); raises
    NotSynthesisable naming the variant when no small assignment exists."""
    from itertools import product
    key = (getattr(S.kernel, "__name__", "?"), tuple((k, str(t), repr(a)) for k, (t, a) in S.spec.items()))
    hit = _DIMS_MEMO.get(key)
    if hit is not None:
        return dict(hit)
    walks = [walk(S.attr(name) if name in S.spec else None) for name in free]
    for combo in product(*walks):
        full = derive(dict(zip(free, combo)))
        if full is not None and all(holds(S.attr(k), v) for k, v in full.items() if k in S.spec):
            _DIMS_MEMO[key] = full
            return dict(full)
    raise NotSynthesisable(f"{S.kernel.__name__}: no small dimensions over {free} specialise every derived size as "
                           f"the variant does ({[(k, S.attr(k)) for k in S.names if k in S.spec]})")


def reach(offsets) -> int:
    """1 + the largest of the element offsets the lanes address (a numpy array), 0 when none does: an exact extent."""
    return int(offsets.max()) + 1 if len(offsets) else 0



# ---------------------------------------------------------------------------
# the synthetic job
# ---------------------------------------------------------------------------
def synthesise(kernel, specialization, cands, recipe=None) -> List[Launch]:
    """Every synthetic launch of a variant: the recipe at every count of the ladder and every layout it reads, each
    completed (every pointer it did not name a buffer of the count, every float or bool a drawn scalar, every
    constexpr the variant's own) and checked to specialise exactly as the variant; duplicates dropped. Raises
    NotSynthesisable by name."""
    recipe = recipe or recipe_of(kernel)
    cname = _count_param(kernel, specialization, recipe)
    probe = Synth(kernel, specialization, 0, LAYOUTS[0])
    ctype = probe.ty(cname) if cname in probe.spec else "?"
    if ctype not in ("i32", "i64", "constexpr"):
        raise NotSynthesisable(f"{kernel.__name__}: its count {cname} is specialised {ctype!r}: no ladder of counts "
                               f"is that type; not provable on synthetic inputs")
    counts = [width(ctype, n) for n in ladder(probe.attr(cname), cands)]
    out: List[Launch] = []
    seen = set()
    for n in counts:
        for layout in getattr(recipe, "layouts", LAYOUTS[:1]):
            S = Synth(kernel, specialization, n, layout)
            args, axes = recipe(S)
            launch = _complete(S, dict(args), axes)
            check(kernel, specialization, launch)
            if launch.key() not in seen:
                seen.add(launch.key())
                out.append(launch)
    return out


def _complete(S: Synth, args: Dict[str, Any], axes) -> Launch:
    """A recipe's arguments completed in parameter order: a pointer it did not name a buffer of the count, a float or
    bool a drawn scalar, a constexpr (declared, or fixed by the variant) the variant's own value; an integer it did
    not name is refused (an integer may be a size)."""
    from nbxtensor.ops.elementwise_launch import per_tile
    full: Dict[str, Any] = {}
    for name in S.names:
        t, a = S.spec[name]
        if name in args:
            full[name] = args[name]
        elif str(t) == "constexpr":
            full[name] = a
        elif str(t).startswith("*"):
            full[name] = S.buf(name, S.n)
        elif str(t) in _INT_TYPES:
            raise NotSynthesisable(f"{S.kernel.__name__}: the recipe gives no value for integer {name} (a size or a "
                                   f"scalar: it must say which)")
        else:
            full[name] = S.scalar(name)
    return Launch(full, axes if axes is not None else (per_tile(S.n),), S.n, S.layout)


def _normal(spec) -> List[Tuple[str, str]]:
    """A specialization as the variant id spells it: a class tile by its role, everything else (type, repr(attr))."""
    return [("tile", v.role) if _is_tile(v) else (str(t), repr(v)) for t, v in spec]


def check(kernel, specialization, launch: Launch) -> None:
    """Refuse by name a synthetic launch whose arguments do not specialise exactly as the variant: another binary."""
    from nbxtensor.backend.launcher import nbx_binder
    args, base = [], 0x100000
    for name, v in launch.args.items():
        if isinstance(v, Buf):
            d, elem = dtype_of(v.ty)
            base += 1 << 20
            args.append(_Fake(base if v.aligned else base + elem, d, v.elements, elem))
        else:
            args.append(v)
    _bound, spec, _opts = nbx_binder(kernel, tuple(args), {})
    got, want = _normal(spec), _normal(specialization)
    if got != want:
        diff = [(n, g, w) for n, g, w in zip(launch.args, got, want) if g != w]
        raise NotSynthesisable(f"{kernel.__name__}: its synthetic launch at count {launch.count} ({launch.layout}) "
                               f"specialises otherwise than the variant {diff}: the recipe makes another binary")


# ---------------------------------------------------------------------------
# the bytes of a buffer
# ---------------------------------------------------------------------------
def fill_bytes(buf: Buf) -> bytes:
    """A buffer's bytes, deterministic from its seed: its values, its indices, or finite values of its dtype (floats
    in [-4, 4), signed integers in [-8, 8), unsigned in [0, 16), bools 0 / 1)."""
    n = max(buf.elements, 1)
    t = buf.ty[1:]
    rng = random.Random(hashlib.sha256(buf.seed.encode()).hexdigest())
    kind = buf.fill[0]
    if kind == "values":
        vals = list(buf.fill[1]) + [0] * (n - len(buf.fill[1]))
    elif kind == "index":
        vals = [rng.randrange(buf.fill[1]) for _ in range(n)]
    elif t.startswith("fp") or t.startswith("bf"):
        vals = [rng.randrange(-2000, 2000) / 500.0 for _ in range(n)]
    elif t == "i1":
        vals = [rng.randrange(2) for _ in range(n)]
    elif t.startswith("u"):
        vals = [rng.randrange(16) for _ in range(n)]
    else:
        vals = [rng.randrange(-8, 8) for _ in range(n)]
    return _pack(t, vals, buf.elem)


_PACK = {"fp32": "f", "fp64": "d", "fp16": "e", "i8": "b", "i16": "h", "i32": "i", "i64": "q", "u8": "B",
         "u16": "H", "u32": "I", "u64": "Q", "i1": "B"}


def _pack(t: str, vals, elem: int) -> bytes:
    """`vals` packed little-endian as elements of type `t` (`elem` bytes each); bf16 as the high half of fp32; an
    8-bit float as its raw byte drawn from the value."""
    if t == "bf16":
        return b"".join(struct.pack("<f", float(v))[2:] for v in vals)
    code = _PACK.get(t)
    if code is None:
        if elem == 1:
            return bytes((int(v * 8) & 0x7F) for v in vals)
        raise NotSynthesisable(f"no fill for elements of type {t}")
    if code in "fde":
        return struct.pack(f"<{len(vals)}{code}", *(float(v) for v in vals))
    return struct.pack(f"<{len(vals)}{code}", *(int(v) for v in vals))
