"""The element-wise class's tile, MARKED, and what each kernel of the class declares about it.

`elementwise_launch.ew_meta` hands one tile and warps to every element-wise kernel of a size class. The setting is
swept on a plain copy; a setting proven on the copy is not proven on the kernel it is applied to (PixArt-XL on a
V100, 23/11/2026: BLOCK_SIZE 4096 at 2 warps, swept on the copy, faulted with an illegal address in
`mul_scalar_strided_nd_kernel` fp32 — the assembler's -O3 build of that tile; correct at -O0). So the tile is
handed out as an `EwTile`: an int of the same value that remembers its class and its role. It survives every way a
call site uses it (`**ew_meta(n)`, `meta['BLOCK_SIZE']`, `BLOCK_X=BLOCK`, `cdiv(n, BLOCK)`; only an explicit
`int()` drops it), and the ONE launcher sees it where a binary is first resolved, so no kernel of the class reaches
the device on an unproven tile (`ops/ew_proof.py`).

An `EwTile` hashes apart from the plain int of its value: a launch carrying it keys its own entry in the launcher's
cache, so the proof runs where that entry is first made even when the same binary was compiled by a launch that
carries no class tile. It compares equal to the int (Triton reads it as the number it is).

Each kernel of the class DECLARES, where it is defined, whether its result is independent of the tile
(`tile_independent`: each output element computed alone from its inputs) or not (`tile_dependent(why)`: two lanes
may write one address, an accumulate, a result defined by order). The proof compares the bits of the candidates'
results only for a kernel that declares independence; a dependent one is proven by the absence of a fault alone,
and the proof's record names it.
"""
from __future__ import annotations

_SALT = 0x5EED_E11E


class EwTile(int):
    """An element-wise class's tile or warps: `int(value)`, plus the class (`ew_class`) and the role
    (`'tile'` | `'warps'`) it was resolved for."""
    # no __slots__: CPython refuses a nonempty one on a subtype of int (a variable-size builtin); the instance
    # dict carries the class and the role

    def __new__(cls, value: int, ew_class: int, role: str):
        """An int of `value` remembering its class and role."""
        self = super().__new__(cls, int(value))
        self.ew_class = int(ew_class)
        self.role = role
        return self

    def __hash__(self):
        """Hashes apart from the plain int, so a launch carrying it keys its own launcher entry."""
        return int.__hash__(self) ^ _SALT

    def __eq__(self, other):
        """Equal to the int of the same value."""
        return int.__eq__(self, other)

    def __ne__(self, other):
        """Not equal as the int is not."""
        return int.__ne__(self, other)

    def __repr__(self):
        """Prints as the int."""
        return int.__repr__(self)

    __str__ = __repr__

    def __reduce__(self):
        """Pickles with its class and role."""
        return (EwTile, (int(self), self.ew_class, self.role))


def _mark(kernel, **attrs):
    """Set the declaration on the kernel and on every object it wraps (`.fn`), so the launcher reads it on whichever
    one a call site launches."""
    obj = kernel
    while obj is not None:
        for k, v in attrs.items():
            setattr(obj, k, v)
        inner = getattr(obj, "fn", None)          # a JITFunction's own `fn` is the Python function: not marked
        obj = inner if hasattr(inner, "params") or hasattr(inner, "fn") else None
    return kernel


def tile_independent(kernel):
    """Declare on a kernel that each output element is computed alone: its bits do not depend on the tile."""
    return _mark(kernel, nbx_tile_independent=True)


def tile_dependent(why: str):
    """Declare on a kernel that its result may depend on the tile (`why`: two lanes writing one address, an
    accumulate, an order): the proof of its tile is the absence of a fault alone."""
    return lambda kernel: _mark(kernel, nbx_tile_independent=False, nbx_tile_dependent_why=why)


def declared_independence(kernel):
    """True / False as the kernel declares, None when it declares nothing."""
    return getattr(kernel, "nbx_tile_independent", None)


def class_tiles(values) -> dict:
    """{role: EwTile} among `values` (a launch's keyword values): empty for a launch outside the class."""
    return {v.role: v for v in values if isinstance(v, EwTile)}
