"""An operand as plain data: its shape, its strides in elements and its dtype name (R-NBXT 11).

What a planner hands the library to ask what an op costs (`ops._cost` and the wrappers' `*_cost` declarations) without
a tensor: no device, no memory, only the layout and the width the ops execute with (`nbxtensor.shape`,
`nbxtensor.dtype`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from nbxtensor.dtype.itemsize import itemsize
from nbxtensor.shape import contiguous_strides, numel

__all__ = ["Spec"]


@dataclass(frozen=True)
class Spec:
    """An operand: its shape, its strides in elements, its dtype name."""
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    dtype: str

    @staticmethod
    def contiguous(shape: Sequence[int], dtype: str) -> "Spec":
        """The row-major operand of `shape` and `dtype` (its strides `nbxtensor.shape.contiguous_strides`)."""
        return Spec(tuple(shape), contiguous_strides(tuple(shape)), dtype)

    @property
    def nbytes(self) -> int:
        """The bytes its elements hold: the shape's element count times the dtype's width."""
        return numel(self.shape) * itemsize(self.dtype)
