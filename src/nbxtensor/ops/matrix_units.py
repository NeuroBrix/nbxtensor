"""The matrix unit a device declares, read and checked against the kernels this library builds for it.

A hardware profile may declare a `matrix_unit` (its shape, operand and accumulator dtypes, measured tiles) for the
kernels Triton cannot lower onto that unit; `matrix_unit()` reads it and refuses by name what the library has no kernel
for. The one decision point every reader goes through: the wrappers, the launch keys, the tile choices of
`ops._configs` and a planner pricing a device that is not this process's (`vendor_profile=`).
"""
from __future__ import annotations

from typing import Optional

__all__ = ["matrix_unit"]


def matrix_unit(vendor_profile: Optional[dict] = None) -> dict:
    """The matrix unit the hardware profile declares for kernels Triton cannot lower onto it (`matrix_unit`:
    shape, operand and accumulator dtypes, measured tiles); `{}` when the profile declares none. A declared
    unit this engine has no kernel for, dtypes its instruction does not compute in, or a flash row whose tile
    the kernel cannot lay out is refused by name: a declared capability is never silently left unused.
    `vendor_profile`: the profile to read (Prism, planning for a device that is not this process's); the
    profile in force when None (the wrappers)."""
    from nbxtensor.ops import _configs           # the profile in force (its module imports Triton: read at the call)
    profile = _configs.active_vendor_profile() if vendor_profile is None else vendor_profile
    mu = dict(profile.get("matrix_unit") or {})
    if not mu:
        return mu
    from nbxtensor.ops.mma_m8n8k4 import UNIT
    if mu.get("shape") != UNIT["shape"]:
        raise ValueError(f"hardware profile: matrix_unit.shape {mu.get('shape')!r} has no kernel "
                         f"(this engine builds {UNIT['shape']!r})")
    for f in ("operand_dtype", "accumulator_dtype"):
        if mu.get(f) != UNIT[f]:
            raise ValueError(f"hardware profile: matrix_unit.{f} {mu.get(f)!r} is not what "
                             f"{UNIT['shape']} computes in ({UNIT[f]!r})")
    for row in mu.get("mm") or []:
        for f in ("block_m", "block_n"):
            t = int(row[f])
            if t < 16 or t % 16 or (t // 16) & (t // 16 - 1):
                raise ValueError(f"hardware profile: matrix_unit.mm {f} {t} is not 16 x a power of two")
    split = mu.get("fp32_split") or {}
    smem = (profile.get("memory") or {}).get("max_shared_memory_per_block")     # the read profile's own budget
    from nbxtensor.ops.flash_attention_m8n8k4 import head_parts, staged_shared_bytes
    for sec, rows in (("flash", mu.get("flash") or []), ("fp32_split.flash", split.get("flash") or [])):
        for row in rows:
            for f in ("block_m", "block_n"):
                t = int(row[f])
                if t < 16 or t % 16 or (t // 16) & (t // 16 - 1):
                    raise ValueError(f"hardware profile: matrix_unit.{sec} {f} {t} is not 16 x a power of two")
            wm = int(row["warps_m"])
            if wm < 1 or wm & (wm - 1) or (int(row["block_m"]) // 16) % wm:
                raise ValueError(f"hardware profile: matrix_unit.{sec} warps_m {wm} does not divide the "
                                 f"{row['block_m']}-row tile's 16-row bands")
            if "head_dim_le" not in row:
                raise ValueError(f"hardware profile: matrix_unit.{sec} row {row!r} names no head_dim_le — the head "
                                 "dim its shared memory is priced at")
            if smem is not None:
                # THE DOOR: the tiles the kernel stages by name (ops/flash_attention_m8n8k4.staged_shared_bytes) at
                # the row's widest head dim, against the profile's budget. A wider row used to be refused by the
                # compiler at the first launch that reached it — a census of "not this time"; here it never loads.
                need = staged_shared_bytes(int(row["block_m"]), int(row["block_n"]),
                                           *head_parts(int(row["head_dim_le"])), split=(sec != "flash"))
                if need > int(smem):
                    raise ValueError(f"hardware profile: matrix_unit.{sec} row head_dim_le {row['head_dim_le']} "
                                     f"{row['block_m']}x{row['block_n']} stages {need} bytes of shared memory, over "
                                     f"memory.max_shared_memory_per_block {smem}")
    for row in split.get("flash") or []:
        # the head-dim columns the unit sums (rounding toward zero) before each round-to-nearest add of S
        if "block_k" not in row:
            raise ValueError(f"hardware profile: matrix_unit.fp32_split.flash row head_dim_le {row.get('head_dim_le')} "
                             "names no block_k (the head-dim columns the unit sums before each round-to-nearest add)")
        bk = int(row["block_k"])
        if bk < UNIT["k"] or bk & (bk - 1):
            raise ValueError(f"hardware profile: matrix_unit.fp32_split.flash block_k {bk} is not a power of two "
                             f">= {UNIT['k']} (the unit's k)")
    layout = (profile.get("conv") or {}).get("weight_layout")
    if layout not in (None, "KCRS"):
        # ops/conv2d_m8n8k4.py reads a filter as one contiguous (C, R, S) run (w.stride(0) alone): a weight the
        # loader relaid channels-innermost would be read in the wrong order. Refused here, where the two
        # declarations meet, until that kernel takes the weight's four strides.
        raise ValueError(f"hardware profile: matrix_unit with conv.weight_layout {layout!r} — the {UNIT['shape']} "
                         f"convolution reads KCRS weights only")
    if split and (not split.get("mm") or not split.get("operands")):
        raise ValueError("hardware profile: matrix_unit.fp32_split declares no mm tiles or no operands")
    for row in split.get("mm") or []:
        for f in ("block_m", "block_n"):
            t = int(row[f])
            if t < 16 or t % 16 or (t // 16) & (t // 16 - 1):
                raise ValueError(f"hardware profile: matrix_unit.fp32_split.mm {f} {t} is not 16 x a power of two")
    for row in (mu.get("mm") or []) + (split.get("mm") or []):
        bk, wm, wn = int(row["block_k"]), int(row["warps_m"]), int(row["warps_n"])
        if bk < 8 or bk & (bk - 1):
            raise ValueError(f"hardware profile: matrix_unit.mm block_k {bk} is not a power of two >= 8")
        if (int(row["block_m"]) // 16) % wm or (int(row["block_n"]) // 16) % wn:
            raise ValueError(f"hardware profile: matrix_unit.mm warps {wm}x{wn} do not divide the "
                             f"{row['block_m']}x{row['block_n']} tile's 16-blocks")
        int(row["group_m"])
    return mu
