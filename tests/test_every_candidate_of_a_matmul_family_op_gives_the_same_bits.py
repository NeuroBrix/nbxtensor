"""Every candidate of a matmul-family op and dtype gives the same bits: the unit a call runs on is the RULE's, never a
timing, so no choice of tuning (a kernel configuration, a tile a certificate names, no certificate, a certificate that
names the FMA kernel) can change the answer.

The candidates are enumerated, each one pinned, and the public op (`mm`, `addmm`) run through it:
- the FMA kernel's own configurations (the autotuner's list, one at a time: a one-configuration tuner does not sweep);
- on a profile that declares a matrix unit and a GEMM it takes (M fills the unit's smallest tile), the certificate
  reading is pinned to every tile of the profile, to the FMA kernel's, and to none: the call runs on the unit in every
  case, and its bits are the unit's.
On a card with no matrix unit (this Mac) there is one path, the FMA kernel, and the test holds every configuration of it
to one set of bits. On a card with the unit (the NVIDIA rack, Volta) a GEMM of M >= 64 runs on the unit, M < 64 on the
FMA kernel: each is its own group and each group must give one set of bits. The two groups are NOT compared with each
other (a different unit is a different answer; the rule fixes which one answers).

Run: python -m pytest tests/test_every_candidate_of_a_matmul_family_op_gives_the_same_bits.py
The card run (NVIDIA, Volta: where the unit exists): the same command, on each card of the rack."""
import numpy as np
import pytest

pytest.importorskip("triton")

from nbxtensor.memory.allocator import DeviceAllocator as DA  # noqa: E402
from nbxtensor.ops import autotune_certified as CERT  # noqa: E402
from nbxtensor.ops import linalg_wrappers as LW  # noqa: E402
from nbxtensor.ops import matmul as MM  # noqa: E402
from nbxtensor.ops.matrix_units import matrix_unit  # noqa: E402
from nbxtensor.tensor import NBXTensor  # noqa: E402

DTYPES = {"fp16": np.float16, "fp32": np.float32}
N, K = 192, 320


def _operands(m, dtype, seed=0):
    rng = np.random.default_rng(seed)
    mk = lambda *s: NBXTensor.from_numpy(np.ascontiguousarray((rng.standard_normal(s) * 0.3).astype(dtype)))  # noqa: E731
    return mk(m, K), mk(K, N), mk(N)


def _bytes(t):
    DA.stream_synchronize(0)
    return np.ascontiguousarray(t.numpy()).tobytes()


def _run(op, a, b, bias):
    return LW.mm(a, b) if op == "mm" else LW.addmm(bias, a, b)


def _tuner(op):
    return MM.matmul_kernel if op == "mm" else MM.addmm_kernel


def _pinned_to_one_config(monkeypatch, tuner, cfg):
    monkeypatch.setattr(tuner, "configs", [cfg])
    tuner.cache.clear()


def _unit_tiles(dtype):
    mu = matrix_unit()
    if not mu:
        return []
    return (mu["fp32_split"] if dtype == "fp32" else mu)["mm"]


def _tile_names_unit(tile):
    fields = CERT._UNIT_TILE_FIELDS
    return lambda *_a, **_k: ("unit", {f: int(tile[f]) for f in fields})


@pytest.mark.parametrize("m", [32, 256])
@pytest.mark.parametrize("dtype", sorted(DTYPES))
@pytest.mark.parametrize("op", ["mm", "addmm"])
def test_every_candidate_of_the_op_gives_the_same_bits(monkeypatch, op, dtype, m):
    a, b, bias = _operands(m, DTYPES[dtype])
    tuner = _tuner(op)
    kept_configs, kept_cache = list(tuner.configs), dict(tuner.cache)
    try:
        answers = {}
        for i, cfg in enumerate(kept_configs):
            _pinned_to_one_config(monkeypatch, tuner, cfg)
            try:
                answers[f"fma config {i} {cfg.kwargs}"] = _bytes(_run(op, a, b, bias))
            except Exception as exc:              # a configuration the backend refuses is no candidate (counted below)
                answers[f"refused {i}"] = None
                print(f"refused: {cfg.kwargs}: {str(exc)[:120]}")
        tiles = _unit_tiles(dtype)
        if tiles and m >= min(int(r["block_m"]) for r in tiles):
            # the unit takes it: a certificate naming each tile, the FMA kernel, or nothing must not change one bit
            for t in tiles:
                monkeypatch.setattr(CERT, "unit_choice", _tile_names_unit(t))
                answers[f"certificate names tile {t}"] = _bytes(_run(op, a, b, bias))
            monkeypatch.setattr(CERT, "unit_choice", lambda *_a, **_k: ("fma", None))
            answers["certificate names the FMA kernel"] = _bytes(_run(op, a, b, bias))
            monkeypatch.setattr(CERT, "unit_choice", lambda *_a, **_k: None)
            answers["no certificate"] = _bytes(_run(op, a, b, bias))
        ran = {k: v for k, v in answers.items() if v is not None}
        assert len(ran) >= 2, f"{op} {dtype} M={m}: fewer than two candidates ran ({sorted(answers)})"
        first_name, first = next(iter(ran.items()))
        differing = [k for k, v in ran.items() if v != first]
        assert not differing, f"{op} {dtype} M={m}: {len(differing)} of {len(ran)} candidates differ in bits from {first_name}: {differing[:4]}"
    finally:
        tuner.configs[:] = kept_configs
        tuner.cache.clear()
        tuner.cache.update(kept_cache)


def _unit_mm_tiles(profile_dir):
    """(file, representation, [block_k of each declared mm tile]) for every matrix unit a device profile declares: the
    unit's own operands (`matrix_unit.mm`) and every other representation on it (`matrix_unit.<name>.mm`)."""
    import yaml
    for f in sorted(profile_dir.rglob("*.yml")):
        mu = (yaml.safe_load(f.read_text()) or {}).get("matrix_unit") or {}
        reps = [("operand_dtype", mu)] + [(k, v) for k, v in mu.items() if isinstance(v, dict) and "mm" in v]
        for name, rep in reps:
            if rep.get("mm"):
                yield f.name, name, [int(t["block_k"]) for t in rep["mm"]]


def test_every_representation_declares_its_unit_tiles_at_one_block_k():
    """Card-less, on the shipped profiles. The unit kernel sums each block_k tile's MMA steps into a partial before adding
    it into the accumulator (matmul_m8n8k4), so the order of the sums follows block_k: tiles of one representation at two
    block_k would let the certificate's choice of tile change the bits again, and only a card would see it."""
    from pathlib import Path
    import nbxtensor
    found = list(_unit_mm_tiles(Path(nbxtensor.__file__).resolve().parent / "backend" / "devices"))
    assert found, "no shipped profile declares a matrix unit's mm tiles: nothing was checked"
    mixed = [(f, name, ks) for f, name, ks in found if len(set(ks)) > 1]
    assert not mixed, mixed
