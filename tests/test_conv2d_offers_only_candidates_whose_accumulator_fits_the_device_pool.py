"""Conv2d offers the tuner only the candidates whose float32 accumulator tile fits the device's threadgroup / shared
memory pool, read from the device profile (`memory.max_shared_memory_per_block`), never a tile or a device named.

Mac M4 Pro, 2026-11-19: past the pool, Metal lowers the accumulator in passes and sums in another order (every tile
of 64 or 128 KB gave other bits than the 16 and 32 KB tiles). The rule must take those out on Apple (32 KB) and keep
every candidate on a V100 (96 KB), whose bits were equal for all of them. Runs on any host: the pool is read from
the profile files, not from a device.

Run: python -m pytest tests/test_conv2d_offers_only_candidates_whose_accumulator_fits_the_device_pool.py"""
from pathlib import Path

import pytest
import yaml

pytest.importorskip("triton")
from nbxtensor.ops import conv2d as CV  # noqa: E402

DEVICES = Path(CV.__file__).resolve().parents[1] / "backend" / "devices"


def _pool(profile):
    return int(yaml.safe_load((DEVICES / profile).read_text())["memory"]["max_shared_memory_per_block"])


def _offered(monkeypatch, profile):
    pool = _pool(profile)
    monkeypatch.setattr(CV, "fits_shared_memory", lambda nbytes: int(nbytes) <= pool)
    return CV.conv2d_candidates(CV._CONV2D_AUTOTUNE)


def test_a_v100_keeps_every_candidate(monkeypatch):
    assert len(_offered(monkeypatch, "nvidia/volta.yml")) == len(CV._CONV2D_AUTOTUNE)


@pytest.mark.parametrize("profile", sorted(str(p.relative_to(DEVICES)) for p in (DEVICES / "apple").glob("*.yml")))
def test_an_apple_gpu_keeps_only_tiles_whose_accumulator_fits_its_pool(monkeypatch, profile):
    pool = _pool(profile)
    kept = _offered(monkeypatch, profile)
    left = [c for c in CV._CONV2D_AUTOTUNE if c not in kept]
    assert all(c.kwargs["BLOCK_SIZE_BHW"] * c.kwargs["BLOCK_SIZE_OUTF"] * 4 <= pool for c in kept)
    assert left and all(c.kwargs["BLOCK_SIZE_BHW"] * c.kwargs["BLOCK_SIZE_OUTF"] * 4 > pool for c in left)


def test_a_list_with_nothing_left_is_refused_by_name(monkeypatch):
    monkeypatch.setattr(CV, "fits_shared_memory", lambda nbytes: False)
    with pytest.raises(RuntimeError, match="max_shared_memory_per_block"):
        CV.conv2d_candidates(CV._CONV2D_AUTOTUNE)
