"""Every NVIDIA compute capability the library can be compiled for reads, from the library's own device files, the
shared memory per block and the registers per block NVIDIA documents for it. Conv2d's candidate list and the MoE
kernels prune by `memory.max_shared_memory_per_block` and refuse by name where no device file declares it
(`_configs.fits_shared_memory`): a capability with no file is a card where they stop instead of running. A figure
larger than the card holds offers tiles it cannot hold; smaller deletes tiles that work.

The oracle is NVIDIA's documentation, quoted in DOCUMENTED with its source. The capabilities are the ones the ptxas
Triton ships can compile for (read from its own `--help`), so a toolchain that gains a target fails here until the
library describes it.

A minor reads a file of its major only where that file documents it (`memory.by_capability`); a minor no file
documents is refused by name, never handed its major's figures.

What it catches: a capability with no device file or a wrong figure (Pascal and Blackwell had none until 2026-11-19;
sm_86 read sm_80's 164 KB; sm_107 read sm_100's 227 KB where NVIDIA documents 327) - red; an undocumented minor given
its major's figures without a word - red."""
import errno
import re
import subprocess
from pathlib import Path

import pytest

from nbxtensor.ops import _configs

# SOURCES: CUDA C++ Programming Guide 12.9, "Technical Specifications per Compute Capability"
# (https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-programming-guide/index.html) for 5.x to 7.2 and 10.1 (the
# guide's "10.x" column); CUDA Programming Guide, Table 34 "Memory Information per Compute Capability"
# (https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html) for 7.5 to 12.x.
# Both read 2026-11-19. "Maximum amount of shared memory per thread block" in KB x 1024, "Maximum number of 32-bit
# registers per thread block" in K x 1024.
KB, K = 1024, 1024
DOCUMENTED = {
    "5.0": (48 * KB, 64 * K), "5.2": (48 * KB, 64 * K), "5.3": (48 * KB, 32 * K),
    "6.0": (48 * KB, 64 * K), "6.1": (48 * KB, 64 * K), "6.2": (48 * KB, 32 * K),
    "7.0": (96 * KB, 64 * K), "7.2": (96 * KB, 64 * K), "7.5": (64 * KB, 64 * K),
    "8.0": (163 * KB, 64 * K), "8.6": (99 * KB, 64 * K), "8.7": (163 * KB, 64 * K), "8.9": (99 * KB, 64 * K),
    "9.0": (227 * KB, 64 * K),
    "10.0": (227 * KB, 64 * K), "10.1": (227 * KB, 64 * K), "10.3": (227 * KB, 64 * K), "10.7": (327 * KB, 64 * K),
    "12.0": (99 * KB, 64 * K), "12.1": (99 * KB, 64 * K),
}


@pytest.fixture(autouse=True)
def _library_files_alone():
    from nbxtensor.backend import device_files
    if device_files._application_dir() is not None:
        pytest.skip("an application gave its own device files: this test reads the library's alone")
    saved = dict(_configs._ACTIVE_PROFILE)
    yield
    _configs._ACTIVE_PROFILE.clear()
    _configs._ACTIVE_PROFILE.update(saved)


def _resolve(cc):
    _configs._ACTIVE_PROFILE.clear()
    budget = _configs._profiles_for(cc)
    return budget, dict(_configs._ACTIVE_PROFILE)


@pytest.mark.parametrize("cc", sorted(DOCUMENTED, key=lambda c: tuple(map(int, c.split(".")))))
def test_the_capability_reads_its_documented_figures(cc):
    smem, regs = DOCUMENTED[cc]
    budget, profile = _resolve(cc)
    assert budget == smem, f"sm_{cc.replace('.', '')}: {budget} bytes of shared memory per block, documented {smem}"
    got = (profile.get("memory") or {}).get("max_registers_per_block")
    assert got == regs, f"sm_{cc.replace('.', '')}: {got} registers per block, documented {regs} ({profile.get('_profile')})"


def test_every_target_of_the_shipped_ptxas_is_documented_here():
    import triton
    ptxas = Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "ptxas"
    if not ptxas.is_file():
        pytest.skip("this Triton ships no NVIDIA ptxas (no CUDA backend on this machine)")
    try:
        out = subprocess.run([str(ptxas), "--help"], capture_output=True, text=True, timeout=60).stdout
    except OSError as e:
        # The wheel ships ptxas built for its own platform (a Linux ELF in every wheel); a host of another system or
        # architecture cannot execute it at all (Mac M4 Pro, 2026-11-19: "[Errno 8] Exec format error"). Only that
        # refusal is a skip: any other failure to run the shipped compiler is a fault and stays one.
        if e.errno != errno.ENOEXEC:
            raise
        pytest.skip(f"the shipped ptxas ({ptxas}) is built for another system and cannot run on this host: {e}")
    targets = sorted({m for m in re.findall(r"\bsm_(\d+)\b", out)}, key=int)
    assert targets, "ptxas --help named no sm_ target"
    caps = [f"{t[:-1]}.{t[-1]}" for t in targets]
    missing = [c for c in caps if c not in DOCUMENTED]
    assert not missing, f"ptxas compiles for {missing}, which no documented figure here covers: describe them"


# A minor of every major that has a file, documented by none of NVIDIA's columns: refused by name.
UNDOCUMENTED = ["5.9", "6.9", "7.7", "8.8", "9.1", "10.9", "12.5"]


@pytest.mark.parametrize("cc", UNDOCUMENTED)
def test_a_minor_no_file_of_its_major_documents_is_refused_by_name(cc):
    budget, profile = _resolve(cc)
    assert budget is None and not profile, f"sm_{cc.replace('.', '')} was handed {budget} ({profile.get('_profile')})"
    assert _configs._UNDOCUMENTED.get(cc), f"no file of major {cc.split('.')[0]} named in the refusal of {cc}"


def test_every_minor_a_file_lists_is_one_nvidia_documents_with_those_figures():
    from nbxtensor.backend import device_files
    for vendor, name, path in device_files.names():
        if vendor != "nvidia":
            continue
        listed = ((device_files.merged(vendor, name).get("memory") or {}).get("by_capability") or {})
        for cc in listed:
            assert cc in DOCUMENTED, f"{path.name} lists {cc}, which no NVIDIA column documents here"
            budget, profile = _resolve(cc)
            assert profile.get("_profile") == path.stem, f"{cc} resolved to {profile.get('_profile')}, listed by {path.stem}"


def test_the_refusal_names_the_capability_in_force_only(monkeypatch):
    """Two refused capabilities in one process: the second refusal speaks of the second alone, never of one the
    process resolved before (the reasons are kept per capability)."""
    from nbxtensor.backend import launcher
    monkeypatch.setattr(_configs, "_UNDOCUMENTED", {})
    said = {}
    for arch in (81, 109):
        monkeypatch.setattr(launcher, "arch_name", lambda a=arch: a)
        _configs._ACTIVE_PROFILE.clear()
        with pytest.raises(RuntimeError) as e:
            _configs.fits_shared_memory(1024)
        said[arch] = str(e.value)
    assert "capability 8.1" in said[81] and "ampere.yml" in said[81]
    assert "capability 10.9" in said[109] and "blackwell.yml" in said[109] and "8.1" not in said[109], said[109]


def test_a_documented_minor_is_told_it_reads_its_own_figures_not_a_fallback(caplog):
    import logging
    _configs._FALLBACK_ANNOUNCED.clear()
    with caplog.at_level(logging.INFO):
        budget, _profile = _resolve("8.6")
    assert budget == 99 * KB
    said = " ".join(r.getMessage() for r in caplog.records if "[PROFILE]" in r.getMessage())
    assert "documents it" in said and "falling back" not in said and "not optimally" not in said, said
