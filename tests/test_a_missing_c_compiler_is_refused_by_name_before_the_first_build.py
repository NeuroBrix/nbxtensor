"""A process with no C compiler is refused by name before Triton's own driver builds its C helper, where it failed inside
that first build ("Failed to find C compiler"), minutes into a run when a measure was the first thing to reach it.

The compiler is resolved as Triton resolves it (`triton/runtime/build.py`, `_find_compiler`: CC when set, else gcc then
clang on PATH); a test reads Triton's own source so the two cannot drift apart. The check runs where the library hands a
launch to Triton's own driver: a kernel that is not the library's, and the differential arm's device selection.

What this test does if the code is wrong: a CC naming a missing compiler accepted (red), the refusal without the name
asked for (red), no CC and nothing on PATH accepted or refused without the names tried (red), a second read of the
environment after a positive answer (red), a hand-back to Triton's driver that does not ask first (red), a backend that
never builds with a C compiler refused (red), names or an order Triton does not use (red).

Run: python -m pytest tests/<this file>"""
import inspect
import os
import stat

import pytest

pytest.importorskip("triton")
from triton.backends.compiler import GPUTarget  # noqa: E402

from nbxtensor.backend import c_compiler as C  # noqa: E402

BUILDS = GPUTarget("cuda", 70, 32)          # a target whose Triton backend builds a C helper (read, not assumed below)


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(C, "_build_impl", lambda: None)
    C.reset()
    yield
    C.reset()


def _executable(tmp_path, name="cc"):
    path = tmp_path / name
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def test_the_resolution_is_the_one_triton_uses():
    from triton.runtime import build
    src = inspect.getsource(build._find_compiler)
    assert 'os.environ.get("CC")' in src
    for name in C.TRITON_NAMES:
        assert f'shutil.which("{name}")' in src
    assert "gcc if gcc is not None else clang" in src and C.TRITON_NAMES == ("gcc", "clang")


def test_the_backend_of_the_target_is_read_to_build_with_a_c_compiler():
    assert C.builds_with_a_c_compiler(BUILDS)


def test_cc_set_to_a_compiler_that_exists_passes(tmp_path):
    cc = _executable(tmp_path)
    assert C.resolve({"CC": cc}) == (cc, f"CC={cc!r}")
    C.ensure(BUILDS, {"CC": cc})


def test_cc_set_to_a_missing_compiler_is_refused_naming_it(tmp_path):
    missing = str(tmp_path / "no-such-compiler")
    with pytest.raises(C.NoCCompiler) as err:
        C.ensure(BUILDS, {"CC": missing})
    assert missing in str(err.value) and "CC=" in str(err.value)


def test_no_cc_and_none_of_the_names_on_path_is_refused_naming_the_names_tried(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))       # an empty directory: no gcc, no clang
    with pytest.raises(C.NoCCompiler) as err:
        C.ensure(BUILDS, {})
    assert all(name in str(err.value) for name in C.TRITON_NAMES)


def test_no_cc_takes_gcc_before_clang_as_triton_does(tmp_path, monkeypatch):
    clang, gcc = _executable(tmp_path, "clang"), _executable(tmp_path, "gcc")
    monkeypatch.setenv("PATH", str(tmp_path))
    assert C.resolve({}) == (gcc, "gcc")
    os.remove(gcc)
    assert C.resolve({}) == (clang, "clang")


def test_a_backend_that_never_builds_with_a_c_compiler_is_not_checked(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    C.ensure(GPUTarget("no-such-backend", 0, 32), {})
    monkeypatch.setattr(C, "_build_impl", lambda: (lambda *a: "lib.so"))   # Triton builds with the registered one
    C.ensure(BUILDS, {})


def test_the_answer_is_read_once_and_a_launch_then_pays_a_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "builds_with_a_c_compiler", lambda target=None: True)
    monkeypatch.setenv("CC", _executable(tmp_path))
    C.ensure()
    assert C._OK[0]
    monkeypatch.setenv("CC", str(tmp_path / "gone"))
    C.ensure()                               # answered once: no second read of the environment


def test_the_launcher_asks_before_it_hands_a_foreign_kernel_to_tritons_driver(tmp_path, monkeypatch):
    import triton
    import triton.language as tl
    from nbxtensor.backend import launcher as L
    if not L.install():
        pytest.skip("NBX_LAUNCHER=triton: the seam is not installed")

    @triton.jit
    def foreign(x):                          # defined here: not the library's, so handed back to Triton
        tl.store(x, 1)

    monkeypatch.setattr(C, "builds_with_a_c_compiler", lambda target=None: True)
    monkeypatch.setenv("CC", str(tmp_path / "gone"))
    with pytest.raises(C.NoCCompiler):
        foreign.run(None, grid=(1,), warmup=False)
    with pytest.raises(C.NoCCompiler):       # not remembered as fine: every hand-back is refused, none falls through
        foreign[(1,)]


def test_the_differential_arm_asks_before_it_reaches_tritons_driver(tmp_path, monkeypatch):
    from nbxtensor.backend import runtime as R
    from nbxtensor.memory.allocator import DeviceAllocator

    class _Memory:
        def set_device(self, idx):
            pass

    monkeypatch.setenv("NBX_LAUNCHER", "triton")
    monkeypatch.setattr(R, "_cached_device_idx", lambda: None)
    monkeypatch.setattr(R, "device_memory", lambda: _Memory())
    monkeypatch.setattr(C, "builds_with_a_c_compiler", lambda target=None: True)
    monkeypatch.setenv("CC", str(tmp_path / "gone"))
    with pytest.raises(C.NoCCompiler):
        DeviceAllocator.ensure_triton_device(0)
