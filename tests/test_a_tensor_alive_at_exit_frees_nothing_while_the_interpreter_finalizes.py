"""A device tensor still alive when the interpreter exits, with the slab interceptor installed, must exit silently.

Interpreter teardown clears modules in reverse import order: `nbxtensor.memory.slab` is cleared while a tensor it
would free is still referenced, and its globals (`ACTIVE_SLABS`) are None. A free reaching the interceptor then raises
"TypeError: 'NoneType' object is not iterable" inside `NBXTensor.__del__`, which Python prints as "Exception ignored"
(Mac M4 Pro, 2026-11-19: the library suite printed it twice after "1428 passed"). The process is exiting and the
driver takes every device buffer back with it, so a tensor freed while the interpreter finalizes frees nothing.

What this test would do if the code were wrong: the child's stderr carries "Exception ignored" and the traceback.
The direct device operations are fakes, so the test needs no device."""
import subprocess
import sys
import textwrap

CHILD = textwrap.dedent("""
    from nbxtensor.memory import slab as S
    from nbxtensor.memory.allocator import DeviceAllocator as DA
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.dtype.nbx_dtype import NBXDtype

    DA.free_direct = staticmethod(lambda ptr: None)
    DA.get_device = staticmethod(lambda: 0)
    DA.install_interceptor(S.SlabInterceptor())
    # The teardown the suite met, made certain: an object dropped while the interpreter finalizes clears the slab
    # module's globals (as module clearing does) and then lets go of the last reference to a live device tensor.
    class Teardown:
        def __init__(self, t, slab_globals):
            self.t, self.g = t, slab_globals
        def __del__(self):
            self.g["ACTIVE_SLABS"] = None
            self.t = None
    import builtins
    builtins._teardown_at_exit = Teardown(
        NBXTensor(1 << 20, (4,), (1,), NBXDtype.float32, device="cuda", owns_data=True), vars(S))
    print("child done")
""")


def test_a_tensor_alive_at_exit_frees_nothing_while_the_interpreter_finalizes():
    r = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "child done" in r.stdout
    assert "Exception ignored" not in r.stderr and "Traceback" not in r.stderr, r.stderr
