"""The compile bound (`backend/compile_bound.py`) gives three answers and never confuses them (C16, the Dell leg of
23/11/2026 on f0dedbb): a compile that does not end within the bound is EXCLUDED; a candidate Triton's compiler refuses
is a candidate that cannot run (as in-process); a worker that DIES on anything else is an ERROR with its own text.
Every "moe_route_kernel: the candidate's compile did not end within 120 s — excluded" line of that leg was a worker that
died at once: `AttributeError: 'Autotuner' object has no attribute 'params'` (launcher.py:998, `_param_table`) — the
worker looked the kernel up by its module-level name, which holds the AUTOTUNER around the @triton.jit function.

What this test does if the code is wrong:
1. the worker's lookup of an autotuned library kernel (`moe_route_kernel`) does not reach its @triton.jit function;
2. on a card, an autotuned kernel's candidate compiled through the bound raises (on the base: excluded as a timeout,
   60 such lines on the Dell's prefill file);
3. a worker that died is read as a timeout (an exclusion that hides a lost candidate), or a timeout as a death;
4. the autotune screen swallows a dead worker as "a candidate that could not run";
5. the launch's own keywords (`grid`, `warmup`, which an autotuner's launch hands on) reach the compile, which refuses
   them ("Keyword argument grid was specified but unrecognised": the second fault the dead worker had hidden).
Run: python -m pytest tests/test_a_compile_bound_tells_a_timeout_from_a_dead_worker.py
"""
import sys

import pytest

from nbxtensor.backend import compile_bound as CB
from nbxtensor.ops import ew_proof_child


class _Fn:
    __name__ = "k"
    params = ()
    fn = type("F", (), {"__module__": "m"})


def test_the_worker_finds_the_jit_function_inside_an_autotuned_kernel():
    from nbxtensor.ops.moe_route import moe_route_kernel
    found = CB.jit_of(moe_route_kernel)
    assert hasattr(found, "params") and not hasattr(moe_route_kernel, "params")
    assert CB.jit_of(found) is found
    with pytest.raises(TypeError, match="holds no @triton.jit function"):
        CB.jit_of(object())


def test_a_dead_worker_is_an_error_and_a_timeout_an_exclusion(monkeypatch):
    monkeypatch.setattr(CB, "bound_s", lambda: 2.0)
    monkeypatch.setattr(CB, "_named", lambda k: True)            # the stand-in, as a kernel a worker can find
    died = "Traceback ...\nAttributeError: 'Autotuner' object has no attribute 'params'\n"
    monkeypatch.setattr(ew_proof_child._WORKER, "ask", lambda job, timeout=None: (1, died))
    with pytest.raises(CB.CompileWorkerDied, match="(?s)not a timeout, not an exclusion.*Autotuner"):
        CB.precompile(_Fn(), (), {"BLOCK_H": 64})
    monkeypatch.setattr(ew_proof_child._WORKER, "ask",
                        lambda job, timeout=None: (ew_proof_child.TIMED_OUT, "timed out after 2 s"))
    with pytest.raises(CB.CompileUnbounded, match="did not end within 2 s"):
        CB.precompile(_Fn(), (), {"BLOCK_H": 64})
    assert ew_proof_child.TIMED_OUT < -64                  # never a signal's exit code


def test_the_screen_raises_a_dead_worker_instead_of_counting_it_unrun():
    import inspect
    from nbxtensor.backend import launcher
    src = inspect.getsource(launcher)
    i = src.index("precompile(tuner.fn, args")
    window = src[i:i + 900]
    assert window.index("except CompileWorkerDied") < window.index("except Exception:")
    assert "raise" in window[window.index("except CompileWorkerDied"):window.index("except Exception:")]


@pytest.mark.skipif(sys.platform != "darwin", reason="the card here is the Apple GPU (Metal)")
def test_on_a_card_an_autotuned_candidate_compiles_through_the_bound():
    pytest.importorskip("triton")
    import numpy as np
    import triton
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.memory.allocator import DeviceAllocator
    from nbxtensor.ops.moe_route import moe_route_kernel
    from nbxtensor.tensor import NBXTensor
    try:
        DeviceAllocator.set_device(0)
    except Exception as e:                                         # pragma: no cover
        pytest.skip(f"no device: {e}")
    if CB.bound_s() is None:                                       # pragma: no cover
        pytest.skip("this profile declares no autotune.compile_bound_s")
    T, H, E, K = 2, 64, 8, 2
    x = NBXTensor.from_numpy(np.ones((T, H), np.float32), device="cuda:0")
    w = NBXTensor.from_numpy(np.ones((E, H), np.float32), device="cuda:0")
    idx = NBXTensor.from_numpy(np.zeros((T, K), np.int64), device="cuda:0")
    wt = NBXTensor.from_numpy(np.zeros((T, K), np.float32), device="cuda:0")
    args = (x, w, wt, idx, wt, T, E, H, K, 1, 1, 0, *x.stride(), *w.stride(), 0, 0.0, 0.0, 1.0)
    kwargs = dict(SIGMOID=False, HAS_BIAS=False, GROUPED=False, GROUP_TOP2=False, RENORM=False,
                  BLOCK_E=triton.next_power_of_2(E), BLOCK_G=1, BLOCK_K=triton.next_power_of_2(K),
                  BLOCK_H=16, num_warps=2, num_stages=1, grid=(T,), warmup=False)   # as the autotuner's launch hands them
    CB.precompile(moe_route_kernel.fn, args, kwargs)               # what the screen hands: tuner.fn


def test_a_kernel_no_module_names_is_compiled_in_process_and_said(monkeypatch, capsys):
    """A kernel written outside a module (a test's inline kernel renamed into `nbxtensor.tests`) is never handed to a
    worker that cannot import it (it died there on ModuleNotFoundError): compiled in-process, said once by name."""
    asked = []
    monkeypatch.setattr(CB, "bound_s", lambda: 2.0)
    monkeypatch.setattr(ew_proof_child._WORKER, "ask", lambda job, timeout=None: asked.append(job) or (0, ""))

    class Inline:
        __name__ = "inline_kernel"
        params = ()
        fn = type("F", (), {"__module__": "nbxtensor.tests"})
    CB.precompile(Inline(), (), {})
    CB.precompile(Inline(), (), {})
    assert asked == []
    assert capsys.readouterr().err.count("inline_kernel: its module 'nbxtensor.tests' does not name it") == 1
    from nbxtensor.ops.moe_route import moe_route_kernel
    assert CB._named(CB.jit_of(moe_route_kernel))            # a library kernel is named: it goes to the worker
