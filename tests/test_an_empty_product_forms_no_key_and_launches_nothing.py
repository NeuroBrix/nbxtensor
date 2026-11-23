"""A GEMM with no output row or no output column does no work: its wrapper returns the empty output before any launch,
and the key formers (`launch_keys.*_launches`, the census's own functions) form no key for it (qa 1249, verdict
transfer/verdicts/1249_qa.txt: `bmm_launches(0, 512, 1, fp32, fp16, False)` formed an M-bucket-0 baddbmm key and
`linalg_wrappers.bmm` reached the autotuner with it, a run-time sweep). A zero CONTRACTED dimension still writes its
M x N values (zeros, or the bias) and keeps its launch and its key.

Red on the base: the formers return a key for M = 0; the wrappers call the kernel."""
import numpy as np
import pytest

from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.ops import launch_keys as LK

F32, F16 = NBXDtype.float32, NBXDtype.float16


@pytest.mark.parametrize("M,N", [(0, 1), (0, 512), (7, 0), (0, 0)])
def test_no_former_keys_an_empty_product(M, N):
    assert LK.bmm_launches(M, 512, N, F32, F16, False) == []
    assert LK.baddbmm_launches(M, 512, N, F32, F32, F32) == []
    assert LK.mm_launches(M, 512, N, F32, F32, False) == []
    assert LK.addmm_launches(M, 512, N, F32, F32, F32, False) == []


def test_the_key_qa_found_is_formed_by_no_call_and_a_zero_contraction_keeps_its_key():
    assert LK.bmm_launches(0, 512, 1, F32, F16, False) == []
    assert LK.bmm_launches(1, 512, 1, F32, F16, False) != []          # a real row: its key, as before
    assert LK.bmm_launches(3, 0, 5, F32, F32, False) != []            # K = 0 writes 3 x 5 zeros: a launch


def _no_launch(monkeypatch):
    from nbxtensor.ops import linalg_wrappers as W

    class _Refuse:
        def __getitem__(self, grid):
            raise AssertionError("an empty product reached the kernel")
    monkeypatch.setattr(W, "baddbmm_kernel", _Refuse())
    return W


def test_an_empty_bmm_and_baddbmm_return_their_empty_output_without_a_launch(monkeypatch):
    from nbxtensor.tensor import NBXTensor
    W = _no_launch(monkeypatch)
    a = NBXTensor.from_numpy(np.zeros((2, 0, 512), dtype=np.float32))
    b = NBXTensor.from_numpy(np.ones((2, 512, 1), dtype=np.float32))
    c = W.bmm(a, b)
    assert tuple(c.shape) == (2, 0, 1)
    bias = NBXTensor.from_numpy(np.ones((2, 0, 1), dtype=np.float32))
    assert tuple(W.baddbmm_wrapper(bias, a, b).shape) == (2, 0, 1)
    b0 = NBXTensor.from_numpy(np.ones((2, 512, 0), dtype=np.float32))
    a1 = NBXTensor.from_numpy(np.ones((2, 6, 512), dtype=np.float32))
    assert tuple(W.bmm(a1, b0).shape) == (2, 6, 0)
