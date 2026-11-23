"""The guard's canary (`ops/ew_proof.py`): a kernel that writes ONE element past its output, on purpose.

Before a run's first element-wise proof, the proof runs this kernel on its synthetic launches, against the guard, in
the worker: the overrun MUST be seen — a device fault on CUDA (the unmapped granule after a `memory.vmm.GuardedRange`:
an illegal address, 700), a shader-validation report on Metal. A canary that is not seen means the guard does not
guard on this device: the proof says "unguarded" and proves nothing there."""
import triton
import triton.language as tl

import nbxtensor.ops._launch_seam  # noqa: F401  this module's launches go through nbxtensor's launcher


@triton.jit
def ew_canary_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """y[n_elements] = x[0]: one lane one element past y's extent, the access the guard must see."""
    one = tl.arange(0, 1)
    tl.store(y_ptr + n_elements + one, tl.load(x_ptr + one))


#: the canary's variant: fp32 pointers aligned, a count divisible by 16 (its buffers end exactly at the guard)
CANARY_SPEC = [("*fp32", "D"), ("*fp32", "D"), ("i32", "D"), ("constexpr", 256)]
