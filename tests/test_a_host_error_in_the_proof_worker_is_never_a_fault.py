"""qa 1308-4 and 1308-5, card-less.

1308-4: the WORKER side of the classification. A host exception inside the proof worker (here, the kernel's module
cannot be imported) is written as "unrun" with exit UNRUN_EXIT, never as a "fault" with exit DEVICE_EXIT. qa's mutant,
the old false green moved into the worker (`except BaseException` writing {"fault": ...}), survived the parent-side
tests; this test runs the real `_child_run` in its own interpreter and is red on that mutant.
1308-5: the set of driver statuses counted as a tile's device fault is exactly CUDA's sticky kernel faults; an
allocation refusal or an invalid argument is not one."""
import json
import os
import pickle
import subprocess
import sys

from nbxtensor.backend import faults
from nbxtensor.ops import ew_proof_child

RUN = "import sys; from nbxtensor.ops import ew_proof_child as c; c._child_run(sys.argv[1], 0)"


def test_a_host_exception_in_the_worker_is_written_unrun_and_exits_unrun(tmp_path):
    spec = {"module": "nbxtensor.ops.no_such_kernel_module", "name": "no_such_kernel", "launches": [],
            "guard": "none", "options": {}}
    pickle.dump(spec, open(tmp_path / "spec.pkl", "wb"))
    json.dump([[256, 2, 3]], open(tmp_path / "cands.json", "w"))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    done = subprocess.run([sys.executable, "-c", RUN, str(tmp_path)], capture_output=True, text=True, env=env)
    lines = [json.loads(l) for l in (tmp_path / "out0.jsonl").read_text().splitlines()]
    assert done.returncode == ew_proof_child.UNRUN_EXIT, (done.returncode, done.stderr[-800:])
    assert any("unrun" in l for l in lines) and not any("fault" in l for l in lines), lines
    assert "ModuleNotFoundError" in next(l["text"] for l in lines if "unrun" in l)


def test_the_device_faults_are_exactly_cudas_sticky_kernel_faults():
    assert faults.KERNEL_FAULTS == frozenset({700, 714, 715, 716, 717, 718, 719})
    for status in (1, 2, 98, 200, 209, 400):          # invalid value, out of memory, invalid function, invalid image...
        assert not faults.DeviceStatusError("x", "cuLaunchKernel", status).kernel_fault()
    assert faults.DeviceStatusError("x", "cuCtxSynchronize", 700).kernel_fault()
