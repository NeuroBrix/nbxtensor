"""Whatever kernel module a process imports first, its kernels launch through nbxtensor's launcher, not upstream Triton's.

The seam (`nbxtensor.backend.launcher.install`) used to be installed only by `ops.wrappers`, the twelve
`ops.*_wrappers` modules, `ops.token_bitmask` and `memory.kv_host_pages`. A process whose first launch came from any
other module ran upstream Triton's launch path, which on Apple refuses an NBXTensor argument: the chunked gated delta
rule imported alone failed its first tuning with "TypeError: Arg 0: expected an MPS tensor, a SharedBuffer or the
packed scalar bytes" (Mac M4 Pro, 2026-11-19), and passed when an earlier import had pulled `ops.wrappers`. Each
kernel module now imports `ops._launch_seam` (or calls the install itself).

What this test would do if the code were wrong: a kernel module without the import is named; a child process that
imports one kernel module, and nothing else of the package, finds the seam not installed. It needs no device."""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "nbxtensor"


def _defines_a_kernel(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for d in node.decorator_list:
                if ast.unparse(d).split("(")[0] in ("triton.jit", "gluon.jit"):
                    return True
    return False


# Only the module's own top-level statements count: they are what runs at import. A seam import or an install call
# inside a function body runs only if that function is called (qa on 1195, 2026-11-19: kv_host_pages' install moved
# into a function stayed green while importing the module alone left the launcher not installed).
def _imports_the_seam(tree):
    return any(isinstance(n, ast.Import) and any(a.name == "nbxtensor.ops._launch_seam" for a in n.names)
               or isinstance(n, ast.ImportFrom) and (n.module or "").endswith("_launch_seam")
               for n in tree.body)


def _installs_the_launcher(tree):
    """A module's own top-level call of the launcher's install (`launcher.install()`, or `install()` imported from it)."""
    installs = {a.asname or a.name for n in tree.body if isinstance(n, ast.ImportFrom)
                and (n.module or "").endswith("backend.launcher") for a in n.names if a.name == "install"}
    for n in tree.body:
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call):
            f = ast.unparse(n.value.func)
            if f.endswith("launcher.install") or f in installs:
                return True
    return False


def test_every_kernel_module_of_the_library_installs_the_launcher():
    """Every module of the package that defines a kernel, at any depth, imports the seam or calls the install itself:
    no module is exempted by name (a module under memory/ with its seam line removed stayed green when only ops/*.py
    was read, qa on the launch-seam hand-in, 2026-11-19)."""
    missing = [str(p.relative_to(SRC)) for p in sorted(SRC.rglob("*.py"))
               if _defines_a_kernel(t := ast.parse(p.read_text()))
               and not (_imports_the_seam(t) or _installs_the_launcher(t))]
    assert missing == [], f"kernel modules that would launch through upstream Triton when imported first: {missing}"


@pytest.mark.parametrize("module", ["nbxtensor.ops.gated_delta_rule_chunked", "nbxtensor.ops.gated_delta_rule",
                                    "nbxtensor.ops.conv2d", "nbxtensor.ops.softmax"])
def test_importing_one_kernel_module_installs_the_launcher(module):
    pytest.importorskip("triton")
    child = (f"import os; os.environ.pop('NBX_LAUNCHER', None)\nimport {module}\n"
             "from nbxtensor.backend import launcher\nprint('installed', launcher._installed)")
    r = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().endswith("installed True"), (r.stdout, r.stderr[-2000:])
