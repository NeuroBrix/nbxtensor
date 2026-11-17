"""The census test file needs triton to run; this one does not: it reads the sources and checks every `G.<name>` the
census test uses is defined at the top level of the op module (a renamed function left behind goes red here)."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST = ROOT / "tests" / "test_the_gated_delta_rule_op_runs_the_form_its_census_certifies.py"
MODULE = ROOT / "src" / "nbxtensor" / "ops" / "gated_delta_rule.py"


def test_every_name_the_census_test_reads_from_the_op_module_exists():
    used = {n.attr for n in ast.walk(ast.parse(TEST.read_text())) if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name) and n.value.id == "G"}
    defined = set()
    for n in ast.parse(MODULE.read_text()).body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            defined.add(n.name)
        elif isinstance(n, ast.Assign):
            defined |= {t.id for t in n.targets if isinstance(t, ast.Name)}
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            defined.add(n.target.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            defined |= {(a.asname or a.name).split(".")[0] for a in n.names}
    assert used, "the census test names nothing from the op module: the check lost its subject"
    assert used - defined == set(), f"the census test reads names the op module lacks: {sorted(used - defined)}"
