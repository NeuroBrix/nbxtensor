"""No Python `and` / `or` produces a value inside a kernel body of the library.

Triton lowers `and` / `or` itself (`code_generator.visit_BoolOp`): on a tensor operand it warns that the construct is
deprecated (Triton names it for removal) and computes both sides elementwise; on a constexpr operand it short-circuits,
so a constexpr that is True drops out of the expression and the other operand alone is the result, with no word. A
kernel that mixes the two gets the short-circuit on one side only: what the dropped operand would have done (its side
effects, its block shape) is gone with no word. `&` / `|` on parenthesised comparisons say what they compute.
A boolean operator in the TEST of an `if`, `while` or conditional expression on constexpr values is scalar Python and
is left alone.

The engine's gate reads the library's `ops` kernels too (its OPS_DIRS) and was red on `ops/gemv_vec.py`, which carried
`tl.static_assert(BLOCK_K >= K_SUB and BLOCK_K % K_SUB == 0, ...)` (engine census, Mac M4 Pro, 2026-11-19); the library
held no test of its own for it, so the library alone, without the engine beside it, did not gate its own kernels.

What this test would do if the code were wrong: a kernel with `and` / `or` producing a value is named with its line."""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "nbxtensor"


def _is_kernel(fn):
    return any(ast.unparse(d).split("(")[0] in ("triton.jit", "gluon.jit", "jit") for d in fn.decorator_list)


def _offenders(tree):
    found = []
    for fn in ast.walk(tree):
        if not (isinstance(fn, ast.FunctionDef) and _is_kernel(fn)):
            continue
        in_test = {id(s) for n in ast.walk(fn) if isinstance(n, (ast.If, ast.While, ast.IfExp))
                   for s in ast.walk(n.test)}
        found += [(fn.name, n.lineno) for n in ast.walk(fn) if isinstance(n, ast.BoolOp) and id(n) not in in_test]
    return found


def test_no_boolean_operator_produces_a_value_in_a_kernel():
    offenders = [f"{p.relative_to(SRC)}:{line} ({name})" for p in sorted(SRC.rglob("*.py"))
                 for name, line in _offenders(ast.parse(p.read_text()))]
    assert offenders == [], f"`and` / `or` producing a value in a kernel body (use & | on parenthesised terms): {offenders}"


def test_the_gate_sees_a_value_and_spares_a_constexpr_test():
    src = ("@triton.jit\ndef k(X, N: tl.constexpr):\n    if N > 1 and N < 8:\n        pass\n"
           "    tl.static_assert(N > 1 and N < 8)\n")
    assert _offenders(ast.parse(src)) == [("k", 5)]
