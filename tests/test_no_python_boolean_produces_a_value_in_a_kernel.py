"""No Python `and` / `or` produces a value inside a kernel body of the library.

Triton lowers `and` / `or` itself (`code_generator.visit_BoolOp`): on a tensor operand it warns that the construct is
deprecated (Triton names it for removal) and computes both sides elementwise; on a constexpr operand it short-circuits,
so a constexpr that is True drops out of the expression and the other operand alone is the result, with no word. A
kernel that mixes the two gets the short-circuit on one side only: what the dropped operand would have done (its side
effects, its block shape) is gone with no word. `&` / `|` on parenthesised comparisons say what they compute.
A boolean operator in the TEST of an `if`, `while` or conditional expression is scalar Python, and left alone, only
when every operand is a compile-time value (a constexpr argument, a `static_range` variable, a constant or a name
made of them): the gate tells by what the operands are, not by where the operator stands. The jit decorator is found
through the module's imports, so `import triton as t` and `@t.jit` is a kernel too.

The engine's gate reads the library's `ops` kernels too (its OPS_DIRS) and was red on `ops/gemv_vec.py`, which carried
`tl.static_assert(BLOCK_K >= K_SUB and BLOCK_K % K_SUB == 0, ...)` (engine census, Mac M4 Pro, 2026-11-19); the library
held no test of its own for it, so the library alone, without the engine beside it, did not gate its own kernels.

What this test would do if the code were wrong: a kernel with `and` / `or` producing a value is named with its line."""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "nbxtensor"


_JIT = ("triton.jit", "triton.experimental.gluon.jit")


def _imports(tree):
    """The module's local names for what it imports, as dotted paths: `import triton as t` -> t: triton,
    `from triton import jit as j` -> j: triton.jit."""
    names = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                names[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
            for a in n.names:
                names[a.asname or a.name] = f"{n.module}.{a.name}"
    return names


def _resolved(node, names):
    parts = ast.unparse(node).split("(")[0].split(".")
    return ".".join([names.get(parts[0], parts[0])] + parts[1:])


def _is_kernel(fn, names):
    return any(_resolved(d, names) in _JIT for d in fn.decorator_list)


def _static_names(fn, names):
    """The names in a kernel body that hold compile-time values: constexpr parameters, a `static_range` loop variable,
    and a name assigned once from compile-time values only."""
    static = {a.arg for a in fn.args.args + fn.args.kwonlyargs
              if a.annotation is not None and ast.unparse(a.annotation).split(".")[-1] == "constexpr"}
    stores = {}
    for n in ast.walk(fn):
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.For)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                for x in ast.walk(t):
                    if isinstance(x, ast.Name):
                        stores[x.id] = stores.get(x.id, 0) + 1
    for n in ast.walk(fn):
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and isinstance(n.iter, ast.Call) \
                and _resolved(n.iter.func, names).split(".")[-1] == "static_range" and stores[n.target.id] == 1:
            static.add(n.target.id)
    grown = True
    while grown:
        grown = False
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                name = n.targets[0].id
                if name not in static and stores[name] == 1 and _is_static(n.value, static):
                    static.add(name)
                    grown = True
    return static


def _is_static(node, static):
    """An expression of constants and compile-time names only, through operators, comparisons and conditionals."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name):
        return node.id in static
    if isinstance(node, ast.Call) and ast.unparse(node.func).split(".")[-1] == "constexpr":
        return True                                  # `tl.constexpr(...)` is a compile-time value by construction
    kids = [c for c in ast.iter_child_nodes(node) if not isinstance(c, (ast.operator, ast.cmpop, ast.boolop,
                                                                          ast.unaryop, ast.expr_context))]
    return isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.IfExp)) \
        and all(_is_static(c, static) for c in kids)


def _offenders(tree):
    names = _imports(tree)
    found = []
    for fn in ast.walk(tree):
        if not (isinstance(fn, ast.FunctionDef) and _is_kernel(fn, names)):
            continue
        static = _static_names(fn, names)
        in_test = {id(s) for n in ast.walk(fn) if isinstance(n, (ast.If, ast.While, ast.IfExp))
                   for s in ast.walk(n.test)}
        found += [(fn.name, n.lineno) for n in ast.walk(fn) if isinstance(n, ast.BoolOp)
                  and not (id(n) in in_test and _is_static(n, static))]
    return found


def test_no_boolean_operator_produces_a_value_in_a_kernel():
    offenders = [f"{p.relative_to(SRC)}:{line} ({name})" for p in sorted(SRC.rglob("*.py"))
                 for name, line in _offenders(ast.parse(p.read_text()))]
    assert offenders == [], f"`and` / `or` producing a value in a kernel body (use & | on parenthesised terms): {offenders}"


def test_the_gate_sees_a_value_and_spares_a_constexpr_test():
    src = ("import triton\nimport triton.language as tl\n\n@triton.jit\ndef k(X, N: tl.constexpr):\n"
           "    if N > 1 and N < 8:\n        pass\n    tl.static_assert(N > 1 and N < 8)\n")
    assert _offenders(ast.parse(src)) == [("k", 8)]


def _kernel(body, header="import triton\nimport triton.language as tl\n\n@triton.jit\n"):
    return _offenders(ast.parse(header + "def k(X, N: tl.constexpr, M):\n" + body))


def test_the_gate_spares_a_test_by_what_its_operands_are_not_by_where_it_stands():
    """A boolean operator in a test is scalar Python only on compile-time values: on a runtime operand (a tensor, a
    program id, a non-constexpr argument) a `while` or conditional-expression test is the same `and` Triton deprecates
    (qa on 1214). Red: a runtime `and` in a `while` or an `if` expression test passes."""
    assert _kernel("    while tl.program_id(0) > 0 and M > 1:\n        pass\n") == [("k", 6)]
    assert _kernel("    y = 1 if M > 0 and N > 1 else 2\n") == [("k", 6)]
    assert _kernel("    if tl.load(X) > 0 or M:\n        pass\n") == [("k", 6)]
    assert _kernel("    if tl.constexpr(M.dtype.is_int()) or tl.constexpr(M.dtype.is_fp64()):\n        pass\n") == []
    assert _kernel("    if N > 1 and N < 8:\n        pass\n") == []
    assert _kernel("    while N > 1 and N < 8:\n        pass\n") == []
    assert _kernel("    K = N * 2\n    y = 1 if K > 1 and N < 8 else 2\n") == []
    assert _kernel("    for i in tl.static_range(4):\n        if i > 0 and N > 1:\n            pass\n") == []


def test_the_gate_finds_a_kernel_whatever_the_decorator_is_imported_as():
    """The jit decorator is found through the module's imports, not by its spelling (qa on 1214): `import triton as t`
    and `@t.jit`, `from triton import jit as j`, `from triton.experimental import gluon as g` and `@g.jit`."""
    body = "def k(X, N):\n    y = N > 1 and N < 8\n"
    for header in ("import triton as t\n\n@t.jit\n", "from triton import jit as j\n\n@j\n",
                   "from triton import jit as j\n\n@j(debug=True)\n",
                   "from triton.experimental import gluon as g\n\n@g.jit\n"):
        assert _offenders(ast.parse(header + body)) == [("k", 5)], header
    assert _offenders(ast.parse("import functools\n\n@functools.cache\n" + body)) == []
