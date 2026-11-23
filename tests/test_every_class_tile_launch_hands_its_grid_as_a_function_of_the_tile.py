"""Every launch whose tile or warps come from the element-wise class (`ew_meta`) hands its grid as a FUNCTION of the
tile — one of the helpers (`ew_grid`, `ew_grid_of`, `_1d_grid`, `_1d_grid_fixed`, `_ew_grid`) or a lambda — never a
tuple computed from the tile it was handed (orchestrator 23/11/2026, the token bitmask: a grid computed from the class
tile by the caller ran the proof's candidates and the moved launch on a wrong grid, and the moved launch wrote part
of its buffer only). The run-time refusal ("grid fixed by the caller", `ops/ew_proof.settle`) is the net for a kernel
added later; this test is the proof it is unreachable on the shipped source.

Read device-free, from the library's source (ast): a launch is a call of `kernel[grid](...)`; it reads the class
when a keyword is `**<a value of ew_meta>` or a value derived from one (`_m['BLOCK_SIZE']`, `BLOCK = _m[...]`).
Red on the base: the hand-computed grids (`(triton.cdiv(n, BLOCK),)`, `(rows, triton.cdiv(vocab, m["BLOCK_SIZE"]))`)."""
import ast
import pathlib

import nbxtensor

HELPERS = {"ew_grid", "ew_grid_of", "_ew_grid_of", "_1d_grid", "_1d_grid_fixed", "_ew_grid"}
META_CALLS = {"ew_meta", "_ew_meta", "_ew", "class_meta"}


def _callee(node):
    f = node.func if isinstance(node, ast.Call) else None
    return f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _scan_function(fn, path, bad):
    meta, grids = set(), {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name, value = node.targets[0].id, node.value
            if any(_callee(c) in META_CALLS for c in ast.walk(value) if isinstance(c, ast.Call)) \
                    or (_names(value) & meta and not isinstance(value, (ast.Tuple, ast.Lambda))):
                meta.add(name)
            grids[name] = value
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)):
            continue
        reads_class = any(
            (kw.arg is None and (any(_callee(c) in META_CALLS for c in ast.walk(kw.value) if isinstance(c, ast.Call))
                                 or _names(kw.value) & meta))
            or (kw.arg is not None and _names(kw.value) & meta)
            for kw in node.keywords)
        if not reads_class:
            continue
        grid = node.func.slice
        if isinstance(grid, ast.Name) and grid.id in grids:
            grid = grids[grid.id]
        ok = isinstance(grid, ast.Lambda) or (isinstance(grid, ast.Call) and _callee(grid) in HELPERS)
        if not ok:
            bad.append(f"{path}:{node.lineno}: {ast.unparse(node.func)[:120]}")


def class_tile_launches_with_a_fixed_grid():
    root = pathlib.Path(nbxtensor.__file__).parent
    bad = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _scan_function(fn, path.relative_to(root), bad)
    return sorted(set(bad))


def test_no_class_tile_launch_is_given_a_fixed_grid():
    bad = class_tile_launches_with_a_fixed_grid()
    assert bad == [], "launches handed a grid computed from the class tile:\n" + "\n".join(bad)


if __name__ == "__main__":
    for line in class_tile_launches_with_a_fixed_grid():
        print(line)
