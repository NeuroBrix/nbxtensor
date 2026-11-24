"""A synthetic launch carries every integer argument at the VARIANT'S OWN width (`ops/ew_synthetic.Synth.carry`; the
project lead, 24/11/2026).

The red (V100, Sana-4K at 4096, the VAE decode's pixel shuffle through a broadcast view): "element-wise variant
pixel_shuffle_broadcast_aware_kernel is not proven: its synthetic launch at count 16 (contiguous) specialises otherwise
than the variant [('stride_out_n', ('i32', "'D'"), ('i64', "'D'"))]". At 4096 the output's batch stride is 2^31, which
Triton's binder types i64; the recipe's small dimensions made it i32 — another binary — and the variant, legitimate,
was refused. The class: ANY integer a recipe draws for a variant specialised i64.

The cure, in one helper: an i64 integer is carried past 2^31 (its small value + 2^31, the same divisibility), a count
stays small and 64-bit by declaration; a wide value the kernel dereferences makes the recipe size a buffer past
`ew_synthetic.SMALL`, and the variant is refused BY NAME (`WideNotProvable`), never proven on another binary.

What each test does if the code is wrong:
1. the pixel shuffle variants with an i64 batch stride (Sana-4K) and the other wide launches of the shipped graphs
   (a causal pad's i64 xs0, a cat input past 2^31 elements) are refused instead of reproduced on small buffers;
2. some class-tiled kernel's recipe, given one of its integers i64 (with every integer at least as large in its own
   geometry, with and without its count i64), makes another binary or fails other than by the named wide refusal; or
   the list of kernels whose wide form is refused by name moved without `tools/ew_wide_scalars.md` moving with it;
3. a class that moves off a failing tile says nothing, or says it at every launch, or does not name the tile it now
   runs on, the tile it left and the variant that fails it.
Device-free.
Run: python -m pytest tests/test_a_synthetic_launch_carries_each_integer_at_the_variants_own_width.py
"""
import ast
import importlib
import pathlib
import re

import pytest

from nbxtensor.backend.launcher import _param_table, nbx_binder
from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.ops import ew_proof, ew_synthetic as E
from nbxtensor.ops.ew_tile import EwTile

CANDS = [(b, w, 3) for b in (256, 512, 1024, 2048, 4096) for w in (2, 4, 8)]
TILE = EwTile(1024, 65536, "tile")
ROOT = pathlib.Path(E.__file__).resolve().parents[1]


def _ptr(i):
    """A pointer argument as the binder reads it: a 16-byte aligned fp32 stand-in, the i-th of a launch."""
    return E._Fake((i + 1) << 20, NBXDtype.float32, 4, 4)


def _shuffle_bcast(h, w, c_pre, b, c):
    """The broadcast-view pixel shuffle's arguments at a (h, w) image of c_pre channels expanded b times (r = 2)."""
    from nbxtensor.ops.pixel_shuffle import pixel_shuffle_broadcast_aware_kernel as k
    oh, ow = 2 * h, 2 * w
    return k, (_ptr(0), _ptr(1), c * oh * ow, c, oh, ow, 2, b, c_pre * h * w, h * w, 0, w, 1,
               c * oh * ow, oh * ow, ow, 1), {"BLOCK_SIZE": TILE}


def _shuffle(h, w, c_in):
    """The plain pixel shuffle's arguments: a contiguous [1, c_in, h, w] to [1, c_in / 4, 2h, 2w]."""
    from nbxtensor.ops.pixel_shuffle import pixel_shuffle_kernel as k
    c, oh, ow = c_in // 4, 2 * h, 2 * w
    return k, (_ptr(0), _ptr(1), c * oh * ow, c, h, w, 2, oh, ow, c_in * h * w, h * w, w, 1,
               c * oh * ow, oh * ow, ow, 1), {"BLOCK_SIZE": TILE}


def _unshuffle(ih, iw, c):
    """The pixel unshuffle's arguments: a contiguous [1, c, ih, iw] to [1, 4c, ih / 2, iw / 2]."""
    from nbxtensor.ops.pixel_unshuffle import pixel_unshuffle_kernel as k
    h, w, co = ih // 2, iw // 2, 4 * c
    return k, (_ptr(0), _ptr(1), co * h * w, c, ih, iw, 2, co, h, w, c * ih * iw, ih * iw, iw, 1,
               co * h * w, h * w, w, 1), {"BLOCK_SIZE": TILE}


def _pad_nd(t, h, w, which="constant"):
    """A causal pad of a [2, 128, t, h, w] video by two frames in front, as `constant_pad_nd_wrapper` launches it."""
    from nbxtensor.ops import pad_op
    k = pad_op.constant_pad_nd_kernel if which == "constant" else pad_op.replication_pad_nd_kernel
    din, dout = [2, 128, t, h, w], [2, 128, t + 2, h, w]
    st = [128 * t * h * w, t * h * w, h * w, w, 1]
    total = 2 * 128 * (t + 2) * h * w
    args = (_ptr(0), _ptr(1), total, *din, *dout, *st, 0, 0, 2, 0, 0) + ((0.0,) if which == "constant" else ())
    return k, args, {"BLOCK_SIZE": TILE, "WIDE": total >= 1 << 31}


def _cat(t, h, w):
    """A cat of a [2, 256, 1, h, w] and a [2, 256, t, h, w] on dim 2, as `NBXTensor.cat` launches it."""
    from nbxtensor.ops.cat_op import cat_copy_kernel_4 as k
    post = h * w
    return k, (_ptr(0), _ptr(1), _ptr(2), _ptr(0), _ptr(0), 1, t, 0, 0, t + 1, post, 0, 1, t + 1, t + 1,
               2 * 256 * post, 2 * 256 * t * post, 0, 0), {"BLOCK_X": TILE}


#: the wide launches the shipped graphs reach (tools/ew_wide_scalars.md), and the same shape classes small
SHIPPED = {
    "Sana-4K pixel shuffle through a broadcast view (the red)": lambda: _shuffle_bcast(2048, 2048, 256, 2, 128),
    "pixel shuffle [1, 512, 2048, 2048]": lambda: _shuffle(2048, 2048, 512),
    "pixel unshuffle [1, 128, 4096, 4096]": lambda: _unshuffle(4096, 4096, 128),
    "CogVideoX causal constant pad [2, 128, 61, 480, 720]": lambda: _pad_nd(61, 480, 720),
    "CogVideoX causal replication pad [2, 128, 61, 480, 720]": lambda: _pad_nd(61, 480, 720, "replication"),
    "CogVideoX cat [2, 256, 1 + 60, 240, 360]": lambda: _cat(60, 240, 360),
}
SMALL_TWINS = [lambda: _shuffle_bcast(64, 48, 8, 2, 4), lambda: _shuffle(64, 48, 16), lambda: _unshuffle(64, 48, 4),
               lambda: _pad_nd(5, 24, 40), lambda: _pad_nd(5, 24, 40, "replication"), lambda: _cat(5, 24, 40)]


def _spec(k, args, kw):
    """The variant a launch of these arguments compiles: the binder's specialization."""
    return list(nbx_binder(k, args, kw)[1])


# --- 1. the red, and the shipped wide launches --------------------------------------------------------------------
def test_the_pixel_shuffle_variant_with_a_64_bit_output_batch_stride_is_reproduced():
    """The red variant: stride_out_n (and the count) i64. Its synthetic launches specialise exactly as it does, the
    batch stride carried past 2^31, one image covering the ladder (N is 0 on every lane), the buffers small."""
    k, args, kw = SHIPPED["Sana-4K pixel shuffle through a broadcast view (the red)"]()
    spec = _spec(k, args, kw)
    names = [p[0] for p in _param_table(k)]
    assert dict(zip(names, spec))["stride_out_n"] == ("i64", "D")             # Triton's view of the 4096 launch
    launches = E.synthesise(k, spec, CANDS)                                    # raised the red before the cure
    assert sorted(int(ln.count) for ln in launches) == [16, 240, 4080, 4112, 8240]
    for ln in launches:
        assert ln.args["stride_out_n"] >= 1 << 31 and ln.args["stride_out_n"] % 16 == 0
        assert ln.args["output_ptr"].elements == ln.count                     # the wide stride never dereferenced
        E.check(k, spec, ln)                                                   # the same binary, once more


@pytest.mark.parametrize("what", sorted(SHIPPED))
def test_every_wide_launch_of_the_shipped_graphs_is_reproduced_on_small_buffers(what):
    """Each wide shape class the shipped graphs reach: its variant (i64 integers included) is reproduced, kilobytes."""
    k, args, kw = SHIPPED[what]()
    spec = _spec(k, args, kw)
    assert any(t == "i64" for t, _a in spec)
    launches = E.synthesise(k, spec, CANDS)
    assert max(b.elements for ln in launches for b in ln.bufs()) < 1 << 20


# --- 2. the census ------------------------------------------------------------------------------------------------
#: the class-tiled kernels some wide form of which (an i64 integer the kernel dereferences) is refused by name, the
#: list of tools/ew_wide_scalars.md
WIDE_REFUSED = sorted("""
_token_bitmask_kernel adaptive_avg_pool2d_kernel bit_reverse_rows_kernel cat_copy_kernel_4 col2im1d_kernel
constant_pad_2d_kernel constant_pad_nd_kernel conv2d_bias_inplace_kernel dft_rfft_matrix_kernel
fft_stage_rows_kernel flip_strided_kernel gather_kernel idft_c2r_matrix_kernel index_add_gather_kernel index_put_kernel
index_select_mid_kernel pixel_shuffle_broadcast_aware_kernel pixel_shuffle_kernel pixel_unshuffle_kernel
replication_pad_nd_kernel scatter_add_kernel scatter_kernel scatter_reduce_amax_kernel scatter_reduce_amin_kernel
silu_mul_split_kernel upsample_nearest2d_kernel upsample_nearest3d_kernel
""".split())


def _class_tiled():
    """{kernel name: (module, its FunctionDef)} of every kernel the class tile reaches: launched with `ew_meta` at a
    call site, or declaring a synthetic recipe."""
    defs, launched = {}, set()
    for p in sorted(ROOT.rglob("*.py")):
        tree = ast.parse(p.read_text())
        mod = "nbxtensor." + ".".join(p.relative_to(ROOT).with_suffix("").parts)
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef) and any("jit" in ast.unparse(d) for d in n.decorator_list):
                defs[n.name] = (mod, n)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript) and any(
                    "ew_meta(" in ast.unparse(a) or "m['BLOCK_SIZE']" in ast.unparse(a)
                    or 'm["BLOCK_SIZE"]' in ast.unparse(a) for a in n.args + [kw.value for kw in n.keywords]):
                v = n.func.value
                launched.add(v.id if isinstance(v, ast.Name) else getattr(v, "attr", None))
    return {n: d for n, d in defs.items()
            if n in launched or any("synthetic(" in ast.unparse(x) for x in d[1].decorator_list)}


def _pointers(node):
    """The parameters a kernel loads or stores through: the leftmost operand of each load / store / atomic address
    (through local names), or a parameter named as a pointer."""
    params = [a.arg for a in node.args.args]
    alias = {n.targets[0].id: n.value for n in ast.walk(node)
             if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}

    def base(e, depth=0):
        """The parameter an address expression starts from, None when it starts from none."""
        while isinstance(e, ast.BinOp) and isinstance(e.op, ast.Add):
            e = e.left
        if isinstance(e, ast.Name):
            if e.id in params:
                return e.id
            if e.id in alias and depth < 5:
                return base(alias[e.id], depth + 1)
        return None
    out = {p for p in params if "ptr" in p}
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.args and (
                n.func.attr in ("load", "store") or n.func.attr.startswith("atomic")):
            out.add(base(n.args[0]))
    return out - {None}


def _base(k, seed):
    """(count name, the variant the recipe's own launch compiles, its integer values) for a seed specialization: the
    recipe run once and its arguments bound as Triton binds them. An integer the recipe gives no value is a float
    of the call site's, a buffer it makes a pointer, an unspecialised flag the recipe needs specialised 'D'."""
    seed, names = list(seed), [p[0] for p in _param_table(k)]
    for _ in range(16):
        try:
            recipe = E.recipe_of(k)
            cname = E._count_param(k, seed, recipe)
            S = E.Synth(k, seed, 4097, E.LAYOUTS[0], counts=(cname,) + tuple(getattr(recipe, "counts", ())),
                        most=8240)
            args, axes = recipe(S)
            ln = E._complete(S, dict(args), axes)
            break
        except E.NotSynthesisable as why:
            m = (re.search(r"no value for integer (\w+)", str(why)) or re.search(r"makes a buffer for (\w+)", str(why))
                 or re.search(r"(\w+) specialised None", str(why)))
            if not m:
                raise
            seed[names.index(m.group(1))] = (("fp32", None) if "no value" in str(why) else ("*fp32", "D")
                                             if "buffer" in str(why) else ("i32", "D"))
    fake, at = [], 0x100000
    for v in ln.args.values():
        if isinstance(v, E.Buf):
            d, elem = E.dtype_of(v.ty)
            at += 1 << 20
            fake.append(E._Fake(at if v.aligned else at + elem, d, v.elements, elem))
        else:
            fake.append(v)
    return cname, _spec(k, tuple(fake), {}), {n: v for n, v in ln.args.items() if not isinstance(v, E.Buf)}


def _bases(name, k, node):
    """The variants the census starts from: the recipe's own launch at seeds whose integers are all '', all 'D', all
    unspecialised, and, for the shape classes the shipped graphs reach wide, a small real launch of each."""
    ptrs = _pointers(node)
    seed = [("constexpr", TILE if pn in ("BLOCK_SIZE", "BLOCK", "BLOCK_X") else {"NDIM": 3, "TAPS": 2}.get(pn, False))
            if cx else ("*fp32", "D") if pn in ptrs else ("i32", None) for pn, cx, *_ in _param_table(k)]
    out = []
    for att in ("", "D", None):
        try:
            b = _base(k, [(t, att) if t == "i32" else (t, a) for t, a in seed])
        except E.NotSynthesisable:
            continue
        if b[1] not in [x[1] for x in out]:
            out.append(b)
    for twin in SMALL_TWINS:
        tk, args, kw = twin()
        if tk.__name__ == name:
            names = [p[0] for p in _param_table(tk)]
            out.append((E._count_param(tk, _spec(tk, args, kw), E.recipe_of(tk)), _spec(tk, args, kw),
                        dict(zip(names, args))))
    return out


def test_every_class_tiled_kernel_reproduces_every_width_of_its_integers_or_is_refused_by_name():
    """For every class-tiled kernel and its recipe, from each base variant: each integer i64 (with every integer at
    least as large in the base's own geometry — a size past 2^31 makes every size above it wide too), with and without
    the count i64, and every integer i64 at once. Each such variant is reproduced exactly (the binder's view of every
    synthetic launch, `ew_synthetic.check`) or refused by name as a wide form that cannot be proven on small buffers;
    every other outcome is listed as a failure, by kernel."""
    failures, refused, seen = [], set(), 0
    for name, (mod, node) in sorted(_class_tiled().items()):
        k = getattr(importlib.import_module(mod), name)
        bases = _bases(name, k, node)
        if not bases:
            failures.append(f"{name}: no base variant: its recipe runs on no seed")
            continue
        names = [p[0] for p in _param_table(k)]
        for cname, s0, vals in bases:
            ints = [i for i, (t, _a) in enumerate(s0) if t in ("i32", "i64") and names[i] != cname]
            ci = names.index(cname)
            count = [] if s0[ci][0] == "constexpr" else [ci]
            groups = [count] if count else []
            for i in ints:
                above = [j for j in ints if int(vals[names[j]]) >= int(vals[names[i]])]
                groups += [above, above + count]
            groups += [ints + count] if ints else []
            for group in groups:
                spec = list(s0)
                for j in group:
                    spec[j] = ("i64", s0[j][1])
                seen += 1
                try:
                    E.synthesise(k, spec, CANDS)                 # checks every launch against `spec`
                except E.WideNotProvable as why:
                    assert "cannot be proven on small buffers" in str(why)
                    refused.add(name)
                except Exception as why:                         # noqa: BLE001  every other end is a failure
                    failures.append(f"{name} with {[names[j] for j in group]} i64: {type(why).__name__}: "
                                    f"{str(why)[:240]}")
    assert seen > 500
    assert failures == [], "\n".join(failures)
    assert sorted(refused) == WIDE_REFUSED, sorted(refused)
    doc = (ROOT.parents[1] / "tools" / "ew_wide_scalars.md").read_text()
    assert all(f"`{n}`" in doc for n in WIDE_REFUSED)


# --- 3. the move's line -------------------------------------------------------------------------------------------
LADDER = [(256, 2, 3), (512, 4, 3), (1024, 4, 3), (1024, 8, 3), (2048, 8, 3)]
CLS = 65536


class _K:
    """A kernel as `settle` reads it: a name and the module of its function."""
    __name__ = "copy_kernel"
    fn = type("F", (), {"__module__": "nbxtensor.tensor"})


def _home(passed):
    """A proof table whose every variant passes the tiles `passed` and records a fault at 1024x8."""
    tab = object.__new__(ew_proof._Table)
    tab.key = "test"
    tab.kept, tab.shipped, tab.unkept, tab.seen_classes = {}, {}, {}, {}
    tab.record = lambda vid: {"passed": [list(t) for t in passed],
                              "faults": [{"tile": [1024, 8, 3], "fault": "an illegal memory access (700)"}]}
    return tab


def _launch(meta, n=4096):
    """The arguments `settle` receives for a launch handed `meta` by `**ew_meta(n)`."""
    from nbxtensor.ops import elementwise_launch as EL
    from nbxtensor.ops.ew_tile import class_tiles
    options = {k: v for k, v in meta.items() if k != "BLOCK_SIZE"}
    bound = {"x": 0, "n": n, "BLOCK_SIZE": meta["BLOCK_SIZE"]}
    spec = [("*fp32", "D"), ("i32", ""), ("constexpr", meta["BLOCK_SIZE"])]
    return EL.ew_grid(n), bound, spec, options, class_tiles(list(bound.values()) + list(options.values()))


def test_a_class_move_is_said_once_naming_its_new_tile_the_tile_it_left_and_the_failing_variant(monkeypatch, capsys):
    """Twenty launches of a class whose tile faults on a variant: ONE line, naming the tile the class now runs on,
    the tile it left, the variant (kernel and id) and its fault."""
    from nbxtensor.ops import elementwise_launch as EL
    monkeypatch.setattr(EL, "class_candidates", lambda: list(LADDER))
    monkeypatch.setattr(EL, "_MEMO", {})
    monkeypatch.setattr(ew_proof, "_SAID", set())
    monkeypatch.setattr(ew_proof, "table", lambda: _home([LADDER[1]]))
    monkeypatch.setattr(EL, "move_class_winner", lambda nb, b, w: EL.class_meta(nb, b, w))
    for _ in range(20):
        grid, bound, spec, options, tiles = _launch(EL.class_meta(CLS, 1024, 8))
        assert ew_proof.settle(_K(), grid, bound, spec, options, tiles)[1]["BLOCK_SIZE"] == 512
    lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("[ew-proof] class")]
    vid, _tile = ew_proof.variant_of(_K(), spec, options, tiles)
    assert len(lines) == 1, lines
    line = lines[0]
    assert f"class {CLS} now runs on tile 512x4 (stages 3)" in line
    assert "left tile 1024x8 stages 3" in line
    assert f"variant {ew_proof.stamp_key({'v': vid})} of copy_kernel" in line
    assert "an illegal memory access (700)" in line
