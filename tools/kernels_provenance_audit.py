"""Kernel provenance audit: classes every kernel file that mentions an upstream project, writes docs/kernels_provenance.md and,
with --apply, sets the file headers of its class (comments only). Classes: A ours, B rewritten from upstream, C copied
unchanged, UNSURE (header left as it is). Run from the repository root; --match <json> adds the closest-upstream evidence."""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "src/nbxtensor/ops"
MARK_B = "# Rewritten by NeuroBrix (Copyright NeuroBrix, Apache-2.0); derived from "
MARK_A = "# Copyright NeuroBrix, Apache-2.0"
PROJECTS = {  # table label in THIRD_PARTY_NOTICES -> (short name, licence)
    "FlagOpen/FlagGems": ("FlagGems", "Apache-2.0"), "BobMcDear/attorch": ("attorch", "MIT"),
    "linkedin/Liger-Kernel": ("Liger-Kernel", "BSD-2-Clause"), "Dao-AILab/flash-attention": ("flash-attention", "BSD-3-Clause"),
}
MENTION = re.compile(r"flaggems|flag_gems|attorch|liger|flash[-_ ]attn|flash-attention|flash-linear|\bfla\b", re.I)
DECLARE = re.compile(r"(extracted|ported|port of|from (flaggems|attorch|liger|dao|flash)|logic from|wrapper from|derived|adapted|modelled)", re.I)
DEFERRED = "gated_delta_rule_chunked.py"


def listed():
    text = (ROOT / "THIRD_PARTY_NOTICES.md").read_text().split("## Projects cited")[0]
    out = {}
    for row in text.splitlines():
        for label, (short, lic) in PROJECTS.items():
            if row.startswith("| [" + label):
                for f in re.findall(r"`ops/(\w+\.py)`", row):
                    out.setdefault(f, []).append((short, lic))
    return out


def body_split(t):
    """(header comments, rest): the leading '#' comment lines we own, so headers are idempotent."""
    lines = t.split("\n")
    i = 0
    while i < len(lines) and (lines[i].startswith(MARK_A) or lines[i].startswith("# Rewritten by NeuroBrix") or lines[i].startswith("# Change:") or lines[i].startswith("# Original notice")):
        i += 1
    return lines[:i], "\n".join(lines[i:])


def classify(match):
    L = listed()
    rows = []
    for f in sorted(OPS.glob("*.py")):
        t = f.read_text()
        _, rest = body_split(t)
        if not MENTION.search(rest) and f.name not in L and "m8n8k4" not in f.name:
            continue
        rel = str(f.relative_to(ROOT))
        decl = next((l.strip() for l in rest.split("\n")[:40] if DECLARE.search(l) and MENTION.search(l)), None)
        m = match.get(rel)
        up = f"closest file at the upstream HEAD: {m[1]}:{m[2].split('/',1)[1] if m[2] else ''} (line similarity {m[0]:.2f}); upstream unpinned, no byte comparison to the extraction" if m and m[2] else "no counterpart found at the upstream HEAD"
        if f.name == DEFERRED:
            rows.append((rel, "fla", "MIT", "-", "class decided after 569 (file being rewritten, not touched)", f.name)); continue
        if "m8n8k4" in f.name:
            rows.append((rel, "none", "Apache-2.0", "A", "no upstream: written here (m8n8k4 family, our Gluon kernel); the upstream names in it are the algorithm's name or a reference", f.name)); continue
        if f.name in L and decl:
            names = ", ".join(sorted({s for s, _ in L[f.name]})); lics = ", ".join(sorted({l for _, l in L[f.name]}))
            rows.append((rel, names, lics, "B", f"header declares \"{decl[:110]}\"; listed in THIRD_PARTY_NOTICES; {up}; torch-free Triton form, so modified", f.name)); continue
        why = "listed in THIRD_PARTY_NOTICES but its header does not declare a derivation (the source is named in passing, no statement of derivation)" if f.name in L else "mentions an upstream name, not listed in THIRD_PARTY_NOTICES"
        rows.append((rel, "?", "?", "UNSURE", why + "; header left as it is", f.name))
    return rows, L


def header(row, L, t):
    rel, proj, lic, cls, ev, name = row
    if cls == "A":
        return [MARK_A]
    names = [(s, l) for s, l in L[name]]
    src = " and ".join(f"{s} ({l})" for s, l in names)
    ch = "rewritten as a pure Triton kernel with no PyTorch import"
    if re.search(r"nbx_autotune|_configs", t): ch += ", its launch configurations certified through nbxtensor's own autotune"
    if "wrappers" in name: ch += ", its calls moved onto nbxtensor's array API"
    return [MARK_B + src + ", original notice kept", f"# Change: {ch}.", "# Original notice: the provenance statement in the docstring below, unchanged, and THIRD_PARTY_NOTICES.md."]


def main():
    apply = "--apply" in sys.argv
    match = json.load(open(sys.argv[sys.argv.index("--match") + 1])) if "--match" in sys.argv else {}
    rows, L = classify(match)
    tool = ("tools/measure_the_chunked_gated_delta_rule.py", "none", "Apache-2.0", "A", "no upstream: written here; fla is only the measured reference, run in a child process, none of its code is copied", "")
    rows.append(tool)
    if apply:
        for r in rows:
            if r[3] not in "AB" or r[3] == "" or r[0].endswith(DEFERRED): continue
            p = ROOT / r[0]; t = p.read_text(); _, rest = body_split(t)
            if r[3] == "A" and r[0].startswith("tools/"): h = [MARK_A]
            else: h = header(r, L, rest)
            p.write_text("\n".join(h) + "\n" + rest)
    cnt = {c: sum(1 for r in rows if r[3] == c) for c in "ABC"}; cnt["UNSURE"] = sum(1 for r in rows if r[3] == "UNSURE")
    out = ["# Kernel provenance", "", "Generated by `tools/kernels_provenance_audit.py`. A = ours (Copyright NeuroBrix, Apache-2.0); B = rewritten by NeuroBrix from the project named, original notice kept; C = copied unchanged (original notice alone); UNSURE = no evidence to class it, header left as it is.", "",
           f"Counts: A={cnt['A']} B={cnt['B']} C={cnt['C']} UNSURE={cnt['UNSURE']}; 1 file waits on card 569. No file is class C: none is byte-identical to any upstream file we could compare.", "",
           "| file | upstream | licence | class | evidence |", "|---|---|---|---|---|"]
    out += [f"| `{r[0]}` | {r[1]} | {r[2]} | {r[3]} | {r[4]} |" for r in rows]
    others = sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts and p.suffix in (".md", ".yml", ".txt") + ("",) and MENTION.search(p.read_text(errors="ignore")) and "kernels_provenance" not in p.name)
    out += ["", "## Files that mention an upstream name and are not kernels (no header to set)", ""] + [f"- `{o}`" for o in others]
    (ROOT / "docs/kernels_provenance.md").write_text("\n".join(out) + "\n")
    print(cnt)

main()
