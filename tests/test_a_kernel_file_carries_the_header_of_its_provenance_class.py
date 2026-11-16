"""Every kernel file of docs/kernels_provenance.md carries the header of its class, and a rewritten or copied one keeps
the upstream notice it derives from (the project named in its text and in THIRD_PARTY_NOTICES.md)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROWS = [m.groups() for m in re.finditer(r"^\| `([^`]+)` \| ([^|]*) \| ([^|]*) \| (A|B|C|UNSURE|-) \|", (ROOT / "docs/kernels_provenance.md").read_text(), re.M)]
NOTICES = (ROOT / "THIRD_PARTY_NOTICES.md").read_text()
LICENCE_MARK = {"FlagGems": "FlagOpen/FlagGems", "attorch": "BobMcDear/attorch", "Liger-Kernel": "linkedin/Liger-Kernel", "flash-attention": "Dao-AILab/flash-attention"}


def _head(rel):
    return [l for l in (ROOT / rel).read_text().split("\n")[:4]]


def test_the_table_counts_every_class_and_names_the_file_it_waits_on():
    classes = [r[3] for r in ROWS]
    assert classes.count("A") >= 6 and classes.count("B") >= 79 and "C" not in classes
    assert [r for r in ROWS if r[3] == "-"][0][0].endswith("gated_delta_rule_chunked.py")


def test_a_class_a_file_carries_the_neurobrix_copyright_alone():
    for rel, _, _, cls in ROWS:
        if cls == "A":
            assert _head(rel)[0] == "# Copyright NeuroBrix, Apache-2.0", rel


def test_a_class_b_file_says_it_was_rewritten_names_its_project_and_keeps_the_original_notice():
    for rel, proj, lic, cls in ROWS:
        if cls != "B":
            continue
        head, text = _head(rel), (ROOT / rel).read_text()
        assert head[0].startswith("# Rewritten by NeuroBrix (Copyright NeuroBrix, Apache-2.0); derived from "), rel
        assert head[1].startswith("# Change: "), rel
        for name in [p.strip() for p in proj.split(",")]:
            assert name in head[0], (rel, name)
            body = "\n".join(text.split("\n")[3:])
            assert re.search(name.replace("-", "[-_ ]?"), body, re.I), f"{rel} lost its original notice naming {name}"
            assert LICENCE_MARK[name] in NOTICES and f"`ops/{Path(rel).name}`" in NOTICES, rel


def test_a_class_c_file_keeps_its_original_notice_alone():
    for rel, _, _, cls in ROWS:
        if cls == "C":
            assert not _head(rel)[0].startswith("# Rewritten by NeuroBrix"), rel
            assert re.search(r"copyright", "\n".join(_head(rel)), re.I), rel


def test_an_unsure_file_is_left_without_a_neurobrix_header():
    for rel, _, _, cls in ROWS:
        if cls == "UNSURE":
            assert not _head(rel)[0].startswith(("# Rewritten by NeuroBrix", "# Copyright NeuroBrix")), rel


def test_no_third_party_notice_text_was_shortened():
    for lic in ("Apache License", "MIT License", "BSD 2-Clause", "BSD 3-Clause"):
        assert lic in NOTICES
    assert len(NOTICES.splitlines()) >= 311
