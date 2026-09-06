#!/usr/bin/env python3
"""Fail the build on an undefined \\ref, a missing \\input, or an unbalanced brace count.

LaTeX resolves an undefined reference to '??' and warns rather than failing, which is easy to miss in a long log and
easy to submit. This makes it an error. \\open{} markers are counted and printed, not treated as errors, so a draft
still builds; a submission check should require the count to be zero.
"""
import glob
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def strip_comments(t: str) -> str:
    """Remove LaTeX comments before any analysis: a brace, ref or input inside a comment is not real, and counting it
    produces false failures (an explanatory comment mentioning \\disc{ is not an unbalanced brace)."""
    out = []
    for line in t.split("\n"):
        i, n = 0, len(line)
        while i < n:
            if line[i] == "%" and (i == 0 or line[i - 1] != "\\"):
                break
            i += 1
        out.append(line[:i])
    return "\n".join(out)


def main() -> int:
    tex = sorted(glob.glob(str(HERE / "sections/*.tex"))) + [str(HERE / "main.tex")]
    tables = sorted(glob.glob(str(HERE / "tables/*.tex")))
    labels, refs, missing_inputs, unbalanced = set(), {}, [], []
    for f in tex + tables:
        t = strip_comments(Path(f).read_text())
        labels |= set(re.findall(r"\\label\{([^}]+)\}", t))
        for r in re.findall(r"\\ref\{([^}]+)\}", t):
            refs.setdefault(r, []).append(Path(f).name)
        if t.count("{") != t.count("}"):
            unbalanced.append((Path(f).name, t.count("{"), t.count("}")))
        for inc in re.findall(r"\\input\{([^}]+)\}", t):
            if not (HERE / f"{inc}.tex").exists() and not (HERE / inc).exists():
                missing_inputs.append((Path(f).name, inc))
    errs = []
    for r, where in sorted(refs.items()):
        if r not in labels:
            errs.append(f"undefined \\ref{{{r}}} used in {', '.join(sorted(set(where)))}")
    errs += [f"missing \\input{{{i}}} in {f}" for f, i in missing_inputs]
    errs += [f"unbalanced braces in {f}: {a} open, {b} close" for f, a, b in unbalanced]
    prov = HERE / "tables" / "_provenance.tex"
    if prov.exists() and "+dirty" in prov.read_text():
        errs.append("tables/_provenance.tex records a DIRTY working tree (\\resultscommit contains '+dirty'). "
                    "A paper whose contribution is provenance must not ship that string. Commit the tree, then "
                    "re-run `make tables && make figures` so the stamp names a clean commit.")
    n_open = sum(len(re.findall(r"\\open\{", strip_comments(Path(f).read_text()))) for f in tex)
    n_disc = sum(len(re.findall(r"\\disc\{", strip_comments(Path(f).read_text()))) for f in tex)
    if errs:
        print("check_refs: FAILED", file=sys.stderr)
        for e in errs:
            print(f"  {e}", file=sys.stderr)
        return 1
    print(f"check_refs: ok -- {len(labels)} labels, {len(refs)} distinct refs all resolved; "
          f"{n_open} \\open markers, {n_disc} \\disc markers outstanding")
    return 0


if __name__ == "__main__":
    sys.exit(main())
