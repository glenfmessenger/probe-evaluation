"""Every committed aggregate must ship with the per-case scores it was computed from.

An AUC on its own can be quoted but never re-interrogated. You cannot put a confidence interval on it, re-stratify it,
check it against a different control, or answer a reviewer who asks what the distribution looked like -- and if the
machine that produced it is gone, none of that is recoverable. That is not hypothetical here: Arm 2's per-case scores
were written into the gitignored activations directory, the directory did not survive the run, and the manuscript had
to report analytic approximations where it wanted bootstraps (Appendix A, panel B).

This module encodes the rule that came out of it. Activation tensors may be ignored; per-case scores may not.
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Keys that mean "this file reports an aggregate over cases". Any of them makes per-case scores mandatory.
AGGREGATE_KEYS = {"auc", "auc_all", "auc_length_stratified", "auc_vs_xstest", "auc_vs_jbb"}
# Keys that satisfy the requirement, whether the scores are inline or in a sibling file.
SCORE_KEYS = {"scores", "per_case", "raw_scores"}

# Once the record of what was lost: Arm 2's three directories were produced before the scores/activations split
# existed and their scores died with the VM. They were re-run on 2026-09-11 (GATES.md Amendment 7b) with the scores
# committed under results/gates/scores/, so the exemption is empty and must stay empty. A results directory that
# lands here instead of committing its scores is the failure this file was written to prevent.
KNOWN_GAPS = set()


def _has_scores(obj) -> bool:
    if isinstance(obj, dict):
        if any(k in obj and obj[k] for k in SCORE_KEYS):
            return True
        return any(_has_scores(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_scores(v) for v in obj[:4])
    return False


def _has_aggregate(obj) -> bool:
    if isinstance(obj, dict):
        if any(k in obj for k in AGGREGATE_KEYS):
            return True
        return any(_has_aggregate(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_aggregate(v) for v in obj[:4])
    return False


def _result_files():
    root = REPO / "results"
    if not root.is_dir():
        pytest.skip("results/ not present")
    return sorted(p for p in root.rglob("*.json") if "activations" not in p.parts)


def _file_carries_scores(path: Path) -> bool:
    if path.suffix == ".npz":
        return True
    try:
        return _has_scores(json.load(open(path)))
    except (json.JSONDecodeError, OSError):
        return False


def _covered(f: Path, d) -> bool:
    """A committed aggregate is covered when its per-case scores can be found from it, by any of four routes."""
    if _has_scores(d):                                                  # 1. inline
        return True
    declared = d.get("source") if isinstance(d, dict) else None         # 2. declared, and the target really has them
    for rel in ([declared] if isinstance(declared, str) else (declared or [])):
        for hit in sorted(REPO.glob(rel)) or [REPO / rel]:
            if hit.exists() and _file_carries_scores(hit):
                return True
    for sib in f.parent.glob("*.json"):                                 # 3. a sibling in the same directory
        if sib != f and _file_carries_scores(sib):
            return True
    for pat in ("*scores*.json", "*scores*.npz"):                       # 4. a scores file beside or below
        if list(f.parent.glob(pat)) or list(f.parent.glob("scores/" + pat)):
            return True
    return False


def test_every_aggregate_ships_with_its_per_case_scores():
    offenders = []
    for f in _result_files():
        rel = f.relative_to(REPO).as_posix()
        if any(rel.startswith(g + "/") for g in KNOWN_GAPS):
            continue
        try:
            d = json.load(open(f))
        except json.JSONDecodeError:
            continue
        if _has_aggregate(d) and not _covered(f, d):
            offenders.append(rel)
    assert not offenders, (
        "these results report an AUC with no per-case scores committed anywhere, so the number cannot be "
        "re-analysed:\n  " + "\n  ".join(offenders) +
        "\n\nWrite the scores next to the aggregate (see gates.runb.scores_dir), or declare a \"source\" naming the "
        "committed file that holds them. Activations may be gitignored; scores may not. Do not add the directory to "
        "KNOWN_GAPS to make this pass.")


def test_the_guard_would_catch_a_regression(tmp_path):
    """The rule is only worth having if it fires. An aggregate with no scores, no source and no sibling must fail."""
    orphan = tmp_path / "orphan.json"
    orphan.write_text(json.dumps({"results": {"test": {"final": {"auc": 0.87, "n_pos": 10, "n_neg": 10}}}}))
    d = json.load(open(orphan))
    assert _has_aggregate(d)
    assert not _covered(orphan, d)
    withscores = tmp_path / "withscores.json"
    withscores.write_text(json.dumps({"auc": 0.87, "scores": {"pos": [0.1, 0.2], "neg": [0.0]}}))
    assert _covered(withscores, json.load(open(withscores)))


def test_known_gaps_are_exactly_the_documented_ones():
    """The exemption list was the record of what was lost; the loss was repaired by re-running Arm 2 on 2026-09-11.
    If it ever grows, the rule has been broken and the growth is what needs review."""
    assert KNOWN_GAPS == set()
    for kind in ("b2", "b2_grouped", "b2_templated"):
        import re
        files = [f for f in (REPO / "results/gates/scores").glob(f"{kind}_*_test_scores.json")
                 if re.match(rf"{kind}_(google|meta_llama)_", f.name)]       # b2_* must not swallow b2_grouped_*
        assert len(files) == 7, kind


def test_gitignore_covers_activations_but_not_scores():
    ig = (REPO / ".gitignore").read_text()
    assert "results/gates/activations/" in ig and "results/phase2/activations/" in ig
    for line in ig.splitlines():
        t = line.split("#")[0].strip()
        if t and "scores" in t:
            pytest.fail(f".gitignore excludes a scores path, which the results rule forbids: {line!r}")


def test_run_configs_separate_scores_from_activations():
    import yaml
    for name, key in (("gates_config.yaml", "scores_dir"), ("phase2_config.yaml", "scores_dir")):
        cfg = yaml.safe_load((REPO / name).read_text())
        assert key in cfg, f"{name} must declare {key} so per-case scores land outside the ignored activations dir"
        assert "activations" not in cfg[key], f"{name}: {key} points inside the activations tree ({cfg[key]!r})"
