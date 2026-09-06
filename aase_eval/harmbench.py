"""HarmBench + benign-counterpart loaders (fail-loud, real data only).

HarmBench (centerforaisafety/HarmBench, pinned as the git submodule
``third_party/harmbench``) ships ``data/behavior_datasets/harmbench_behaviors_text_all.csv``
with 400 rows and columns::

    Behavior, FunctionalCategory, SemanticCategory, Tags, ContextString, BehaviorID

FunctionalCategory: standard (200), contextual (100, needs ContextString
prepended), copyright (100, requests to reproduce copyrighted text — not a
harmful-content request and excluded by default).  ``_text_val.csv`` /
``_text_test.csv`` are the official 80/320 split of the same 400 behaviors.

Benign counterpart for FPR (see eval_config.yaml ``harmbench.benign_set``):
  * ``jbb_benign``   — JailbreakBench JBB-Behaviors benign split, 100 prompts,
                      one benign twin per harmful behavior category
                      (``third_party/benign_sets/jbb_benign_behaviors.csv``,
                      columns Index, Goal, Target, Behavior, Category, Source).
  * ``xstest_safe``  — XSTest v2 safe prompts, 250 prompts designed to look
                      unsafe but be benign (``third_party/benign_sets/xstest_v2_prompts.csv``,
                      columns id, prompt, type, label, focus, note; label == "safe").
The 31 hand-written benign prompts of the original script are NOT a valid
option here; they live in ``legacy_authored_eval_set.json`` behind an explicit flag.
"""
import csv
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from .errors import DataLoadError, empty_error, missing_file_error, schema_error

HARMBENCH_COLUMNS = ("Behavior", "FunctionalCategory", "SemanticCategory", "Tags", "ContextString", "BehaviorID")
HARMBENCH_SUBSETS = {"all": "harmbench_behaviors_text_all.csv", "val": "harmbench_behaviors_text_val.csv", "test": "harmbench_behaviors_text_test.csv"}
JBB_COLUMNS = ("Index", "Goal", "Target", "Behavior", "Category", "Source")
XSTEST_COLUMNS = ("id", "prompt", "type", "label", "focus", "note")


def _read_csv(path: Path, required_cols, what: str) -> List[Dict[str, str]]:
    if not path.exists():
        raise missing_file_error(path, what)
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        missing = [c for c in required_cols if c not in cols]
        if missing:
            raise schema_error(path, required_cols, cols, context=what)
        rows = list(reader)
    if not rows:
        raise empty_error(path, what)
    return rows


def load_harmbench_behaviors(harmbench_dir, subset: str = "all", functional_categories=("standard", "contextual"),
                             prepend_context: bool = True) -> List[Dict[str, Any]]:
    """Return HarmBench behaviors as prompt records with label 1.

    ``functional_categories`` selects which FunctionalCategory values are kept.
    Contextual behaviors get their ContextString prepended (two newlines) when
    ``prepend_context`` is true, matching how HarmBench presents them to models.
    """
    if subset not in HARMBENCH_SUBSETS:
        raise DataLoadError(f"unknown HarmBench subset {subset!r}; choose from {sorted(HARMBENCH_SUBSETS)}")
    path = Path(harmbench_dir) / "data" / "behavior_datasets" / HARMBENCH_SUBSETS[subset]
    rows = _read_csv(path, HARMBENCH_COLUMNS, f"HarmBench behaviors ({subset})")
    keep = set(functional_categories)
    out = []
    for r in rows:
        if r["FunctionalCategory"] not in keep:
            continue
        text = r["Behavior"].strip()
        if r["FunctionalCategory"] == "contextual" and prepend_context and r["ContextString"].strip():
            text = r["ContextString"].strip() + "\n\n" + text
        out.append({
            "id": r["BehaviorID"], "label": 1, "prompt": text,
            "functional_category": r["FunctionalCategory"], "semantic_category": r["SemanticCategory"],
            "tags": r["Tags"], "source": f"harmbench:{subset}",
        })
    if not out:
        raise DataLoadError(f"HarmBench filter {sorted(keep)} on {path} produced zero behaviors")
    return out


def harmbench_breakdown(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n": len(records),
        "by_functional_category": dict(Counter(r["functional_category"] for r in records)),
        "by_semantic_category": dict(Counter(r["semantic_category"] for r in records)),
    }


def load_jbb_benign(path) -> List[Dict[str, Any]]:
    rows = _read_csv(Path(path), JBB_COLUMNS, "JailbreakBench benign behaviors")
    return [{"id": f"jbb_benign_{r['Index']}", "label": 0, "prompt": r["Goal"].strip(), "category": r["Category"], "source": "jbb_benign"} for r in rows]


def load_xstest_safe(path) -> List[Dict[str, Any]]:
    rows = _read_csv(Path(path), XSTEST_COLUMNS, "XSTest prompts")
    out = [{"id": f"xstest_{r['id']}", "label": 0, "prompt": r["prompt"].strip(), "category": r["type"], "source": "xstest_safe"}
           for r in rows if r["label"].strip().lower() == "safe"]
    if not out:
        raise DataLoadError(f"no rows with label == 'safe' in {path}")
    return out


def load_benign_set(name: str, datasets_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Dispatch on the benign-set name used in eval_config.yaml."""
    from .config import resolve_path

    if name == "jbb_benign":
        return load_jbb_benign(resolve_path(datasets_cfg["jbb_benign_csv"]))
    if name == "xstest_safe":
        return load_xstest_safe(resolve_path(datasets_cfg["xstest_csv"]))
    raise DataLoadError(f"unknown benign set {name!r}; valid: jbb_benign, xstest_safe (the authored 31-prompt list is legacy-only)")


def load_harmbench_eval_set(cfg: Dict[str, Any], datasets_cfg: Dict[str, Any], harmbench_dir=None) -> Dict[str, Any]:
    """Build the AF evaluation set: HarmBench harmful (label 1) + configured benign set (label 0)."""
    from .config import resolve_path

    d = Path(harmbench_dir) if harmbench_dir else resolve_path(datasets_cfg["harmbench_dir"])
    harmful = load_harmbench_behaviors(d, cfg.get("subset", "all"), tuple(cfg.get("functional_categories", ("standard", "contextual"))),
                                       bool(cfg.get("prepend_context", True)))
    names = cfg.get("benign_sets") or ([cfg["benign_set"]] if cfg.get("benign_set") else None)
    if not names:
        raise DataLoadError("harmbench.benign_sets must list at least one benign set (xstest_safe, jbb_benign)")
    benign = []
    for name in names:
        for r in load_benign_set(name, datasets_cfg):
            benign.append({**r, "benign_set": name})
    max_h, max_b = cfg.get("max_harmful"), cfg.get("max_benign")
    if max_h:
        harmful = harmful[: int(max_h)]
    if max_b:
        benign = benign[: int(max_b)]
    summary = {"harmbench_dir": str(d), "subset": cfg.get("subset", "all"), "benign_sets": list(names),
               "n_harmful": len(harmful), "n_benign": len(benign), "n_benign_by_set": dict(Counter(r["benign_set"] for r in benign)),
               "harmful_breakdown": harmbench_breakdown(harmful),
               "benign_categories": dict(Counter(r["category"] for r in benign))}
    return {"harmful": harmful, "benign": benign, "summary": summary}
