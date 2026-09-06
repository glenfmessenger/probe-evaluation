"""The single remaining authored prompt set, quarantined behind an explicit flag.

``legacy_authored_eval_set.json`` holds the 56 hand-written "HarmBench-style"
prompts (8 × 7 categories) and 31 hand-written benign prompts that used to be
embedded in ``package/eval_harmbench.py`` and silently used whenever HarmBench
was not found.  They are kept only so that old numbers can be reproduced on
request.  There is no code path that reaches this file without the caller
passing ``allow_legacy=True`` (CLI: ``--use-legacy-authored-set``), and the
returned records are tagged ``source == "legacy_authored"`` so they can never
be reported as HarmBench.
"""
import json
from pathlib import Path
from typing import Any, Dict, List

from .errors import DataLoadError

LEGACY_PATH = Path(__file__).resolve().parent / "legacy_authored_eval_set.json"


def load_legacy_authored_set(allow_legacy: bool = False, path: Path = LEGACY_PATH) -> Dict[str, List[Dict[str, Any]]]:
    if not allow_legacy:
        raise DataLoadError(
            "the authored 56+31 prompt list is legacy-only; pass --use-legacy-authored-set explicitly. "
            "It is never used as a fallback for missing HarmBench data.")
    if not path.exists():
        raise DataLoadError(f"legacy authored set not found: {path}")
    with open(path) as f:
        data = json.load(f)
    harmful = [{"id": f"legacy_{cat}_{i}", "label": 1, "prompt": p, "functional_category": "authored",
                "semantic_category": cat, "source": "legacy_authored"}
               for cat, ps in data["harmful_by_category"].items() for i, p in enumerate(ps)]
    benign = [{"id": f"legacy_benign_{i}", "label": 0, "prompt": p, "category": "authored", "source": "legacy_authored"}
              for i, p in enumerate(data["benign"])]
    return {"harmful": harmful, "benign": benign, "note": data.get("note", "")}
