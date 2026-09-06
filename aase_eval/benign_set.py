"""Pinned AAG benign evaluation set (``datasets/aag_benign_eval.jsonl``): schema, loader, FPR breakdown.

Schema (one JSON object per line; every field required unless marked optional):

    id                  str   unique, e.g. "injuser_03", "agentdojo_workspace_017", "authored_TodoistSearchTasks_02"
    label               int   always 0
    subset              str   "injecagent_user" | "agentdojo" | "authored"
    source              str   finer-grained origin tag, e.g. "injecagent:user_cases.jsonl",
                              "agentdojo:0.1.35/v1.2.2/workspace", "authored:scaffold=TodoistSearchTasks"
    user_tool           str   tool name the case is about (InjecAgent user tool, or AgentDojo tool)
    user_instruction    str   the user's request (real InjecAgent / AgentDojo task prompt)
    tool_response       str   the benign tool output that is scored (no injection)
    hard_negative       bool  content superficially resembles instructions (imperatives / trigger words)
    hard_negative_kind  str|null  "natural" (found in AgentDojo data) or an authored kind; null when not a hard negative
    provenance          obj   how the case was produced; keys differ by subset (see build_benign_set.py), always
                              includes "origin" and, for authored cases, "author" and "reviewed_by"
    format_note         str   how the tool_response is formatted relative to the InjecAgent injection cases

Scaffold guarantee: ``injecagent_user`` and ``authored`` cases are built from the same 17 real
``Tool Response Template`` strings as the injection cases, with the ``<Attacker Instruction>`` slot
filled (authored) or removed (injecagent_user); the harness applies the identical prompt template.
AgentDojo cases keep AgentDojo's own user prompts and tool outputs (converted to Python dict/list
reprs, ISO dates) — see ``format_note`` on each case and datasets/README.md.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .errors import DataLoadError, empty_error, missing_file_error

SUBSETS = ("injecagent_user", "agentdojo", "authored")
REQUIRED_FIELDS = {
    "id": str, "label": int, "subset": str, "source": str, "user_tool": str, "user_instruction": str,
    "tool_response": str, "hard_negative": bool, "provenance": dict, "format_note": str,
}
OPTIONAL_FIELDS = {"hard_negative_kind": (str, type(None))}
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "datasets" / "aag_benign_eval.jsonl"


def normalise_text(s: str) -> str:
    """Lower-case, strip punctuation and collapse whitespace — used for contamination checks."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def validate_record(rec: Dict[str, Any], path, lineno: int) -> None:
    if not isinstance(rec, dict):
        raise DataLoadError(f"{path}:{lineno}: record is not a JSON object")
    missing = [k for k in REQUIRED_FIELDS if k not in rec]
    if missing:
        raise DataLoadError(f"{path}:{lineno}: missing required fields {missing}; found {sorted(rec)}")
    for k, t in REQUIRED_FIELDS.items():
        if not isinstance(rec[k], t) or (t is int and isinstance(rec[k], bool)):
            raise DataLoadError(f"{path}:{lineno}: field {k!r} must be {t.__name__}, got {type(rec[k]).__name__}")
    for k, t in OPTIONAL_FIELDS.items():
        if k in rec and not isinstance(rec[k], t):
            raise DataLoadError(f"{path}:{lineno}: field {k!r} has wrong type {type(rec[k]).__name__}")
    unknown = set(rec) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS)
    if unknown:
        raise DataLoadError(f"{path}:{lineno}: unknown fields {sorted(unknown)}")
    if rec["label"] != 0:
        raise DataLoadError(f"{path}:{lineno}: benign record has label {rec['label']!r}, expected 0")
    if rec["subset"] not in SUBSETS:
        raise DataLoadError(f"{path}:{lineno}: subset {rec['subset']!r} not in {SUBSETS}")
    if rec["hard_negative"] != bool(rec.get("hard_negative_kind")):
        raise DataLoadError(f"{path}:{lineno}: hard_negative={rec['hard_negative']} but hard_negative_kind={rec.get('hard_negative_kind')!r}")
    if not rec["tool_response"].strip() or not rec["user_instruction"].strip():
        raise DataLoadError(f"{path}:{lineno}: empty user_instruction or tool_response")
    if "origin" not in rec["provenance"]:
        raise DataLoadError(f"{path}:{lineno}: provenance lacks 'origin'")


def load_benign_set(path=DEFAULT_PATH, subsets: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    """Load and validate the pinned benign set. Raises DataLoadError on anything unexpected."""
    path = Path(path)
    if not path.exists():
        raise missing_file_error(path, "AAG benign evaluation set", "build it with: python -m aase_eval.build_benign_set")
    recs, ids = [], set()
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise DataLoadError(f"{path}:{n}: malformed JSON: {e}") from e
            validate_record(rec, path, n)
            if rec["id"] in ids:
                raise DataLoadError(f"{path}:{n}: duplicate id {rec['id']!r}")
            ids.add(rec["id"])
            recs.append(rec)
    if not recs:
        raise empty_error(path, "AAG benign evaluation set")
    if subsets is not None:
        want = set(subsets)
        bad = want - set(SUBSETS)
        if bad:
            raise DataLoadError(f"unknown benign subsets requested: {sorted(bad)}")
        recs = [r for r in recs if r["subset"] in want]
        if not recs:
            raise DataLoadError(f"benign set {path} has no records in subsets {sorted(want)}")
    return recs


def to_prompt_records(recs: List[Dict[str, Any]], template: str) -> List[Dict[str, Any]]:
    """Render benign records with the SAME prompt template the harness uses for injections."""
    from .injecagent import format_prompt

    return [
        {
            "id": r["id"], "label": 0,
            "prompt": format_prompt(r["user_instruction"], r["tool_response"], template),
            "user_tool": r["user_tool"], "subset": r["subset"], "source": r["source"],
            "hard_negative": r["hard_negative"], "hard_negative_kind": r.get("hard_negative_kind"),
        }
        for r in recs
    ]


def fpr_breakdown(neg_records: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    """FPR overall, per subset, hard-negative-only and plain-only, from scored benign records.

    Each record needs ``score`` and, when present, ``subset`` / ``hard_negative``.
    """
    def rate(rows):
        n = len(rows)
        fp = sum(1 for r in rows if r["score"] > threshold)
        return {"n": n, "false_positives": fp, "fpr": (fp / n) if n else None}

    out = {"threshold": float(threshold), "overall": rate(neg_records), "per_subset": {}, "hard_negative_only": None, "plain_only": None}
    subsets = sorted({r.get("subset", "unknown") for r in neg_records})
    for s in subsets:
        out["per_subset"][s] = rate([r for r in neg_records if r.get("subset", "unknown") == s])
    hn = [r for r in neg_records if r.get("hard_negative")]
    plain = [r for r in neg_records if not r.get("hard_negative")]
    out["hard_negative_only"] = rate(hn)
    out["plain_only"] = rate(plain)
    return out
