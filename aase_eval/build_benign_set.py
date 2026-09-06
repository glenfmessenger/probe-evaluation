#!/usr/bin/env python3
"""Build ``datasets/aag_benign_eval.jsonl`` (approved Stage-2 composition: 17 InjecAgent user cases +
63 AgentDojo ground-truth tool outputs stratified by suite + 40 authored cases over InjecAgent scaffolds).

Deterministic: same inputs -> byte-identical output.  Inputs:
  * InjecAgent user cases (eval_config.yaml datasets.injecagent_dir)
  * datasets/sources/agentdojo_v1.2.2_extract.json — produced once by aase_eval/extract_agentdojo.py
    inside a venv with agentdojo==0.1.35 (that package is NOT a repo dependency); pinned by sha256
  * aase_eval/authored_benign_cases.py

Usage:  python -m aase_eval.build_benign_set [--out datasets/aag_benign_eval.jsonl]
Writes the dataset plus ``<out>.build.json`` with exclusion/backfill accounting.
"""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .authored_benign_cases import AUTHOR, AUTHORED_CASES, HARD_NEGATIVE_KINDS, REVIEWED_BY, REVIEW_DATE
from .config import REPO_ROOT, load_config, resolve_path
from .errors import DataLoadError
from .injecagent import PLACEHOLDER, load_attacker_cases, load_user_cases

AGENTDOJO_EXTRACT = REPO_ROOT / "datasets" / "sources" / "agentdojo_v1.2.2_extract.json"
STRATA = {"workspace": 24, "travel": 20, "banking": 10, "slack": 9}          # approved stratification (sum 63)
MAX_CHARS = 4000                                                            # amendment 2
PASSWORD_EXCLUDE_TOOLS = {"update_password"}                                # amendment 2: call args carry a fictional password
FICTIONAL_PASSWORDS = ("1j1l-2k3j",)                                        # amendment 2: AgentDojo's fictional credential string
NATURAL_HN = re.compile(r"\b(please|make sure|don'?t|do not|remember to|you must|click|pay the|update your|can you|let me know|reply|confirm)\b", re.I)
CONFIRMATION_BUDGET = {"workspace": 0, "travel": 5, "banking": 2, "slack": 3}  # short write-confirmations kept for realism


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_injecagent_user(injecagent_dir) -> list:
    out = []
    for i, u in enumerate(load_user_cases(injecagent_dir)):
        out.append({
            "id": f"injuser_{i:02d}", "label": 0, "subset": "injecagent_user", "source": "injecagent:user_cases.jsonl",
            "user_tool": u["user_tool"], "user_instruction": u["user_instruction"],
            "tool_response": u["tool_response_template"].replace(PLACEHOLDER, ""),
            "hard_negative": False, "hard_negative_kind": None,
            "provenance": {"origin": "InjecAgent user_cases.jsonl (uiuc-kang-lab/InjecAgent @ f19c9f2), row " + str(i),
                           "transform": f"'{PLACEHOLDER}' placeholder removed from Tool Response Template"},
            "format_note": "identical scaffold to the injection cases (same template, slot empty); Python-dict repr",
        })
    return out


def build_agentdojo(extract_path: Path) -> tuple:
    if not extract_path.exists():
        raise DataLoadError(f"AgentDojo extraction not found: {extract_path} (run aase_eval/extract_agentdojo.py in a venv with agentdojo)")
    ex = json.loads(extract_path.read_text())
    rows = ex["rows"]
    ver, suite_ver = ex["agentdojo_version"], ex["suite_version"]
    excl = {"over_max_chars": [], "password": []}
    kept = []
    for r in rows:
        if r["n_chars"] > MAX_CHARS:
            excl["over_max_chars"].append(f"{r['suite']}/{r['tool']} ({r['n_chars']} chars)")
        elif r["tool"] in PASSWORD_EXCLUDE_TOOLS or any(pw in r["tool_response"] or pw in r["user_instruction"] for pw in FICTIONAL_PASSWORDS):
            excl["password"].append(f"{r['suite']}/{r['tool']} task={r['user_task']} args={json.dumps(r['args'])}")
        else:
            kept.append(r)
    by_suite = defaultdict(list)
    for r in kept:
        by_suite[r["suite"]].append(r)
    selected, accounting = [], {}
    for suite, want in STRATA.items():
        pool = by_suite[suite]
        natural = [r for r in pool if NATURAL_HN.search(r["tool_response"])]
        substantive = [r for r in pool if r not in natural and r["n_chars"] >= 80]
        confirm = [r for r in pool if r not in natural and r["n_chars"] < 80]
        key = lambda r: hashlib.sha256(f"{r['suite']}|{r['tool']}|{json.dumps(r['args'], sort_keys=True)}".encode()).hexdigest()
        substantive.sort(key=key); confirm.sort(key=key); natural.sort(key=key)
        take = natural[:want]
        n_conf = min(CONFIRMATION_BUDGET[suite], len(confirm), want - len(take))
        take += substantive[: want - len(take) - n_conf]
        take += confirm[:n_conf]
        if len(take) < want:          # backfill from whatever is left in the same suite
            rest = [r for r in substantive + confirm if r not in take]
            take += rest[: want - len(take)]
        if len(take) != want:
            raise DataLoadError(f"AgentDojo suite {suite}: wanted {want}, only {len(take)} available after exclusions")
        accounting[suite] = {"available_after_exclusions": len(pool), "selected": len(take), "natural_hard_negatives": len(natural),
                             "substantive_selected": sum(1 for r in take if r in substantive), "confirmations_selected": sum(1 for r in take if r in confirm)}
        for r in take:
            hn = r in natural
            selected.append({
                "id": None, "label": 0, "subset": "agentdojo", "source": f"agentdojo:{ver}/{suite_ver}/{suite}",
                "user_tool": r["tool"], "user_instruction": r["user_instruction"], "tool_response": r["tool_response"],
                "hard_negative": hn, "hard_negative_kind": "natural" if hn else None,
                "provenance": {"origin": f"AgentDojo {ver} (MIT), suite {suite_ver}/{suite}, user task {r['user_task']}, ground-truth call {r['tool']}({json.dumps(r['args'])}) executed on the default environment with no injections",
                               "transform": "pydantic/datetime outputs converted to Python dict/list reprs with ISO dates (aase_eval/extract_agentdojo.py); plain-string outputs kept verbatim",
                               "hard_negative_detection": "regex on imperative phrases (build_benign_set.NATURAL_HN)" if hn else None},
                "format_note": ("AgentDojo user prompt + AgentDojo tool output; output is a Python dict/list repr (single-quoted keys, ISO dates) like InjecAgent's, "
                                "but with AgentDojo field names and no Tool Response Template scaffold" if not r["tool_response"].startswith(("Dear", "Bill", "The")) else
                                "AgentDojo user prompt + plain-text tool output (file/web content), not a dict repr"),
            })
    selected.sort(key=lambda x: (x["source"], x["user_tool"], x["user_instruction"], x["tool_response"]))
    counters = Counter()
    for s in selected:
        suite = s["source"].rsplit("/", 1)[-1]
        s["id"] = f"agentdojo_{suite}_{counters[suite]:03d}"; counters[suite] += 1
    return selected, {"agentdojo_version": ver, "suite_version": suite_ver, "extract_sha256": _sha(extract_path), "rows_extracted": len(rows),
                      "excluded": {k: {"count": len(v), "cases": v} for k, v in excl.items()}, "per_suite": accounting}


def build_authored(injecagent_dir) -> list:
    users = {u["user_tool"]: u for u in load_user_cases(injecagent_dir)}
    out, per_tool = [], Counter()
    for user_tool, kind, hn, slot, style in AUTHORED_CASES:
        if user_tool not in users:
            raise DataLoadError(f"authored case references unknown InjecAgent user tool {user_tool!r}")
        if hn != (kind in HARD_NEGATIVE_KINDS):
            raise DataLoadError(f"authored case kind/flag mismatch: {user_tool} {kind} {hn}")
        u = users[user_tool]
        out.append({
            "id": f"authored_{user_tool}_{per_tool[user_tool]:02d}", "label": 0, "subset": "authored", "source": f"authored:scaffold={user_tool}",
            "user_tool": user_tool, "user_instruction": u["user_instruction"],
            "tool_response": u["tool_response_template"].replace(PLACEHOLDER, slot),
            "hard_negative": hn, "hard_negative_kind": kind if hn else None,
            "provenance": {"origin": "authored", "author": AUTHOR, "reviewed_by": REVIEWED_BY, "review_date": REVIEW_DATE,
                           "scaffold": f"InjecAgent user_cases.jsonl row for {user_tool} (real User Instruction + Tool Response Template)",
                           "slot_text": slot, "kind": kind, "style": style,
                           "transform": f"slot text inserted at the '{PLACEHOLDER}' placeholder exactly as an injection would be"},
            "format_note": "identical scaffold to the injection cases (same template, slot filled with authored text); Python-dict repr",
        })
        per_tool[user_tool] += 1
    return out


def build(out_path: Path, config_path=None) -> dict:
    cfg = load_config(config_path)
    inj_dir = resolve_path(cfg["datasets"]["injecagent_dir"])
    users = build_injecagent_user(inj_dir)
    dojo, dojo_acc = build_agentdojo(AGENTDOJO_EXTRACT)
    authored = build_authored(inj_dir)
    recs = users + dojo + authored
    # contamination + duplicate guards (also enforced by tests)
    from .benign_set import normalise_text, validate_record
    attacks = [normalise_text(a["attacker_instruction"]) for a in load_attacker_cases(inj_dir)]
    for r in recs:
        validate_record(r, out_path, 0)
        nt = normalise_text(r["tool_response"] + " " + r["user_instruction"])
        for a in attacks:
            if a and a in nt:
                raise DataLoadError(f"benign case {r['id']} contains an InjecAgent attacker instruction: {a[:80]}")
    keys = [(r["user_instruction"], r["tool_response"]) for r in recs]
    if len(set(keys)) != len(keys):
        raise DataLoadError("duplicate (user_instruction, tool_response) pairs in benign set")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    stats = {"output": str(out_path.relative_to(REPO_ROOT)) if out_path.is_relative_to(REPO_ROOT) else str(out_path), "sha256": _sha(out_path),
             "counts": {"total": len(recs), **Counter(r["subset"] for r in recs)},
             "hard_negatives": {"total": sum(r["hard_negative"] for r in recs), **Counter(r["hard_negative_kind"] for r in recs if r["hard_negative"])},
             "agentdojo": dojo_acc, "authored_author": AUTHOR}
    Path(str(out_path) + ".build.json").write_text(json.dumps(stats, indent=1))
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "datasets" / "aag_benign_eval.jsonl"))
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    stats = build(Path(args.out), args.config)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
