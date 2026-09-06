"""Stage-2 gates for the pinned AAG benign set (datasets/aag_benign_eval.jsonl)."""
import json
import hashlib
from collections import Counter
from pathlib import Path

import pytest

from aase_eval import DataLoadError
from aase_eval.benign_set import DEFAULT_PATH, REQUIRED_FIELDS, SUBSETS, fpr_breakdown, load_benign_set, normalise_text, to_prompt_records
from aase_eval.injecagent import (PLACEHOLDER, DEFAULT_PROMPT_TEMPLATE, build_injection_prompts, format_prompt,
                                  load_attacker_cases, load_injecagent_eval_set, load_test_cases, load_user_cases)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def benign():
    return load_benign_set(DEFAULT_PATH)


def test_file_pinned_in_manifest(benign):
    sha = hashlib.sha256(DEFAULT_PATH.read_bytes()).hexdigest()
    prov = (REPO / "manifest" / "provenance.csv").read_text()
    assert sha in prov, "datasets/aag_benign_eval.jsonl sha256 must be recorded in manifest/provenance.csv"
    build = json.loads((str(DEFAULT_PATH) + ".build.json") and Path(str(DEFAULT_PATH) + ".build.json").read_text())
    assert build["sha256"] == sha


def test_schema_and_counts(benign):
    assert len(benign) >= 100 and len(benign) == 120
    for r in benign:
        assert set(REQUIRED_FIELDS) <= set(r)
        assert r["label"] == 0 and r["subset"] in SUBSETS
    c = Counter(r["subset"] for r in benign)
    assert c == {"injecagent_user": 17, "agentdojo": 63, "authored": 40}
    assert Counter(r["source"].rsplit("/", 1)[-1] for r in benign if r["subset"] == "agentdojo") == \
        {"workspace": 24, "travel": 20, "banking": 10, "slack": 9}


def test_hard_negative_subset_present_and_tagged(benign):
    hn = [r for r in benign if r["hard_negative"]]
    assert len(hn) == 36
    kinds = Counter(r["hard_negative_kind"] for r in hn)
    assert kinds["natural"] == 10 and sum(v for k, v in kinds.items() if k != "natural") == 26
    assert all(r["hard_negative_kind"] for r in hn) and all(not r.get("hard_negative_kind") for r in benign if not r["hard_negative"])
    # provenance distinguishes found vs authored hard negatives
    for r in hn:
        if r["hard_negative_kind"] == "natural":
            assert r["subset"] == "agentdojo" and r["provenance"]["hard_negative_detection"]
        else:
            assert r["subset"] == "authored" and r["provenance"]["origin"] == "authored" and r["provenance"]["author"]
    for r in benign:
        if r["subset"] == "authored":
            assert r["provenance"]["reviewed_by"] == "Glen Messenger" and r["provenance"]["review_date"] == "2026-09-03"
    assert all(k in kinds for k in ("quoted_imperative_email", "trigger_word_doc", "maintenance_notice", "marketing_cta",
                                    "logistics_instruction", "code_readme_commands", "security_awareness", "ai_meta_mention"))


def test_no_duplicates(benign):
    ids = [r["id"] for r in benign]
    assert len(set(ids)) == len(ids)
    pairs = [(r["user_instruction"], r["tool_response"]) for r in benign]
    assert len(set(pairs)) == len(pairs)
    prompts = [p["prompt"] for p in to_prompt_records(benign, DEFAULT_PROMPT_TEMPLATE)]
    assert len(set(prompts)) == len(prompts)
    # and none coincides with an injection prompt
    inj = {p["prompt"] for p in build_injection_prompts(load_test_cases(REPO / "runtime/aag/InjecAgent", "base"))}
    assert not (inj & set(prompts))


def test_no_attacker_instruction_contamination(benign, injecagent_dir):
    attacks = [a["attacker_instruction"] for a in load_attacker_cases(injecagent_dir)]
    assert len(attacks) == 62
    norm_attacks = [normalise_text(a) for a in attacks]
    for r in benign:
        text = r["tool_response"] + "\n" + r["user_instruction"]
        for a, na in zip(attacks, norm_attacks):
            assert a not in text, f"{r['id']} contains attacker instruction verbatim: {a[:60]}"
            assert na not in normalise_text(text), f"{r['id']} contains attacker instruction (normalised): {a[:60]}"
    # no leftover placeholder anywhere
    assert all(PLACEHOLDER not in r["tool_response"] for r in benign)


def test_authored_and_injecagent_user_cases_match_positive_scaffold(benign, injecagent_dir):
    """Amendment 4: authored + injecagent_user benign cases are schema- and scaffold-identical to the positives."""
    users = {u["user_tool"]: u for u in load_user_cases(injecagent_dir)}
    positives = load_test_cases(injecagent_dir, "base")
    pos_by_tool = {}
    for p in positives:
        pos_by_tool.setdefault(p["user_tool"], p)
    # every positive is exactly template-with-slot-filled (so the template IS the scaffold)
    for p in positives:
        assert p["tool_response"] == users[p["user_tool"]]["tool_response_template"].replace(PLACEHOLDER, p["attacker_instruction"])
    for r in benign:
        if r["subset"] not in ("authored", "injecagent_user"):
            continue
        u = users[r["user_tool"]]
        pos = pos_by_tool[r["user_tool"]]
        assert r["user_instruction"] == u["user_instruction"] == pos["user_instruction"]
        slot = r["provenance"]["slot_text"] if r["subset"] == "authored" else ""
        assert r["tool_response"] == u["tool_response_template"].replace(PLACEHOLDER, slot)
        # prefix/suffix around the slot identical to the positive case of the same scaffold
        pre, post = u["tool_response_template"].split(PLACEHOLDER, 1)
        assert r["tool_response"].startswith(pre) and r["tool_response"].endswith(post)
        assert pos["tool_response"].startswith(pre) and pos["tool_response"].endswith(post)
        assert not r["tool_response"].startswith('"')  # same quote-stripped surface form as positives
    # the harness renders both classes with the same template and the same field set
    data = load_injecagent_eval_set({"dir": str(injecagent_dir), "settings": ["base"]}, injecagent_dir)
    pos_rec, neg_rec = data["injections"][0], data["benign"][0]
    assert {"id", "label", "prompt"} <= set(pos_rec) and {"id", "label", "prompt"} <= set(neg_rec)
    for rec in data["benign"]:
        src = next(b for b in benign if b["id"] == rec["id"])
        assert rec["prompt"] == format_prompt(src["user_instruction"], src["tool_response"], DEFAULT_PROMPT_TEMPLATE)
    assert pos_rec["prompt"].startswith("User: ") and neg_rec["prompt"].startswith("User: ")
    assert pos_rec["prompt"].endswith("\nAssistant:") and neg_rec["prompt"].endswith("\nAssistant:")


def test_agentdojo_cases_carry_format_notes(benign):
    for r in benign:
        if r["subset"] == "agentdojo":
            assert r["format_note"] and "AgentDojo" in r["format_note"]
            assert "AgentDojo" in r["provenance"]["origin"] and r["provenance"]["transform"]
            assert not r["tool_response"].startswith('"')
            assert len(r["tool_response"]) <= 4000
            assert r["user_tool"] != "update_password"


def test_default_config_uses_pinned_set(cfg, injecagent_dir):
    data = load_injecagent_eval_set(cfg["injecagent"], injecagent_dir)
    s = data["summary"]
    assert s["benign_source"] == "aag_benign_eval" and s["n_benign"] == 120 and s["n_benign_hard_negative"] == 36
    assert s["benign_subsets"] == {"injecagent_user": 17, "agentdojo": 63, "authored": 40}
    alt = load_injecagent_eval_set(dict(cfg["injecagent"], benign_source="injecagent_user17"), injecagent_dir)
    assert alt["summary"]["n_benign"] == 17 and alt["summary"]["benign_source"] == "injecagent_user17"
    sub = load_injecagent_eval_set(dict(cfg["injecagent"], benign_subsets=["authored"]), injecagent_dir)
    assert sub["summary"]["n_benign"] == 40
    with pytest.raises(DataLoadError):
        load_injecagent_eval_set(dict(cfg["injecagent"], benign_source="synthetic"), injecagent_dir)


def test_loader_raises_on_missing_or_malformed(tmp_path):
    with pytest.raises(DataLoadError, match="not found"):
        load_benign_set(tmp_path / "nope.jsonl")
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "x", "label": 0}\n')
    with pytest.raises(DataLoadError, match="missing required fields"):
        load_benign_set(bad)
    bad.write_text("{not json}\n")
    with pytest.raises(DataLoadError, match="malformed JSON"):
        load_benign_set(bad)
    good = json.loads(DEFAULT_PATH.read_text().splitlines()[0])
    wrong = dict(good, label=1)
    bad.write_text(json.dumps(wrong) + "\n")
    with pytest.raises(DataLoadError, match="label"):
        load_benign_set(bad)
    bad.write_text(json.dumps(good) + "\n" + json.dumps(good) + "\n")
    with pytest.raises(DataLoadError, match="duplicate id"):
        load_benign_set(bad)
    bad.write_text("")
    with pytest.raises(DataLoadError):
        load_benign_set(bad)
    with pytest.raises(DataLoadError, match="unknown benign subsets"):
        load_benign_set(DEFAULT_PATH, subsets=["synthetic"])


def test_fpr_breakdown_four_ways():
    neg = [
        {"score": 0.9, "subset": "authored", "hard_negative": True},
        {"score": 0.1, "subset": "authored", "hard_negative": False},
        {"score": 0.7, "subset": "agentdojo", "hard_negative": True},
        {"score": 0.2, "subset": "agentdojo", "hard_negative": False},
        {"score": 0.2, "subset": "injecagent_user", "hard_negative": False},
    ]
    b = fpr_breakdown(neg, 0.5)
    assert b["overall"] == {"n": 5, "false_positives": 2, "fpr": 0.4}
    assert b["per_subset"]["authored"]["fpr"] == 0.5 and b["per_subset"]["injecagent_user"]["fpr"] == 0.0
    assert b["hard_negative_only"] == {"n": 2, "false_positives": 2, "fpr": 1.0}
    assert b["plain_only"] == {"n": 3, "false_positives": 0, "fpr": 0.0}
    assert set(b) == {"threshold", "overall", "per_subset", "hard_negative_only", "plain_only"}


def test_builder_is_deterministic(tmp_path):
    from aase_eval.build_benign_set import build
    out = tmp_path / "rebuilt.jsonl"
    build(out)
    assert out.read_bytes() == DEFAULT_PATH.read_bytes(), "rebuilding must reproduce the pinned file byte for byte"
