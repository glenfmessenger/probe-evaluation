#!/usr/bin/env python3
"""ARM 2 — AAG ceiling and extraction-position protocol (no new data).

Question. Phase 1 §4-5 measured real-InjecAgent AAG AUC of 0.52-0.72 with a probe read at the FINAL token of

    "User: {user_instruction}\\nTool Result: {tool_response}\\nAssistant:"

The injected instruction sits in the middle of the tool response (median start 33 %, median end 85 % of the tool-response
string; never at its end), so the final-token reading is separated from the injection by tens of tokens of unrelated tool
output. Arm 2 asks whether that weak result is a *representation ceiling* (the model does not linearly encode "there is an
injection here" anywhere we can read it) or an *extraction artefact* (it does, but not at the final token).

Split. Real InjecAgent `base` setting only, 1,054 cases, stratified 50/50 train/test by `attack_type` (6 values: Others
255, Data Security Harm 187, Physical Data 187, Physical Harm 170, Financial Harm 153, Financial Data 102), seeded. The
120-case benign set (datasets/aag_benign_eval.jsonl) is split 50/50 the same way, stratified jointly by `subset` and
`hard_negative` so both halves carry the same mix of InjecAgent user cases, AgentDojo suites, authored scaffolds and the
36 hard negatives. Probes are trained on the train halves only and every reported number is on the held-out halves. The
`enhanced` setting is held out entirely as a second test set (never trained on), so a design that only works on the
"hacking prompt" wrapper is visible as such.

Four extraction designs. All read the same layer as Phase 1 (each model's AAG probe `layer_index`, unchanged; a design
that needs a different layer is a different claim and is out of scope except for the 3-layer check noted below), and all
train and evaluate with the same design — no cross-design probe transfer.

  (a) final  — hidden state at the final prompt token (the Phase 1 convention). Baseline.
  (b) mean   — mean of the hidden states over the token span covering {tool_response}. Tests whether the signal is
               present but diluted at the final position.
  (c) maxpos — probe score computed at every token position in the tool-response span; the case score is the maximum.
               Note this changes the score's distribution (a max of n draws), so its threshold is recalibrated on the
               train half and its AUC is the honest comparison, not its detection rate at a transferred threshold.
  (d) oracle — hidden state at the token where the injected instruction ENDS. Uses label knowledge (the attacker
               instruction string) to place the read, so it is NOT deployable; it is the diagnostic upper bound. If (d)
               is also weak, the ceiling is in the representation, not in the extraction position.

Benign-side convention for (d). Benign cases contain no injection, so there is no "injection end" to read. Reading them
at the end of the tool response instead would make position itself informative (harmful read mid-span, benign read at
span end) and inflate the oracle. Primary convention: for each benign case, draw a relative position from the empirical
distribution of injection-end fractions of the harmful TRAIN half (seeded), and read at that fraction of its own
tool-response span. Secondary sanity check, reported alongside: read benign at the tool-response end. If the two differ
materially the oracle is reading position, not content, and that is reported as the finding.

Span location. `attacker_instruction` appears verbatim in `tool_response` in 1,054/1,054 base cases (verified below), so
spans are located by exact character search, then mapped to token indices with the tokenizer's offset mapping
(`return_offsets_mapping=True`). Everything fails loud: a missing or ambiguous span raises rather than falling back to a
heuristic position.

  python -m gates.arm2_protocol [--dry-run]
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aase_eval.config import load_config, resolve_path  # noqa: E402
from aase_eval.errors import DataLoadError  # noqa: E402
from aase_eval.injecagent import DEFAULT_PROMPT_TEMPLATE, format_prompt, load_test_cases  # noqa: E402

DESIGNS = ("final", "mean", "maxpos", "oracle")
DEFAULT_SEED = 20260904


# ---------------------------------------------------------------- split
def _stratified_half(items: Sequence[Dict[str, Any]], key, seed: int) -> Tuple[List[Dict], List[Dict]]:
    """Deterministic 50/50 split within each stratum (odd strata put the extra item in train)."""
    buckets: Dict[Any, List[Dict]] = {}
    for it in items:
        buckets.setdefault(key(it), []).append(it)
    train, test = [], []
    for k in sorted(buckets, key=lambda x: str(x)):
        group = sorted(buckets[k], key=lambda r: r.get("id") or r.get("case_id"))
        random.Random(f"{seed}|{k}").shuffle(group)
        n_train = (len(group) + 1) // 2
        train += group[:n_train]; test += group[n_train:]
    return train, test


def build_split(cfg: Dict[str, Any] = None, seed: int = DEFAULT_SEED, group_by_attacker: bool = False) -> Dict[str, Any]:
    """Train/test halves of the real InjecAgent base cases and of the 120-case benign set.

    group_by_attacker=False (the pre-registered split): cases are split individually, stratified by attack_type. The
    1,054 cases are built from only 62 distinct attacker instructions crossed with 17 user instructions, so ALL 62
    injection strings appear in both halves — no case, tool response or (user, attacker) pair is shared, but a probe can
    still recognise an injection string it was trained on. This split measures recognition, not generalisation.

    group_by_attacker=True (the leakage control added during Stage B): the 62 attacker instructions are split 50/50,
    stratified by attack_type, and a case goes to the half its injection string belongs to. No injection string is ever
    seen in training, so this measures generalisation to unseen injections. Both are reported."""
    cfg = cfg or load_config()
    inj_dir = resolve_path(cfg["datasets"]["injecagent_dir"])
    base = load_test_cases(inj_dir, "base")
    enhanced = load_test_cases(inj_dir, "enhanced")
    if len(base) != 1054 or len(enhanced) != 1054:
        raise DataLoadError(f"expected 1054 base and 1054 enhanced InjecAgent cases, got {len(base)} and {len(enhanced)}")
    for c in base:
        c["id"] = c["case_id"]
    if group_by_attacker:
        groups = {}
        for c in base:
            groups.setdefault(c["attacker_instruction"], []).append(c)
        keys = [{"id": k, "attack_type": v[0]["attack_type"]} for k, v in sorted(groups.items())]
        gtr, _ = _stratified_half(keys, lambda g: g["attack_type"], seed)
        tr_keys = {g["id"] for g in gtr}
        pos_train = [c for c in base if c["attacker_instruction"] in tr_keys]
        pos_test = [c for c in base if c["attacker_instruction"] not in tr_keys]
    else:
        pos_train, pos_test = _stratified_half(base, lambda c: c["attack_type"], seed)

    benign_path = resolve_path(cfg["injecagent"]["benign_set_path"])   # the pinned 120-case set (eval_config.yaml)
    benign = [json.loads(l) for l in open(benign_path) if l.strip()]
    if len(benign) != 120:
        raise DataLoadError(f"expected 120 benign cases in {benign_path}, got {len(benign)}")
    neg_train, neg_test = _stratified_half(benign, lambda r: (r["subset"], bool(r["hard_negative"])), seed)
    return {"seed": seed, "group_by_attacker": group_by_attacker,
            "positives": {"train": pos_train, "test": pos_test}, "negatives": {"train": neg_train, "test": neg_test},
            "enhanced_holdout": enhanced, "benign_path": str(benign_path), "injecagent_dir": str(inj_dir)}


def split_summary(sp: Dict[str, Any]) -> Dict[str, Any]:
    from collections import Counter
    def cnt(rows, key):
        return dict(sorted(Counter(key(r) for r in rows).items()))
    return {
        "seed": sp["seed"],
        "positives": {"train": len(sp["positives"]["train"]), "test": len(sp["positives"]["test"]),
                      "by_attack_type_train": cnt(sp["positives"]["train"], lambda c: c["attack_type"]),
                      "by_attack_type_test": cnt(sp["positives"]["test"], lambda c: c["attack_type"])},
        "negatives": {"train": len(sp["negatives"]["train"]), "test": len(sp["negatives"]["test"]),
                      "by_subset_train": cnt(sp["negatives"]["train"], lambda r: r["subset"]),
                      "by_subset_test": cnt(sp["negatives"]["test"], lambda r: r["subset"]),
                      "hard_negatives_train": sum(bool(r["hard_negative"]) for r in sp["negatives"]["train"]),
                      "hard_negatives_test": sum(bool(r["hard_negative"]) for r in sp["negatives"]["test"])},
        "enhanced_holdout": len(sp["enhanced_holdout"]),
        "overlap_case_ids": len({c["case_id"] for c in sp["positives"]["train"]} & {c["case_id"] for c in sp["positives"]["test"]}),
        "group_by_attacker": bool(sp.get("group_by_attacker")),
        "shared_attacker_instructions": len({c["attacker_instruction"] for c in sp["positives"]["train"]} &
                                            {c["attacker_instruction"] for c in sp["positives"]["test"]}),
        "shared_user_instructions": len({c["user_instruction"] for c in sp["positives"]["train"]} &
                                        {c["user_instruction"] for c in sp["positives"]["test"]}),
    }


# ---------------------------------------------------------------- prompts and character spans
def positive_prompt(case: Dict[str, Any], template: str = DEFAULT_PROMPT_TEMPLATE) -> str:
    return format_prompt(case["user_instruction"], case["tool_response"], template)


def benign_prompt(rec: Dict[str, Any], template: str = DEFAULT_PROMPT_TEMPLATE) -> str:
    return format_prompt(rec["user_instruction"], rec["tool_response"], template)


def char_spans(prompt: str, tool_response: str, attacker_instruction: str = None) -> Dict[str, Tuple[int, int]]:
    """Character spans of the tool response and (for injection cases) of the injected instruction inside the prompt.

    Fails loud: the substring must be present exactly once in its container."""
    if prompt.count(tool_response) != 1:
        raise DataLoadError(f"tool response occurs {prompt.count(tool_response)} times in the prompt (need exactly 1): {tool_response[:80]!r}")
    t0 = prompt.index(tool_response); spans = {"tool_response": (t0, t0 + len(tool_response))}
    if attacker_instruction is not None:
        if tool_response.count(attacker_instruction) != 1:
            raise DataLoadError(f"attacker instruction occurs {tool_response.count(attacker_instruction)} times in the tool response "
                                f"(need exactly 1): {attacker_instruction[:80]!r}")
        a0 = t0 + tool_response.index(attacker_instruction)
        spans["injection"] = (a0, a0 + len(attacker_instruction))
    return spans


def token_span(offsets: Sequence[Tuple[int, int]], char_span: Tuple[int, int]) -> Tuple[int, int]:
    """Half-open token index range covering a character span, from a fast tokenizer's offset mapping.

    Pure function so the mapping logic is testable on CPU without a tokenizer. Special tokens carry the offset (0, 0)
    and are skipped. Raises if the span maps to no token."""
    c0, c1 = char_span
    idx = [i for i, (s, e) in enumerate(offsets) if e > s and s < c1 and e > c0]
    if not idx:
        raise DataLoadError(f"character span {char_span} maps to no token (offsets cover 0..{max((e for _, e in offsets), default=0)})")
    return idx[0], idx[-1] + 1


def read_positions(design: str, offsets: Sequence[Tuple[int, int]], spans: Dict[str, Tuple[int, int]],
                   n_tokens: int, benign_fraction: float = None) -> Dict[str, Any]:
    """Which token position(s) design `design` reads, given a prompt's offset mapping and character spans.

    Returns {"kind": "single"|"span", "index": i} or {"kind": "span", "start": a, "end": b} — the GPU stage turns this
    into a hidden-state read (single position, mean over the span, or per-position scores for maxpos)."""
    if design not in DESIGNS:
        raise ValueError(f"unknown design {design!r}, expected one of {DESIGNS}")
    if design == "final":
        return {"kind": "single", "index": n_tokens - 1}
    tstart, tend = token_span(offsets, spans["tool_response"])
    if design in ("mean", "maxpos"):
        return {"kind": "span", "start": tstart, "end": tend}
    # oracle
    if "injection" in spans:
        _, iend = token_span(offsets, spans["injection"])
        return {"kind": "single", "index": iend - 1, "oracle": True}
    if benign_fraction is None:
        raise DataLoadError("oracle design on a benign case requires benign_fraction (drawn from the harmful train half)")
    i = tstart + int(round(benign_fraction * max(0, tend - 1 - tstart)))
    return {"kind": "single", "index": min(max(i, tstart), tend - 1), "oracle": True, "benign_fraction": benign_fraction}


def benign_fraction_sampler(positives_train: Sequence[Dict[str, Any]], seed: int = DEFAULT_SEED):
    """Empirical distribution of injection-end position, as a fraction of the tool-response span, over the harmful train
    half. Used to place the oracle read on benign cases so that read position carries no label information."""
    fr = []
    for c in positives_train:
        tr, ai = c["tool_response"], c["attacker_instruction"]
        if ai not in tr:
            raise DataLoadError(f"case {c['case_id']}: attacker instruction not found verbatim in the tool response")
        fr.append((tr.index(ai) + len(ai)) / max(1, len(tr)))
    rng = random.Random(seed)
    return fr, (lambda: rng.choice(fr))


# ---------------------------------------------------------------- desk checks
def verbatim_check(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    import statistics
    ok, fr_start, fr_end, lens = 0, [], [], []
    for c in cases:
        tr, ai = c["tool_response"], c["attacker_instruction"]
        if tr.count(ai) == 1:
            ok += 1
            fr_start.append(tr.index(ai) / max(1, len(tr))); fr_end.append((tr.index(ai) + len(ai)) / max(1, len(tr)))
        lens.append(len(tr))
    return {"n": len(cases), "verbatim_once": ok,
            "injection_start_fraction": {"median": round(statistics.median(fr_start), 3), "min": round(min(fr_start), 3), "max": round(max(fr_start), 3)},
            "injection_end_fraction": {"median": round(statistics.median(fr_end), 3), "min": round(min(fr_end), 3), "max": round(max(fr_end), 3),
                                       "at_span_end": sum(f >= 0.999 for f in fr_end)},
            "tool_response_chars": {"median": statistics.median(lens), "max": max(lens)}}


def protocol_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", default="results/gates/arm2_split.json")
    a = ap.parse_args(argv)
    cfg = load_config()
    sp = build_split(cfg, a.seed)
    summ = split_summary(sp)
    vb = verbatim_check(sp["positives"]["train"] + sp["positives"]["test"])
    fr, _ = benign_fraction_sampler(sp["positives"]["train"], a.seed)
    print(json.dumps({"split": summ, "verbatim_check": vb, "designs": list(DESIGNS),
                      "benign_oracle_fraction_pool": {"n": len(fr), "median": round(sorted(fr)[len(fr) // 2], 3)},
                      "protocol_sha256": protocol_hash()[:16]}, indent=2))
    ex = sp["positives"]["train"][0]
    p = positive_prompt(ex); cs = char_spans(p, ex["tool_response"], ex["attacker_instruction"])
    print(f"\nexample prompt ({ex['case_id']}), char spans {cs}:\n  {p[:300]!r}")
    print(f"  injected text: {p[cs['injection'][0]:cs['injection'][1]][:120]!r}")
    out = Path(a.out)
    if a.dry_run:
        print("\noutput ->", out, "(dry run: not written)")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    ids = {k: {s: [c.get("case_id") or c.get("id") for c in sp[k][s]] for s in ("train", "test")} for k in ("positives", "negatives")}
    json.dump({"seed": a.seed, "summary": summ, "verbatim_check": vb, "case_ids": ids, "protocol_sha256": protocol_hash()},
              open(out, "w"), indent=2)
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
