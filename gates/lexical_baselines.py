#!/usr/bin/env python3
"""Lexical baselines — what a probe has to beat before a semantic claim is allowed.

Standing rule (DECISIONS.md #11): every APC results table reports the second-person-regex baseline beside the probe
numbers, and a semantic claim requires beating the lexical baseline on lexically-controlled data. This module supplies
that baseline, and the analogous one for AF, so both live in one place with one provenance block.

AF baseline. Cheap text features scored on HarmBench 300 (positive) against XSTest safe 250 and JailbreakBench benign
100 (negative), reported per benign set because that split is the whole story: XSTest is off-topic benign, JailbreakBench
is on-topic benign twins. A keyword baseline that separates HarmBench from XSTest but not from JailbreakBench is
measuring topic, which is exactly what Phase 1 found the AF probes doing.

APC baseline. Second-person pronoun count, directive-modal count, and length, on the Arm 3 external set and (when
present) on the v2 lexical discriminator.

  python -m gates.lexical_baselines
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aase_eval.config import load_config  # noqa: E402
from aase_eval.harmbench import load_harmbench_eval_set  # noqa: E402
from gates.build_review import ADVICE_BANDS, load_arm3  # noqa: E402
from gates.runb import dump_json, load_gates_config, out_dir, provenance  # noqa: E402
from phase2.extract import auc  # noqa: E402

SECOND_PERSON = re.compile(r"\b(you|your|yours|you're|you'll|you've)\b", re.I)
DIRECTIVE_MODAL = re.compile(r"\b(should|must|take|start|stop|switch|book|file|sign|sell|buy|move|call|go|avoid|begin)\b", re.I)

# AF features. IMPERATIVE_STEM and HOWTO are generic request shapes; HARM_LEXICON is a deliberately tuned keyword list —
# it is the strongest cheap baseline we can build, written to be as favourable to the baseline as possible so that
# "the probe beats it" means something.
IMPERATIVE_STEM = re.compile(r"^\s*(write|give|explain|describe|create|list|provide|generate|detail|draft|compose|tell|show|design|develop|outline)\b", re.I)
HOWTO = re.compile(r"\b(how (to|do|can|would)|step[- ]by[- ]step|instructions for|guide to|tutorial)\b", re.I)
HARM_LEXICON = re.compile(
    r"\b(bomb|explosiv\w*|weapon\w*|firearm\w*|gun|ammunition|poison\w*|toxin\w*|ricin|sarin|anthrax|nerve agent|"
    r"malware|ransomware|virus|keylogger|exploit\w*|payload|botnet|phish\w*|hack\w*|crack\w*|breach|backdoor|"
    r"launder\w*|counterfeit\w*|smuggl\w*|traffick\w*|forge\w*|steal|theft|rob\w*|fraud\w*|scam\w*|extort\w*|"
    r"drug\w*|meth\w*|cocaine|heroin|fentanyl|opioid\w*|overdose|"
    r"kill\w*|murder\w*|assassinat\w*|torture|abuse|assault|harass\w*|stalk\w*|threaten\w*|suicide|self[- ]harm|"
    r"undetect\w*|untraceable|without getting caught|evade|bypass|disable|conceal|hide)\b", re.I)


def features_af(texts):
    return {
        "imperative_stem": np.array([1.0 * bool(IMPERATIVE_STEM.search(t)) for t in texts]),
        "howto_pattern": np.array([1.0 * bool(HOWTO.search(t)) for t in texts]),
        "harm_lexicon_hits": np.array([float(len(HARM_LEXICON.findall(t))) for t in texts]),
        "harm_lexicon_any": np.array([1.0 * bool(HARM_LEXICON.search(t)) for t in texts]),
        "char_length": np.array([float(len(t)) for t in texts]),
        "harm_lexicon_plus_howto": np.array([float(len(HARM_LEXICON.findall(t))) + 0.5 * bool(HOWTO.search(t)) for t in texts]),
    }


def af_baseline():
    base = load_config()
    d = load_harmbench_eval_set(base["harmbench"], base["datasets"])
    harmful = [r["prompt"] for r in d["harmful"]]
    xs = [r["prompt"] for r in d["benign"] if r["benign_set"] == "xstest_safe"]
    jbb = [r["prompt"] for r in d["benign"] if r["benign_set"] == "jbb_benign"]
    fh, fx, fj = features_af(harmful), features_af(xs), features_af(jbb)
    from gates.runb import stratified_auc
    Lh = np.array([float(len(t)) for t in harmful]); Lx = np.array([float(len(t)) for t in xs]); Lj = np.array([float(len(t)) for t in jbb])
    out = {"n_harmful": len(harmful), "n_xstest": len(xs), "n_jbb": len(jbb), "features": {},
           "note": "auc_length_stratified is the column comparable with the probes' headline number; a pure length "
                   "feature necessarily falls to chance there, which is the point of stratifying"}
    for k in fh:
        allneg = np.concatenate([fx[k], fj[k]])
        sc = np.concatenate([fh[k], allneg]); yy = np.concatenate([np.ones(len(fh[k])), np.zeros(len(allneg))])
        LL = np.concatenate([Lh, Lx, Lj])
        scj = np.concatenate([fh[k], fj[k]]); yj = np.concatenate([np.ones(len(fh[k])), np.zeros(len(fj[k]))])
        LJ = np.concatenate([Lh, Lj])
        out["features"][k] = {"auc_all_benign": auc(fh[k], allneg), "auc_vs_xstest": auc(fh[k], fx[k]),
                              "auc_vs_jbb": auc(fh[k], fj[k]),
                              "auc_length_stratified_all": stratified_auc(sc, yy, LL)[0],
                              "auc_length_stratified_vs_jbb": stratified_auc(scj, yj, LJ)[0]}
    import math
    def keyf(kv, field):
        v = kv[1][field]
        return -1.0 if (v is None or (isinstance(v, float) and math.isnan(v))) else v
    best_all = max(out["features"].items(), key=lambda kv: keyf(kv, "auc_length_stratified_all"))
    best_jbb = max(out["features"].items(), key=lambda kv: keyf(kv, "auc_length_stratified_vs_jbb"))
    out["best_feature_all_benign"] = {"feature": best_all[0], **best_all[1]}
    out["best_feature_vs_jbb"] = {"feature": best_jbb[0], **best_jbb[1]}
    return out


def af_comparison(cfg, base):
    """The baseline beside the probes, on the numbers GATES.md judges."""
    import glob
    rows = {}
    for p in sorted(glob.glob(str(out_dir(cfg) / "b1" / "*.json"))):
        d = json.load(open(p))
        c = d["templated"]["cells"]["paired"]; j = d["raw"]["cells"]["january"]
        rows[d["model"]] = {"paired_templated_auc_vs_jbb": c["auc_vs_jbb"],
                            "paired_templated_auc_strat": c["length_all"]["auc_length_stratified"],
                            "january_raw_auc_vs_jbb": j["auc_vs_jbb"]}
    if rows:
        bj = base["best_feature_vs_jbb"]
        ref = bj["auc_length_stratified_vs_jbb"]
        beats = {m: v["paired_templated_auc_strat"] - base["best_feature_all_benign"]["auc_length_stratified_all"]
                 for m, v in rows.items()}
        beats_j = {m: v["paired_templated_auc_vs_jbb"] - bj["auc_vs_jbb"] for m, v in rows.items()}
        return {"per_model": rows, "lexical_best_vs_jbb": bj,
                "lexical_best_all_benign": base["best_feature_all_benign"],
                "paired_templated_minus_lexical_length_stratified": {"min": min(beats.values()), "max": max(beats.values()),
                                                                     "mean": sum(beats.values()) / len(beats)},
                "paired_templated_minus_lexical_vs_jbb_raw": {"min": min(beats_j.values()), "max": max(beats_j.values()),
                                                              "mean": sum(beats_j.values()) / len(beats_j)},
                "lexical_reference_length_stratified_vs_jbb": ref}
    return {}


# ---------------------------------------------------------------- omnibus lexical baseline
# TF-IDF (word unigrams + bigrams) into L2-regularised logistic regression, cross-validated. Implemented in numpy
# because the pinned environment has no scikit-learn and this repository does not add dependencies for a baseline.
# The vocabulary and the IDF weights are fitted inside each training fold, so no fold sees held-out text.
OMNIBUS_L2 = 1.0
OMNIBUS_ITERS = 300
OMNIBUS_LR = 0.5
WORD = re.compile(r"[a-z0-9']+")


def _tokens(t):
    w = WORD.findall(t.lower())
    return w + [f"{a}_{b}" for a, b in zip(w, w[1:])]


def _fit_vocab(texts, min_df=2):
    df = {}
    for t in texts:
        for tok in set(_tokens(t)):
            df[tok] = df.get(tok, 0) + 1
    vocab = {tok: i for i, tok in enumerate(sorted(k for k, v in df.items() if v >= min_df))}
    n = len(texts)
    idf = np.zeros(len(vocab))
    for tok, i in vocab.items():
        idf[i] = np.log((1 + n) / (1 + df[tok])) + 1.0
    return vocab, idf


def _transform(texts, vocab, idf):
    X = np.zeros((len(texts), len(vocab)))
    for r, t in enumerate(texts):
        for tok in _tokens(t):
            j = vocab.get(tok)
            if j is not None:
                X[r, j] += 1.0
    X *= idf
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(nrm, 1e-9)


def _logreg(X, y, l2=OMNIBUS_L2, iters=OMNIBUS_ITERS, lr=OMNIBUS_LR):
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = X.T @ (p - y) / n + l2 * w / n
        gb = float((p - y).mean())
        w -= lr * g; b -= lr * gb
    return w, b


def omnibus_baseline(texts, y, folds=None, seed=0, groups=None):
    """Cross-validated TF-IDF logistic regression. folds=None means leave-one-out.

    `groups` (e.g. a minimal-pair id per case) switches to LEAVE-GROUP-OUT, which is the only valid scheme when the set
    contains near-duplicates: under leave-one-out, a held-out case whose near-twin carrying the OPPOSITE label sits in
    the training fold is reliably anti-predicted, and the run reports an extreme inverted AUC that says nothing about
    lexical separability. See GATES.md, v2 construction standard."""
    texts = list(texts); y = np.asarray(y, float); n = len(texts)
    idx = np.arange(n)
    if groups is not None:
        groups = np.asarray(groups)
        splits = [(idx[groups != g], idx[groups == g]) for g in sorted(set(groups.tolist()))]
        scheme = f"leave-group-out ({len(splits)} groups)"
    elif folds is None or folds >= n:
        splits = [(np.delete(idx, i), np.array([i])) for i in range(n)]
        scheme = "leave-one-out"
    else:
        rng = np.random.default_rng(seed); order = rng.permutation(idx)
        parts = np.array_split(order, folds)
        splits = [(np.setdiff1d(idx, te), te) for te in parts]
        scheme = f"{folds}-fold"
    pred = np.zeros(n)
    for tr, te in splits:
        vocab, idf = _fit_vocab([texts[i] for i in tr])
        if not vocab:
            continue
        Xtr = _transform([texts[i] for i in tr], vocab, idf)
        Xte = _transform([texts[i] for i in te], vocab, idf)
        w, b = _logreg(Xtr, y[tr])
        pred[te] = Xte @ w + b
    return {"scheme": scheme, "n": n, "n_pos": int(y.sum()), "auc": auc(pred[y == 1], pred[y == 0]),
            "features": "tfidf word unigrams+bigrams, min_df=2, L2 logistic (numpy)"}


def apc_baseline(rows, label_fn, scope_name):
    y = np.array([1 if label_fn(r) else 0 for r in rows])
    txt = [r["text"] for r in rows]
    feats = {"second_person_count": np.array([float(len(SECOND_PERSON.findall(t))) for t in txt]),
             "directive_modal_count": np.array([float(len(DIRECTIVE_MODAL.findall(t))) for t in txt]),
             "char_length": np.array([float(len(t)) for t in txt])}
    out = {"scope": scope_name, "n": len(rows), "n_pos": int(y.sum()),
           **{k: auc(v[y == 1], v[y == 0]) for k, v in feats.items()}}
    out["omnibus_tfidf_logreg"] = omnibus_baseline(txt, y, folds=None if len(rows) <= 120 else 10)
    return out


def apc_baselines():
    rows, _ = load_arm3()
    out = {"v1": {}}
    for topic in ("medical", "financial", "legal"):
        sel = [r for r in rows if r["topic"] == topic and r["band"] != "refusal"]
        hard = [r for r in rows if r["topic"] == topic and r["band"] in ("hedged_advice", "information_dense")]
        out["v1"][topic] = {"all_bands": apc_baseline(sel, lambda r: r["band"] in ADVICE_BANDS, f"{topic}/all"),
                            "hard_bands": apc_baseline(hard, lambda r: r["band"] == "hedged_advice", f"{topic}/hard")}
    v2p = REPO / "gates/data/arm3_v2_discriminator.json"
    if v2p.exists():
        d = json.load(open(v2p)); cs = d["cases"]
        inc = [c for c in cs if c["cell"] in ("advice_no_2p", "information_2p")]
        out["v2"] = {"all_cells": apc_baseline(cs, lambda r: r["proposed_label"] == "advice", "v2/all"),
                     "incongruent_cells_only": apc_baseline(inc, lambda r: r["proposed_label"] == "advice", "v2/incongruent"),
                     "note": "v2 is built so second-person marking is uninformative overall and inverted on the "
                             "incongruent cells; a lexical-cue detector scores ~0.5 and ~0.0 respectively"}
    return out


def main() -> int:
    cfg = load_gates_config()
    af = af_baseline()
    res = {"af": af, "af_vs_probes": af_comparison(cfg, af), "apc": apc_baselines(),
           # The probe side of af_vs_probes aggregates the b1 cells; those files carry the per-case scores.
           # tests/test_results_completeness.py reads this key, so an aggregate is never orphaned from its scores.
           "source": ["results/gates/b1/*.json"],
           "definitions": {"second_person": SECOND_PERSON.pattern, "harm_lexicon": HARM_LEXICON.pattern[:200] + "…",
                           "note": "HARM_LEXICON is deliberately tuned to be the strongest cheap baseline available, so "
                                   "that beating it is meaningful"},
           "provenance": provenance(cfg, {"part": "lexical_baselines"})}
    p = out_dir(cfg) / "lexical_baselines.json"
    dump_json(res, p)
    print("AF lexical baselines (HarmBench 300 vs benign):")
    for k, v in af["features"].items():
        print(f"  {k:<26} AUC all {v['auc_all_benign']:.3f} (strat {v['auc_length_stratified_all']:.3f}) | "
              f"vs XSTest {v['auc_vs_xstest']:.3f} | vs JBB {v['auc_vs_jbb']:.3f} (strat {v['auc_length_stratified_vs_jbb']:.3f})")
    print(f"  best length-stratified overall: {af['best_feature_all_benign']['feature']} "
          f"({af['best_feature_all_benign']['auc_length_stratified_all']:.3f}); best length-stratified vs JBB: "
          f"{af['best_feature_vs_jbb']['feature']} ({af['best_feature_vs_jbb']['auc_length_stratified_vs_jbb']:.3f})")
    if res["af_vs_probes"]:
        m = res["af_vs_probes"]["paired_templated_minus_lexical_length_stratified"]
        print(f"  paired/templated probe minus best lexical, length-stratified: min {m['min']:+.3f} max {m['max']:+.3f} mean {m['mean']:+.3f}")
    print("\nAPC lexical baselines:")
    for topic, v in res["apc"]["v1"].items():
        for scope, c in v.items():
            print(f"  v1 {topic:<10} {scope:<11} n={c['n']:3d} second-person AUC {c['second_person_count']:.3f} "
                  f"modal {c['directive_modal_count']:.3f} length {c['char_length']:.3f}")
    if "v2" in res["apc"]:
        for scope, c in res["apc"]["v2"].items():
            if isinstance(c, dict):
                print(f"  v2 {scope:<24} n={c['n']:3d} second-person AUC {c['second_person_count']:.3f} "
                      f"modal {c['directive_modal_count']:.3f} length {c['char_length']:.3f}")
    print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
