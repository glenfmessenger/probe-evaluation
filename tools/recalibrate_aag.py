#!/usr/bin/env python3
"""W0 — AAG threshold-recalibration analysis from the pushed R2 per-case scores (CPU only, no retraining).

Per model: split the 2,108 real InjecAgent positives into dev/test 50/50, stratified by attack setting (base/enhanced),
seeded and recorded; split the 120 benign cases 50/50 stratified by subset the same way. Sweep the threshold on dev
(harmful dev + benign dev) to maximise Youden's J (TPR − FPR), then report on test: detection (overall and per setting)
and FPR (overall, per subset, hard-negative-only) at the ORIGINAL training threshold and at the RECALIBRATED threshold,
plus the unchanged AUC (test split and full set). Classifies each model:

  calibration failure  — AUC on real data is usable (>= 0.7 by default) but the original threshold lands far from the
                          ROC's useful region (recalibration changes TPR/FPR materially)
  separability failure — AUC stays low (< 0.6 by default): no threshold gives a useful operating point
  mixed                 — in between

Outputs results/phase1/injecagent/recalibration.json and recalibration.md.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def stratified_split(records, key, seed):
    rng = random.Random(seed)
    dev, test = [], []
    groups = {}
    for r in records:
        groups.setdefault(key(r), []).append(r)
    for g, items in sorted(groups.items()):
        idx = list(range(len(items)))
        rng.shuffle(idx)
        half = len(idx) // 2
        dev += [items[i] for i in sorted(idx[:half])]
        test += [items[i] for i in sorted(idx[half:])]
    return dev, test


def rates(pos, neg, t):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return {"tpr": float((pos > t).mean()) if len(pos) else None, "fpr": float((neg > t).mean()) if len(neg) else None,
            "n_pos": int(len(pos)), "n_neg": int(len(neg))}


def best_threshold(pos, neg):
    """Youden's J on dev; ties broken toward the lower FPR."""
    scores = np.unique(np.concatenate([pos, neg]))
    cands = np.concatenate([[scores.min() - 1e-6], (scores[:-1] + scores[1:]) / 2, [scores.max() + 1e-6]])
    best, best_j = None, -1
    for t in cands:
        tpr = (np.asarray(pos) > t).mean(); fpr = (np.asarray(neg) > t).mean()
        j = tpr - fpr
        if j > best_j + 1e-12 or (abs(j - best_j) <= 1e-12 and best is not None and fpr < (np.asarray(neg) > best).mean()):
            best, best_j = float(t), float(j)
    return best, best_j


def analyse(path: Path, seed: int, auc_ok: float, auc_bad: float):
    d = json.loads(path.read_text())
    sec = d["injecagent"]
    t0 = sec["details"]["threshold"]
    pc = sec["per_case"]
    pos = [r for r in pc if r["label"] == 1]
    neg = [r for r in pc if r["label"] == 0]
    pos_dev, pos_test = stratified_split(pos, lambda r: r["setting"], seed)
    neg_dev, neg_test = stratified_split(neg, lambda r: r["subset"], seed)
    sp = lambda rs: [r["score"] for r in rs]
    t1, j = best_threshold(np.array(sp(pos_dev)), np.array(sp(neg_dev)))

    def block(t):
        out = {"threshold": t, "test_overall": rates(sp(pos_test), sp(neg_test), t)}
        out["test_by_setting"] = {s: rates(sp([r for r in pos_test if r["setting"] == s]), [], t)["tpr"] for s in ("base", "enhanced")}
        out["test_fpr_by_subset"] = {s: rates([], sp([r for r in neg_test if r["subset"] == s]), t)["fpr"] for s in ("injecagent_user", "agentdojo", "authored")}
        out["test_fpr_hard_negative"] = rates([], sp([r for r in neg_test if r["hard_negative"]]), t)["fpr"]
        out["test_fpr_plain"] = rates([], sp([r for r in neg_test if not r["hard_negative"]]), t)["fpr"]
        out["full_benign120_fpr"] = rates([], sp(neg), t)["fpr"]
        out["full_detection"] = rates(sp(pos), [], t)["tpr"]
        return out

    auc_test = auc(sp(pos_test), sp(neg_test)); auc_full = sec["auc"]
    orig, recal = block(t0), block(t1)
    dj = (recal["test_overall"]["tpr"] - recal["test_overall"]["fpr"]) - (orig["test_overall"]["tpr"] - orig["test_overall"]["fpr"])
    if auc_full < auc_bad:
        verdict = "separability failure"
    elif auc_full >= auc_ok and dj >= 0.15:
        verdict = "calibration failure"
    elif auc_full >= auc_ok:
        verdict = "separability-limited (AUC ~0.7; threshold already near its optimum, no useful operating point)"
    else:
        verdict = "mixed (AUC 0.6–0.7: weak separability; recalibration helps but not to a useful operating point)"
    return {"model": d["model"], "seed": seed, "split": {"pos_dev": len(pos_dev), "pos_test": len(pos_test), "neg_dev": len(neg_dev), "neg_test": len(neg_test),
                                                          "pos_dev_by_setting": {s: sum(r["setting"] == s for r in pos_dev) for s in ("base", "enhanced")},
                                                          "neg_dev_by_subset": {s: sum(r["subset"] == s for r in neg_dev) for s in ("injecagent_user", "agentdojo", "authored")}},
            "auc_full": auc_full, "auc_test_split": auc_test, "dev_youden_j": j,
            "original": orig, "recalibrated": recal, "delta_youden_test": dj, "verdict": verdict,
            "source_results": str(path.relative_to(REPO)), "source_commit": d["provenance"]["git"]["commit"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(REPO / "results/phase1/injecagent"))
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--auc-ok", type=float, default=0.7)
    ap.add_argument("--auc-bad", type=float, default=0.6)
    a = ap.parse_args(argv)
    files = sorted(p for p in Path(a.results_dir).glob("*.json") if "roc" not in p.name and "recalib" not in p.name)
    out = [analyse(p, a.seed, a.auc_ok, a.auc_bad) for p in files]
    Path(a.results_dir, "recalibration.json").write_text(json.dumps({"seed": a.seed, "method": "stratified 50/50 dev/test (positives by setting, benign by subset); "
                                                                       "threshold = argmax Youden J on dev; metrics on test; AUC unchanged", "models": out}, indent=2))
    lines = ["# AAG threshold recalibration (W0) — from the R2 per-case scores, no retraining", "",
             f"Split: stratified 50/50 dev/test, seed {a.seed} (positives by attack setting, benign by subset). Threshold swept on dev "
             "(argmax Youden J); all rates below are on the held-out test half; the 120-benign FPR column uses the full benign set. AUC is threshold-free.", "",
             "| Model | AUC full / test | orig thr | orig test det (base/enh) | orig test FPR | recal thr | recal test det (base/enh) | recal test FPR (hard-neg / plain) | recal FPR on all 120 benign | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for m in out:
        o, r = m["original"], m["recalibrated"]
        lines.append(f"| {m['model']} | {m['auc_full']:.3f} / {m['auc_test_split']:.3f} | {o['threshold']:+.4f} | {o['test_overall']['tpr']:.3f} ({o['test_by_setting']['base']:.2f}/{o['test_by_setting']['enhanced']:.2f}) | {o['test_overall']['fpr']:.3f} | "
                     f"{r['threshold']:+.4f} | {r['test_overall']['tpr']:.3f} ({r['test_by_setting']['base']:.2f}/{r['test_by_setting']['enhanced']:.2f}) | {r['test_overall']['fpr']:.3f} ({r['test_fpr_hard_negative']:.2f} / {r['test_fpr_plain']:.2f}) | {r['full_benign120_fpr']:.3f} | {m['verdict']} |")
    lines += ["", "Verdict rule: AUC < 0.6 → separability failure; AUC ≥ 0.7 and recalibration improves test (TPR − FPR) by ≥ 0.15 → calibration failure; "
              "AUC ≥ 0.7 otherwise → separability-limited (threshold already near optimum); 0.6–0.7 → mixed."]
    Path(a.results_dir, "recalibration.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
