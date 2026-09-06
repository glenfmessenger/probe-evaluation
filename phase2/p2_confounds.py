#!/usr/bin/env python3
"""P2c — confound control for the P2 (and P2b) layer sweeps. CPU only; reads results/phase2/p2_scores{,_mlp}.npz.

Two questions the raw AUCs cannot answer:
  1. Length. AILuminate harmful prompts are much longer than the XSTest/JBB benign prompts (length alone separates the
     classes at AUC ~0.95), and an encoded prompt's token count scales with its plaintext length. A probe cell whose score
     tracks length would post a high harmful-vs-benign AUC without reading any harm. Control: Spearman(score, length) and a
     length-stratified AUC (quantile bins of text length within each condition, per-bin Mann-Whitney AUC pooled with
     n_pos*n_neg weights; bins with fewer than `min_per_class` of either class are dropped and reported).
  2. Semantic coupling. If a block represented the *decoded* content of an encoded prompt, the probe score on the encoded
     prompt should track the same prompt's plain-text score at that block. Control: Spearman(score_encoded, score_plain)
     within harmful and within benign prompts, per block.

Outputs results/phase2/p2_confounds.json and prints a markdown summary.

  python -m phase2.p2_confounds [--dry-run]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase2.common import build_eval_set, dump_json, format_for_mode, load_phase2_config, out_dir, provenance  # noqa: E402
from phase2.extract import auc  # noqa: E402

N_BINS = 5
MIN_PER_CLASS = 5


def spearman(a, b) -> float:
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def stratified_auc(s, y, L, n_bins=N_BINS, min_per_class=MIN_PER_CLASS):
    """Length-stratified AUC: quantile bins of L over all records of the condition; per-bin AUC pooled by n_pos*n_neg."""
    edges = np.quantile(L, np.linspace(0, 1, n_bins + 1)); edges[-1] += 1
    num, den, bins = 0.0, 0.0, []
    for k in range(n_bins):
        m = (L >= edges[k]) & (L < edges[k + 1])
        p, n = s[m & (y == 1)], s[m & (y == 0)]
        b = {"len_range": [int(edges[k]), int(edges[k + 1] - 1)], "n_pos": int(len(p)), "n_neg": int(len(n)), "auc": None}
        if len(p) >= min_per_class and len(n) >= min_per_class:
            b["auc"] = auc(p, n); w = len(p) * len(n); num += b["auc"] * w; den += w
        bins.append(b)
    return (num / den if den else float("nan")), bins


def analyse_feature(cfg, data, npz_path: Path, feature: str):
    z = np.load(npz_path)
    by_id = {r["id"]: r for r in data["records"]}
    blocks = cfg["blocks"]; out = {"feature": feature, "scores_file": npz_path.name, "cells": {}, "summary": {}}
    for mode in cfg["modes"]:
        ids = list(z[f"ids|{mode}|phase1"]); pos = {i: k for k, i in enumerate(ids)}
        cond = np.array([by_id[i]["condition"] for i in ids]); y = np.array([by_id[i]["label"] for i in ids])
        L = np.array([len(format_for_mode(by_id[i]["text"], mode)) for i in ids])           # characters of what the model saw
        plain_idx = np.array([pos[i.replace(f"|{by_id[i]['condition']}|", "|none|")] for i in ids])
        for c in cfg["conditions"]:
            m = cond == c
            len_auc = auc(L[m & (y == 1)], L[m & (y == 0)])
            for pname in ("january", "inmode"):
                best_raw, best_strat = None, None
                for b in blocks:
                    s = z[f"{pname}|{mode}|phase1|{b}"].astype(np.float64); sm, ym, Lm = s[m], y[m], L[m]
                    sp = s[plain_idx[m]]
                    a_raw = auc(sm[ym == 1], sm[ym == 0]); a_str, bins = stratified_auc(sm, ym, Lm)
                    cell = {"auc": a_raw, "auc_length_stratified": a_str, "bins": bins, "length_auc": len_auc,
                            "rho_score_length": spearman(sm, Lm), "rho_score_length_harmful": spearman(sm[ym == 1], Lm[ym == 1]),
                            "rho_encoded_vs_plain_harmful": spearman(sm[ym == 1], sp[ym == 1]) if c != "none" else 1.0,
                            "rho_encoded_vs_plain_benign": spearman(sm[ym == 0], sp[ym == 0]) if c != "none" else 1.0}
                    out["cells"][f"{pname}|{mode}|phase1|{b}|{c}"] = cell
                    if best_raw is None or a_raw > best_raw[1]:
                        best_raw = (b, a_raw, cell)
                    if not np.isnan(a_str) and (best_strat is None or a_str > best_strat[1]):
                        best_strat = (b, a_str, cell)
                out["summary"][f"{pname}|{mode}|{c}"] = {
                    "length_auc": len_auc,
                    "best_raw": {"block": best_raw[0], "auc": best_raw[1], "auc_length_stratified": best_raw[2]["auc_length_stratified"],
                                 "rho_score_length": best_raw[2]["rho_score_length"], "rho_encoded_vs_plain_harmful": best_raw[2]["rho_encoded_vs_plain_harmful"]},
                    "best_length_stratified": {"block": best_strat[0], "auc_length_stratified": best_strat[1], "auc": best_strat[2]["auc"],
                                               "rho_encoded_vs_plain_harmful": best_strat[2]["rho_encoded_vs_plain_harmful"]} if best_strat else None,
                    "max_rho_encoded_vs_plain_harmful_over_blocks": max(out["cells"][f"{pname}|{mode}|phase1|{b}|{c}"]["rho_encoded_vs_plain_harmful"] for b in blocks),
                }
    return out


def markdown(res) -> str:
    L = []
    for feature, r in res["features"].items():
        L += [f"#### Length control and encoded-vs-plain coupling — feature `{feature}` ({r['scores_file']})", "",
              "| Probe | Mode | Condition | length-only AUC | best raw AUC @ block | same cell: length-stratified AUC | rho(score, length) | best length-stratified AUC @ block | max over blocks of rho(encoded score, plain score) within harmful |",
              "|---|---|---|---|---|---|---|---|---|"]
        for k, v in r["summary"].items():
            p, m, c = k.split("|"); br, bs = v["best_raw"], v["best_length_stratified"]
            L.append(f"| {p} | {m} | {c} | {v['length_auc']:.2f} | {br['auc']:.3f} @ b{br['block']} | {br['auc_length_stratified']:.3f} | {br['rho_score_length']:+.2f} | "
                     + (f"{bs['auc_length_stratified']:.3f} @ b{bs['block']}" if bs else "—") + f" | {v['max_rho_encoded_vs_plain_harmful_over_blocks']:+.2f} |")
        L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_phase2_config(a.config); od = out_dir(cfg)
    data = build_eval_set(cfg)
    files = {"residual": od / "p2_scores.npz", "mlp_out": od / "p2_scores_mlp.npz"}
    print(f"P2c confounds: eval set {data['summary']['n_records']} records; score files: " + ", ".join(f"{k}={'present' if p.exists() else 'absent'}" for k, p in files.items()))
    print(f"  length-stratified AUC: {N_BINS} quantile bins of text length per condition, bins need >= {MIN_PER_CLASS} per class")
    if a.dry_run:
        print("output ->", od / "p2_confounds.json")
        return 0
    res = {"features": {}, "method": {"n_bins": N_BINS, "min_per_class": MIN_PER_CLASS, "length": "characters of the mode-formatted text the model saw",
                                       "stratified_auc": "per-bin Mann-Whitney AUC pooled with n_pos*n_neg weights", "coupling": "Spearman(score on encoded prompt, score on the same prompt's plain text at the same block)"}}
    for feature, p in files.items():
        if p.exists():
            res["features"][feature] = analyse_feature(cfg, data, p, feature)
    if not res["features"]:
        print("no score files found", file=sys.stderr); return 1
    res["provenance"] = provenance(cfg, {"part": "p2c", "inputs": [p.name for p in files.values() if p.exists()]})
    dump_json(res, od / "p2_confounds.json")
    print(markdown(res))
    print("wrote", od / "p2_confounds.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
