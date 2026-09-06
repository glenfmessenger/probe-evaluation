#!/usr/bin/env python3
"""ARM 3 (B3) — APC external evaluation and cross-topic transfer.

The shipped APC probes were trained on 15 short authored (violating, compliant) pairs per policy and reported AUC 1.00
on a 6-example held-out split. This scores them, and freshly trained paired/standard probes from the same 15 medical
pairs, on the signed-off external set: 240 medical model responses in four bands plus 20 refusals, and 60 financial +
60 legal for cross-topic transfer.

Cases are model RESPONSES, so `raw` mode is the primary framing (it matches how the shipped probe was trained); the
templated mode wraps the text as the model's own turn and is reported as a secondary.

Reported per probe per model:
  * external medical AUC, advice vs information, refusals excluded — raw and length-stratified (the PASS criterion);
  * cross-topic transfer to financial and legal, same treatment;
  * the per-band breakdown, with `hedged_advice` and `information_dense` called out — a probe that scores well overall
    while failing both hard bands has learned hedging or specificity, not the speech act;
  * GATES.md Amendment 1: the refusal band explicitly — score distribution and flag rate on the 20 medical refusals,
    per probe per model, as its own table. A probe that flags refusals as violations would fire on exactly the
    responses the policy wants, which is a deployment finding and is not visible inside an AUC that excludes them.

  python -m gates.b3_apc_external [--dry-run] [--models M ...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gates.build_review import ADVICE_BANDS, INFO_BANDS, load_arm3  # noqa: E402
from gates.runb import (act_dir, check_scores, dump_json, format_mode, fpr_at_threshold, length_report,  # noqa: E402
                        load_gates_config, load_probe, out_dir, provenance, safe, train_rule)
from phase2.extract import auc, dist_stats, score, unit  # noqa: E402


def training_pairs(policy: str):
    """The original 15 (violating, compliant) pairs the shipped probe was trained on."""
    import importlib.util
    p = Path(__file__).resolve().parents[1] / "aase_vllm/scripts/train_probes_vllm.py"
    spec = importlib.util.spec_from_file_location("_tp", p)
    import ast
    tree = ast.parse(p.read_text())
    ns = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "").endswith("_PAIRS"):
            ns[node.targets[0].id] = ast.literal_eval(node.value)
    key = f"{policy.upper()}_PAIRS"
    if key not in ns:
        raise KeyError(f"{key} not found in {p}")
    return ns[key]


def cases():
    rows, missing = load_arm3()
    if missing:
        raise FileNotFoundError(f"Arm 3 case files missing: {missing}")
    return rows


def evaluate_probe(sc, rows, thr, topic):
    """AUC on advice vs information (refusals excluded), plus per-band rates and the refusal table."""
    sel = [i for i, r in enumerate(rows) if r["topic"] == topic and r["band"] != "refusal"]
    s = sc[sel]; y = np.array([1 if rows[i]["band"] in ADVICE_BANDS else 0 for i in sel])
    L = np.array([len(rows[i]["text"]) for i in sel])
    cell = {"n": len(sel), "n_advice": int(y.sum()), "n_information": int((1 - y).sum()), "threshold": thr,
            "auc": auc(s[y == 1], s[y == 0]), "flag_rate_advice": fpr_at_threshold(s[y == 1], thr),
            "flag_rate_information": fpr_at_threshold(s[y == 0], thr), "length": length_report(s, y, L), "bands": {}}
    for band in ("clear_advice", "hedged_advice", "clear_information", "information_dense", "refusal"):
        idx = [i for i, r in enumerate(rows) if r["topic"] == topic and r["band"] == band]
        if not idx:
            continue
        bs = sc[idx]
        cell["bands"][band] = {"n": len(idx), "flag_rate": fpr_at_threshold(bs, thr), "scores": dist_stats(bs),
                               "expected_label": "advice" if band in ADVICE_BANDS else ("information" if band in INFO_BANDS else "refusal")}
    hard = {}
    for band, want in (("hedged_advice", 1), ("information_dense", 0)):
        idx = [i for i, r in enumerate(rows) if r["topic"] == topic and r["band"] == band]
        if idx:
            other = [i for i, r in enumerate(rows) if r["topic"] == topic and r["band"] ==
                     ("information_dense" if want else "hedged_advice")]
            hard[band] = {"n": len(idx), "correct_rate": (fpr_at_threshold(sc[idx], thr) if want else 1 - fpr_at_threshold(sc[idx], thr)),
                          "auc_vs_other_hard_band": auc(sc[idx], sc[other]) if want else auc(sc[other], sc[idx])}
    cell["hard_bands"] = hard
    return cell


def refusal_band_table(sc, rows, thr):
    """GATES.md Amendment 1 — the refusal band on its own, medical only."""
    idx = [i for i, r in enumerate(rows) if r["topic"] == "medical" and r["band"] == "refusal"]
    s = sc[idx]
    adv = [i for i, r in enumerate(rows) if r["topic"] == "medical" and r["band"] in ADVICE_BANDS]
    inf = [i for i, r in enumerate(rows) if r["topic"] == "medical" and r["band"] in INFO_BANDS]
    return {"n": len(idx), "threshold": thr, "flag_rate": fpr_at_threshold(s, thr), "scores": dist_stats(s),
            "auc_vs_advice": auc(sc[adv], s), "auc_vs_information": auc(sc[inf], s),
            "mean_minus_information_mean": float(s.mean() - sc[inf].mean()),
            "flagged_ids": [rows[i]["id"] for i in idx if sc[i] > thr]}


def run_model(cfg, model_name, rows):
    from phase2.extract import extract_last_token, load_model
    b3 = cfg["b3"]; policy = b3["policy"]
    shipped = load_probe(cfg, "apc", model_name, policy)
    layer = shipped["layer_index"]
    pairs = training_pairs(policy)
    model, tok, info = load_model(model_name, cfg.get("dtype", "bfloat16"))
    print(f"  {model_name}: APC/{policy} layer_index {layer} (layer_pct {shipped['meta'].get('layer_pct')}), "
          f"{len(pairs)} training pairs", flush=True)
    out = {"model": model_name, "policy": policy, "layer_index": layer, "n_training_pairs": len(pairs), "modes": {}}
    for mode in cfg["modes"]:
        tp = extract_last_token(model, tok, [format_mode(p[0], mode, model_name, tok) for p in pairs], [layer],
                                cfg["batch_size"], cfg["max_length"], feature=cfg["feature"])[:, 0, :]
        tn = extract_last_token(model, tok, [format_mode(p[1], mode, model_name, tok) for p in pairs], [layer],
                                cfg["batch_size"], cfg["max_length"], feature=cfg["feature"])[:, 0, :]
        acts = extract_last_token(model, tok, [format_mode(r["text"], mode, model_name, tok) for r in rows], [layer],
                                  cfg["batch_size"], cfg["max_length"], progress=f"{safe(model_name)}/{mode}",
                                  feature=cfg["feature"])[:, 0, :]
        np.savez_compressed(act_dir(cfg) / f"b3_{safe(model_name)}_{mode}.npz", acts=acts,
                            ids=np.array([r["id"] for r in rows]), layer=layer)
        probes = {"january": {**shipped, "source": "shipped Phase 1 APC probe"}}
        for rule in b3["rules"]:
            probes[rule] = train_rule(tp, tn, rule, pairs=True)
        cells = {}
        for pname, pr in probes.items():
            sc = score(acts, pr["direction"], "january")
            check_scores(f"{model_name}|{mode}|apc-{pname}", sc)
            c = {"threshold": pr["threshold"], "train_auc": pr.get("train_auc"),
                 "medical": evaluate_probe(sc, rows, pr["threshold"], "medical"),
                 "financial": evaluate_probe(sc, rows, pr["threshold"], "financial"),
                 "legal": evaluate_probe(sc, rows, pr["threshold"], "legal"),
                 "refusal_band": refusal_band_table(sc, rows, pr["threshold"]),
                 "scores": {r["id"]: float(v) for r, v in zip(rows, sc)}}
            cells[pname] = c
            m, f, l = c["medical"], c["financial"], c["legal"]
            print(f"    [{mode}] {pname:9s} medical AUC {m['auc']:.3f} (strat {m['length']['auc_length_stratified']:.3f}) "
                  f"| transfer fin {f['auc']:.3f} leg {l['auc']:.3f} | hard bands hedged {m['hard_bands']['hedged_advice']['correct_rate']:.2f} "
                  f"dense {m['hard_bands']['information_dense']['correct_rate']:.2f} | refusal flag {c['refusal_band']['flag_rate']:.2f}",
                  flush=True)
        out["modes"][mode] = {"probes": {k: {kk: vv for kk, vv in v.items() if kk != "direction"} for k, v in probes.items()},
                              "cells": cells}
    del model
    import torch; torch.cuda.empty_cache()
    return out


def pick_secondaries(cfg):
    """The two best AF models from Arm 1, by AUC vs JBB benign (the Arm 1 criterion that matters), read from B1 output."""
    import json
    d = out_dir(cfg) / "b1"
    best = []
    for p in sorted(d.glob("*.json")):
        r = json.load(open(p))
        v = max((r[m]["cells"][pn]["auc_vs_jbb"] for m in cfg["modes"] if m in r for pn in ("paired", "standard")),
                default=None)
        if v is not None:
            best.append((v, r["model"]))
    best.sort(reverse=True)
    return [m for _, m in best[:cfg["b3"]["n_secondary_models"]]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--models", nargs="*", default=None)
    a = ap.parse_args(argv)
    cfg = load_gates_config(a.config)
    rows = cases()
    n = {t: sum(r["topic"] == t for r in rows) for t in ("medical", "financial", "legal")}
    models = a.models
    if not models:
        models = [cfg["b3"]["primary_model"]]
        if not a.dry_run:
            models += [m for m in pick_secondaries(cfg) if m != cfg["b3"]["primary_model"]][:cfg["b3"]["n_secondary_models"]]
    print(f"B3 cases: {len(rows)} ({n}); bands advice={sorted(ADVICE_BANDS)} information={sorted(INFO_BANDS)} + refusal; "
          f"policy {cfg['b3']['policy']}; feature {cfg['feature']}; modes {cfg['modes']}")
    print(f"B3 models: {models} (primary {cfg['b3']['primary_model']}, secondaries chosen from B1 by AUC vs JBB benign)")
    od = out_dir(cfg) / "b3"
    if a.dry_run:
        print(f"training pairs available: {len(training_pairs(cfg['b3']['policy']))}")
        print("output ->", od / "<model>.json")
        return 0
    for m in models:
        out = run_model(cfg, m, rows)
        out["case_counts"] = n
        out["provenance"] = provenance(cfg, {"part": "b3", "model": m})
        p = od / f"{safe(m)}.json"; dump_json(out, p)
        print("  wrote", p, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
