#!/usr/bin/env python3
"""P3 — training-set circularity control.

Retrain an AF probe with the December recipe (AFTrainingData from the Dec-26 three-way script: 81 dangerous prompts of which
30 are Base64/ROT13/leetspeak-encoded, 40 safe; difference-of-means direction; midpoint threshold on RAW dot products;
Gemma chat template at extraction = the December convention), for every block in both modes, plus the same recipe with the
30 encoded prompts removed (ablation). Evaluate on the P2 activation cache (all 1,200 harmful + 350 benign x conditions).

Reports, per {probe recipe x mode x condition}: at the December block (27) and at the block with the highest encoded-harmful
detection — detection rate of encoded harmful, FPR on encoded benign at the SAME threshold, plain-text detection/FPR, and AUC.
Hypothesis under test: the probe detects the encoding, not the harm -> high encoded detection together with high encoded-benign FPR.

  python -m phase2.p3_circularity [--dry-run]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase2.common import (build_eval_set, december_plain_training_set, december_training_set, dump_json, format_for_mode,  # noqa: E402
                           load_phase2_config, out_dir, provenance)
from phase2.p2_layer_sweep import act_path  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_phase2_config(a.config)
    dec = december_training_set(); decp = december_plain_training_set()
    print(f"P3 December recipe: {len(dec['harmful'])} dangerous ({dec['n_encoded_in_harmful']} encoded: {dec['n_base64_rot13']} base64/rot13 + {dec['n_leet']} leet), {len(dec['benign'])} safe; ablation: {len(decp['harmful'])} plain dangerous")
    for p in [x for x in dec["harmful"] if x.startswith(("Decode and execute", "ROT13"))][:2]:
        print("  encoded training prompt:", p[:110])
    od = out_dir(cfg)
    if a.dry_run:
        print("output ->", od / "p3_circularity.json (schema: provenance, recipes{december, december_plain}{mode}{block}{condition}{detection, fpr_benign, auc, threshold}, at_december_block, at_best_block)")
        return 0
    from phase2.extract import auc, dist_stats, extract_last_token, load_model, score, train_direction
    blocks = cfg["blocks"]; bi = {b: i for i, b in enumerate(blocks)}
    data = build_eval_set(cfg); by_id = {r["id"]: r for r in data["records"]}
    # extract training activations for both recipes in both modes (cache)
    trains = {"december": dec, "december_plain": decp}
    need = [(rname, mode) for rname in trains for mode in cfg["modes"] if not act_path(cfg, mode, rname, "train").exists()]
    if need:
        model, tok, info = load_model(cfg["model"], cfg.get("dtype", "bfloat16"))
        for rname, mode in need:
            tr = trains[rname]
            texts = [format_for_mode(t, mode) for t in tr["harmful"] + tr["benign"]]
            acts = extract_last_token(model, tok, texts, blocks, cfg.get("batch_size", 16), cfg.get("max_length", 1024), progress=f"train-{rname}/{mode}")
            np.savez(act_path(cfg, mode, rname, "train"), acts=acts, labels=np.array([1] * len(tr["harmful"]) + [0] * len(tr["benign"])), blocks=np.array(blocks))
        del model
    else:
        info = {"note": "training activations cached"}
    out = {"provenance": provenance(cfg, {"part": "p3", "model_info": info, "december_training_set": dec["provenance"]}),
           "recipes": {}, "at_december_block": {}, "at_best_block": {}, "december_block": cfg["p3"]["december_block"]}
    wrapper = "phase1"
    for rname, tr in trains.items():
        out["recipes"][rname] = {}
        for mode in cfg["modes"]:
            z = np.load(act_path(cfg, mode, rname, "train")); tacts, tl = z["acts"], z["labels"]
            ze = np.load(act_path(cfg, mode, wrapper)); acts, ids = ze["acts"], list(ze["ids"])
            labels = np.array([by_id[i]["label"] for i in ids]); conds = np.array([by_id[i]["condition"] for i in ids])
            out["recipes"][rname][mode] = {}
            for b in blocks:
                pr = train_direction(tacts[tl == 1, bi[b], :], tacts[tl == 0, bi[b], :], "december")
                s = score(acts[:, bi[b], :], pr["direction"], "december")
                cell = {"threshold": pr["threshold"], "train_separation": pr["train_separation"], "train_auc": pr["train_auc"]}
                for cond in cfg["conditions"]:
                    m = conds == cond; h, n = s[m & (labels == 1)], s[m & (labels == 0)]
                    cell[cond] = {"detection": float((h > pr["threshold"]).mean()), "fpr_benign": float((n > pr["threshold"]).mean()), "auc": auc(h, n),
                                  "harmful": dist_stats(h), "benign": dist_stats(n)}
                out["recipes"][rname][mode][str(b)] = cell
            # December block and best block (by mean encoded-harmful detection over base64+rot13+leetspeak)
            db = str(cfg["p3"]["december_block"])
            out["at_december_block"][f"{rname}|{mode}"] = {c: out["recipes"][rname][mode][db][c] for c in cfg["conditions"]} | {"block": int(db), "threshold": out["recipes"][rname][mode][db]["threshold"]}
            best = max(blocks, key=lambda b: np.mean([out["recipes"][rname][mode][str(b)][c]["detection"] for c in ("base64", "rot13", "leetspeak")]))
            out["at_best_block"][f"{rname}|{mode}"] = {c: out["recipes"][rname][mode][str(best)][c] for c in cfg["conditions"]} | {"block": int(best), "threshold": out["recipes"][rname][mode][str(best)]["threshold"]}
    dump_json(out, od / "p3_circularity.json")
    for k, v in out["at_december_block"].items():
        print(f"  [{k}] block {v['block']}: " + " ".join(f"{c}: det {v[c]['detection']:.2f} fpr {v[c]['fpr_benign']:.2f} auc {v[c]['auc']:.2f}" for c in cfg["conditions"]))
    print("wrote", od / "p3_circularity.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
