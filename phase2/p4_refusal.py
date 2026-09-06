#!/usr/bin/env python3
"""P4 — refusal separation: the probe's marginal value over the model's own behaviour.

Stratified sample (seeded): per_cell prompts x 4 conditions x {harmful, benign} (default 25 -> 200), phase1 wrapper.
Each prompt is generated through the chat template (greedy, max_new_tokens); refusal = December regexes on the
continuation. Probe scores come from the P2 cache: the January probe at block 21 (raw mode) and the in-mode retrained
probe at the block with the best plain-text AUC in templated mode (recomputed here from the cached activations).
Report detection conditioned on refusal status, per condition and label.

  python -m phase2.p4_refusal [--dry-run]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase2.common import REPO, build_eval_set, dump_json, is_refusal, load_phase2_config, out_dir, provenance, stratified_sample  # noqa: E402
from phase2.p2_layer_sweep import CONVENTIONS, act_path, sfx  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--feature", default=None, choices=["residual", "mlp_out"], help="activation feature for both probes (default: config `feature`, else residual)")
    a = ap.parse_args(argv)
    cfg = load_phase2_config(a.config); p4 = cfg["p4"]
    if a.feature:
        cfg["feature"] = a.feature
    cfg.setdefault("feature", "residual")
    print(f"P4 feature: {cfg['feature']} — {CONVENTIONS[cfg['feature']]}")
    data = build_eval_set(cfg)
    sample = stratified_sample(data["records"], p4["per_cell"], p4["seed"], wrapper="phase1")
    cells = {}
    for r in sample:
        cells[(r["condition"], r["label"])] = cells.get((r["condition"], r["label"]), 0) + 1
    print(f"P4 sample: {len(sample)} prompts, cells {sorted(cells.items())}, seed {p4['seed']}")
    od = out_dir(cfg)
    if a.dry_run:
        print("output ->", od / f"p4_refusal{sfx(cfg)}.json (schema: provenance, items[{{id, condition, label, refusal, output, score_january_b21_raw, score_inmode_templated}}], table{{condition}}{{label}}{{refused|not_refused}}{{n, detection_january, detection_inmode}})")
        return 0
    import json
    from phase2.extract import generate, load_model, load_january_probe, score, train_direction
    # Generations are feature-independent (they only depend on the prompt); cache them so P4b (mlp_out) re-scores the
    # same continuations instead of regenerating (greedy, so a regeneration would be identical anyway).
    gen_cache = od / "p4_generations.json"
    if gen_cache.exists():
        gc = json.load(open(gen_cache))
        if [r["id"] for r in sample] == gc["ids"] and gc["max_new_tokens"] == p4["max_new_tokens"]:
            outs = gc["outputs"]; info = gc["model_info"]; print(f"  reusing cached generations {gen_cache.name} (n={len(outs)})")
        else:
            outs = None
    else:
        outs = None
    if outs is None:
        model, tok, info = load_model(cfg["model"], cfg.get("dtype", "bfloat16"))
        outs = generate(model, tok, [r["text"] for r in sample], max_new_tokens=p4["max_new_tokens"], batch_size=max(1, cfg.get("batch_size", 16) // 2))
        del model
        dump_json({"ids": [r["id"] for r in sample], "max_new_tokens": p4["max_new_tokens"], "model_info": info, "outputs": outs}, gen_cache)
    blocks = cfg["blocks"]; bi = {b: i for i, b in enumerate(blocks)}
    jan = load_january_probe(REPO, cfg["p2"]["january_probe_model"])
    zr = np.load(act_path(cfg, "raw", "phase1")); ids_r = list(zr["ids"]); pos_r = {i: k for k, i in enumerate(ids_r)}
    zt = np.load(act_path(cfg, "templated", "phase1")); ids_t = list(zt["ids"]); pos_t = {i: k for k, i in enumerate(ids_t)}
    # in-mode templated probe at its best plain-text block (from the January training cache)
    ztr = np.load(act_path(cfg, "templated", "january", "train")); by_id = {r["id"]: r for r in data["records"]}
    labels_t = np.array([by_id[i]["label"] for i in ids_t]); conds_t = np.array([by_id[i]["condition"] for i in ids_t])
    from phase2.extract import auc as _auc
    best_b, best_auc, best_pr = None, -1, None
    for b in blocks:
        pr = train_direction(ztr["acts"][ztr["labels"] == 1, bi[b], :], ztr["acts"][ztr["labels"] == 0, bi[b], :], "january")
        s = score(zt["acts"][:, bi[b], :], pr["direction"], "january"); m = conds_t == "none"
        v = _auc(s[m & (labels_t == 1)], s[m & (labels_t == 0)])
        if v > best_auc:
            best_b, best_auc, best_pr = b, v, pr
    items, table = [], {}
    for r, o in zip(sample, outs):
        ref = is_refusal(o)
        sj = float(score(zr["acts"][pos_r[r["id"]], bi[jan["layer_index"]], :][None], jan["direction"], "january")[0])
        si = float(score(zt["acts"][pos_t[r["id"]], bi[best_b], :][None], best_pr["direction"], "january")[0])
        items.append({"id": r["id"], "condition": r["condition"], "label": r["label"], "refusal": ref, "output": o[:300],
                      "score_january_b21_raw": sj, "det_january": sj > jan["threshold"], "score_inmode_templated": si, "det_inmode": si > best_pr["threshold"]})
        t = table.setdefault(r["condition"], {}).setdefault(str(r["label"]), {}).setdefault("refused" if ref else "not_refused", {"n": 0, "det_january": 0, "det_inmode": 0})
        t["n"] += 1; t["det_january"] += int(sj > jan["threshold"]); t["det_inmode"] += int(si > best_pr["threshold"])
    for c in table:
        for lab in table[c]:
            for st in table[c][lab]:
                t = table[c][lab][st]; t["detection_january"] = t["det_january"] / t["n"]; t["detection_inmode"] = t["det_inmode"] / t["n"]
    out = {"provenance": provenance(cfg, {"part": "p4" if cfg["feature"] == "residual" else "p4b", "feature": cfg["feature"], "model_info": info, "generation_cache": gen_cache.name}),
           "feature": cfg["feature"], "indexing_convention": CONVENTIONS[cfg["feature"]],
           "sample": {"n": len(sample), "per_cell": p4["per_cell"], "seed": p4["seed"]},
           "probes": {"january": {"block": jan["layer_index"], "threshold": jan["threshold"], "mode": "raw"},
                      "inmode_templated": {"block": best_b, "threshold": best_pr["threshold"], "plain_auc_full_set": best_auc, "mode": "templated"}},
           "table": table, "items": items}
    dump_json(out, od / f"p4_refusal{sfx(cfg)}.json")
    for c in cfg["conditions"]:
        for lab in ("1", "0"):
            for st, t in table.get(c, {}).get(lab, {}).items():
                print(f"  {c:<9} label={lab} {st:<12} n={t['n']:3d} det_january={t['detection_january']:.2f} det_inmode={t['detection_inmode']:.2f}")
    print("wrote", od / f"p4_refusal{sfx(cfg)}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
