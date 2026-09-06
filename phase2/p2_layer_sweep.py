#!/usr/bin/env python3
"""P2 — layer sweep x template matrix on gemma-2-9b-it.

For every record of the eval set (1,200 AILuminate harmful + 350 benign) x 4 conditions x wrappers, in both modes
(raw / templated), one forward pass captures the post-block last-token activation of blocks 14..35. Activations are cached
as float16 .npz under results/phase2/activations/ (gitignored). Then, per {block x mode x condition x wrapper}:

  * probe "january":  the Phase 1 January AF direction (block 21, raw-trained) applied to every block's activation
                      (cosine score; threshold from its metadata) — note a block-21 direction on other blocks is only a
                      sanity/continuity measure, the principled per-block measure is the next probe;
  * probe "inmode":   a direction retrained per block *and per mode* from the January 20+20 training prompts
                      (raw mode -> trained on raw text; templated mode -> trained on templated text), January threshold rule.

Outputs: results/phase2/p2_summary.json (AUC harmful-vs-benign per cell + distribution stats + detection/FPR at threshold),
results/phase2/p2_scores.npz (per-case scores for every cell, with the record ids), heatmaps
results/phase2/p2_heatmap_<mode>_<probe>.png, and a backend-consistency check against Phase 1's vLLM scores at block 21.

  python -m phase2.p2_layer_sweep [--dry-run] [--skip-extract]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase2.common import (BLOCKS, REPO, build_eval_set, dump_json, format_for_mode, january_training_set, load_phase2_config,  # noqa: E402
                           out_dir, provenance)
from aase_eval import resolve_path  # noqa: E402


def sfx(cfg) -> str:
    """File-name suffix for the activation feature: '' for the residual stream, '_mlp' for the MLP-branch output (P2b/P4b)."""
    return "" if cfg.get("feature", "residual") == "residual" else "_mlp"


def act_path(cfg, mode, wrapper, which="eval"):
    d = resolve_path(cfg["activations_dir"]); d.mkdir(parents=True, exist_ok=True)
    return d / f"{which}_{mode}_{wrapper}{sfx(cfg)}.npz"


CONVENTIONS = {
    "residual": "block k = 0-based decoder layer; activation = post-block residual stream = HF hidden_states[k+1] = HF/December forward-hook output on layers[k]; last token of the mode-formatted text",
    "mlp_out": "block k = 0-based decoder layer; activation = block k's MLP-branch output before the residual add (Gemma-2: post_feedforward_layernorm output) = vLLM decoder-layer output[0] = what the Phase 1 / January vLLM hook captured; last token of the mode-formatted text",
}


def extract_all(cfg, data, model, tok):
    from phase2.extract import extract_last_token
    blocks = cfg["blocks"]
    for wrapper in cfg["wrappers"]:
        recs = [r for r in data["records"] if r["wrapper"] == wrapper]
        if not recs:
            continue
        for mode in cfg["modes"]:
            p = act_path(cfg, mode, wrapper)
            if p.exists():
                print(f"  cached {p.name}"); continue
            texts = [format_for_mode(r["text"], mode) for r in recs]
            acts = extract_last_token(model, tok, texts, blocks, cfg.get("batch_size", 16), cfg.get("max_length", 1024), progress=f"{mode}/{wrapper}", feature=cfg.get("feature", "residual"))
            np.savez(p, acts=acts, ids=np.array([r["id"] for r in recs]), blocks=np.array(blocks))
            print(f"  wrote {p} {acts.shape}")
    jan = january_training_set()
    for mode in cfg["modes"]:
        p = act_path(cfg, mode, "january", "train")
        if p.exists():
            continue
        texts = [format_for_mode(t, mode) for t in jan["harmful"] + jan["benign"]]
        acts = extract_last_token(model, tok, texts, blocks, cfg.get("batch_size", 16), cfg.get("max_length", 1024), progress=f"train-january/{mode}", feature=cfg.get("feature", "residual"))
        np.savez(p, acts=acts, labels=np.array([1] * len(jan["harmful"]) + [0] * len(jan["benign"])), blocks=np.array(blocks))


def analyse(cfg, data):
    from phase2.extract import auc, dist_stats, load_january_probe, score, train_direction
    blocks = cfg["blocks"]; bi = {b: i for i, b in enumerate(blocks)}
    jan_probe = load_january_probe(REPO, cfg["p2"]["january_probe_model"])
    by_id = {r["id"]: r for r in data["records"]}
    summary = {"blocks": blocks, "modes": cfg["modes"], "conditions": cfg["conditions"], "wrappers": cfg["wrappers"],
               "feature": cfg.get("feature", "residual"), "indexing_convention": CONVENTIONS[cfg.get("feature", "residual")],
               "january_probe": {"path": jan_probe["path"], "layer_index": jan_probe["layer_index"], "threshold": jan_probe["threshold"], "meta": jan_probe["meta"]},
               "inmode_probes": {}, "cells": {}}
    scores_out = {}
    # in-mode probes per block/mode from the January prompts
    probes = {"january": {m: {b: jan_probe for b in blocks} for m in cfg["modes"]}, "inmode": {}}
    for mode in cfg["modes"]:
        z = np.load(act_path(cfg, mode, "january", "train"))
        acts, labels = z["acts"], z["labels"]
        probes["inmode"][mode] = {}
        for b in blocks:
            pr = train_direction(acts[labels == 1, bi[b], :], acts[labels == 0, bi[b], :], "january")
            probes["inmode"][mode][b] = pr
            summary["inmode_probes"][f"{mode}|{b}"] = {k: v for k, v in pr.items() if k != "direction"}
    for wrapper in cfg["wrappers"]:
        for mode in cfg["modes"]:
            p = act_path(cfg, mode, wrapper)
            if not p.exists():
                continue
            z = np.load(p); acts, ids = z["acts"], list(z["ids"])
            labels = np.array([by_id[i]["label"] for i in ids]); conds = np.array([by_id[i]["condition"] for i in ids])
            bsets = np.array([by_id[i].get("benign_set", "") for i in ids])
            for pname in ("january", "inmode"):
                for b in blocks:
                    pr = probes[pname][mode][b]
                    s = score(acts[:, bi[b], :], pr["direction"], pr["rule"])
                    scores_out[f"{pname}|{mode}|{wrapper}|{b}"] = s.astype(np.float32)
                    for cond in cfg["conditions"]:
                        m = conds == cond
                        if not m.any():
                            continue
                        h, n = s[m & (labels == 1)], s[m & (labels == 0)]
                        t = pr["threshold"]
                        cell = {"auc": auc(h, n), "auc_vs_xstest": auc(h, s[m & (labels == 0) & (bsets == "xstest_safe")]),
                                "auc_vs_jbb": auc(h, s[m & (labels == 0) & (bsets == "jbb_benign")]),
                                "threshold": t, "detection": float((h > t).mean()), "fpr": float((n > t).mean()),
                                "harmful": dist_stats(h), "benign": dist_stats(n)}
                        summary["cells"][f"{pname}|{mode}|{wrapper}|{b}|{cond}"] = cell
            scores_out[f"ids|{mode}|{wrapper}"] = np.array(ids)
    # max AUC over blocks per (probe, mode, wrapper, condition)
    summary["max_auc_over_blocks"] = {}
    for key in list(summary["cells"]):
        pname, mode, wrapper, b, cond = key.split("|")
        k2 = f"{pname}|{mode}|{wrapper}|{cond}"
        v = summary["cells"][key]["auc"]
        cur = summary["max_auc_over_blocks"].get(k2)
        if cur is None or v > cur["auc"]:
            summary["max_auc_over_blocks"][k2] = {"auc": v, "block": int(b), "detection_at_thr": summary["cells"][key]["detection"], "fpr_at_thr": summary["cells"][key]["fpr"]}
    return summary, scores_out, probes


def backend_check(cfg, scores_out, data):
    """January probe, raw mode, phase1 wrapper, block 21 vs Phase 1's vLLM per-case scores on the same prompt ids."""
    p1 = REPO / "results/phase1/ailuminate/google_gemma_2_9b_it.json"
    if not p1.exists():
        return {"available": False}
    sec = json.load(open(p1))["ailuminate"]
    v = {}
    for r in sec["per_case"]:
        rid, enc = r["id"].split("|")
        v[(rid, enc)] = r["score"]
    ids = scores_out["ids|raw|phase1"]; s = scores_out["january|raw|phase1|21"]
    a, b = [], []
    for i, rid in enumerate(ids):
        src, enc, _ = rid.split("|")
        if (src, enc) in v:
            a.append(v[(src, enc)]); b.append(float(s[i]))
    a, b = np.array(a), np.array(b)
    return {"available": True, "n_matched": int(len(a)), "pearson_r": float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else None,
            "mean_abs_diff": float(np.abs(a - b).mean()), "phase1_mean": float(a.mean()), "phase2_mean": float(b.mean()),
            "feature": cfg.get("feature", "residual"),
            "note": "Phase 1 = vLLM 0.13 forward hook on layers[21] taking output[0] (= MLP-branch output, bf16); Phase 2 feature 'residual' = HF hidden_states[22] (a different tensor), 'mlp_out' = HF hook on the same MLP-branch module (the like-for-like comparison); same probe, raw text"}


def heatmaps(cfg, summary, od):
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    blocks = cfg["blocks"]; conds = cfg["conditions"]
    for pname in ("january", "inmode"):
        for mode in cfg["modes"]:
            M = np.full((len(conds), len(blocks)), np.nan)
            for i, c in enumerate(conds):
                for j, b in enumerate(blocks):
                    cell = summary["cells"].get(f"{pname}|{mode}|phase1|{b}|{c}")
                    if cell:
                        M[i, j] = cell["auc"]
            fig, ax = plt.subplots(figsize=(12, 3.6))
            im = ax.imshow(M, vmin=0.3, vmax=1.0, cmap="RdYlGn", aspect="auto")
            ax.set_xticks(range(len(blocks))); ax.set_xticklabels(blocks); ax.set_yticks(range(len(conds))); ax.set_yticklabels(conds)
            for i in range(len(conds)):
                for j in range(len(blocks)):
                    if not np.isnan(M[i, j]):
                        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
            feat = cfg.get("feature", "residual")
            ax.set_xlabel(f"block (0-based); feature = {feat}"); ax.set_title(f"AUC harmful(1200) vs benign(350) — gemma-2-9b-it, mode={mode}, probe={pname}, wrapper=phase1, feature={feat}")
            plt.colorbar(im, ax=ax, fraction=0.02); plt.tight_layout()
            fig.savefig(od / f"p2_heatmap{sfx(cfg)}_{mode}_{pname}.png", dpi=140); plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-extract", action="store_true", help="analyse cached activations only")
    ap.add_argument("--feature", default=None, choices=["residual", "mlp_out"], help="activation feature (default: config `feature`, else residual)")
    a = ap.parse_args(argv)
    cfg = load_phase2_config(a.config)
    if a.feature:
        cfg["feature"] = a.feature
    cfg.setdefault("feature", "residual")
    print(f"P2 feature: {cfg['feature']} — {CONVENTIONS[cfg['feature']]}")
    data = build_eval_set(cfg)
    jan = january_training_set()
    s = data["summary"]
    print(f"P2 eval set: {s['n_records']} records = ({s['n_harmful_prompts']} harmful + {s['n_benign_prompts']} benign) x conditions {s['conditions']} x wrappers {s['n_by_wrapper']}; december wrapper differs: {s['december_wrapper_differs']}")
    print(f"P2 modes {cfg['modes']}, blocks {cfg['blocks'][0]}..{cfg['blocks'][-1]} ({len(cfg['blocks'])}), forward passes = {s['n_records']} x {len(cfg['modes'])} = {s['n_records'] * len(cfg['modes'])} (+ {2 * (len(jan['harmful']) + len(jan['benign']))} for the January training prompts)")
    for mode in cfg["modes"]:
        r = data["records"][0]
        print(f"  sample [{mode}] {r['id']}: {format_for_mode(r['text'], mode)[:150]!r}")
    od = out_dir(cfg)
    if a.dry_run:
        print("output ->", od / f"p2_summary{sfx(cfg)}.json", od / f"p2_scores{sfx(cfg)}.npz", od / f"p2_heatmap{sfx(cfg)}_<mode>_<probe>.png; activations cached in", cfg["activations_dir"])
        return 0
    if not a.skip_extract:
        from phase2.extract import chat_template_check, load_model
        model, tok, info = load_model(cfg["model"], cfg.get("dtype", "bfloat16"))
        tc = chat_template_check(tok)
        print("chat template check:", {k: v for k, v in tc.items() if k != "tokenizer_template"})
        extract_all(cfg, data, model, tok)
        del model
        import torch; torch.cuda.empty_cache()
    else:
        info, tc = {"note": "skip-extract"}, {}
    summary, scores_out, probes = analyse(cfg, data)
    summary["backend_check_vs_phase1_block21"] = backend_check(cfg, scores_out, data)
    summary["provenance"] = provenance(cfg, {"part": "p2" if cfg["feature"] == "residual" else "p2b", "feature": cfg["feature"], "model_info": info, "chat_template_check": tc, "january_training_set": jan["provenance"]})
    summary["eval_set"] = s
    dump_json(summary, od / f"p2_summary{sfx(cfg)}.json")
    np.savez_compressed(od / f"p2_scores{sfx(cfg)}.npz", **scores_out)
    heatmaps(cfg, summary, od)
    print("max AUC over blocks (phase1 wrapper):")
    for k, v in sorted(summary["max_auc_over_blocks"].items()):
        if "|phase1|" in k:
            print(f"  {k:<32} auc {v['auc']:.3f} @ block {v['block']}")
    print("backend check:", summary["backend_check_vs_phase1_block21"])
    print("wrote", od / f"p2_summary{sfx(cfg)}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
