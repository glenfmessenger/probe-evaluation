#!/usr/bin/env python3
"""ARM 1 (B1) — paired contrastive AF: does the paper's own paired rule recover the harm signal that standard
contrastive training missed?

Per model x extraction mode, three directions are scored on the identical Phase 1 AF matrix:

  january   — the shipped Phase 1 AF probe, unchanged (the row every number is compared against);
  paired    — v = mean_i(h_i^+ - h_i^-) over the 40 signed-off pairs (paper eq. 1);
  standard  — v = mean(h^+) - mean(h^-) over the same 80 texts (paper eq. 2, the Phase 1 rule);
  offtopic  — v = mean(h^+) - mean(h^-) with the SAME 40 harmful members but the 20 off-topic benign prompts the
              January AF probe was actually trained on, in place of the on-topic twins.

FINDING, recorded here because it changes what Arm 1 can claim: for complete, aligned, equal-sized pairs the paper's
paired rule and its standard rule are the SAME ESTIMATOR — mean_i(x_i - y_i) is identically mean(x) - mean(y). The
`paired` and `standard` rows below therefore agree to the last bit by construction, on every model and mode, and no
experiment can separate them. The paper's claimed methodological contribution ("paired contrastive training isolates
safety-relevant signals ... shared features cancel out in the mean difference computation") is a property of choosing
*paired data*, not of the estimator; with balanced classes the estimator is the ordinary difference of class means.

That is why `offtopic` exists. It holds the harmful side and the rule fixed and varies only the negative set, so it
isolates the effect the paper attributes to pairing: on-topic benign twins versus the off-topic benign prompts the
shipped probes were trained on. `paired`/`standard` vs `offtopic` is the comparison that carries the Arm 1 hypothesis;
`paired` vs `standard` is reported as the tautology it is.
The layer is the model's own AF probe `layer_index`, the feature is `mlp_out` (the space the shipped probes live in).

Both default controls are applied: every AUC is reported raw and length-stratified, and a seeded subsample of prompts
is generated so detection can be conditioned on whether the model itself refused. The Arm 1 threshold also needs a
matched operating point: the JBB-benign FPR is read at the threshold whose XSTest FPR equals the January probe's XSTest
FPR for that model, so criterion 2 cannot be met by simply moving the threshold.

  python -m gates.b1_paired_af [--dry-run] [--models M ...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aase_eval.config import load_config, resolve_path  # noqa: E402
from aase_eval.harmbench import load_harmbench_eval_set  # noqa: E402
from gates.arm1_pairs import PAIRS  # noqa: E402
from gates.runb import (act_dir, check_scores, dump_json, format_mode, fpr_at_threshold, length_report,  # noqa: E402
                        load_gates_config, load_probe, matched_operating_point, out_dir, provenance, refusal_table,
                        safe, train_rule)
from phase2.common import january_training_set  # noqa: E402
from phase2.extract import auc, score  # noqa: E402

PROBE_NAMES = ("january", "paired", "standard", "offtopic")
OFFTOPIC_BENIGN = january_training_set()["benign"]     # the 20 benign prompts the shipped AF probes were trained on


def eval_set():
    base = load_config()
    d = load_harmbench_eval_set(base["harmbench"], base["datasets"])
    recs = ([{"text": r["prompt"], "label": 1, "set": "harmbench", "category": r.get("category"), "id": f"hb_{i}"}
             for i, r in enumerate(d["harmful"])] +
            [{"text": r["prompt"], "label": 0, "set": r["benign_set"], "category": r.get("category"), "id": f"bn_{i}"}
             for i, r in enumerate(d["benign"])])
    return recs, d["summary"]


def run_model(cfg, model_name, recs, results):
    from phase2.extract import extract_last_token, load_model
    b1 = cfg["b1"]
    probe = load_probe(cfg, "af", model_name)
    layer = probe["layer_index"]
    model, tok, info = load_model(model_name, cfg.get("dtype", "bfloat16"))
    print(f"  {model_name}: AF layer_index {layer}, {info}", flush=True)
    pos_txt = [p[1] for p in PAIRS]; neg_txt = [p[2] for p in PAIRS]
    per_mode, probes_by_mode = {}, {}
    for mode in cfg["modes"]:
        tr_pos = extract_last_token(model, tok, [format_mode(t, mode, model_name, tok) for t in pos_txt], [layer],
                                    cfg["batch_size"], cfg["max_length"], feature=cfg["feature"])[:, 0, :]
        tr_neg = extract_last_token(model, tok, [format_mode(t, mode, model_name, tok) for t in neg_txt], [layer],
                                    cfg["batch_size"], cfg["max_length"], feature=cfg["feature"])[:, 0, :]
        texts = [format_mode(r["text"], mode, model_name, tok) for r in recs]
        acts = extract_last_token(model, tok, texts, [layer], cfg["batch_size"], cfg["max_length"],
                                  progress=f"{safe(model_name)}/{mode}", feature=cfg["feature"])[:, 0, :]
        np.savez_compressed(act_dir(cfg) / f"b1_{safe(model_name)}_{mode}.npz", acts=acts,
                            ids=np.array([r["id"] for r in recs]), train_pos=tr_pos, train_neg=tr_neg, layer=layer)
        tr_off = extract_last_token(model, tok, [format_mode(t, mode, model_name, tok) for t in OFFTOPIC_BENIGN], [layer],
                                    cfg["batch_size"], cfg["max_length"], feature=cfg["feature"])[:, 0, :]
        probes = {"january": {**probe, "source": "shipped Phase 1 AF probe"}}
        for rule in b1["rules"]:
            probes[rule] = train_rule(tr_pos, tr_neg, rule, pairs=True)
        probes["offtopic"] = train_rule(tr_pos, tr_off, "standard")
        probes["offtopic"]["negatives"] = "January AF training benign prompts (off-topic)"
        d_pair = probes["paired"]["direction"]; d_std = probes["standard"]["direction"]
        probes["paired"]["identical_to_standard"] = bool(np.allclose(d_pair, d_std, atol=1e-5))
        y = np.array([r["label"] for r in recs]); L = np.array([len(r["text"]) for r in recs])
        sets = np.array([r["set"] for r in recs])
        cells = {}
        jan_xstest_fpr = None
        for pname in PROBE_NAMES:
            pr = probes[pname]
            s = score(acts, pr["direction"], "january")
            check_scores(f"{model_name}|{mode}|{pname}", s)
            xs, jb, hp = s[sets == "xstest_safe"], s[sets == "jbb_benign"], s[y == 1]
            thr = pr["threshold"]
            if pname == "january":
                jan_xstest_fpr = fpr_at_threshold(xs, thr)
            cell = {"threshold": thr, "train_auc": pr.get("train_auc"), "train_separation": pr.get("train_separation"),
                    "detection": fpr_at_threshold(hp, thr), "fpr_all": fpr_at_threshold(s[y == 0], thr),
                    "fpr_xstest": fpr_at_threshold(xs, thr), "fpr_jbb": fpr_at_threshold(jb, thr),
                    "auc_all": auc(hp, s[y == 0]), "auc_vs_xstest": auc(hp, xs), "auc_vs_jbb": auc(hp, jb),
                    "length_all": length_report(s, y, L),
                    "length_vs_jbb": length_report(np.concatenate([hp, jb]), np.concatenate([np.ones(len(hp)), np.zeros(len(jb))]),
                                                   np.concatenate([L[y == 1], L[sets == "jbb_benign"]])),
                    "matched_operating_point": matched_operating_point(hp, xs, jb, jan_xstest_fpr),
                    "scores": {"harmful": list(map(float, hp)), "xstest": list(map(float, xs)), "jbb": list(map(float, jb))}}
            cells[pname] = cell
            print(f"    [{mode}] {pname:9s} AUC {cell['auc_all']:.3f} (strat {cell['length_all']['auc_length_stratified']:.3f}) "
                  f"vsJBB {cell['auc_vs_jbb']:.3f} | det {cell['detection']:.2f} FPR x/j {cell['fpr_xstest']:.2f}/{cell['fpr_jbb']:.2f} "
                  f"| matched-XSTest {cell['matched_operating_point']['xstest_fpr']:.2f} -> JBB {cell['matched_operating_point']['jbb_fpr']:.2f}",
                  flush=True)
        probes_by_mode[mode] = probes
        per_mode[mode] = {"probes": {k: {kk: vv for kk, vv in v.items() if kk != "direction"} for k, v in probes.items()},
                          "cells": cells}
    per_mode["refusal"] = refusal_leg(cfg, model, tok, recs, probes_by_mode[cfg["modes"][0]], model_name)
    del model
    import torch; torch.cuda.empty_cache()
    results[model_name] = per_mode
    return per_mode


def refusal_leg(cfg, model, tok, recs, probes, model_name):
    """Generate on a seeded per-set subsample and condition each probe's flag on whether the model refused.

    Scores come from the cached activations of the first mode, so the refusal leg costs one generation pass, not a
    second extraction; the mode used is recorded in the output."""
    import random
    from phase2.common import is_refusal
    from phase2.extract import generate
    b1 = cfg["b1"]; n = b1["refusal_sample_per_set"]
    rng = random.Random(b1["seed"])
    sample = []
    for st in ("harmbench", "xstest_safe", "jbb_benign"):
        rows = [r for r in recs if r["set"] == st]
        sample += rng.sample(rows, min(n, len(rows)))
    outs = generate(model, tok, [r["text"] for r in sample], max_new_tokens=b1["max_new_tokens"],
                    batch_size=max(1, cfg["batch_size"] // 2))
    idx = {r["id"]: i for i, r in enumerate(recs)}
    z = np.load(act_dir(cfg) / f"b1_{safe(model_name)}_{cfg['modes'][0]}.npz")
    acts = z["acts"]
    tables = {}
    for pname in PROBE_NAMES:
        pr = probes[pname]
        items = []
        for r, o in zip(sample, outs):
            s = float(score(acts[idx[r["id"]]][None], pr["direction"], "january")[0])
            items.append({"set": r["set"], "label": r["label"], "refusal": is_refusal(o), "flagged": s > pr["threshold"],
                          "score": s})
        tables[pname] = refusal_table(items)
    return {"n": len(sample), "mode": cfg["modes"][0], "max_new_tokens": b1["max_new_tokens"],
            "refusal_rate_by_set": {st: float(np.mean([is_refusal(o) for r, o in zip(sample, outs) if r["set"] == st]))
                                    for st in ("harmbench", "xstest_safe", "jbb_benign")},
            "tables": tables, "samples": [{"set": r["set"], "text": r["text"][:160], "output": o[:200]}
                                          for r, o in list(zip(sample, outs))[:6]]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--models", nargs="*", default=None)
    a = ap.parse_args(argv)
    cfg = load_gates_config(a.config)
    models = a.models or cfg["models"]
    recs, summ = eval_set()
    print(f"B1 eval set: {summ['n_harmful']} HarmBench + {summ['n_benign']} benign {summ['n_benign_by_set']}; "
          f"training {len(PAIRS)} pairs = {2 * len(PAIRS)} texts; modes {cfg['modes']}; feature {cfg['feature']}")
    print(f"B1 probes: {list(PROBE_NAMES)}; offtopic negatives = {len(OFFTOPIC_BENIGN)} January AF benign prompts")
    print("B1 note: paired and standard are the same estimator for complete aligned pairs "
          "(mean of within-pair differences == difference of class means); offtopic is the ablation that varies the "
          "negative set with the harmful side and the rule held fixed.")
    print(f"B1 models ({len(models)}): {models}")
    od = out_dir(cfg) / "b1"
    if a.dry_run:
        print("output ->", od / "<model>.json", "and", od / "summary.json")
        return 0
    results = {}
    for m in models:
        p = od / f"{safe(m)}.json"
        per_mode = run_model(cfg, m, recs, results)
        dump_json({"model": m, "eval_set": summ, "provenance": provenance(cfg, {"part": "b1", "model": m}),
                   **per_mode}, p)
        print("  wrote", p, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
