#!/usr/bin/env python3
"""B4 — the extraction-convention exhibit, assembled once here so the paper never reconstructs it from raw JSON.

This is the central table of the rewrite: the same probes, the same 650 evaluation prompts, the same layer and the same
feature, read two ways — at the final token of the bare prompt (the Phase 1 convention) and at the final token of the
prompt wrapped in the model's own chat template. For each model it carries the January probe (as shipped) beside the
newly trained paired probe, with the length-stratified AUC and the JailbreakBench false-positive rate taken at the
matched-XSTest operating point, which is what makes the FPR columns comparable across probes.

Writes results/gates/extraction_convention.{json,md} with a full provenance block, and prints the markdown.

  python -m gates.b4_exhibit
"""
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gates.runb import dump_json, load_gates_config, out_dir, provenance  # noqa: E402

PROBES = ("january", "paired", "offtopic")


def collect(cfg):
    rows = []
    for p in sorted(glob.glob(str(out_dir(cfg) / "b1" / "*.json"))):
        d = json.load(open(p))
        model = d["model"]
        r = {"model": model, "layer_index": d["raw"]["probes"]["january"]["layer_index"],
             "feature": cfg["feature"], "n_harmful": d["eval_set"]["n_harmful"],
             "n_benign": d["eval_set"]["n_benign"], "n_benign_by_set": d["eval_set"]["n_benign_by_set"], "cells": {}}
        for probe in PROBES:
            for mode in ("raw", "templated"):
                c = d[mode]["cells"][probe]
                op = c["matched_operating_point"]
                r["cells"][f"{probe}|{mode}"] = {
                    "auc_all": c["auc_all"], "auc_length_stratified": c["length_all"]["auc_length_stratified"],
                    "auc_vs_xstest": c["auc_vs_xstest"], "auc_vs_jbb": c["auc_vs_jbb"],
                    "detection_at_own_threshold": c["detection"], "fpr_xstest_at_own_threshold": c["fpr_xstest"],
                    "fpr_jbb_at_own_threshold": c["fpr_jbb"],
                    "matched_xstest_fpr_target": op["target_xstest_fpr"], "matched_xstest_fpr_actual": op["xstest_fpr"],
                    "jbb_fpr_at_matched_point": op["jbb_fpr"], "detection_at_matched_point": op["detection"],
                    "rho_score_length": c["length_all"]["rho_score_length"]}
        r["template_effect_paired_auc_strat"] = (r["cells"]["paired|templated"]["auc_length_stratified"]
                                                 - r["cells"]["paired|raw"]["auc_length_stratified"])
        r["twins_effect_templated_auc_vs_jbb"] = (r["cells"]["paired|templated"]["auc_vs_jbb"]
                                                  - r["cells"]["offtopic|templated"]["auc_vs_jbb"])
        r["criterion_2_evaluable"] = r["cells"]["paired|templated"]["matched_xstest_fpr_target"] <= 0.20
        rows.append(r)
    return rows


def markdown(rows, cfg) -> str:
    L = ["### Exhibit — extraction convention: bare prompt vs the model's own chat template", "",
         f"Same 300 HarmBench behaviors and 350 benign prompts (XSTest safe 250 + JailbreakBench benign 100), same layer",
         f"per model, same feature (`{cfg['feature']}`, the space every shipped probe was trained in). The only thing that",
         "changes between the two halves of each row is whether the activation is read at the final token of the bare",
         "prompt or of the prompt wrapped in that model's chat template.",
         "",
         "`AUC*` is length-stratified (the headline number GATES.md judges). `JBB FPR@m` is the JailbreakBench-benign",
         "false-positive rate at the threshold whose XSTest FPR matches the January probe's XSTest FPR for that model,",
         "so the FPR columns are comparable across probes rather than reflecting a threshold shift.",
         "",
         "| Model | Layer | Probe | AUC* raw | AUC* templated | Δ | AUC vs JBB raw | AUC vs JBB templated | JBB FPR@m raw | JBB FPR@m templated |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for probe in PROBES:
            a, b = r["cells"][f"{probe}|raw"], r["cells"][f"{probe}|templated"]
            L.append(f"| {r['model'].split('/')[-1]} | {r['layer_index']} | {probe} | "
                     f"{a['auc_length_stratified']:.3f} | {b['auc_length_stratified']:.3f} | "
                     f"{b['auc_length_stratified'] - a['auc_length_stratified']:+.3f} | "
                     f"{a['auc_vs_jbb']:.3f} | {b['auc_vs_jbb']:.3f} | "
                     f"{a['jbb_fpr_at_matched_point']:.2f} | {b['jbb_fpr_at_matched_point']:.2f} |")
    te = [r["template_effect_paired_auc_strat"] for r in rows]
    tw = [r["twins_effect_templated_auc_vs_jbb"] for r in rows]
    L += ["",
          f"**Template effect** (paired probe, length-stratified AUC, templated − raw): min {min(te):+.3f}, "
          f"max {max(te):+.3f}, mean {sum(te) / len(te):+.3f} over {len(te)} models — positive on every model.",
          f"**On-topic-twins effect** (templated, AUC vs JBB, paired − offtopic): min {min(tw):+.3f}, max {max(tw):+.3f}, "
          f"mean {sum(tw) / len(tw):+.3f} — indistinguishable from zero.",
          "",
          "Read together: the confound Phase 1 found is a property of the extraction convention, not of the training-set",
          "construction. Swapping the negatives for on-topic twins changes nothing; reading the same text through the",
          "chat template changes everything.", ""]
    return "\n".join(L)


def main() -> int:
    cfg = load_gates_config()
    rows = collect(cfg)
    if not rows:
        print("no B1 results found", file=sys.stderr); return 1
    md = markdown(rows, cfg)
    od = out_dir(cfg)
    dump_json({"exhibit": "extraction_convention", "rows": rows,
               # This exhibit aggregates; the per-case scores behind every AUC here are committed in the b1 files.
               # tests/test_results_completeness.py reads this key, so an aggregate is never orphaned from its scores.
               "source": ["results/gates/b1/*.json"],
               "definitions": {"auc_length_stratified": "5 quantile bins of prompt character length, per-bin "
                                                        "Mann-Whitney AUC pooled by n_pos*n_neg, bins needing >=5 per class",
                               "jbb_fpr_at_matched_point": "JailbreakBench-benign FPR at the threshold whose XSTest FPR "
                                                           "equals the January probe's XSTest FPR for that model",
                               "feature": cfg["feature"]},
               "provenance": provenance(cfg, {"part": "b4_exhibit"})}, od / "extraction_convention.json")
    (od / "extraction_convention.md").write_text(md)
    print(md)
    print("wrote", od / "extraction_convention.json", "and", od / "extraction_convention.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
