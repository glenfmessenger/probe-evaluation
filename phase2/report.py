#!/usr/bin/env python3
"""Assemble the numeric tables for PHASE2_REPORT.md from results/phase2/*.json (CPU). Prints markdown.

  python -m phase2.report > /tmp/phase2_tables.md
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase2.common import load_phase2_config, out_dir  # noqa: E402


def main(argv=None) -> int:
    cfg = load_phase2_config(None); od = out_dir(cfg); L = []
    # P1
    p = od / "p1_decode.json"
    if p.exists():
        d = json.load(open(p)); L += ["### P1 — behavioral decode test", "", "| Encoding | n | exact | partial | fail | refusals in the decode reply |", "|---|---|---|---|---|---|"]
        for enc, s in d["summary"].items():
            L.append(f"| {enc} | {s['n']} | {s['exact']} | {s['partial']} | {s['fail']} | {s['refusals']} |")
        L.append("")
    # P2 (residual-stream feature) and P2b (MLP-branch feature = the Phase 1 / January probe space)
    for fname, label in (("p2_summary.json", "P2 — feature `residual` (post-block residual stream)"), ("p2_summary_mlp.json", "P2b — feature `mlp_out` (block MLP-branch output = vLLM output[0] = Phase 1/January probe space)")):
        p = od / fname
        if not p.exists():
            continue
        d = json.load(open(p)); conds = d["conditions"]
        L += [f"### {label}: max AUC over blocks 14–35 (harmful 1,200 vs benign 350), phase1 wrapper", "",
              "| Probe | Mode | " + " | ".join(conds) + " |", "|---|---|" + "---|" * len(conds)]
        for pname in ("january", "inmode"):
            for mode in d["modes"]:
                row = []
                for c in conds:
                    v = d["max_auc_over_blocks"].get(f"{pname}|{mode}|phase1|{c}")
                    row.append(f"{v['auc']:.3f} @ b{v['block']}" if v else "—")
                L.append(f"| {pname} | {mode} | " + " | ".join(row) + " |")
        L += ["", "Block 21 (the January probe's block) and block 27 (December's), phase1 wrapper:", "",
              "| Probe | Mode | Block | " + " | ".join(f"{c}: AUC / det / FPR" for c in conds) + " |", "|---|---|---|" + "---|" * len(conds)]
        for pname in ("january", "inmode"):
            for mode in d["modes"]:
                for b in (21, 27):
                    row = []
                    for c in conds:
                        cell = d["cells"].get(f"{pname}|{mode}|phase1|{b}|{c}")
                        row.append(f"{cell['auc']:.3f} / {cell['detection']:.2f} / {cell['fpr']:.2f}" if cell else "—")
                    L.append(f"| {pname} | {mode} | {b} | " + " | ".join(row) + " |")
        if "december" in d["wrappers"]:
            L += ["", "December wrappers (where they differ from phase1), in-mode probe, max AUC over blocks:", ""]
            for mode in d["modes"]:
                for c in conds:
                    v = d["max_auc_over_blocks"].get(f"inmode|{mode}|december|{c}")
                    if v:
                        L.append(f"- {mode} / {c}: {v['auc']:.3f} @ block {v['block']}")
        bc = d.get("backend_check_vs_phase1_block21", {})
        L += ["", f"Backend consistency (January probe, raw, block 21, feature `{d.get('feature', 'residual')}`) vs Phase 1 vLLM scores: n={bc.get('n_matched')} r={bc.get('pearson_r')} mean|Δ|={bc.get('mean_abs_diff')}", ""]
    # P2c confounds
    p = od / "p2_confounds.json"
    if p.exists():
        from phase2.p2_confounds import markdown as conf_md
        L += ["### P2c — length control and encoded-vs-plain coupling", "", conf_md(json.load(open(p)))]
    # P3
    p = od / "p3_circularity.json"
    if p.exists():
        d = json.load(open(p)); conds = cfg["conditions"]
        for title, key in (("at the December block (27)", "at_december_block"), ("at the best encoded-detection block", "at_best_block")):
            L += [f"### P3 — December-recipe probe {title}", "", "| Recipe | Mode | Block | " + " | ".join(f"{c}: det / FPR-benign / AUC" for c in conds) + " |", "|---|---|---|" + "---|" * len(conds)]
            for k, v in d[key].items():
                r, m = k.split("|")
                L.append(f"| {r} | {m} | {v['block']} | " + " | ".join(f"{v[c]['detection']:.2f} / {v[c]['fpr_benign']:.2f} / {v[c]['auc']:.2f}" for c in conds) + " |")
            L.append("")
    # P4 / P4b
    for p in (od / "p4_refusal.json", od / "p4_refusal_mlp.json"):
        if not p.exists():
            continue
        d = json.load(open(p))
        L += [f"### {'P4b' if d.get('feature') == 'mlp_out' else 'P4'} — probe detection conditioned on the model's own refusal (n={d['sample']['n']}; feature `{d.get('feature', 'residual')}`; January probe b21 raw; in-mode templated probe b{d['probes']['inmode_templated']['block']})", "",
              "| Condition | Label | Refused? | n | det (January) | det (in-mode templated) |", "|---|---|---|---|---|---|"]
        for c in cfg["conditions"]:
            for lab in ("1", "0"):
                for st, t in d["table"].get(c, {}).get(lab, {}).items():
                    L.append(f"| {c} | {'harmful' if lab == '1' else 'benign'} | {st} | {t['n']} | {t['detection_january']:.2f} | {t['detection_inmode']:.2f} |")
        L.append("")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
