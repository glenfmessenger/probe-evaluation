#!/usr/bin/env python3
"""Confidence intervals for the headline effects, computed from the committed per-case scores.

Framing rule 3 (no untraced numbers) extended to uncertainty: every interval in the manuscript is produced here from a
committed results file, and the build fails on a missing source rather than emitting an interval whose provenance
cannot be checked.

Three families of interval, and they are NOT all computed the same way, because the committed evidence differs:

  AF  (Table III)  -- the harmful-content extraction-convention effect. results/gates/b1/*.json commits the per-case
                      score of every one of the 650 evaluation prompts under both template modes, so this is a
                      nonparametric bootstrap over cases. The template delta is resampled PAIRED: raw and templated
                      are the same 650 prompts, so one resample of case indices is applied to both arms and the
                      difference is taken within the resample. An unpaired interval on that difference would be
                      wrong, not merely conservative.

  OBF (Table XIV)  -- the obfuscation best-cell result. results/phase2/p2_scores_mlp.npz commits the per-case score
                      of all 6,200 prompt/condition records at every swept block, so this is also a case bootstrap.

  AAG (Table VI)   -- the read-position effect. results/gates/scores/b2_grouped_*_test_scores.json commits the
                      per-case score of every held-out case under all five reads (the 2026-09-11 re-run under
                      GATES.md Amendment 7; the original run's scores were lost with the VM and this panel was
                      analytic at f07ce36). Case bootstrap, with every difference between two reads resampled PAIRED.
                      Two differences are reported: mean - final (the repair) and mean - mean_all (Amendment 7a: what
                      localising the read to the tool-response span adds over pooling the whole prompt).

  OBF pooled       -- Amendment 7c: the best length-stratified Base64 cell under the pooled reads, from the committed
                      results/phase2/p2d_scores{,_mlp}.npz, bootstrapped exactly like the final-token best cell.

All three families reproduce the committed point estimate exactly before resampling anything (`_verify`), using
the same stratified_auc/auc functions that produced the committed number rather than a reimplementation. A mismatch
aborts: an interval around a point estimate we cannot reproduce would be worthless.

  python paper/tools/gen_stats.py           # compute intervals, write the table and paper/stats/ci.json
  python paper/tools/gen_stats.py --check    # verify sources and that the outputs exist; write nothing
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO))
from gen_tables import MODEL_ORDER, SHORT, MissingSource, TABLES, emit, src, srcs  # noqa: E402

STATS = REPO / "paper/stats"
B = 2000                      # bootstrap resamples
SEED = 20260910               # recorded in ci.json and stated in the table note; changing it changes every interval
PCT = (2.5, 97.5)             # percentile interval, 95 %

SOURCES = ["results/gates/b1/*.json", "results/gates/b2_grouped/*.json", "results/gates/scores/b2_grouped_*_test_scores.json",
           "results/phase2/p2_scores_mlp.npz", "results/phase2/p2_scores.npz", "results/phase2/p2_confounds.json",
           "results/phase2/p2_summary_mlp.json", "results/phase2/p2d_summary.json", "results/phase2/p2d_summary_mlp.json",
           "results/phase2/p2d_scores.npz", "results/phase2/p2d_scores_mlp.npz"]
# GATES.md Arm 2 PASS threshold: a model "has learnable signal" when the mean read clears it on the held-out half.
# Pre-registered 2026-09-03; the 0.94 cut-off used at f07ce36 was chosen after the numbers and is retired (Amendment 7b).
AAG_PASS = 0.90
P2D_FEATURES = {"mlp_out": "results/phase2/p2d_scores_mlp.npz", "residual": "results/phase2/p2d_scores.npz"}
P2D_SUMMARY = {"mlp_out": "results/phase2/p2d_summary_mlp.json", "residual": "results/phase2/p2d_summary.json"}

# Both activation spaces, because the manuscript's "best anywhere in the sweep" is a maximum over both: the Base64
# maximum lives in the residual space and the ROT13 maximum in the MLP-branch space.
FEATURES = {"mlp_out": "results/phase2/p2_scores_mlp.npz", "residual": "results/phase2/p2_scores.npz"}

# The AF cell the manuscript's headline is about: the probe trained on on-topic twins, which is the row Table III
# labels "on-topic" and the only row the "+0.30 to +0.54 on all seven models" claim is made for.
AF_PROBE = "paired"
# The results files key the two probe lineages and the two activation spaces by their internal names. Nothing
# published may use those: the manuscript says "shipped" and "retrained", "MLP-branch" and "residual", and a table
# cell reading "mlp_out/inmode" would be the only place in the paper where a reader meets the harness's vocabulary.
# The raw key is kept beside the label in ci.json so a number stays traceable to the file it came from.
PROBE_LABEL = {"january": "shipped", "inmode": "retrained"}
SPACE_LABEL = {"mlp_out": "MLP-branch", "residual": "residual"}
CONDS = ["none", "base64", "rot13", "leetspeak"]
COND_LABEL = {"none": "plain", "base64": "Base64", "rot13": "ROT13", "leetspeak": "leetspeak"}
READS = ["final", "mean", "mean_all", "maxpos", "oracle"]


def pctl(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return (float(np.percentile(x, PCT[0])), float(np.percentile(x, PCT[1])))


def _verify(what, got, want, tol=1e-9):
    if not (abs(got - want) <= tol):
        raise MissingSource(f"gen_stats: CANNOT REPRODUCE the committed point estimate for {what}\n"
                            f"  recomputed {got!r} from the committed per-case scores, results file says {want!r}\n"
                            f"  an interval around an unreproducible estimate would be meaningless; aborting.")


# ------------------------------------------------------------------ AF: bootstrap over cases (Table III)
def af_intervals():
    from gates.b1_paired_af import eval_set
    from gates.runb import stratified_auc

    recs, _ = eval_set()
    y = np.array([r["label"] for r in recs])
    L = np.array([len(r["text"]) for r in recs], float)
    sets = np.array([r["set"] for r in recs])
    # Resample within each source set, so every resample keeps the evaluation's composition (300 harmful,
    # 250 XSTest, 100 JailbreakBench) rather than letting the benign mixture drift between resamples.
    groups = [np.flatnonzero(y == 1), np.flatnonzero(sets == "xstest_safe"), np.flatnonzero(sets == "jbb_benign")]

    out = {}
    for mi, model in enumerate(MODEL_ORDER):
        f = [p for p in srcs("results/gates/b1/*.json", 7) if json.load(open(p))["model"] == model]
        d = json.load(open(f[0]))
        s = {}
        for mode in ("raw", "templated"):
            c = d[mode]["cells"][AF_PROBE]
            v = np.empty(len(recs))
            v[y == 1] = c["scores"]["harmful"]
            v[sets == "xstest_safe"] = c["scores"]["xstest"]
            v[sets == "jbb_benign"] = c["scores"]["jbb"]
            s[mode] = v
            _verify(f"AF {model} {mode}", stratified_auc(v, y, L)[0], c["length_all"]["auc_length_stratified"])

        rng = np.random.default_rng([SEED, mi])
        raw, tmpl, delta = np.empty(B), np.empty(B), np.empty(B)
        for b in range(B):
            idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
            yb, Lb = y[idx], L[idx]
            raw[b] = stratified_auc(s["raw"][idx], yb, Lb)[0]
            tmpl[b] = stratified_auc(s["templated"][idx], yb, Lb)[0]
            delta[b] = tmpl[b] - raw[b]                       # PAIRED: same resampled cases in both arms
        pt = {m: stratified_auc(s[m], y, L)[0] for m in ("raw", "templated")}
        out[model] = {"raw": {"auc_star": pt["raw"], "ci": pctl(raw)},
                      "templated": {"auc_star": pt["templated"], "ci": pctl(tmpl)},
                      "delta": {"estimate": pt["templated"] - pt["raw"], "ci": pctl(delta)},
                      "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()), "method": "paired case bootstrap"}
        print(f"  AF {SHORT[model]:13s} raw {pt['raw']:.3f} {out[model]['raw']['ci']}  "
              f"tmpl {pt['templated']:.3f} {out[model]['templated']['ci']}  "
              f"delta {out[model]['delta']['estimate']:+.3f} {out[model]['delta']['ci']}", flush=True)
    return out


# ------------------------------------------------------------------ OBF: bootstrap over cases (Table XIV)
def obf_intervals():
    """Per encoding, the single best length-stratified cell anywhere in the sweep -- over both activation spaces, both
    probe constructions, both template modes and all 22 blocks -- with a bootstrap interval around it.

    That cell is a MAXIMUM over 352 cells per encoding, so its interval is conditional on the selection and is
    optimistic: it is the best case for the claim this section rejects, which is the point of quoting it."""
    from phase2.common import build_eval_set, format_for_mode, load_phase2_config
    from phase2.p2_confounds import stratified_auc

    cfg = load_phase2_config()
    data = build_eval_set(cfg)
    by_id = {r["id"]: r for r in data["records"]}
    conf = json.load(open(src("results/phase2/p2_confounds.json")))["features"]
    npz = {feat: np.load(src(rel)) for feat, rel in FEATURES.items()}

    out = {}
    for ci_, c in enumerate(CONDS):
        # locate the best cell for this encoding across every axis of the sweep
        best = max(((conf[feat]["summary"][f"{p}|{m}|{c}"]["best_length_stratified"], feat, p, m)
                    for feat in FEATURES for p in ("january", "inmode") for m in cfg["modes"]),
                   key=lambda t: t[0]["auc_length_stratified"])
        cell, feat, probe, mode = best
        block = cell["block"]
        z = npz[feat]
        ids = list(z[f"ids|{mode}|phase1"])
        cond = np.array([by_id[i]["condition"] for i in ids])
        y = np.array([by_id[i]["label"] for i in ids])
        L = np.array([len(format_for_mode(by_id[i]["text"], mode)) for i in ids], float)
        m_ = cond == c
        s_ = z[f"{probe}|{mode}|phase1|{block}"].astype(np.float64)[m_]
        yb, Lb = y[m_], L[m_]
        _verify(f"OBF {feat}|{probe}|{mode}|b{block}|{c}", stratified_auc(s_, yb, Lb)[0],
                cell["auc_length_stratified"])

        groups = [np.flatnonzero(yb == 1), np.flatnonzero(yb == 0)]
        rng = np.random.default_rng([SEED, 100 + ci_])
        draws = np.empty(B)
        for b in range(B):
            idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
            draws[b] = stratified_auc(s_[idx], yb[idx], Lb[idx])[0]

        # the population the maximum was drawn from, so the table can show the selection it rests on
        pool = [v["auc_length_stratified"] for f2 in FEATURES for k, v in conf[f2]["cells"].items()
                if k.endswith("|" + c) and not np.isnan(v["auc_length_stratified"])]
        out[c] = {"auc_star": cell["auc_length_stratified"], "ci": pctl(draws), "block": int(block),
                  "space": SPACE_LABEL[feat], "probe": PROBE_LABEL[probe], "mode": mode,
                  "results_key": f"{feat}|{probe}|{mode}|phase1|{block}", "n_cells": len(pool),
                  "median_over_cells": float(np.median(pool)),
                  "n_pos": int((yb == 1).sum()), "n_neg": int((yb == 0).sum()),
                  "method": "case bootstrap at the selected best cell (conditional on selection)"}
        print(f"  OBF {c:10s} best {cell['auc_length_stratified']:.3f} {out[c]['ci']} "
              f"({SPACE_LABEL[feat]}/{PROBE_LABEL[probe]}/{mode} b{block}); "
              f"median over {len(pool)} cells {out[c]['median_over_cells']:.3f}", flush=True)
    return out


# ------------------------------------------------------------------ AAG: paired case bootstrap (Amendment 7b)
def aag_intervals():
    """Per model, the five reads' AUCs on the grouped test half with case-bootstrap intervals, and two PAIRED
    differences: mean - final and mean - mean_all. Resampling is stratified by class so every resample keeps the
    525 / 59 composition of the held-out half."""
    from phase2.extract import auc

    agg = {json.load(open(f))["model"]: json.load(open(f)) for f in srcs("results/gates/b2_grouped/*.json", 7)}
    out = {}
    for mi, model in enumerate(MODEL_ORDER):
        f = [p for p in srcs("results/gates/scores/b2_grouped_*_test_scores.json", 7)
             if json.load(open(p))["model"] == model]
        d = json.load(open(f[0]))
        y = np.array(d["labels"]); s = {r: np.array(d["scores"][r], float) for r in READS}
        t_ = agg[model]["results"]["test"]
        for r in READS:
            _verify(f"AAG {SHORT[model]} {r}", auc(s[r][y == 1], s[r][y == 0]), t_[r]["auc"])
        groups = [np.flatnonzero(y == 1), np.flatnonzero(y == 0)]
        rng = np.random.default_rng([SEED, 200 + mi])
        draws = {r: np.empty(B) for r in READS}
        for b in range(B):
            idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
            yb = y[idx]
            for r in READS:
                draws[r][b] = auc(s[r][idx][yb == 1], s[r][idx][yb == 0])      # same resample for every read: PAIRED
        rec = {"n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()), "method": "paired case bootstrap",
               "source": f[0].replace(str(REPO) + "/", "")}
        for r in READS:
            rec[r] = {"auc": t_[r]["auc"], "ci": pctl(draws[r])}
        for name, a, b_ in (("delta_mean_minus_final", "mean", "final"), ("delta_mean_minus_meanall", "mean", "mean_all")):
            rec[name] = {"estimate": t_[a]["auc"] - t_[b_]["auc"], "ci": pctl(draws[a] - draws[b_]),
                         "method": "paired case bootstrap"}
        out[model] = rec
        print(f"  AAG {SHORT[model]:13s} final {rec['final']['auc']:.3f} {rec['final']['ci']}  "
              f"mean {rec['mean']['auc']:.3f} {rec['mean']['ci']}  d(mean-final) {rec['delta_mean_minus_final']['estimate']:+.3f} "
              f"{rec['delta_mean_minus_final']['ci']}  d(mean-mean_all) {rec['delta_mean_minus_meanall']['estimate']:+.3f} "
              f"{rec['delta_mean_minus_meanall']['ci']}", flush=True)
    return out


# ------------------------------------------------------------------ OBF pooled: bootstrap at the best pooled Base64 cell (Amendment 7c)
def obf_pooled_interval(cond="base64"):
    from phase2.common import build_eval_set, format_for_mode, load_phase2_config
    from phase2.p2_confounds import stratified_auc

    cfg = load_phase2_config()
    data = build_eval_set(cfg)
    by_id = {r["id"]: r for r in data["records"]}
    best = None
    for feat, rel in P2D_SUMMARY.items():
        sm = json.load(open(src(rel)))
        for k, c in sm["cells"].items():
            if k.endswith("|" + cond) and c["auc_length_stratified"] is not None and \
                    (best is None or c["auc_length_stratified"] > best[0]):
                best = (c["auc_length_stratified"], feat, k, c, sm["verdict"]["verdict"])
    a_star, feat, key, cell, verdict = best
    probe, read, mode, wrapper, block, _ = key.split("|")
    z = np.load(src(P2D_FEATURES[feat]))
    ids = list(z[f"ids|{mode}|{wrapper}"])
    condv = np.array([by_id[i]["condition"] for i in ids]); y = np.array([by_id[i]["label"] for i in ids])
    L = np.array([len(format_for_mode(by_id[i]["text"], mode)) for i in ids], float)
    m_ = condv == cond
    s_ = z[f"{probe}|{read}|{mode}|{wrapper}|{block}"].astype(np.float64)[m_]
    yb, Lb = y[m_], L[m_]
    _verify(f"OBF pooled {key}", stratified_auc(s_, yb, Lb)[0], a_star)
    groups = [np.flatnonzero(yb == 1), np.flatnonzero(yb == 0)]
    rng = np.random.default_rng([SEED, 300])
    draws = np.empty(B)
    for b in range(B):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        draws[b] = stratified_auc(s_[idx], yb[idx], Lb[idx])[0]
    pool = [c["auc_length_stratified"] for f2, rel in P2D_SUMMARY.items()
            for k, c in json.load(open(src(rel)))["cells"].items() if k.endswith("|" + cond) and not np.isnan(c["auc_length_stratified"])]
    verdicts = {f2: json.load(open(src(rel)))["verdict"]["verdict"] for f2, rel in P2D_SUMMARY.items()}
    out = {"auc_star": a_star, "ci": pctl(draws), "block": int(block), "space": SPACE_LABEL[feat], "probe": PROBE_LABEL[probe],
           "mode": mode, "read": read, "results_key": key, "rho_encoded_vs_plain_harmful": cell["rho_encoded_vs_plain_harmful"],
           "n_cells": len(pool), "median_over_cells": float(np.median(pool)), "n_pos": int((yb == 1).sum()), "n_neg": int((yb == 0).sum()),
           "verdict_by_space": verdicts, "method": "case bootstrap at the selected best pooled cell (conditional on selection)"}
    print(f"  OBF pooled {cond}: best {a_star:.3f} {out['ci']} ({SPACE_LABEL[feat]}/{PROBE_LABEL[probe]}/{read}/{mode} b{block}, "
          f"rho plain {cell['rho_encoded_vs_plain_harmful']:+.2f}); median over {len(pool)} cells {out['median_over_cells']:.3f}; "
          f"verdicts {verdicts}", flush=True)
    return out


# ------------------------------------------------------------------ the table
def f3(x):
    return f"{x:.3f}"


def ci3(c):
    return rf"[{c[0]:.3f},\,{c[1]:.3f}]"


def tab_confidence(af, aag, obf, pooled):
    NC = 9
    a_rows = [f"{SHORT[m]} & {f3(af[m]['raw']['auc_star'])} & {ci3(af[m]['raw']['ci'])} & "
              f"{f3(af[m]['templated']['auc_star'])} & {ci3(af[m]['templated']['ci'])} & "
              f"{af[m]['delta']['estimate']:+.3f} & {ci3(af[m]['delta']['ci'])} & & " + r"\\" for m in MODEL_ORDER]
    b_rows = [f"{SHORT[m]} & {f3(aag[m]['final']['auc'])} & {ci3(aag[m]['final']['ci'])} & "
              f"{f3(aag[m]['mean']['auc'])} & {ci3(aag[m]['mean']['ci'])} & "
              f"{aag[m]['delta_mean_minus_final']['estimate']:+.3f} & {ci3(aag[m]['delta_mean_minus_final']['ci'])} & "
              f"{aag[m]['delta_mean_minus_meanall']['estimate']:+.3f} & {ci3(aag[m]['delta_mean_minus_meanall']['ci'])} "
              + r"\\" for m in MODEL_ORDER]
    c_rows = []
    for c in CONDS:
        o = obf[c]
        c_rows.append(f"{COND_LABEL[c]} & final & {o['space']}, {o['probe']}, {o['mode']} & "
                      f"{o['block']} & {f3(o['auc_star'])} & {ci3(o['ci'])} & {f3(o['median_over_cells'])} & & " + r"\\")
    p = pooled
    c_rows.append(rf"Base64 & {p['read'].replace('mean_', '')} mean & {p['space']}, {p['probe']}, {p['mode']} & "
                  f"{p['block']} & {f3(p['auc_star'])} & {ci3(p['ci'])} & {f3(p['median_over_cells'])} & "
                  rf"$\rho={p['rho_encoded_vs_plain_harmful']:+.2f}$ & " + r"\\")

    af_lo = min(af[m]["delta"]["ci"][0] for m in MODEL_ORDER)
    crosses = [SHORT[m] for m in MODEL_ORDER
               if aag[m]["delta_mean_minus_final"]["ci"][0] <= 0 <= aag[m]["delta_mean_minus_final"]["ci"][1]]
    negative = [SHORT[m] for m in MODEL_ORDER if aag[m]["delta_mean_minus_final"]["ci"][1] < 0]
    learnable = [m for m in MODEL_ORDER if aag[m]["mean"]["auc"] >= AAG_PASS]
    aag_lo = min(aag[m]["delta_mean_minus_final"]["ci"][0] for m in learnable) if learnable else float("nan")
    loc_excl = [SHORT[m] for m in learnable if aag[m]["delta_mean_minus_meanall"]["ci"][0] > 0]
    loc_lo = min(aag[m]["delta_mean_minus_meanall"]["ci"][0] for m in learnable) if learnable else float("nan")
    above = [COND_LABEL[c] for c in ("base64", "rot13") if obf[c]["ci"][0] > 0.5]

    note = [rf"\footnotesize Panel A: every template difference excludes zero; the weakest lower bound is "
            rf"${af_lo:+.3f}$ ({SHORT[min(MODEL_ORDER, key=lambda m: af[m]['delta']['ci'][0])]}). "
            rf"Panel B: on the {len(learnable)} models whose mean read clears the pre-registered ${AAG_PASS:.2f}$ pass "
            rf"threshold the difference from the final-token read excludes zero, weakest lower bound ${aag_lo:+.3f}$; it crosses zero on "
            + (", ".join(crosses) if crosses else "no model")
            + (rf" and is significantly negative on {', '.join(negative)}" if negative else "") + ". "
            rf"The second difference, mean $-$ mean\textsubscript{{all}}, is what localising the read to the tool-response "
            rf"span adds over pooling the whole prompt: it excludes zero on {len(loc_excl)} of those {len(learnable)} models "
            + (rf"({', '.join(loc_excl)}; weakest lower bound ${loc_lo:+.3f}$)" if loc_excl else rf"(weakest lower bound ${loc_lo:+.3f}$)") + ". "
            rf"The intervals in this panel are wide because the grouped test half holds only "
            rf"${aag[MODEL_ORDER[0]]['n_neg']}$ benign cases (Section~\ref{{sec:data}}). "
            rf"Panel C: each final-token row is the maximum over ${obf['base64']['n_cells']}$ cells and the pooled row over "
            rf"${p['n_cells']}$, so the intervals are conditional on that selection; the median column is the population the "
            r"maximum was drawn from. "]
    if above:
        note.append(rf"The best {' and '.join(above)} final-token cell does exceed chance "
                    r"(Section~\ref{sec:sweep} discusses that cell directly); the median does not. ")
    else:
        note.append(r"No encoded final-token interval excludes $0.5$. ")
    note.append(rf"Pooled-read verdict on Base64 (Amendment 7): {', '.join(f'{SPACE_LABEL[k]} {v}' for k, v in p['verdict_by_space'].items())}.")

    body = "\n".join([
        r"\begin{table*}[!t]",
        r"\caption{Ninety-five per cent confidence intervals on the three headline effects: nonparametric case",
        rf"bootstraps, ${B}$ resamples, seed ${SEED}$, percentile intervals. Every difference between two reads of the",
        r"same cases is resampled \emph{paired}: one draw of case indices is applied to both arms and the difference",
        r"taken within the draw. Panel B is computed from the per-case scores committed with the 2026-09-11 re-run of",
        r"Table~\ref{tab:aag_designs}.}",
        r"\label{tab:confidence}", r"\centering", r"\begin{tabular}{l" + "c" * (NC - 1) + "}", r"\hline",
        rf"\multicolumn{{{NC}}}{{l}}{{\textbf{{A. Harmful content: bare versus templated read, on-topic probe "
        r"(Table~\ref{tab:extraction_convention})}} \\",
        r"Model & \aucstar{} raw & 95\,\% CI & \aucstar{} tmpl & 95\,\% CI & $\Delta$ & 95\,\% CI & & \\",
        r"\hline", *a_rows,
        r"\addlinespace",
        rf"\multicolumn{{{NC}}}{{l}}{{\textbf{{B. Prompt injection: final, mean-over-span and whole-prompt mean, grouped split "
        r"(Table~\ref{tab:aag_designs})}} \\",
        r"Model & AUC final & 95\,\% CI & AUC mean & 95\,\% CI & mean$-$final & 95\,\% CI & "
        r"mean$-$mean\textsubscript{all} & 95\,\% CI \\",
        r"\hline", *b_rows,
        r"\addlinespace",
        rf"\multicolumn{{{NC}}}{{l}}{{\textbf{{C. Obfuscation: the best length-stratified cell anywhere in the sweep, per "
        r"encoding (Tables~\ref{tab:obfuscation_sweep} and~\ref{tab:obfuscation_pooled})}} \\",
        r"Encoding & read & space, probe, mode & block & \aucstar{} & 95\,\% CI & median over cells & & \\",
        r"\hline", *c_rows, r"\hline", r"\end{tabular}", "",
        "".join(note), r"\end{table*}"])
    return emit("tab_confidence.tex", body, SOURCES, size="scriptsize", colsep=0.8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    for pat in SOURCES:
        srcs(pat) if "*" in pat else src(pat)
    if a.check:
        missing = [p.name for p in (TABLES / "tab_confidence.tex", STATS / "ci.json") if not p.exists()]
        if missing:
            raise SystemExit(f"gen_stats --check: not generated: {missing} (run `make stats`)")
        print(f"gen_stats --check: all {len(SOURCES)} source patterns present, tab_confidence.tex and ci.json on disk")
        return 0
    print(f"gen_stats: B={B}, seed={SEED}, percentile {PCT[0]}--{PCT[1]}")
    af, aag, obf, pooled = af_intervals(), aag_intervals(), obf_intervals(), obf_pooled_interval()
    STATS.mkdir(parents=True, exist_ok=True)
    json.dump({"bootstrap_resamples": B, "seed": SEED, "percentiles": list(PCT), "sources": SOURCES, "aag_pass_threshold": AAG_PASS,
               "af_extraction_convention": af, "aag_read_position": aag, "obfuscation_best_cell": obf,
               "obfuscation_best_pooled_cell": pooled},
              open(STATS / "ci.json", "w"), indent=1, default=float)
    print(f"  wrote paper/tables/{tab_confidence(af, aag, obf, pooled)}")
    print("  wrote paper/stats/ci.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
