#!/usr/bin/env python3
"""
ROC curves from real per-case benchmark scores (Phase-0 replacement).

This script does NOT run any model and does NOT contain any prompts.  It reads
the per-model results JSON written by ``benchmark_vllm.py`` (which records a
``per_case`` list with ``score``, ``label`` and ``id`` for every scored prompt),
recomputes AUC from those scores, plots ROC curves, and writes
``<output>_data.json`` with the raw scores it used.  If it is pointed at a
directory with no usable results, or a results file without per-case scores,
it raises instead of drawing anything.

The previous version of this file fabricated its AAG/AF test sets from literal
templates; it is quarantined at ``archive/synthetic/generate_roc_curves_vllm_SYNTHETIC.py``.

Usage:
    python generate_roc_curves_vllm.py --results-dir results/phase1 --output roc_curves_phase1
    python generate_roc_curves_vllm.py --results results/phase1/google_gemma_2_9b_it.json ...
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from aase_eval import DataLoadError  # noqa: E402

BENCHMARK_TO_PROBE = {"harmbench": "af", "injecagent": "aag"}
TITLES = {"af": "AF - Harmful Content Detection (HarmBench)", "aag": "AAG - Prompt Injection Detection (InjecAgent)"}


def auc_mann_whitney(pos: np.ndarray, neg: np.ndarray) -> float:
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def roc_points(pos: np.ndarray, neg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """FPR/TPR at every distinct threshold (descending), plus the (0,0) and (1,1) ends."""
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(-scores, kind="mergesort")
    scores, labels = scores[order], labels[order]
    tps = np.cumsum(labels)
    fps = np.cumsum(1 - labels)
    distinct = np.where(np.diff(scores))[0]
    idx = np.concatenate([distinct, [len(scores) - 1]])
    tpr = np.concatenate([[0.0], tps[idx] / len(pos)])
    fpr = np.concatenate([[0.0], fps[idx] / len(neg)])
    return fpr, tpr


def load_results_file(path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract {probe: {positive, negative, threshold, n_..., source}} from one benchmark results JSON."""
    with open(path) as f:
        data = json.load(f)
    if "model" not in data:
        raise DataLoadError(f"{path}: not a benchmark_vllm.py results file (no 'model' key); keys={sorted(data)}")
    out = {}
    for bench, probe in BENCHMARK_TO_PROBE.items():
        b = data.get(bench)
        if not isinstance(b, dict) or "error" in b:
            continue
        per_case = b.get("per_case")
        if not per_case:
            raise DataLoadError(f"{path}: '{bench}' has no per_case scores — was it produced by the pre-Phase-0 script? "
                                f"keys={sorted(b)}")
        pos = np.array([r["score"] for r in per_case if r["label"] == 1], dtype=float)
        neg = np.array([r["score"] for r in per_case if r["label"] == 0], dtype=float)
        if len(pos) == 0 or len(neg) == 0:
            raise DataLoadError(f"{path}: '{bench}' per_case has {len(pos)} positives and {len(neg)} negatives")
        out[probe] = {"positive": pos, "negative": neg, "threshold": float(b["details"]["threshold"]),
                      "data_summary": b.get("data_summary", {}), "n_positive": len(pos), "n_negative": len(neg)}
    if not out:
        raise DataLoadError(f"{path}: contains no scoreable benchmark section (harmbench/injecagent)")
    return data["model"], out


def collect(results_files: List[Path]) -> Dict[str, Dict[str, Dict]]:
    if not results_files:
        raise DataLoadError("no results files given (use --results-dir or --results); refusing to plot nothing")
    all_results = {}
    for p in results_files:
        if not p.exists():
            raise DataLoadError(f"results file not found: {p}")
        model, res = load_results_file(p)
        all_results[model] = res
    return all_results


def plot(all_results: Dict[str, Dict[str, Dict]], output: str) -> Dict[str, Dict[str, float]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probes = sorted({p for r in all_results.values() for p in r})
    fig, axes = plt.subplots(1, len(probes), figsize=(6 * len(probes), 5), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_results), 1)))
    aucs: Dict[str, Dict[str, float]] = {}
    for ax, probe in zip(axes[0], probes):
        for (model, res), color in zip(all_results.items(), colors):
            if probe not in res:
                continue
            fpr, tpr = roc_points(res[probe]["positive"], res[probe]["negative"])
            auc = auc_mann_whitney(res[probe]["positive"], res[probe]["negative"])
            aucs.setdefault(model, {})[probe] = auc
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{model.split('/')[-1]} ({auc:.3f}; n={res[probe]['n_positive']}+{res[probe]['n_negative']})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title(TITLES.get(probe, probe)); ax.legend(loc="lower right", fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(f"{output}.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{output}.pdf", bbox_inches="tight")
    print(f"Saved: {output}.png and {output}.pdf")
    return aucs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ROC curves from benchmark_vllm.py per-case results (no model, no prompts)")
    ap.add_argument("--results-dir", default=None, help="directory of <model_safe>.json results files")
    ap.add_argument("--results", nargs="*", default=[], help="explicit results files")
    ap.add_argument("--output", default="roc_curves_real")
    args = ap.parse_args(argv)
    files = [Path(p) for p in args.results]
    if args.results_dir:
        d = Path(args.results_dir)
        if not d.is_dir():
            raise DataLoadError(f"results dir not found: {d}")
        files += sorted(d.glob("*.json"))
    all_results = collect(files)
    aucs = plot(all_results, args.output)
    save = {m: {p: {"positive": r[p]["positive"].tolist(), "negative": r[p]["negative"].tolist(), "threshold": r[p]["threshold"],
                    "auc": aucs[m][p], "data_summary": r[p]["data_summary"]} for p in r} for m, r in all_results.items()}
    with open(f"{args.output}_data.json", "w") as f:
        json.dump(save, f, indent=2)
    print(f"Saved: {args.output}_data.json")
    print(f"\n{'Model':<40}" + "".join(f"{p:>12}" for p in sorted({p for r in aucs.values() for p in r})))
    for m, r in aucs.items():
        print(f"{m:<40}" + "".join(f"{r.get(p, float('nan')):>12.3f}" for p in sorted({p for rr in aucs.values() for p in rr})))
    return 0


if __name__ == "__main__":
    sys.exit(main())
