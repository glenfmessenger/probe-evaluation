#!/usr/bin/env python3
"""Generate every figure in the manuscript from the committed, provenance-stamped results files.

PAPER_BRIEF framing rule 3 extended to graphics: no data point is placed by hand. Both figures are drawn here from
results JSON, and the build fails if a declared source is missing rather than emitting a stale or partial figure.

  python paper/tools/gen_figures.py            # regenerate both
  python paper/tools/gen_figures.py --check     # verify sources and that both PDFs exist; write nothing
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIGURES = REPO / "paper/figures"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_tables import MODEL_ORDER, SHORT, src  # noqa: E402  (shared source resolution: same failure mode, same message)

SOURCES = ["results/phase2/p2_confounds.json", "results/gates/extraction_convention.json",
           "results/gates/b2_grouped/*.json"]

# The retrained in-mode probe in the MLP-branch space is the most favourable configuration in the sweep, so it is the
# one the figure shows: the claim is that even the best case fails, and a figure of a weaker probe would not carry it.
PROBE, FEATURE, WRAPPER = "inmode", "mlp_out", "phase1"
CONDS = ["none", "base64", "rot13", "leetspeak"]
COND_LABEL = {"none": "plain", "base64": "Base64", "rot13": "ROT13", "leetspeak": "leetspeak"}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})
    return plt


# matplotlib stamps a CreationDate into every PDF, so regenerating an unchanged figure still rewrites the bytes and
# marks the working tree dirty. That is what kept re-introducing "+dirty" into the provenance stamp. Suppressing the
# timestamp makes figure generation idempotent: identical inputs produce byte-identical output.
SAVE = {"bbox_inches": "tight", "metadata": {"CreationDate": None}}


def fig_obfuscation():
    """Section 7: length-stratified AUC per block per encoding, one panel per template mode."""
    import numpy as np
    plt = _mpl()
    rel = "results/phase2/p2_confounds.json"
    d = json.load(open(src(rel)))["features"][FEATURE]["cells"]
    blocks = sorted({int(k.split("|")[3]) for k in d})
    modes = ["raw", "templated"]
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.1), constrained_layout=True)
    im = None
    for ax, mode in zip(axes, modes):
        M = np.full((len(CONDS), len(blocks)), np.nan)
        for i, c in enumerate(CONDS):
            for j, b in enumerate(blocks):
                cell = d.get(f"{PROBE}|{mode}|{WRAPPER}|{b}|{c}")
                if cell and cell["auc_length_stratified"] is not None:
                    M[i, j] = cell["auc_length_stratified"]
        im = ax.imshow(M, vmin=0.3, vmax=0.9, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(blocks))); ax.set_xticklabels(blocks, fontsize=5.5)
        ax.set_yticks(range(len(CONDS))); ax.set_yticklabels([COND_LABEL[c] for c in CONDS], fontsize=6.5)
        for i in range(len(CONDS)):
            for j in range(len(blocks)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=4.2)
        ax.set_title(f"{mode} extraction", fontsize=7, pad=3)
        ax.set_xlabel("decoder block", fontsize=6.5)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label("length-stratified AUC", fontsize=6.5); cb.ax.tick_params(labelsize=5.5)
    out = FIGURES / "fig_obfuscation.pdf"
    fig.savefig(out, **SAVE); plt.close(fig)
    return out.name


def fig_extraction():
    """Section 5: the central exhibit as dumbbells. Panel A harmful-content, panel B prompt injection."""
    plt = _mpl()
    rel_a = "results/gates/extraction_convention.json"
    rows = {r["model"]: r for r in json.load(open(src(rel_a)))["rows"]}
    grp = {}
    import glob
    files = sorted(glob.glob(str(REPO / "results/gates/b2_grouped/*.json")))
    if len(files) != 7:
        from gen_tables import MissingSource
        raise MissingSource(f"gen_figures: expected 7 grouped-split results, found {len(files)}")
    for f in files:
        x = json.load(open(f)); grp[x["model"]] = x["results"]["test"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), constrained_layout=True)
    order = list(reversed(MODEL_ORDER))
    y = range(len(order))

    a_before = [rows[m]["cells"]["paired|raw"]["auc_length_stratified"] for m in order]
    a_after = [rows[m]["cells"]["paired|templated"]["auc_length_stratified"] for m in order]
    b_before = [grp[m]["final"]["auc"] for m in order]
    b_after = [grp[m]["mean"]["auc"] for m in order]

    for ax, before, after, (lb, la), title, xlab in (
            (axes[0], a_before, a_after, ("bare prompt", "chat template"),
             "A. Harmful content: text the probe reads", "length-stratified AUC"),
            (axes[1], b_before, b_after, ("final token", "mean over span"),
             "B. Prompt injection: position read", "AUC, unseen injections")):
        for i, (b0, b1) in enumerate(zip(before, after)):
            ax.plot([b0, b1], [i, i], color="0.6", lw=1.0, zorder=1)
        ax.scatter(before, list(y), s=18, color="#b2182b", zorder=2, label=lb)
        ax.scatter(after, list(y), s=18, color="#2166ac", zorder=2, label=la)
        ax.set_yticks(list(y)); ax.set_yticklabels([SHORT[m] for m in order], fontsize=6.5)
        ax.set_xlim(0.3, 1.02); ax.set_xlabel(xlab, fontsize=6.5)
        ax.set_title(title, fontsize=7, pad=3)
        ax.grid(axis="x", lw=0.3, color="0.85"); ax.set_axisbelow(True)
        ax.legend(fontsize=6, loc="lower left", frameon=False)
        ax.tick_params(labelsize=6)
    out = FIGURES / "fig_extraction.pdf"
    fig.savefig(out, **SAVE); plt.close(fig)
    return out.name


GENERATORS = [fig_extraction, fig_obfuscation]
EXPECTED = ["fig_extraction.pdf", "fig_obfuscation.pdf"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    import glob
    for pat in SOURCES:
        if "*" in pat:
            if not glob.glob(str(REPO / pat)):
                from gen_tables import MissingSource
                raise MissingSource(f"gen_figures: MISSING SOURCE {pat} (no files matched)")
        else:
            src(pat)
    if a.check:
        missing = [f for f in EXPECTED if not (FIGURES / f).exists()]
        if missing:
            raise SystemExit(f"gen_figures --check: figures not generated: {missing}")
        print(f"gen_figures --check: all {len(SOURCES)} source patterns present, {len(EXPECTED)} figures on disk")
        return 0
    FIGURES.mkdir(parents=True, exist_ok=True)
    for g in GENERATORS:
        print(f"  wrote paper/figures/{g()}")
    print(f"gen_figures: {len(GENERATORS)} figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
