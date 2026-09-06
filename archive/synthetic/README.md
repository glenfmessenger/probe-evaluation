# archive/synthetic — quarantined synthetic-score artefacts

Nothing in this directory is a measurement on real benchmark data. It is kept only so the history documented in
`VERIFICATION_INJECAGENT.md` and `SECOND_PASS_REPORT.md` stays reproducible. Do not import from here; do not cite.

| File | Was | Why quarantined |
|---|---|---|
| `generate_roc_curves_vllm_SYNTHETIC.py` | `aase_vllm/scripts/generate_roc_curves_vllm.py` (2026-01-04) | built its "InjecAgent" AAG set from 5 literal templates × 13 and 5 safe literals × 40, and its AF set from 8 literals × 7; never read a dataset |
| `roc_curves_vllm_data_SYNTHETIC.json` | `aase_vllm/roc_curves_vllm_data.json` | the scores that script produced (5-periodic clusters; source of the paper's AAG AUC column) |
| `roc_curves_vllm_SYNTHETIC.pdf` / `.png` | `aase_vllm/roc_curves_vllm.{pdf,png}` = the paper's `figure4_roc_curves.pdf` | plotted from the file above |

The replacement `aase_vllm/scripts/generate_roc_curves_vllm.py` consumes per-case scores from `benchmark_vllm.py`
results files and raises if it is pointed at nothing.
