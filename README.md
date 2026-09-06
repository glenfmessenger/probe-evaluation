# Minimal-data activation probes: evaluation harness and results

Code, authored data, probes, pre-registered thresholds, decision log and provenance-stamped results for
*Diagnosing and Repairing Minimal-Data Activation Probes for LLM Safety*.

Every number in the paper is generated from a results file in this repository by `paper/tools/gen_tables.py`
and `paper/tools/gen_figures.py`. Both abort if a source file is missing, so the manuscript cannot be built
against stale or absent results.

## Start here

| File | What it is |
|---|---|
| `GATES.md` | the pre-registered thresholds, and all six amendments with the superseded wording struck through rather than deleted |
| `GATES_REPORT.md` | the three-arm result: the verdict for each arm and what it licenses the paper to claim |
| `DECISIONS.md` | every evaluation decision, its rationale, who took it and when |
| `PHASE1_REPORT.md`, `PHASE2_REPORT.md` | the earlier measurement passes the study builds on |
| `AILUMINATE_DISCREPANCY.md`, `VERIFICATION_INJECAGENT.md` | two methodological audits the paper cites |
| `BENIGN_SOURCES.md` | how the 120-case agent benign set was constructed |

## Layout

```
aase_eval/      dataset loaders (fail-loud: a missing or malformed source raises, never falls back)
aase_vllm/      probe training and benchmark scripts, and the shipped probes under pretrained/
gates/          the three-arm study: protocol, authored data, runners, validity checks
phase2/         the obfuscation study
paper/          table and figure generators, their generated output, and the manuscript source
results/        every provenance-stamped results file (git commit, config and dataset sha256s, packages, GPU)
datasets/       authored and derived evaluation data
third_party/    third-party benchmark data, with licences retained
tests/          109 CPU tests, including byte-compare fixtures pinning the authored data to its sign-off
```

`package/`, `scripts/` and `archive/` hold superseded code from earlier iterations. They are retained because
`tests/test_no_fallback.py` polices them — that suite exists to stop a silent synthetic-data fallback from
returning — but nothing in the current evaluation imports them.

## Running

```
pip install -r requirements-pinned.txt
python -m pytest -q tests            # CPU only
```

GPU stages are driven by `gates/run_gates.sh` and `phase2/run_phase2.sh`, which verify provenance, apply the
degenerate-score stop rule, and commit after each milestone.

## Data provenance

Third-party benchmark data is included where its licence permits redistribution, with the upstream licence
retained: HarmBench behaviour datasets (behaviour CSVs only), XSTest v2, JailbreakBench, InjecAgent, and the
AILuminate v1.0 DEMO prompt set. Each results file records the sha256 of every dataset it read, so any
substitution is detectable.

## One file is not an evaluation set

`gates/data/arm3_v2_discriminator.json` is marked `NOT FOR EVALUATION`. It is retained as the artifact of the
construction-failure finding reported in the paper, and was never scored. Do not use it to evaluate a
classifier.
