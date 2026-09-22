# AASE — Activation-based AI Safety Enforcement (consolidated)

Single-repo consolidation of the AASE research codebase (activation probes for harmful content **AF**, agent-action gating **AAG**, policy compliance **APC**, hallucination detection **AHD**, and instruction-hierarchy **IHE** experiments) that was scattered across `~/Downloads`. Built 2026-09-02; see `CONSOLIDATION_LOG.md` for every decision and `manifest/provenance.csv` for the source of every file.

## Layout

| Path | What it is | Origin |
|---|---|---|
| `runtime/` | **Canonical.** Verbatim copy of the GPU-VM home directory (Dec 26–31 2025): AF benchmarks, `af_distribution/` (AF vectors + runtime), `aag/` (agent gating, InjecAgent benchmark), `apc/`, `ahd/`, `ihe/`, all result JSONs, plus `.bash_history` / `.viminfo` | `Downloads/home/glen/` |
| `release/aase-release/` | Packaged release v0.1.0 (2026-01-04) with **63 pretrained probe vectors** (`pretrained/{af,aag,apc}/*.npy`) for 7 models. Preserved intact. | `Downloads/testingaase/aase-release/` |
| `package/aase/` | Unified `aase` Python package (core/probes/integrations incl. llm-d + vLLM, scripts, tests), restored from `aase_complete.zip`; sibling `package/*.py` are the calibration / retraining / evaluation scripts that lived next to it (incl. `eval_harmbench.py`) | `Downloads/files (15)/` |
| `aase_vllm/` | `aase-vllm` package (vLLM `apply_model()` hooks), version 0.1.1, with nested `pretrained/` probes, ROC outputs and gemma-2-9b probes | `Downloads/aase 2/` + `aase-vllm-0.1.1/` |
| `scripts/` | Loose experiment scripts, deduped, grouped by family: `benchmarks/` (`benchmark_comparison` v1–v5, defense comparisons, MLCommons, latency), `validation/` (`phase1_validation` v1–v4, Gemma-7B / Llama-3 / quantization), `af/`, `ahd/`, `apc/`, `infra/` | Downloads root |
| `notebooks/` | `aase_demo.ipynb` v1–v3 (live demo with pretrained probes) | Downloads root |
| `archive/superseded/` | Losing versions of the 13 name conflicts, kept for reference | – |
| `manifest/` | `provenance.csv`, `deduplicated.csv`, `sources_sha256.csv`, `build_repo.py` | generated |
| `EXECUTION_HISTORY.md` | Chronological report of what actually ran on the VM, per script family, with the version resolutions | from `.bash_history` + `.viminfo` |
| `docs/HARMBENCH_SUBSET.md` | How the "HarmBench" eval set is defined (spoiler: static hard-coded lists, not sampling) | – |

## Which version is current?

| Family | Current | Basis |
|---|---|---|
| `agent_gating_poc` | `runtime/aag/agent_gating_poc_v6.py` | last executed (history L317) |
| `phase1_live_agent` | `runtime/aag/phase1_live_agent_v5.py` | last executed (L340) |
| `injecagent_benchmark` | `runtime/aag/injecagent_benchmark_v4.py` | last executed (L384); no v2 ever existed |
| `apc_poc` | `runtime/apc/apc_poc_v3.py` | last executed (L321) |
| `af_benchmark_category` | `runtime/af_benchmark_category_v2.py` | last executed (L67) |
| `simpleqa_ahd_benchmark` | `runtime/ahd/simpleqa_ahd_benchmark.py` | last executed (L436) |
| `benchmark_comparison` | `scripts/benchmarks/benchmark_comparison_v5.py` | not in VM history; docstring changelog chain v1→v5 |
| `phase1_validation` | `scripts/validation/phase1_validation_v4.py` | not in VM history; docstring changelog chain |
| vLLM package | `aase_vllm/` (0.1.1) | later than `aase 2`, same code |

## Runtime notes

- `runtime/aag/InjecAgent/` is a third-party working tree (uiuc-kang-lab/InjecAgent @ f19c9f2); see its `UPSTREAM.txt`.
- Secrets that were in the VM home (`.git-credentials`, `.docker/config.json`, `.ssh/`) and gcloud logs were **not** copied.
- Model weights are not included anywhere; scripts pull from Hugging Face (`google/gemma-*`, `meta-llama/Llama-3.*`).

**Rename note (2026-09-03, Phase 1):** the vLLM package directory that pass one imported as `vllm/` is now `aase_vllm/`. A top-level `vllm/` in the repo shadowed the installed vLLM package whenever the repo root was on `sys.path` (which the harness does), so `import vllm` inside the harness resolved to the AASE package instead of vLLM. Old paths in the historical reports refer to the same files.
