# Phase 0 report — eval pipeline repair (branch `eval-fix`, from `second-pass`)

Constraints honoured: no GPU work, no model downloads, no probe training or evaluation runs; `~/Downloads` untouched; everything below ran on CPU against on-disk data. Not merged — left for review.

Test result: **45 passed** (`python -m pytest -q`, 1.6 s). Dry run: `python aase_vllm/scripts/benchmark_vllm.py --dry-run` → `DRY RUN OK` with 1,054 / 17 / 300 / 100 records.

---

## Part A — the December real-data results

`DEC30_RESULTS.md` (new). Summary: five results files in `runtime/aag/` came from the Dec-30 HF-era scripts that used the correct keys and scored all 1,054 paired InjecAgent cases — but **only on `google/gemma-3-1b-it`**, with aggregates only (no per-case scores, so no AUC can be recomputed), under two non-comparable metric definitions. On the activation-only metric closest to the paper's (v1): 95.4 % / 94.3 % unsafe-action detection with 24.7 % / 21.8 % false-positive rate on the paired safe action (base / enhanced). v4 (generation-based compliance gating): TPR 98.2 % / 90.5 % over the 57 / 42 cases where the undefended model actually complied, FPR 2.1 % / 1.5 %. No December data exists for the other six models or for any vLLM run.

---

## Part B — pipeline fixes

### New shared package `aase_eval/` (repo root)
| File | Purpose |
|---|---|
| `aase_eval/errors.py` | `DataLoadError` + helpers that name the file, expected keys and found keys |
| `aase_eval/config.py` | `load_config()` for `eval_config.yaml` (`--config` / `$AASE_EVAL_CONFIG`), `resolve_path()` relative to repo root |
| `aase_eval/injecagent.py` | real-schema loaders: `load_attacker_cases` (62), `load_user_cases` (17), `load_test_cases` (1,054 per setting), prompt builders, `load_injecagent_eval_set(cfg)`; schema documented in the module docstring |
| `aase_eval/harmbench.py` | `load_harmbench_behaviors` (CSV columns enforced), `load_jbb_benign`, `load_xstest_safe`, `load_harmbench_eval_set(cfg)` |
| `aase_eval/legacy.py` + `legacy_authored_eval_set.json` | the old 56 + 31 authored prompts, loadable only with `allow_legacy=True` |

### B1 — loader key fix
- `aase_vllm/scripts/benchmark_vllm.py`: rewritten (see below); InjecAgent data now comes from `load_aag_data()` (line 111) → `aase_eval.injecagent.load_injecagent_eval_set`, which reads `"Attacker Instruction"` first and `"attacker_instruction"` second (`aase_eval/injecagent.py` lines 51–65, `_pick` line 73).
- `aase_vllm/scripts/train_probes_vllm.py` lines 483–503: the inline JSONL loop is replaced by `load_attacker_cases(injecagent_dir)`; the probe metadata now records `training_provenance` (line 554) with the real/synthetic counts, so a future `n_positive: 50` can never again be ambiguous. Default is **not** to use InjecAgent in training (`eval_config.yaml aag_training.use_injecagent: false`, CLI `--injecagent-dir`), because the 62 attacker instructions appear verbatim in every one of the 1,054 test cases — training on them contaminates the InjecAgent evaluation. **Flagged for review**: the January probes were trained without InjecAgent (by accident); this keeps that property on purpose.
- Verified in isolation: with the real column the loader yields 62 attacker cases and 1,054 distinct paired prompts; the wrong-key path now raises with both key sets in the message (`tests/test_injecagent_loader.py::test_wrong_key_names_raise_with_found_keys`).

### B2 — real benign side
`user_cases.jsonl` schema (17 rows): `User Tool`, `User Instruction`, `Tool Parameters`, `Tool Response Template`, `Thought`, `Level`; every template contains the literal placeholder `<Attacker Instruction>`. `build_benign_prompts` (`aase_eval/injecagent.py` line 204) removes the placeholder (`injecagent.benign_placeholder_fill`, default empty) and formats with the same template as the injections. Count is config-controlled (`benign_sample_count`, default all 17; seeded sampling when smaller). **Flagged**: InjecAgent has only 17 benign contexts, so FPR resolution is 1/17 ≈ 5.9 %; the paper's "200 benign" never had a real source.

Surface-form finding made during this work: all 1,054 injected `Tool Response` strings are wrapped in a pair of literal double quotes and none of the 17 benign templates are — a perfect label cue. `load_test_cases(strip_wrapping_quotes=True)` strips it by default (`injecagent.strip_wrapping_quotes` in config; raw kept as `tool_response_raw`). Reviewers should decide whether upstream's raw form is preferable.

### B3 — silent fallbacks removed (every site raises)
| File : line | Before | Now |
|---|---|---|
| `aase_vllm/scripts/benchmark_vllm.py` | `"InjecAgent not found, using built-in"` (2 × 31), HarmBench 8 × 7 fallback, 8 × 25 benign literals | no literals in the file; `DataLoadError` propagates out of `run_all_models` (lines 364–405; data is loaded at lines 368–369 before any model) — a data failure aborts the whole run before any model loads |
| `aase_vllm/scripts/train_probes_vllm.py` :493 | silent empty list → synthetic-only | `load_attacker_cases` raises |
| `package/eval_injecagent.py` :42, :54–59, :63 | 62 dicts with empty injection scored; 200 synthetic benign | real paired cases; `format_injection_prompt` raises on an empty injection; benign = 17 real user cases |
| `package/eval_harmbench.py` :47, :59, :295 | embedded 56 + 31 list on any failure; JBB from HF hub with fallback | real HarmBench or raise; JBB from pinned CSVs; authored list only via `--use-legacy-authored-set` |
| `package/retrain_aag_injecagent.py` :38, :292 | wrong key; "Few InjecAgent cases found, using synthetic data only" | real loader; raises if < 10 cases |
| `package/retrain_aag_all_models.py` :120, `package/tune_aag_threshold.py` :31 | wrong key | real loader (raises) |
| `package/debug_injecagent.py` :28, :38, :88–99 | wrong keys (`attacker_instruction`, `attacker_tool`) | real column first, alias second |
| `scripts/benchmarks/benchmark_comparison.py` :193, :215, :239, :256 | four `_get_*_fallback()` returns | `RuntimeError` naming the dataset |
| `scripts/benchmarks/mlcommons_benchmark.py` :90 | `get_fallback_prompts()` | `RuntimeError` |

`tests/test_no_fallback.py` greps the live pipeline for every literal and fallback phrase from the verification report and fails if any returns.

### B4 — ROC generation
- Old script + its outputs moved with `git mv` to `archive/synthetic/` (`generate_roc_curves_vllm_SYNTHETIC.py`, `roc_curves_vllm_data_SYNTHETIC.json`, `roc_curves_vllm_SYNTHETIC.{pdf,png}`) with a README; provenance rows updated; `paper-src/README.md` pointer updated.
- New `aase_vllm/scripts/generate_roc_curves_vllm.py`: consumes `per_case` scores from `benchmark_vllm.py` results (`--results-dir` / `--results`), recomputes AUC (exact Mann–Whitney, no sklearn), plots, writes `<output>_data.json`. Raises on an empty directory, a missing file, or a results file without `per_case` (tests in `tests/test_dry_run.py`).

### B5 — HarmBench
- `third_party/harmbench` = git submodule of `centerforaisafety/HarmBench` at **`8e1604d1171fe8a48d8febecd22f600e462bdcdd`** (recorded in `.gitmodules`, `manifest/third_party.csv`, `manifest/provenance.csv`). Clone with `git submodule update --init` on the VM.
- Real-data path verified through `package/eval_harmbench.py`'s own `load_harmbench_data`: **400 behaviors in `harmbench_behaviors_text_all.csv`** = standard 200 + contextual 100 + copyright 100; with the config default (`functional_categories: [standard, contextual]`) **300 behaviors** load, semantic breakdown: cybercrime_intrusion 67, illegal 65, misinformation_disinformation 65, chemical_biological 56, harassment_bullying 25, harmful 22. Contextual behaviors get their `ContextString` prepended. Copyright behaviors are excluded by default because they ask for verbatim copyrighted text, not harmful content — **flagged for review**.
- Benign counterpart — **choice flagged for review**: default `harmbench.benign_set: jbb_benign` = JailbreakBench JBB-Behaviors benign split (100 prompts, one benign twin per harmful category, same benchmark family the repo already used); alternative `xstest_safe` = XSTest v2 safe prompts (250, deliberately unsafe-looking benign requests — the stricter FPR test). Both pinned as CSVs in `third_party/benign_sets/` with sha256 in the manifests. The hand-written 31 are no longer selectable as a benign set.
- The 87-prompt authored list now lives in `aase_eval/legacy_authored_eval_set.json`; reachable only through `--use-legacy-authored-set` (`benchmark_vllm.py`, `eval_harmbench.py`), records tagged `source: legacy_authored`.

### B6 — config
`eval_config.yaml`: the paper's 7 models, dataset paths, InjecAgent setting/counts/template, HarmBench subset/categories/benign set, AAG training policy, probe dir + per-model layer/threshold overrides, output dir and `record_prompt_text`. `benchmark_vllm.py`, `train_probes_vllm.py`, `eval_harmbench.py`, `eval_injecagent.py` all read it; CLI flags only override.

### `benchmark_vllm.py` rewrite (other changes)
Loads all data once before any model; writes one JSON per model under `results/<run_name>/` with `per_case` records (id, label, score, attack_type/category, sha256 of the prompt, and the prompt text itself when `record_prompt_text: true`), `data_summary`, distinct-score counts (the 5-cluster symptom becomes visible immediately), per-attack-type detection, and an exact AUC. gemma-2-9b is back in the model list (config). Latency benchmark unchanged in method.

---

## Part C — verification gates (all CPU)

| Gate | Test(s) | Result |
|---|---|---|
| C1 attacker/test-case loader: 62 attacker cases; 1,054 distinct real prompts per setting; zero synthetic strings; enhanced ≠ base; quote wrapper stripped | `tests/test_injecagent_loader.py` (6 tests) | pass |
| C1 benign loader returns the 17 real user cases, placeholder removed | `test_benign_user_cases_are_real_and_placeholder_free` | pass |
| C1 missing / renamed file, wrong key, empty file, missing dir → `DataLoadError` (never substitution); snake_case alias accepted | 6 tests | pass |
| C2 HarmBench loads 400 / 300 / val 80 / test 320; wrong columns raise; missing dir raises; benign sets 100 / 250; legacy list refused without flag and unreachable from the HarmBench loader | `tests/test_harmbench_loader.py` (10 tests) | pass |
| B3 no synthetic literals / fallbacks in 11 pipeline files, `scripts/benchmarks` fallbacks gone, synthetic ROC quarantined | `tests/test_no_fallback.py` (14 tests) | pass |
| C3 `--dry-run` full data path with counts + 3 samples per split; injecagent-only; loud failure on a missing dir; legacy flag path; ROC script refuses nothing / no-per_case; ROC consumes real per-case results; AUC helper exact | `tests/test_dry_run.py` (8 tests) | pass |

Run: `python -m pytest -q` from the repo root (`pytest.ini` disables a broken unrelated plugin on this machine).

---

## Files changed / added
Added: `aase_eval/` (6 files), `tests/` (5 files), `eval_config.yaml`, `pytest.ini`, `DEC30_RESULTS.md`, `PHASE0_REPORT.md`, `VERIFICATION_INJECAGENT.md` (from the previous session, now committed), `archive/synthetic/README.md`, `manifest/third_party.csv`, `third_party/benign_sets/` (3 CSVs), `.gitmodules` + submodule `third_party/harmbench`, new `aase_vllm/scripts/generate_roc_curves_vllm.py`.
Modified: `aase_vllm/scripts/benchmark_vllm.py` (rewrite), `aase_vllm/scripts/train_probes_vllm.py`, `package/eval_injecagent.py`, `package/eval_harmbench.py`, `package/retrain_aag_injecagent.py`, `package/retrain_aag_all_models.py`, `package/tune_aag_threshold.py`, `package/debug_injecagent.py`, `scripts/benchmarks/benchmark_comparison.py`, `scripts/benchmarks/mlcommons_benchmark.py`, `manifest/provenance.csv`, `paper-src/README.md`, `.gitignore`.
Moved: 4 synthetic ROC artefacts → `archive/synthetic/`.

## Open items for review before Phase 1
1. Benign set for AF FPR: `jbb_benign` (default) vs `xstest_safe` — or report both.
2. HarmBench `copyright` behaviors excluded by default.
3. InjecAgent benign side is 17 cases (FPR granularity 5.9 %); the quote-wrapper normalisation is on by default.
4. AAG probes must be **retrained** in Phase 1 (existing `aase_vllm/pretrained/aag/*.npy` were trained on synthetic injections only); training stays InjecAgent-free unless `aag_training.use_injecagent` is set.
5. `package/*` HF-era scripts were repaired for keys/fallbacks but not otherwise modernised; the vLLM path is the one wired to the config.


---

# Appendix A — Stage 2: pinned 120-case benign side for AAG (2026-09-02)

Survey: `BENIGN_SOURCES.md` (Stage 1) with the build accounting appended. Dataset: `datasets/aag_benign_eval.jsonl`
(120 cases, sha256 `f5dcfe9dfed4b3ef…`, schema in `datasets/README.md`). Test suite now **56 passed**.

| Change | Where |
|---|---|
| Builder (deterministic; exclusions, stratification, backfill, contamination + duplicate guards) | `aase_eval/build_benign_set.py` |
| AgentDojo extraction (run once in a venv with `agentdojo==0.1.35`, MIT; not a repo dependency) → pinned raw dump | `aase_eval/extract_agentdojo.py`, `datasets/sources/agentdojo_v1.2.2_extract.json` |
| 40 authored cases (26 hard negatives over 8 kinds, 14 plain) with author + `reviewed_by: null` provenance | `aase_eval/authored_benign_cases.py` |
| Loader with full schema validation; `fpr_breakdown()` (overall / per subset / hard-negative-only / plain-only) | `aase_eval/benign_set.py` |
| `injecagent.benign_source: aag_benign_eval` (default) / `injecagent_user17`; optional `benign_subsets` filter | `eval_config.yaml`, `aase_eval/injecagent.py` (`load_injecagent_eval_set`) |
| Results JSON: `details.fpr_breakdown` four ways, benign `subset`/`hard_negative` carried per case; dry run prints stratified samples | `aase_vllm/scripts/benchmark_vllm.py` (`run_injecagent`, `dry_run`, `--sample-width`) |
| Loader now also unescapes the `\"` sequences upstream's wrapper introduced (124 across the base set); every positive is asserted equal to its template-with-slot in both settings | `aase_eval/injecagent.py::_strip_wrapping_quotes`, `tests/test_injecagent_loader.py` |
| Tests: schema, counts, stratification, hard-negative tagging (found vs authored), no duplicates, no attacker-instruction contamination (verbatim + normalised), scaffold/format parity with positives, config wiring, loader raises on missing/malformed, FPR breakdown, deterministic rebuild | `tests/test_benign_set.py` (11 tests) |
| Manifests: dataset + build accounting + extraction pinned; AgentDojo row | `manifest/provenance.csv`, `manifest/third_party.csv` |

Amendments honoured: (1) 10 natural + 26 authored = 36 tagged hard negatives, provenance distinguishes them
(`hard_negative_detection` vs `author`); (2) exclusions recorded with counts and backfill in `BENIGN_SOURCES.md`;
(3) four-way FPR in the results schema; (4) scaffold-identity test for authored + injecagent_user cases, AgentDojo
formatting differences documented per case in `format_note` and in `datasets/README.md`.

Open for review: the 40 authored cases are unreviewed by a human (`provenance.reviewed_by` is null); AgentDojo's
fictional IBANs, a "Secret key" Slack message and password-reset emails were kept as legitimate benign content.
