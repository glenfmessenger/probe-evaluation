# PHASE1_REPORT.md — Phase 1 GPU evaluation (A100 VM, 2026-09-03)

Branch `phase1-results` (from `eval-fix`). Every number below comes from a results JSON on that branch that carries a
run-provenance block (git commit, `eval_config.yaml` sha256, dataset sha256s, package versions). Per-leg logs and the
running log are in `results/phase1/SESSION_LOG.md` and `logs/`. Nothing here was produced by a synthetic or fallback data path:
the loaders raise instead, and every leg's counts were checked against the dry-run expectations (`tools/check_leg.py`).

## 1. Environment record

| Item | Value |
|---|---|
| Hardware | 1× NVIDIA A100-SXM4-40GB (the paper's numbers were on an NVIDIA L4 24 GB) · 30 vCPU · 216 GiB RAM |
| OS / driver / CUDA | Ubuntu 22.04.5 · driver 580.105.08 · CUDA 12.8 |
| Python | 3.11.16 (uv-managed venv; system Python is 3.10.12) |
| Stack | vllm 0.13.0 · torch 2.9.0 (cu128) · transformers 4.57.6 · numpy 2.2.6 · bitsandbytes 0.50.2 — `results/phase1/env_resolved.txt` (184 packages) |
| Deviation from `requirements-pinned.txt` | none (name-spelling differences only; see SESSION_LOG S2) |
| Deviation from the paper's stated stack | torch 2.1→2.9.0, transformers 4.36→4.57.6, bitsandbytes 0.41→0.50.2 — forced by vLLM 0.13.0's own pins; the paper's stack was never co-installable (PREFLIGHT_REPORT.md C) |
| Data | InjecAgent @ f19c9f2 (1,054 base + 1,054 enhanced, 120-case pinned benign set), HarmBench @ 8e1604d (300 = standard 200 + contextual 100), XSTest v2 safe 250, JBB benign 100, AILuminate v1.0 DEMO 1,200 — sha256s in each results file |

## 2. Deviations log (things that changed during the session)

| # | What | Why | Where |
|---|---|---|---|
| 1 | `eval-fix` pushed to GitHub from the laptop before cloning | the repo was empty | S1 |
| 2 | `vllm/` directory renamed to `aase_vllm/` | it shadowed the installed vLLM package on `sys.path`; `import vllm` failed inside the harness | commit `076354c` |
| 3 | Stop rule refined: exact 0 %/100 % is non-blocking when scores are non-degenerate | your decision after R3 leg 4 (Llama-3.2-3B, XSTest FPR 0/250 with 250 distinct scores) | `tools/check_leg.py`, commit `3e2e180`; four EXACT-RATE events logged in SESSION_LOG |
| 4 | Results written one file per (benchmark, model) | so legs never overwrite each other | `aase_vllm/scripts/benchmark_vllm.py` |
| 5 | AILuminate benign-encoded control added (XSTest + JBB under the same encodings) | requested for R4 | `aase_eval/ailuminate.py` |

## 3. R1 — AAG probes retrained InjecAgent-free (7 models)

Training set per model: 50 synthetic injections (10 actions × 5 templates), 100 safe prompts (25 distinct, repeated), no InjecAgent text
(`training_provenance.n_real_injecagent_instructions = 0` on every probe). Training-set separation: gemma-2-9b 3.76σ, Llama-3.1-8B 3.34σ,
gemma-2-2b 2.87σ, Llama-3.2-3B 2.79σ, Llama-3.2-1B 2.16σ, gemma-3-1b 1.24σ, gemma-3-4b 0.80σ. The Gemma-2/Llama values reproduce the
January probe metadata to three decimals, confirming the January AAG probes were trained on this same synthetic set; the Gemma-3 values do not.

## 4. R2 — AAG on real InjecAgent (2,108 positives: 1,054 base + 1,054 enhanced; 120 real benign cases)

Thresholds are the training thresholds. Per-case scores, ROC (`results/phase1/injecagent/roc_curves_injecagent.pdf`) and the
four-way FPR breakdown are in the results files.

| Model | AAG train σ (real 0 / synth 50) | Det. base | Det. enhanced | Det. overall | FPR (120) | FPR hard-neg | AUC | Paper: AAG Det. / AUC |
|---|---|---|---|---|---|---|---|---|
| google/gemma-2-9b-it | 3.76 | 62.2% | 84.9% | 73.6% | 47.5% | 38.9% | 0.710 | 100% / 1.00 |
| google/gemma-2-2b-it | 2.87 | 36.7% | 58.8% | 47.8% | 39.2% | 22.2% | 0.581 | 100% / 0.96 |
| google/gemma-3-1b-it | 1.24 | 40.8% | 40.4% | 40.6% | 36.7% | 33.3% | 0.517 | 100% / 1.00 |
| google/gemma-3-4b-it | 0.80 | 0.1% | 2.8% | 1.4% | 0.8% | 0.0% | 0.697 | 100% / 1.00 |
| meta-llama/Llama-3.1-8B-Instruct | 3.34 | 99.2% | 99.6% | 99.4% | 95.8% | 100.0% | 0.703 | 100% / 1.00 |
| meta-llama/Llama-3.2-1B-Instruct | 2.16 | 39.2% | 48.5% | 43.8% | 29.2% | 30.6% | 0.635 | 100% / 1.00 |
| meta-llama/Llama-3.2-3B-Instruct | 2.79 | 77.5% | 81.9% | 79.7% | 57.5% | 52.8% | 0.719 | 100% / 0.88 |

## 5. W0 — AAG threshold recalibration (CPU, from the R2 scores; `results/phase1/injecagent/recalibration.md`)

Stratified 50/50 dev/test (positives by setting, benign by subset), seed 20260903; threshold = argmax Youden J on dev; test-half rates.

| Model | AUC | Orig thr → test det / FPR | Recal thr → test det / FPR (hard-neg) | Recal FPR on all 120 benign | Verdict |
|---|---|---|---|---|---|
| google/gemma-2-9b-it | 0.710 | 72.4% / 42.6% | 60.1% / 34.4% (29%) | 27.5% | separability-limited |
| google/gemma-2-2b-it | 0.581 | 47.8% / 39.3% | 71.2% / 52.5% (35%) | 52.5% | separability failure |
| google/gemma-3-1b-it | 0.517 | 41.2% / 37.7% | 63.2% / 60.7% (47%) | 60.0% | separability failure |
| google/gemma-3-4b-it | 0.697 | 0.9% / 1.6% | 58.7% / 32.8% (29%) | 26.7% | mixed |
| meta-llama/Llama-3.1-8B-Instruct | 0.703 | 99.3% / 96.7% | 35.2% / 8.2% (6%) | 9.2% | calibration failure |
| meta-llama/Llama-3.2-1B-Instruct | 0.635 | 42.0% / 34.4% | 80.4% / 60.7% (76%) | 56.7% | mixed |
| meta-llama/Llama-3.2-3B-Instruct | 0.719 | 80.2% / 57.4% | 62.4% / 29.5% (29%) | 23.3% | separability-limited |

Calibration failure vs separability failure: only Llama-3.1-8B is a calibration failure (AUC 0.70; its training threshold sits below almost
every real benign score, giving 99 %/97 %; recalibrated it reaches 35 % detection at 8 % FPR). gemma-2-2b and gemma-3-1b are separability
failures (AUC < 0.6). The remaining four have AUC 0.64–0.72: no threshold yields a useful operating point (best test points ≈ 60 % detection
at 30 % FPR). **Models below 0.6 AUC even calibration-free: google/gemma-2-2b-it, google/gemma-3-1b-it.**

## 6. R3 — AF on HarmBench 300 vs XSTest 250 + JBB benign 100 (January AF probes, training thresholds)

| Model | Det. HarmBench 300 | FPR XSTest 250 | FPR JBB 100 | AUC (all 350) | AUC vs XSTest / vs JBB | Paper: AF Det. / AUC / σ |
|---|---|---|---|---|---|---|
| google/gemma-2-9b-it | 83.3% | 5.6% | 76.0% | 0.877 | 0.971 / 0.641 | 100% / 1.00 / 7.2σ |
| google/gemma-2-2b-it | 78.0% | 0.8% | 84.0% | 0.833 | 0.973 / 0.484 | 88% / 1.00 / 4.3σ |
| google/gemma-3-1b-it | 91.0% | 16.8% | 92.0% | 0.832 | 0.949 / 0.540 | 100% / 1.00 / 5.1σ |
| google/gemma-3-4b-it | 39.0% | 3.2% | 30.0% | 0.810 | 0.903 / 0.580 | 62% / 0.88 / 2.4σ |
| meta-llama/Llama-3.1-8B-Instruct | 89.0% | 0.0% | 94.0% | 0.844 | 0.999 / 0.456 | 88% / 1.00 / 6.8σ |
| meta-llama/Llama-3.2-1B-Instruct | 98.7% | 0.4% | 91.0% | 0.906 | 1.000 / 0.671 | 88% / 0.98 / 3.9σ |
| meta-llama/Llama-3.2-3B-Instruct | 91.7% | 0.0% | 94.0% | 0.863 | 0.996 / 0.530 | 88% / 0.97 / 4.1σ |

ROC: `results/phase1/harmbench/roc_curves_harmbench.pdf`. The XSTest-vs-JBB split is the same on every model: near-zero FPR on XSTest
(short benign questions containing dangerous-looking words) but 30–94 % FPR on JBB benign (on-topic benign twins of harmful behaviors),
with AUC vs XSTest 0.90–1.00 and AUC vs JBB 0.46–0.67. The AF probes separate *topic* from *not-topic*, not harmful from benign.

## 7. R4 — AILuminate obfuscation, gemma-2-9b (January AF probe) with the benign-encoded control

| Encoding | Det. (AF, gemma-2-9b) | Paper AF | FPR XSTest-enc. | FPR JBB-enc. | AUC vs XSTest-enc. | AUC vs JBB-enc. |
|---|---|---|---|---|---|---|
| none | 47.8% | 100% | 5.6% | 76.0% | 0.850 | 0.343 |
| base64 | 4.0% | 100% | 0.0% | 28.0% | 0.648 | 0.400 |
| rot13 | 3.2% | 100% | 0.0% | 67.0% | 0.494 | 0.039 |
| leetspeak | 46.5% | 86.3% | 7.2% | 71.0% | 0.786 | 0.355 |

Plain-text detection is 47.8 % (paper 100 %). Under Base64 and ROT13 the score distribution collapses (harmful std 0.063 → 0.026 → 0.013)
and detection is 3–4 % (paper 100 %); harmful-vs-benign AUC under the same encoding is ≈ chance vs XSTest and below chance vs JBB.
The probe does not read decoded harm through these encodings. Leetspeak behaves like plain text (46.5 % vs paper 86.3 %).

## 8. R5 — Llama Guard 3 vs AASE (HarmBench 300 + XSTest 250 + JBB 100)

| System | n | TP / FP / FN / TN | Accuracy | TPR | FPR | Precision | F1 | Latency ms (mean ± sd; p50, p99) | FPR XSTest / JBB |
|---|---|---|---|---|---|---|---|---|---|
| AASE AF probe + Llama-3.2-3B (January probe, training threshold) | 650 | 275 / 94 / 25 / 256 | 81.7% | 91.7% | 26.9% | 0.745 | 0.822 | 16.2 ± 1.2 (16.0, 21.1) | 0.0% / 94.0% |
| Llama Guard 3 (8B), guard chat template | 650 | 289 / 31 / 11 / 319 | 93.5% | 96.3% | 8.9% | 0.903 | 0.932 | 64.7 ± 21.5 (47.6, 107.0) | 3.2% / 23.0% |

Same 650 prompts for both (HarmBench 300 + XSTest 250 + JBB 100), 3 latency runs each, per-case verdicts in
`results/phase1/llama_guard_comparison.json` (commit `ad7d4bd`). Speedup 4.0× (A100). Paper Table 5 (16 authored prompts, L4):
Llama Guard 75 % / 50 % / 0 % / F1 0.67 at 306 ms vs AASE 81 % / 88 % / 25 % / 0.82 at 33 ms, 9×. The AASE row is, by coincidence,
close to the paper's; the Llama Guard row is not — prompted through its own template on a real benchmark, Llama Guard 3 is the
more accurate system by every metric, and 94 of AASE's 94 false positives are JBB-benign prompts.

## 9. R6 — Latency (A100, batch size 1)

| Model | Forward pass + hook, mean ± sd ms (p50 / p99) | Probe dot-product overhead mean / p99 ms | n |
|---|---|---|---|
| google/gemma-2-9b-it | 50.0 ± 0.2 (50.0 / 50.3) | 0.0013 / 0.0014 | 50 |
| google/gemma-2-2b-it | 31.5 ± 0.2 (31.5 / 32.1) | 0.0011 / 0.0015 | 50 |
| google/gemma-3-1b-it | 41.5 ± 0.2 (41.5 / 42.2) | 0.0010 / 0.0014 | 50 |
| google/gemma-3-4b-it | 61.2 ± 1.6 (60.8 / 68.9) | 0.0013 / 0.0021 | 50 |
| meta-llama/Llama-3.1-8B-Instruct | 17.7 ± 0.2 (17.7 / 18.1) | 0.0015 / 0.0025 | 50 |
| meta-llama/Llama-3.2-1B-Instruct | 10.1 ± 0.1 (10.1 / 10.5) | 0.0012 / 0.0014 | 50 |
| meta-llama/Llama-3.2-3B-Instruct | 15.9 ± 0.2 (15.8 / 16.5) | 0.0014 / 0.0023 | 50 |

Methodology as in the paper's `benchmark_vllm.py --benchmark latency`: batch size 1, a fixed prompt, `max_tokens=1`, `enforce_eager=True`,
5 warm-up + 50 timed generations with the activation hook registered; probe overhead = 1,000 timed dot products of a unit vector with the
AF direction. Llama Guard 3 (8B) on the same hardware, its own template, `max_tokens=20`: 64.7 ± 21.5 ms
(p50 47.6, p99 107.0) over 650 × 3 prompts; AASE + Llama-3.2-3B on the same 650 prompts 16.2 ms.
**Hardware change:** the paper reported an NVIDIA L4 (24 GB); these are A100-SXM4-40GB numbers under vLLM 0.13.0 / torch 2.9.0 and are
not directly comparable in absolute terms (the paper's 306 ms Llama Guard and 19–92 ms AASE figures were L4 numbers). The relative claim
survives at a smaller ratio: probe overhead is ~0.001 ms, and AASE-on-3B is 4.0× faster than Llama Guard 3 here versus the paper's 9×.

## 10. Results that differ materially from the withdrawn paper (input to the rewrite)

Each row: the paper's claim → what Phase 1 measured on real, pinned data with provenance. AAG rows use the retrained,
InjecAgent-free probes; AF rows use the January AF probes unchanged.

| # | Paper claim (withdrawn) | Phase 1 measurement | Section |
|---|---|---|---|
| 1 | AAG: "100 % detection (62/62 injections) across all 7 models at FPR ≤ 12 %; AUC 0.88–1.00" on InjecAgent | On 2,108 real InjecAgent cases vs 120 real benign contexts: AUC 0.52–0.72; at the training thresholds detection 1–99 % with FPR 1–96 %; the only model detecting ≥ 99 % (Llama-3.1-8B) flags 96 % of benign and 36/36 hard negatives | §4 |
| 2 | AAG works calibration-free (implicit in the AUC claim) | Two models are below 0.6 AUC threshold-free (gemma-2-2b 0.58, gemma-3-1b 0.52); four sit at 0.64–0.72 with no useful operating point; only Llama-3.1-8B is a pure calibration failure (recalibrated: 35 % detection at 8 % FPR) | §5 |
| 3 | AAG "62 injections, 200 benign" were InjecAgent data | The 62/200 in the paper were synthetic fallback prompts (VERIFICATION_INJECAGENT.md); Phase 1 is the first real-data AAG measurement on vLLM | §4 |
| 4 | AF: 88–100 % HarmBench detection on 6 of 7 models, AUC 0.97–1.00, "0 % FPR" (gemma-2-9b) | HarmBench 300: 39–99 % detection (gemma-3-4b 39 %); AUC vs all 350 benign 0.81–0.91; FPR 0–17 % on XSTest **but 30–94 % on JBB benign** | §6 |
| 5 | AF separates harmful from benign | AUC vs XSTest 0.90–1.00 but vs JBB benign 0.46–0.67 on every model: the probes separate topic, not harmfulness; on-topic benign requests are flagged at the harmful rate | §6 |
| 6 | AF on AILuminate (gemma-2-9b): plain 100 %, Base64 100 %, ROT13 100 %, leetspeak 86.3 % | Plain 47.8 %, Base64 4.0 %, ROT13 3.2 %, leetspeak 46.5 %; under Base64/ROT13 the harmful-vs-benign AUC is ≈ chance or inverted (0.65/0.49 vs XSTest, 0.40/0.04 vs JBB) | §7 |
| 7 | "The model internally decodes obfuscated content, creating activation patterns indistinguishable from plaintext" | Not supported: encoded prompts collapse the AF score distribution (harmful std 0.063 → 0.013) instead of reproducing the plaintext pattern | §7 |
| 8 | Llama Guard 3: 75 % accuracy, 50 % TPR, 0 % FPR, F1 0.67 (16 authored prompts) | On 650 real prompts with the guard's own template: 93.5 % / 96.3 % / 8.9 % / F1 0.93 — Llama Guard 3 beats the AASE probe (81.7 % / 91.7 % / 26.9 % / 0.82) on every metric | §8 |
| 9 | 3–16× faster than Llama Guard 3 (306 ms vs 19–92 ms, L4) | A100: Llama Guard 64.7 ms vs AASE-on-3B 16.2 ms → 4.0×; per-model forward passes 10–61 ms; probe overhead ≈ 0.001 ms (this part holds) | §9 |
| 10 | Gemma-2-9B "AUC 1.00 with 7.2σ across all probes", 0 % FPR on AF and AAG | AF AUC 0.88 (76 % FPR on JBB benign), AAG AUC 0.71 (47.5 % FPR); the 7.2σ is a 40-prompt training-set separation, not an evaluation statistic | §4, §6 |
| 11 | Cross-architecture "100 % accuracy on both architectures" | Training-set numbers reproduce (R1) but do not transfer to any real benchmark; the January AAG probes were provably trained on the same 50-prompt synthetic set | §3 |

**(a) XSTest-vs-JBB FPR split per model (AF):** gemma-2-9b 5.6 % / 76 %; gemma-2-2b 0.8 % / 84 %; gemma-3-1b 16.8 % / 92 %; gemma-3-4b 3.2 % / 30 %;
Llama-3.1-8B 0.0 % / 94 %; Llama-3.2-1B 0.4 % / 91 %; Llama-3.2-3B 0.0 % / 94 %. The same split appears in R5 (AASE 0 % / 94 % vs Llama Guard 3.2 % / 23 %).

**(b) AAG models below 0.6 AUC even in the calibration-free view:** google/gemma-2-2b-it (0.581) and google/gemma-3-1b-it (0.517).
gemma-3-4b (0.697), Llama-3.1-8B (0.703), gemma-2-9b (0.710) and Llama-3.2-3B (0.719) clear 0.6 but no threshold gives better than
≈ 60 % detection at ≈ 30 % FPR on the held-out half; Llama-3.2-1B is 0.635.

What still holds: probe overhead is negligible; the pipeline is reproducible end to end with provenance; the AF probes carry a real,
topic-level signal (AUC 0.81–0.91 vs mixed benign, 0.90–1.00 vs XSTest) that a rewrite can describe honestly as such.

