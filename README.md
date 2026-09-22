# Diagnosing and Repairing Minimal-Data Activation Probes for LLM Safety

Code, data, probes, pre-registered thresholds and provenance-stamped results for the paper of that name
(submitted to *IEEE Access*, 2026). Everything in the manuscript is generated from the files here: no number in
the paper is typed by hand.

## What the paper shows

Linear probes on language-model activations are cheap enough to run on every request, and a minimal-data recipe —
tens of contrastive examples — is cheap enough that many people will try it. The question is what has to be true of
an evaluation before a number computed on one means what it appears to mean.

Across seven open models and three tasks, reported outcomes turn out to be dominated by two things most papers
never report: **extraction convention** (which text the activation is read from, at which position, from which
tensor) and **evaluation design** (what the benign side contains, whether length is controlled, whether the split
leaks).

Two failures are repaired by changing where the activation is read, not how the probe is trained. Three claims are
retired: paired contrastive construction is algebraically the difference-of-means estimator; obfuscation robustness
is manufactured by length or by training on the encodings being tested; and short-text policy-compliance
evaluations cannot certify a semantic claim for any classifier, because removing the lexical cue removes the label.

## Verifying this artifact

The checks the paper relies on run on a laptop, in seconds, with no GPU and no model weights:

```bash
python -m pytest tests/ -q          # 119 tests
python -m gates.signoff             # Stage A sign-off pins, byte-compared
```

`tests/test_results_completeness.py` enforces the rule that came out of losing a run's scores: no results directory
may report an AUC without the per-case scores behind it. Activations may be gitignored; scores may not.

Every results file carries a provenance block — git commit, config sha256, dataset sha256s, package versions,
CUDA and GPU — and the dataset hashes recorded there match the files shipped beside them.

## Rebuilding the paper's numbers

```bash
cd paper
make tables      # 20 tables, from results/ only; fails if a declared source is missing
make figures     # both figures, byte-deterministic
make stats       # bootstrap confidence intervals, 2,000 resamples at a recorded seed
make check       # sources, references, markers, and that the stamp is not dirty
```

`make` then builds the PDF, if you supply `ieeeaccess.cls` and `IEEEtran.bst` from the IEEE Access author portal.
Those are not redistributable and are therefore not here; the build tells you where to get them.

## Layout

| Path | What it is |
|---|---|
| `paper/` | Manuscript source. `tools/gen_tables.py`, `gen_figures.py` and `gen_stats.py` generate every table, figure and interval from `results/`. |
| `results/gates/` | The three-arm study. `b1` harmful content, `b2*` prompt injection (per-case, grouped and templated splits), `b3` policy compliance, `scores/` the per-case scores behind every Arm 2 AUC. |
| `results/phase2/` | Obfuscation: the layer sweep, the confound controls, the pooled-read study, and the training-circularity reconstruction. |
| `results/phase1/` | The original benchmark runs, including the Llama Guard comparison and latency measurements. |
| `gates/` | Stage A/B harness, the authored evaluation data, and `signoff.py` — the sign-off record with byte-compare pins. |
| `phase2/` | Obfuscation sweep, confound analysis and pooled reads. |
| `datasets/` | The constructed agent benign set (120 cases, 36 hard negatives) and its builder. |
| `aase_vllm/pretrained/` | The shipped probe direction vectors the paper evaluates. |
| `runtime/`, `package/`, `scripts/`, `archive/` | The original research code as it ran, kept for provenance rather than reuse. |
| `manifest/` | Where every file in this repository came from, with hashes. |

## The pre-registration

`GATES.md` holds the thresholds, fixed before any GPU work, with every superseded sentence struck through rather
than deleted, and seven amendments. `GATES_REPORT.md` records what the runs returned. `DECISIONS.md` logs the calls
made along the way.

Three pre-registered designs could not answer their own questions — a split that leaked injection strings, a
comparison whose two arms were algebraically identical, and an instrument whose validity dissolved as its confounds
were controlled. They are reported in the paper as findings. This is what the amendment mechanism is for.

## Two warnings

**The policy-compliance instrument drafts in `gates/data/` are not valid evaluation sets.** They are released as
artifacts of a construction failure: each drives one lexical cue to chance and is separated by the next. Draft 3
dissolves the labels themselves. Do not use them to evaluate anything.

**Benchmark artifacts.** InjecAgent's injected tool responses are quote-wrapped where the benign templates are not,
a perfect label cue unrelated to injection, and its 1,054 cases cross only 62 attacker instructions with 17 user
instructions, so a case-level split measures recognition rather than generalisation. Both are reported upstream
(uiuc-kang-lab/InjecAgent#7). The harness normalises the first and splits on the second.

## Not included

Raw activation tensors (hundreds of GB, regenerable from the code and the pinned inputs), the IEEE template files,
and the vendored HarmBench tree beyond its behaviour datasets.

## License and attribution

Third-party code under `runtime/aag/InjecAgent/` and `third_party/` is vendored as its authors wrote it and carries
their licenses. Benchmarks used: HarmBench, XSTest, JailbreakBench, InjecAgent, AgentDojo, AILuminate.

An AI assistant built the evaluation harness and ran the experiments under the author's pre-registered protocol;
the author designed the study, made every decision and sign-off, and verified all results. The paper's AI-use
disclosure states this in full. Some authored data files record it per record, in an `author` field.
