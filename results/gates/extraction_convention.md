### Exhibit — extraction convention: bare prompt vs the model's own chat template

Same 300 HarmBench behaviors and 350 benign prompts (XSTest safe 250 + JailbreakBench benign 100), same layer
per model, same feature (`mlp_out`, the space every shipped probe was trained in). The only thing that
changes between the two halves of each row is whether the activation is read at the final token of the bare
prompt or of the prompt wrapped in that model's chat template.

`AUC*` is length-stratified (the headline number GATES.md judges). `JBB FPR@m` is the JailbreakBench-benign
false-positive rate at the threshold whose XSTest FPR matches the January probe's XSTest FPR for that model,
so the FPR columns are comparable across probes rather than reflecting a threshold shift.

| Model | Layer | Probe | AUC* raw | AUC* templated | Δ | AUC vs JBB raw | AUC vs JBB templated | JBB FPR@m raw | JBB FPR@m templated |
|---|---|---|---|---|---|---|---|---|---|
| gemma-2-2b-it | 13 | january | 0.754 | 0.875 | +0.121 | 0.485 | 0.717 | 0.80 | 0.04 |
| gemma-2-2b-it | 13 | paired | 0.663 | 0.972 | +0.309 | 0.825 | 0.909 | 0.00 | 0.00 |
| gemma-2-2b-it | 13 | offtopic | 0.828 | 0.979 | +0.152 | 0.690 | 0.910 | 0.67 | 0.00 |
| gemma-2-9b-it | 21 | january | 0.841 | 0.982 | +0.142 | 0.640 | 0.937 | 0.76 | 0.05 |
| gemma-2-9b-it | 21 | paired | 0.645 | 0.983 | +0.338 | 0.800 | 0.906 | 0.03 | 0.07 |
| gemma-2-9b-it | 21 | offtopic | 0.827 | 0.982 | +0.155 | 0.824 | 0.888 | 0.43 | 0.10 |
| gemma-3-1b-it | 13 | january | 0.689 | 0.706 | +0.017 | 0.539 | 0.719 | 0.89 | 0.00 |
| gemma-3-1b-it | 13 | paired | 0.619 | 0.923 | +0.304 | 0.789 | 0.865 | 0.05 | 0.01 |
| gemma-3-1b-it | 13 | offtopic | 0.637 | 0.935 | +0.298 | 0.600 | 0.866 | 0.68 | 0.04 |
| gemma-3-4b-it | 17 | january | 0.703 | 0.687 | -0.016 | 0.578 | 0.613 | 0.29 | 0.02 |
| gemma-3-4b-it | 17 | paired | 0.676 | 0.972 | +0.296 | 0.603 | 0.868 | 0.22 | 0.00 |
| gemma-3-4b-it | 17 | offtopic | 0.692 | 0.977 | +0.285 | 0.592 | 0.862 | 0.26 | 0.00 |
| Llama-3.1-8B-Instruct | 16 | january | 0.770 | 0.387 | -0.383 | 0.456 | 0.454 | 0.99 | 0.00 |
| Llama-3.1-8B-Instruct | 16 | paired | 0.595 | 0.974 | +0.379 | 0.769 | 0.933 | 0.00 | 0.02 |
| Llama-3.1-8B-Instruct | 16 | offtopic | 0.767 | 0.977 | +0.209 | 0.588 | 0.936 | 0.96 | 0.05 |
| Llama-3.2-1B-Instruct | 8 | january | 0.773 | 0.753 | -0.019 | 0.672 | 0.878 | 0.91 | 0.70 |
| Llama-3.2-1B-Instruct | 8 | paired | 0.547 | 0.955 | +0.408 | 0.770 | 0.878 | 0.00 | 0.75 |
| Llama-3.2-1B-Instruct | 8 | offtopic | 0.780 | 0.949 | +0.169 | 0.754 | 0.864 | 0.57 | 0.99 |
| Llama-3.2-3B-Instruct | 14 | january | 0.757 | 0.959 | +0.202 | 0.531 | 0.947 | 0.99 | 0.02 |
| Llama-3.2-3B-Instruct | 14 | paired | 0.437 | 0.980 | +0.544 | 0.702 | 0.928 | 0.00 | 0.05 |
| Llama-3.2-3B-Instruct | 14 | offtopic | 0.778 | 0.981 | +0.204 | 0.666 | 0.923 | 0.97 | 0.24 |

**Template effect** (paired probe, length-stratified AUC, templated − raw): min +0.296, max +0.544, mean +0.368 over 7 models — positive on every model.
**On-topic-twins effect** (templated, AUC vs JBB, paired − offtopic): min -0.003, max +0.018, mean +0.005 — indistinguishable from zero.

Read together: the confound Phase 1 found is a property of the extraction convention, not of the training-set
construction. Swapping the negatives for on-topic twins changes nothing; reading the same text through the
chat template changes everything.
