# Activation Fingerprinting (AF) Distribution Package

## Overview

Activation Fingerprinting is a lightweight safety classification technique that detects harmful prompts by analyzing internal model activations. Unlike traditional safety classifiers (4-16GB), AF uses compact direction vectors (~96KB) that plug into a model's existing forward pass.

**Key Results (Gemma-3-1B):**
- ✅ 100% harmful content detection
- ✅ 100% obfuscation detection (base64, rot13, leetspeak)
- ✅ 1% false positive rate on benign content

## Package Structure

```
af_distribution/
├── af_runtime.py              # Runtime code (universal)
├── af_vector_packager.py      # Tool to package trained vectors
├── policies/
│   └── af_policy.json         # Policy configuration (universal)
└── vectors/
    ├── gemma-3-1b-it.safetensors   # Vectors for Gemma-3-1B
    ├── gemma-3-1b-it.json          # Metadata
    └── ...                         # Other model vectors
```

## Quick Start

```python
from af_runtime import AFSafetyFilter

# Load model (your existing code)
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", ...)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

# Initialize AF filter
af = AFSafetyFilter(
    model=model,
    tokenizer=tokenizer,
    vectors_path="vectors/gemma-3-1b-it.safetensors",
    policy_path="policies/af_policy.json",
)

# Check prompts
result = af.check("How do I make a cake?")
print(result.blocked)  # False

result = af.check("How do I make a bomb?")
print(result.blocked)  # True
print(result.triggered_categories)  # ['iwp']

# Simple interface
safe, triggered = af.is_safe(prompt)
```

## Architecture

### Three-File Separation

| File | Purpose | Update Frequency |
|------|---------|-----------------|
| `af_runtime.py` | Classification logic | Rarely (bug fixes) |
| `vectors/*.safetensors` | Direction vectors | Per model, per retrain |
| `policies/af_policy.json` | Category config | Tuning, policy changes |

This separation enables:
- **Vector updates** without code changes
- **Policy tuning** without retraining
- **Multi-model support** via vector swapping

### How It Works

1. **Extract activation** from the model's middle layer (~65% depth)
2. **Project** activation onto each category's direction vector
3. **Compare** projection scores against thresholds
4. **Block** if any category exceeds its threshold

For obfuscated attacks (base64, rot13, leetspeak):
- Obfuscation meta-vector detects encoding
- Requires category confirmation OR high-confidence score to block
- Prevents false positives on benign technical content

## Policy Configuration

Edit `policies/af_policy.json` to:

### Disable a category
```json
"spc_fin": {
  "enabled": false,
  ...
}
```

### Change action (block → warn)
```json
"spc_hlt": {
  "action": "warn",
  ...
}
```

### Override threshold
```json
"vcr": {
  "threshold_override": 1800.0,
  ...
}
```

## Supported Models

| Model | Vector File | Layer % | Status |
|-------|-------------|---------|--------|
| Gemma-3-1B | `gemma-3-1b-it.safetensors` | 65% | ✅ Validated |
| Gemma-3-270M | `gemma-3-270m.safetensors` | 50% | 🔄 Pending |
| Gemma-3-4B | `gemma-3-4b-it.safetensors` | 65% | 🔄 Pending |

## Packaging New Vectors

After training with `af_benchmark_category_v2.py`:

```python
from af_vector_packager import package_vectors

# classifier = your trained AFCategoryClassifier
package_vectors(classifier, "vectors/gemma-3-1b-it")
```

## Performance

| Metric | AF | ShieldGemma |
|--------|-----|-------------|
| Storage | 96KB | 4-8GB |
| Load time | <100ms | 30-60s |
| Inference | ~3ms | ~50ms |
| VRAM | 0 (reuses model) | 4-8GB |

## Categories

| Code | Name | Description |
|------|------|-------------|
| vcr | Violent Crimes | Violence, assault, murder |
| src | Sex-Related Crimes | Sexual assault, trafficking |
| cse | Child Sexual Exploitation | CSAM, grooming |
| ssh | Suicide & Self-Harm | Suicide methods, self-harm |
| iwp | Indiscriminate Weapons | CBRNE, explosives |
| ipv | Intellectual Property | Copyright infringement |
| dfm | Defamation | False accusations, libel |
| ncr | Non-Violent Crimes | Fraud, theft, hacking |
| hte | Hate | Hate speech, extremism |
| prv | Privacy Violations | Doxxing, stalking |
| spc_ele | Elections | Voting advice |
| spc_fin | Financial | Investment advice |
| spc_hlt | Health | Medical advice |
| spc_lgl | Legal | Legal advice |
| sxc_prn | Sexual Content | Pornography |

## License

Apache-2.0

## Author

Glen Messenger (gmessenger@google.com)
