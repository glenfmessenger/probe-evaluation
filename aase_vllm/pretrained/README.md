# AASE Pretrained Direction Vectors

This directory contains pre-trained direction vectors for various models.

## Directory Structure

```
pretrained/
├── af/          # Activation Fingerprinting (harmful content)
├── aag/         # Agent Action Gating (prompt injection)
└── apc/         # Activation Policy Compliance (enterprise policies)
```

## File Naming Convention

Files are named: `{model_name}_{policy}.{ext}`

For example:
- `gemma_2_2b_it.npy` - Direction vector
- `gemma_2_2b_it.json` - Metadata (threshold, layer, etc.)

## Adding Pre-trained Vectors

After training a probe, save it to the appropriate directory:

```python
from aase import AF

af = AF()
af.train(harmful_prompts, benign_prompts, extractor)
af.save("aase/pretrained/af/gemma_2_2b_it")
```

## Loading Pre-trained Vectors

```python
from aase import AF

# Load specific model
af = AF.load("aase/pretrained/af/gemma_2_2b_it")

# Or use from_pretrained (searches pretrained dir)
af = AF.from_pretrained("google/gemma-2-2b-it")
```

## Validated Models

The following models have been validated:

| Model | AF | AAG | APC |
|-------|----|----|-----|
| google/gemma-2-2b-it | ✓ | ✓ | ✓ |
| google/gemma-3-1b-it | ✓ | ✓ | ✓ |
| meta-llama/Llama-3-8B-Instruct | ✓ | ✓ | ✓ |

## Recalibration

When deploying to a new backend (e.g., llm-d with quantization),
you may need to recalibrate the threshold:

```python
from aase.integrations.llmd import recalibrate_for_llmd

recalibrated = recalibrate_for_llmd(
    probe=af,
    positive_samples=test_harmful,
    negative_samples=test_benign,
    extractor=llmd_extractor,
    alpha=0.3  # Blend factor
)
```
