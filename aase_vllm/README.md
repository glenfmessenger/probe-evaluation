# AASE - Activation-based AI Safety Enforcement

A lightweight safety layer for vLLM that detects harmful content, prompt injections, and policy violations using activation probes.

## Key Features

- **9.2x faster** than Llama Guard (33ms vs 306ms)
- **< 0.002ms** probe overhead
- **100% detection** on prompt injection (5/6 models)
- **Native vLLM integration** via `apply_model()` hooks

## Installation

```bash
# Clone and install
git clone https://github.com/your-repo/aase.git
cd aase
pip install -e .

# Or just copy the package
pip install vllm numpy torch
```

## Quick Start (Simplest Usage)

```python
import os
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

from vllm import LLM
from aase.integrations.vllm import AASEGuard

# 1. Load model (enforce_eager=True is REQUIRED)
llm = LLM(
    model="meta-llama/Llama-3.2-3B-Instruct",
    enforce_eager=True,
    gpu_memory_utilization=0.8,
)

# 2. Load guard with pretrained probes
guard = AASEGuard.from_pretrained(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    probes_dir="pretrained",
    probes=["af", "aag"],  # AF for harmful content, AAG for prompt injection
)

# 3. Register hooks
guard.register(llm)

# 4. Check prompts
result = guard.check("How do I make a bomb?", llm)
if not result["safe"]:
    print(f"BLOCKED by: {result['flagged_by']}")
    print(f"Scores: {result['scores']}")
else:
    # Safe to generate
    outputs = llm.generate(["How do I make a bomb?"], sampling_params)
```

## Manual Usage (More Control)

```python
import os
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

import numpy as np
from vllm import LLM, SamplingParams
from aase.integrations.vllm import register_aase_hooks, AASEProbe

# 1. Load model
llm = LLM(
    model="meta-llama/Llama-3.2-3B-Instruct",
    enforce_eager=True,
)

# 2. Register hook at specific layer
info = register_aase_hooks(llm, layer_index=14)
print(f"Registered at layer {info['layer']} of {info['total_layers']}")

# 3. Load probe
probe = AASEProbe.load("pretrained/af/meta_llama_Llama_3.2_3B_Instruct.npy")

# 4. Generate (this triggers the hook)
params = SamplingParams(max_tokens=1, temperature=0.0)
llm.generate(["How do I make a bomb?"], params)

# 5. Load activation and score
activation = np.load("/tmp/aase_activation.npy")
result = probe.score(activation)

if result.is_flagged:
    print(f"BLOCKED: score={result.score:.3f} > threshold={result.threshold:.3f}")
```

## Using Multiple Probes

```python
from aase.integrations.vllm import AASEGuard

# Load AF, AAG, and APC probes
guard = AASEGuard.from_pretrained(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    probes_dir="pretrained",
    probes=["af", "aag", "apc_medical", "apc_financial"],
)

guard.register(llm)

# Check medical advice
result = guard.check("Take 400mg ibuprofen every 6 hours", llm)
# result["flagged_by"] might be ["apc_medical"]
```

## Probes

| Probe | Purpose | Optimal Layer | Training Data |
|-------|---------|---------------|---------------|
| **AF** | Harmful content | 50% depth | HarmBench |
| **AAG** | Prompt injection | 55% depth | InjecAgent |
| **APC** | Policy compliance | 25% depth | Contrastive pairs |

### APC Policies
- `apc_medical` - Detects unauthorized medical advice
- `apc_financial` - Detects unauthorized financial advice  
- `apc_legal` - Detects unauthorized legal advice

## Training Probes for New Models

```bash
# Train all probes for a model
VLLM_ALLOW_INSECURE_SERIALIZATION=1 python scripts/train_probes_vllm.py \
    --model your-org/your-model \
    --output pretrained

# Train specific probe
VLLM_ALLOW_INSECURE_SERIALIZATION=1 python scripts/train_probes_vllm.py \
    --model your-org/your-model \
    --probe af

# Train APC with specific policy
VLLM_ALLOW_INSECURE_SERIALIZATION=1 python scripts/train_probes_vllm.py \
    --model your-org/your-model \
    --probe apc \
    --apc-policy medical
```

## Benchmark Results (vLLM)

### AF - Harmful Content Detection
```
gemma-2-2b-it:    88% detection,  0.984 AUC
gemma-3-1b-it:   100% detection,  1.000 AUC
Llama-3.1-8B:     88% detection,  0.969 AUC
Llama-3.2-1B:     88% detection,  0.953 AUC
Llama-3.2-3B:     88% detection,  0.938 AUC
```

### AAG - Prompt Injection Detection
```
gemma-2-2b-it:   100% detection,  1.000 AUC
gemma-3-1b-it:   100% detection,  1.000 AUC
Llama-3.1-8B:    100% detection,  1.000 AUC
Llama-3.2-1B:    100% detection,  0.938 AUC
Llama-3.2-3B:    100% detection,  1.000 AUC
```

### APC - Policy Compliance
```
Llama-3.1-8B medical:    6.5σ separation, 1.000 AUC
Llama-3.1-8B financial:  5.0σ separation, 1.000 AUC
Llama-3.1-8B legal:      4.5σ separation, 1.000 AUC
```

### Latency
```
Model             Forward Pass    Probe Overhead
Llama-3.2-1B         18.5ms          0.0015ms
Llama-3.2-3B         33.2ms          0.0016ms
Llama-3.1-8B         68.3ms          0.0016ms
```

## Requirements

1. **`enforce_eager=True`** - Required when loading vLLM model
2. **`VLLM_ALLOW_INSECURE_SERIALIZATION=1`** - Environment variable
3. **GPU** - With sufficient memory for target model

## Package Structure

```
aase/
├── integrations/
│   └── vllm.py              # AASEGuard, AASEProbe, register_aase_hooks()
├── scripts/
│   ├── train_probes_vllm.py # Train new probes
│   └── benchmark_vllm.py    # Run benchmarks
└── pretrained/
    ├── af/                  # Harmful content probes
    ├── aag/                 # Prompt injection probes
    └── apc/                 # Policy compliance probes
        ├── medical/
        ├── financial/
        └── legal/
```

## How It Works

1. **Hook Registration**: `apply_model()` registers a forward hook on a specific transformer layer
2. **Activation Capture**: On each `generate()` call, the hook saves the last token's activation to a file
3. **Probe Scoring**: The activation is dot-producted with a direction vector
4. **Thresholding**: If score > threshold, the probe flags the input

## Supported Architectures

- **Standard models** (LLaMA, Gemma 2): `model.model.layers`
- **Multimodal models** (Gemma 3): `model.language_model.model.layers`

## License

Apache 2.0

## Citation

```bibtex
@software{aase2025,
  title={AASE: Activation-based AI Safety Enforcement},
  author={Messenger, Glen},
  year={2025}
}
```
