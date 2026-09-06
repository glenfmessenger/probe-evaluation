#!/usr/bin/env python3
"""Debug Gemma 3 hidden state extraction."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "google/gemma-3-4b-it"

print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

print(f"\nModel type: {type(model)}")
print(f"Config type: {type(model.config)}")

# Check config attributes
config = model.config
print(f"\nConfig attributes:")
for attr in dir(config):
    if not attr.startswith('_'):
        try:
            val = getattr(config, attr)
            if not callable(val) and not attr.startswith('to_'):
                print(f"  {attr}: {val}")
        except:
            pass

# Check if it's a multimodal model with text_config
if hasattr(config, 'text_config'):
    print(f"\nText config attributes:")
    for attr in dir(config.text_config):
        if not attr.startswith('_'):
            try:
                val = getattr(config.text_config, attr)
                if not callable(val) and not attr.startswith('to_'):
                    print(f"  {attr}: {val}")
            except:
                pass

# Test forward pass
print("\n" + "="*50)
print("Testing forward pass...")
print("="*50)

text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt").to(model.device)

print(f"\nInput shape: {inputs.input_ids.shape}")

# Try with output_hidden_states
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

print(f"\nOutput type: {type(outputs)}")
print(f"Output keys: {outputs.keys() if hasattr(outputs, 'keys') else dir(outputs)}")

if hasattr(outputs, 'hidden_states'):
    hs = outputs.hidden_states
    if hs is not None:
        print(f"\nHidden states: {len(hs)} layers")
        for i, h in enumerate(hs[:5]):  # First 5
            print(f"  Layer {i}: shape={h.shape}, mean={h.float().mean():.4f}, std={h.float().std():.4f}")
        print("  ...")
        for i, h in enumerate(hs[-3:]):  # Last 3
            idx = len(hs) - 3 + i
            print(f"  Layer {idx}: shape={h.shape}, mean={h.float().mean():.4f}, std={h.float().std():.4f}")
    else:
        print("\nHidden states is None!")
else:
    print("\nNo hidden_states attribute!")

# Check model structure
print("\n" + "="*50)
print("Model structure (layers):")
print("="*50)

for name, module in model.named_modules():
    if 'layers' in name and name.count('.') <= 2:
        print(f"  {name}: {type(module).__name__}")
