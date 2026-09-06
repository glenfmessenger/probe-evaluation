#!/usr/bin/env python3
"""Quick check of Llama file paths."""

from pathlib import Path

probes_dir = "pretrained"
model_name = "meta-llama/Llama-3.1-8B-Instruct"

# Method 1: replace / and -
model_safe = model_name.replace("/", "_").replace("-", "_")
print(f"Model: {model_name}")
print(f"Converted (replace / and -): {model_safe}")

# Check file
af_path = Path(probes_dir) / "af" / f"{model_safe}.npy"
print(f"Looking for: {af_path}")
print(f"Exists: {af_path.exists()}")

# List actual files
print(f"\nActual files in {probes_dir}/af/:")
for f in sorted(Path(probes_dir).glob("af/*.npy")):
    print(f"  {f.name}")
    if "Llama" in f.name:
        print(f"    ^ This one!")
        
# Try exact match
import os
actual_files = os.listdir(f"{probes_dir}/af")
print(f"\nos.listdir results:")
for f in actual_files:
    if "Llama" in f or "llama" in f:
        print(f"  {f}")
        
# Check character by character
expected = "meta_llama_Llama_3.1_8B_Instruct.npy"
print(f"\nExpected filename: {expected}")
print(f"Expected length: {len(expected)}")

for f in actual_files:
    if "3.1" in f and "8B" in f:
        print(f"Actual filename:   {f}")
        print(f"Actual length: {len(f)}")
        print(f"Match: {f == expected}")
        if f != expected:
            for i, (a, b) in enumerate(zip(expected, f)):
                if a != b:
                    print(f"  Diff at position {i}: expected '{a}' got '{b}'")
