#!/usr/bin/env python3
"""Debug probe file paths and naming."""

from pathlib import Path

PROBES_DIR = "pretrained"

# Check what files exist
print("=" * 60)
print(" Probe Files Found")
print("=" * 60)

for probe_type in ["af", "aag"]:
    print(f"\n{probe_type.upper()} probes:")
    probe_dir = Path(PROBES_DIR) / probe_type
    for f in sorted(probe_dir.glob("*.npy")):
        print(f"  {f.stem}")

# Check model name conversions
print("\n" + "=" * 60)
print(" Model Name Conversions")
print("=" * 60)

models = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct", 
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
]

for model in models:
    # Current conversion method
    model_safe = model.replace("/", "_").replace("-", "_")
    
    # Check if file exists
    af_exists = (Path(PROBES_DIR) / "af" / f"{model_safe}.npy").exists()
    aag_exists = (Path(PROBES_DIR) / "aag" / f"{model_safe}.npy").exists()
    
    status = "✓" if af_exists and aag_exists else "✗"
    print(f"\n{model}")
    print(f"  Converted: {model_safe}")
    print(f"  AF exists:  {af_exists}")
    print(f"  AAG exists: {aag_exists}")

# Also check for any file that contains "Llama"
print("\n" + "=" * 60)
print(" Files containing 'Llama' or 'llama'")
print("=" * 60)

for probe_type in ["af", "aag"]:
    probe_dir = Path(PROBES_DIR) / probe_type
    for f in probe_dir.glob("*[Ll]lama*"):
        print(f"  {probe_type}/{f.name}")
