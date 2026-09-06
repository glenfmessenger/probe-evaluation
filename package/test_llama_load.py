#!/usr/bin/env python3
"""Direct test loading Llama probes."""

import os
from pathlib import Path
import numpy as np

print(f"Current working directory: {os.getcwd()}")

probes_dir = "pretrained"
models = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]

for model_name in models:
    model_safe = model_name.replace("/", "_").replace("-", "_")
    
    af_npy = Path(probes_dir) / "af" / f"{model_safe}.npy"
    af_json = Path(probes_dir) / "af" / f"{model_safe}.json"
    aag_npy = Path(probes_dir) / "aag" / f"{model_safe}.npy"
    aag_json = Path(probes_dir) / "aag" / f"{model_safe}.json"
    
    print(f"\n{model_name}")
    print(f"  AF npy:  {af_npy} -> exists={af_npy.exists()}")
    print(f"  AF json: {af_json} -> exists={af_json.exists()}")
    print(f"  AAG npy: {aag_npy} -> exists={aag_npy.exists()}")
    print(f"  AAG json: {aag_json} -> exists={aag_json.exists()}")
    
    # Try to actually load
    if af_npy.exists():
        try:
            data = np.load(str(af_npy))
            print(f"  AF loaded OK: shape={data.shape}")
        except Exception as e:
            print(f"  AF load error: {e}")

# Also test using .with_suffix
print("\n\nTesting .with_suffix() method:")
for model_name in models:
    model_safe = model_name.replace("/", "_").replace("-", "_")
    af_path = Path(probes_dir) / "af" / model_safe
    
    test_path = af_path.with_suffix(".npy")
    print(f"  {af_path} -> with_suffix('.npy') -> {test_path}")
    print(f"    exists: {test_path.exists()}")
