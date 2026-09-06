#!/usr/bin/env python3
"""
Reset Gemma 3 AAG thresholds to training-validated values.

The tuning set was too small and not representative, causing
the thresholds to be set incorrectly.

For Gemma 3 models, we'll use the midpoint between positive
and negative means from training, which gave 99.5% accuracy.
"""

import json
from pathlib import Path

PROBES_DIR = "pretrained"

models_to_fix = [
    "google_gemma_3_1b_it",
    "google_gemma_3_4b_it",
]

print("=" * 60)
print(" Fixing Gemma 3 AAG Thresholds")
print("=" * 60)

for model_safe in models_to_fix:
    json_path = Path(PROBES_DIR) / "aag" / f"{model_safe}.json"
    
    with open(json_path) as f:
        meta = json.load(f)
    
    # Calculate optimal threshold as midpoint between means
    pos_mean = meta["positive_mean"]
    neg_mean = meta["negative_mean"]
    optimal_threshold = (pos_mean + neg_mean) / 2
    
    print(f"\n{model_safe}:")
    print(f"  Positive mean (injection): {pos_mean:.6f}")
    print(f"  Negative mean (safe):      {neg_mean:.6f}")
    print(f"  Separation:                {meta['separation_sigma']:.2f}σ")
    print(f"  Training AUC:              {meta['auc']:.4f}")
    print(f"  Training accuracy:         {meta['accuracy']:.1%}")
    print(f"  Old threshold:             {meta['threshold']:.6f}")
    print(f"  New threshold (midpoint):  {optimal_threshold:.6f}")
    
    # Update
    meta["threshold"] = optimal_threshold
    meta["threshold_source"] = "Midpoint of training means (reset from bad tuning)"
    
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"  ✓ Updated")

print("\n" + "=" * 60)
print(" Expected Results")
print("=" * 60)
print("""
With midpoint threshold:
- Gemma 3 1B: ~62% accuracy (from training), AUC 0.89
- Gemma 3 4B: ~99% accuracy (from training), AUC 0.999

Note: Gemma 3 1B has inherently weak AAG separation (1.6σ).
This is a model limitation, not a threshold issue.
""")
