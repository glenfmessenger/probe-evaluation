#!/usr/bin/env python3
"""Quick fix for AAG threshold - it's too sensitive."""

import json
from pathlib import Path

probes_dir = Path("pretrained")
model_safe = "google_gemma_2_2b_it"

# AAG needs a higher threshold - it's flagging normal prompts
# Based on the scores:
#   cake (safe): 0.0154
#   bomb (safe for AAG - not injection): 0.0220  
#   injection: 0.0386
# Threshold should be around 0.03 to only catch injections

aag_json = probes_dir / "aag" / f"{model_safe}.json"
with open(aag_json) as f:
    meta = json.load(f)

print(f"Current AAG threshold: {meta['threshold']}")
meta['threshold'] = 0.030  # Only flag clear injections
print(f"New AAG threshold: {meta['threshold']}")

with open(aag_json, 'w') as f:
    json.dump(meta, f, indent=2)

print("✓ Updated. Restart server.")
