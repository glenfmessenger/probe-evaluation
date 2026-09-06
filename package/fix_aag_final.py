#!/usr/bin/env python3
"""Raise AAG threshold slightly to reduce false positives on benign prompts."""

import json
from pathlib import Path

aag_json = Path("pretrained/aag/google_gemma_2_2b_it.json")

with open(aag_json) as f:
    meta = json.load(f)

print(f"Current threshold: {meta['threshold']}")

# Raise threshold to only catch clear injections
# Based on debug output:
#   Injections: min=-0.0001, one at +0.1586
#   Benign: up to +0.0231
# 
# Set threshold at 0.05 to only catch the strong injection signals
meta['threshold'] = 0.05

print(f"New threshold: {meta['threshold']}")

with open(aag_json, 'w') as f:
    json.dump(meta, f, indent=2)

print("✓ Updated. Restart server.")
