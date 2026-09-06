#!/usr/bin/env python3
"""
Batch pretrain AASE probes for multiple models.

Usage:
    # Train all supported models (requires significant GPU memory)
    python pretrain_batch.py --all
    
    # Train specific models
    python pretrain_batch.py --models google/gemma-2-9b-it meta-llama/Llama-3.1-8B-Instruct
    
    # Train only small/edge models
    python pretrain_batch.py --edge-only
    
    # List supported models
    python pretrain_batch.py --list

Supported Models:
    Gemma 2: 2b-it, 9b-it, 27b-it
    Gemma 3: 1b-it, 4b-it, 12b-it, 27b-it
    Llama 3.1: 8B-Instruct
    Llama 3.2: 1B-Instruct, 3B-Instruct (edge)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Model configurations: (model_id, gpu_memory_gb, description)
SUPPORTED_MODELS = {
    # Gemma 2 family
    "google/gemma-2-2b-it": (4, "Gemma 2 2B - Edge/Mobile"),
    "google/gemma-2-9b-it": (20, "Gemma 2 9B - Standard"),
    "google/gemma-2-27b-it": (56, "Gemma 2 27B - Large"),
    
    # Gemma 3 family (text-only modes)
    "google/gemma-3-1b-it": (3, "Gemma 3 1B - Edge/Mobile"),
    "google/gemma-3-4b-it": (10, "Gemma 3 4B - Edge/Mobile"),
    "google/gemma-3-12b-it": (28, "Gemma 3 12B - Standard"),
    "google/gemma-3-27b-it": (56, "Gemma 3 27B - Large"),
    
    # Llama 3.1 family
    "meta-llama/Llama-3.1-8B-Instruct": (18, "Llama 3.1 8B - Standard"),
    
    # Llama 3.2 family (edge models)
    "meta-llama/Llama-3.2-1B-Instruct": (3, "Llama 3.2 1B - Edge/Mobile"),
    "meta-llama/Llama-3.2-3B-Instruct": (8, "Llama 3.2 3B - Edge/Mobile"),
}

EDGE_MODELS = [
    "google/gemma-2-2b-it",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]

STANDARD_MODELS = [
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
]


def list_models():
    """List all supported models."""
    print("\nSupported Models for AASE Pretraining:")
    print("=" * 70)
    print(f"{'Model':<45} {'VRAM':<8} {'Type'}")
    print("-" * 70)
    for model, (vram, desc) in SUPPORTED_MODELS.items():
        print(f"{model:<45} {vram:>4} GB   {desc}")
    print()


def check_gpu_memory():
    """Check available GPU memory."""
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            return total
    except:
        pass
    return None


def pretrain_model(model_id: str, output_dir: str):
    """Pretrain probes for a single model."""
    print(f"\n{'='*70}")
    print(f" Training: {model_id}")
    print(f"{'='*70}")
    
    cmd = [
        sys.executable, "pretrain_robust.py",
        "--model", model_id,
        "--output", output_dir,
    ]
    
    env = os.environ.copy()
    env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    
    result = subprocess.run(cmd, env=env)
    
    if result.returncode == 0:
        print(f"✓ {model_id} complete")
        return True
    else:
        print(f"✗ {model_id} failed")
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch pretrain AASE probes")
    parser.add_argument("--models", nargs="+", help="Specific models to train")
    parser.add_argument("--all", action="store_true", help="Train all supported models")
    parser.add_argument("--edge-only", action="store_true", help="Train only edge/mobile models")
    parser.add_argument("--standard", action="store_true", help="Train standard models (2B, 9B, 8B)")
    parser.add_argument("--output", default="pretrained", help="Output directory")
    parser.add_argument("--list", action="store_true", help="List supported models")
    parser.add_argument("--max-vram", type=int, help="Max VRAM in GB to consider")
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
        return
    
    # Determine which models to train
    models_to_train = []
    
    if args.models:
        models_to_train = args.models
    elif args.all:
        models_to_train = list(SUPPORTED_MODELS.keys())
    elif args.edge_only:
        models_to_train = EDGE_MODELS
    elif args.standard:
        models_to_train = STANDARD_MODELS
    else:
        print("Specify --models, --all, --edge-only, or --standard")
        print("Use --list to see available models")
        return
    
    # Filter by VRAM if specified
    if args.max_vram:
        models_to_train = [
            m for m in models_to_train 
            if m in SUPPORTED_MODELS and SUPPORTED_MODELS[m][0] <= args.max_vram
        ]
    
    # Check GPU
    gpu_mem = check_gpu_memory()
    if gpu_mem:
        print(f"\nDetected GPU with {gpu_mem:.1f} GB VRAM")
    
    # Validate models
    valid_models = []
    for model in models_to_train:
        if model not in SUPPORTED_MODELS:
            print(f"Warning: {model} not in supported list, skipping")
            continue
        
        vram_needed = SUPPORTED_MODELS[model][0]
        if gpu_mem and vram_needed > gpu_mem * 0.9:
            print(f"Warning: {model} needs ~{vram_needed}GB VRAM, you have {gpu_mem:.1f}GB")
            response = input(f"  Try anyway? (y/N): ")
            if response.lower() != 'y':
                continue
        
        valid_models.append(model)
    
    if not valid_models:
        print("No valid models to train")
        return
    
    print(f"\nWill train {len(valid_models)} models:")
    for m in valid_models:
        vram, desc = SUPPORTED_MODELS[m]
        print(f"  - {m} ({vram}GB) - {desc}")
    
    response = input("\nProceed? (Y/n): ")
    if response.lower() == 'n':
        return
    
    # Train each model
    results = {}
    for model in valid_models:
        success = pretrain_model(model, args.output)
        results[model] = success
    
    # Summary
    print("\n" + "=" * 70)
    print(" BATCH TRAINING COMPLETE")
    print("=" * 70)
    
    succeeded = [m for m, s in results.items() if s]
    failed = [m for m, s in results.items() if not s]
    
    print(f"\nSucceeded: {len(succeeded)}")
    for m in succeeded:
        print(f"  ✓ {m}")
    
    if failed:
        print(f"\nFailed: {len(failed)}")
        for m in failed:
            print(f"  ✗ {m}")
    
    print(f"\nProbes saved to: {args.output}/")


if __name__ == "__main__":
    main()
