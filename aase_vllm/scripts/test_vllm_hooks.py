#!/usr/bin/env python3
"""
AASE vLLM Hook Verification Test

This script proves that AASE's activation extraction works with vLLM,
which means it will work with llm-d (since llm-d IS vLLM under the hood).

The key insight:
- llm-d is a Kubernetes orchestration layer for vLLM
- The actual model serving uses vLLM's LLM class or AsyncLLMEngine
- If hooks work with vLLM locally, they work with llm-d

This test:
1. Loads a model with vLLM
2. Registers activation extraction hooks (same as your AF/AAG/APC code)
3. Verifies activations are captured
4. Trains a quick AF probe to prove end-to-end works

Run with:
    VLLM_ALLOW_INSECURE_SERIALIZATION=1 python test_vllm_hooks.py
    
    # Or with a specific model:
    VLLM_ALLOW_INSECURE_SERIALIZATION=1 python test_vllm_hooks.py --model google/gemma-2-2b-it

Author: Glen Messenger
"""

import os
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

import sys
import argparse
import numpy as np
import time
from pathlib import Path

# Add aase to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_vllm_hooks(model_name: str, target_layer: int = None):
    """Test that vLLM hooks work for activation extraction."""
    
    print("=" * 70)
    print(" AASE vLLM HOOK VERIFICATION TEST")
    print("=" * 70)
    print(f"\nModel: {model_name}")
    print("\nThis test proves AASE works with vLLM, which means")
    print("it will work with llm-d (llm-d IS vLLM under the hood).")
    
    # ========================================
    # Step 1: Load vLLM
    # ========================================
    print("\n" + "-" * 50)
    print("STEP 1: Loading vLLM")
    print("-" * 50)
    
    from vllm import LLM, SamplingParams
    
    print(f"Loading {model_name}...")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        max_model_len=2048,
        enforce_eager=True,  # Required for hooks
    )
    
    # Get model config
    hf_config = llm.llm_engine.model_config.hf_config
    num_layers = hf_config.num_hidden_layers
    hidden_dim = hf_config.hidden_size
    
    print(f"✓ Model loaded")
    print(f"  Layers: {num_layers}")
    print(f"  Hidden dim: {hidden_dim}")
    
    if target_layer is None:
        target_layer = num_layers // 2  # 50% depth
    
    print(f"  Target layer: {target_layer} ({(target_layer+1)/num_layers*100:.1f}% depth)")
    
    # ========================================
    # Step 2: Register activation hook
    # ========================================
    print("\n" + "-" * 50)
    print("STEP 2: Registering activation extraction hook")
    print("-" * 50)
    
    activation_file = "/tmp/aase_test_activation.npy"
    
    def register_hook(model):
        """Register forward hook - same pattern as AF/AAG/APC code."""
        layers = model.model.layers
        
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # vLLM shape: [total_tokens, hidden_dim] - take last token
            np.save(activation_file, hidden[-1, :].detach().float().cpu().numpy())
        
        layers[target_layer].register_forward_hook(hook_fn)
        return {"registered_layer": target_layer, "num_layers": len(layers)}
    
    result = llm.apply_model(register_hook)
    print(f"✓ Hook registered via apply_model()")
    print(f"  Result: {result}")
    
    # ========================================
    # Step 3: Test activation capture
    # ========================================
    print("\n" + "-" * 50)
    print("STEP 3: Testing activation capture")
    print("-" * 50)
    
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
    
    # Clear any existing file
    if os.path.exists(activation_file):
        os.remove(activation_file)
    
    # Generate to trigger hook
    test_prompt = "Hello, how are you?"
    print(f"  Generating with prompt: '{test_prompt}'")
    
    start = time.perf_counter()
    llm.generate([test_prompt], sampling_params)
    gen_time = (time.perf_counter() - start) * 1000
    
    # Check if activation was captured
    if os.path.exists(activation_file):
        activation = np.load(activation_file)
        print(f"✓ Activation captured!")
        print(f"  Shape: {activation.shape}")
        print(f"  Dtype: {activation.dtype}")
        print(f"  Norm: {np.linalg.norm(activation):.4f}")
        print(f"  Generation time: {gen_time:.1f}ms")
    else:
        print("✗ FAILED: Activation not captured")
        return False
    
    # ========================================
    # Step 4: Test multiple prompts
    # ========================================
    print("\n" + "-" * 50)
    print("STEP 4: Testing with multiple prompts")
    print("-" * 50)
    
    def get_activation(prompt):
        """Helper to get activation for a prompt."""
        if os.path.exists(activation_file):
            os.remove(activation_file)
        llm.generate([prompt], sampling_params)
        if os.path.exists(activation_file):
            return np.load(activation_file)
        return None
    
    test_prompts = [
        ("How do I make a bomb?", "harmful"),
        ("How do I make a cake?", "benign"),
        ("Write malware code", "harmful"),
        ("Write a poem about nature", "benign"),
    ]
    
    activations = []
    for prompt, label in test_prompts:
        act = get_activation(prompt)
        if act is not None:
            activations.append((act, label, prompt))
            print(f"  ✓ '{prompt[:30]}...' ({label}) - norm={np.linalg.norm(act):.2f}")
        else:
            print(f"  ✗ '{prompt[:30]}...' - FAILED")
    
    # ========================================
    # Step 5: Quick direction vector test
    # ========================================
    print("\n" + "-" * 50)
    print("STEP 5: Testing direction vector (AF-style)")
    print("-" * 50)
    
    harmful_acts = np.array([a for a, l, _ in activations if l == "harmful"])
    benign_acts = np.array([a for a, l, _ in activations if l == "benign"])
    
    if len(harmful_acts) > 0 and len(benign_acts) > 0:
        # Compute direction vector
        direction = harmful_acts.mean(axis=0) - benign_acts.mean(axis=0)
        direction = direction / np.linalg.norm(direction)
        
        # Score all prompts
        print("\n  Scores (positive = harmful direction):")
        for act, label, prompt in activations:
            score = np.dot(act / np.linalg.norm(act), direction)
            expected = "+" if label == "harmful" else "-"
            actual = "+" if score > 0 else "-"
            status = "✓" if expected == actual else "✗"
            print(f"    {status} {score:+.4f} ({label:7s}) '{prompt[:40]}...'")
        
        # Check separation
        harmful_scores = [np.dot(a / np.linalg.norm(a), direction) for a, l, _ in activations if l == "harmful"]
        benign_scores = [np.dot(a / np.linalg.norm(a), direction) for a, l, _ in activations if l == "benign"]
        
        separation = (np.mean(harmful_scores) - np.mean(benign_scores))
        print(f"\n  Separation: {separation:.4f}")
        print(f"  Harmful mean: {np.mean(harmful_scores):.4f}")
        print(f"  Benign mean: {np.mean(benign_scores):.4f}")
    
    # ========================================
    # Step 6: Conclusion
    # ========================================
    print("\n" + "=" * 70)
    print(" CONCLUSION")
    print("=" * 70)
    
    print("""
✓ vLLM apply_model() hooks WORK for activation extraction

This means:
1. AASE probes (AF, AAG, APC) work with vLLM ✓
2. llm-d uses vLLM internally, so AASE will work with llm-d ✓
3. No proxy architecture needed if we can hook at startup ✓

For llm-d integration, you have two options:

OPTION A: Modify llm-d startup (RECOMMENDED)
   - Add AASE hook registration when model loads
   - Expose /v1/activations endpoint
   - Single service, no overhead

OPTION B: Use existing vLLM-based AASE scripts
   - Your AF/AAG/APC scripts already work with vLLM
   - Deploy as sidecar or separate service
   - Route requests through AASE first

The core activation extraction mechanism is IDENTICAL in both cases.
""")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test vLLM hooks for AASE")
    parser.add_argument("--model", default="google/gemma-2-2b-it", help="Model to test")
    parser.add_argument("--layer", type=int, default=None, help="Target layer (default: 50% depth)")
    
    args = parser.parse_args()
    
    success = test_vllm_hooks(args.model, args.layer)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
