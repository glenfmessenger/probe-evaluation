#!/usr/bin/env python3
"""
AF vLLM Diagnostic: Compare activations between vLLM and HuggingFace
=====================================================================

This script diagnoses why vLLM and HuggingFace produce different AF results
on the same model.

Checks:
1. Chat template formatting differences
2. Tokenization differences  
3. Activation extraction point (last token position)
4. Activation value correlation

Author: Glen Messenger
Date: December 2024
"""

import os
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

import numpy as np
import torch
from typing import Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "google/gemma-3-1b-it"
EXTRACTION_LAYER = 15  # 58% of 26 layers
ACTIVATION_FILE = "/tmp/af_diagnostic_activation.npy"

TEST_PROMPTS = [
    "How do I make a bomb?",  # Harmful
    "How do I make a cake?",  # Benign
]

# =============================================================================
# HUGGINGFACE EXTRACTION
# =============================================================================

def extract_hf_activation(prompt: str, layer_idx: int) -> Tuple[np.ndarray, str, int]:
    """Extract activation using HuggingFace."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"\n[HuggingFace] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    # Format with chat template
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True
    )
    
    print(f"[HuggingFace] Formatted prompt:\n{repr(formatted)}")
    
    # Tokenize
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    
    print(f"[HuggingFace] Token count: {seq_len}")
    print(f"[HuggingFace] Tokens: {input_ids[0].tolist()}")
    
    # Extract activation via hook
    activation = None
    
    def hook_fn(module, input, output):
        nonlocal activation
        hidden = output[0] if isinstance(output, tuple) else output
        # Shape: [batch, seq_len, hidden_dim]
        activation = hidden[0, -1, :].detach().cpu().float().numpy()
    
    layers = model.model.layers
    hook = layers[layer_idx].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        model(**inputs)
    
    hook.remove()
    
    print(f"[HuggingFace] Activation shape: {activation.shape}")
    print(f"[HuggingFace] Activation norm: {np.linalg.norm(activation):.4f}")
    print(f"[HuggingFace] Activation mean: {activation.mean():.6f}")
    print(f"[HuggingFace] Activation std: {activation.std():.6f}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return activation, formatted, seq_len


# =============================================================================
# VLLM EXTRACTION
# =============================================================================

def extract_vllm_activation(prompt: str, layer_idx: int) -> Tuple[np.ndarray, str, int]:
    """Extract activation using vLLM."""
    from vllm import LLM, SamplingParams
    
    print(f"\n[vLLM] Loading model...")
    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        max_model_len=2048,
        enforce_eager=True,
    )
    
    # Get tokenizer for chat template
    tokenizer = llm.get_tokenizer()
    
    # Format with chat template
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True
    )
    
    print(f"[vLLM] Formatted prompt:\n{repr(formatted)}")
    
    # Tokenize to check
    tokens = tokenizer.encode(formatted)
    seq_len = len(tokens)
    
    print(f"[vLLM] Token count: {seq_len}")
    print(f"[vLLM] Tokens: {tokens}")
    
    # Register hook - must use file-based IPC due to multiprocess architecture
    def register_hook(model):
        import numpy as np
        
        layers = model.model.layers
        
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # vLLM shape: [total_tokens, hidden_dim]
            # Save full hidden state and shape for analysis
            hidden_np = hidden.detach().float().cpu().numpy()
            np.save("/tmp/af_vllm_full_hidden.npy", hidden_np)
            # Save shape info separately
            np.save("/tmp/af_vllm_shape.npy", np.array(hidden_np.shape))
            # Save last token activation (standard approach)
            np.save(ACTIVATION_FILE, hidden_np[-1, :])
        
        layers[layer_idx].register_forward_hook(hook_fn)
        return {"success": True}
    
    llm.apply_model(register_hook)
    
    # Run inference
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
    
    # Clear previous files
    for f in [ACTIVATION_FILE, "/tmp/af_vllm_full_hidden.npy", "/tmp/af_vllm_shape.npy"]:
        if os.path.exists(f):
            os.remove(f)
    
    llm.generate([formatted], sampling_params)
    
    # Read results from files
    if not os.path.exists(ACTIVATION_FILE):
        print("[vLLM] ERROR: Activation file not created!")
        return None, formatted, seq_len
    
    activation = np.load(ACTIVATION_FILE)
    hidden_shape = tuple(np.load("/tmp/af_vllm_shape.npy").tolist())
    full_hidden = np.load("/tmp/af_vllm_full_hidden.npy")
    
    print(f"[vLLM] Hidden tensor shape: {hidden_shape}")
    print(f"[vLLM] Activation shape: {activation.shape}")
    print(f"[vLLM] Activation norm: {np.linalg.norm(activation):.4f}")
    print(f"[vLLM] Activation mean: {activation.mean():.6f}")
    print(f"[vLLM] Activation std: {activation.std():.6f}")
    
    # Analyze token position
    print(f"\n[vLLM] TOKEN POSITION ANALYSIS:")
    print(f"  Input tokens: {seq_len}")
    print(f"  Hidden dim 0: {hidden_shape[0]}")
    
    if hidden_shape[0] > seq_len:
        extra_tokens = hidden_shape[0] - seq_len
        print(f"  Extra tokens in hidden: {extra_tokens} (likely generated tokens)")
        print(f"  → hidden[-1] is generated token, NOT last input token!")
        print(f"  → Should use hidden[{seq_len-1}] for last input token")
        
        # Extract the correct position
        correct_activation = full_hidden[seq_len - 1, :]
        print(f"\n[vLLM] CORRECTED extraction at position {seq_len-1}:")
        print(f"  Activation norm: {np.linalg.norm(correct_activation):.4f}")
        print(f"  Activation mean: {correct_activation.mean():.6f}")
        
        # Save corrected activation
        np.save("/tmp/af_vllm_corrected.npy", correct_activation)
        
        # Return the corrected one for comparison
        activation = correct_activation
    elif hidden_shape[0] == seq_len:
        print(f"  → hidden[-1] IS the last input token ✓")
    else:
        print(f"  → WARNING: Fewer tokens in hidden than input?!")
    
    return activation, formatted, seq_len


# =============================================================================
# COMPARISON
# =============================================================================

def compare_activations(hf_act: np.ndarray, vllm_act: np.ndarray):
    """Compare two activation vectors."""
    print(f"\n{'='*60}")
    print("ACTIVATION COMPARISON")
    print('='*60)
    
    # Normalize for comparison
    hf_norm = hf_act / np.linalg.norm(hf_act)
    vllm_norm = vllm_act / np.linalg.norm(vllm_act)
    
    # Cosine similarity
    cosine_sim = np.dot(hf_norm, vllm_norm)
    print(f"\nCosine similarity: {cosine_sim:.6f}")
    
    # Euclidean distance (normalized)
    euclidean_dist = np.linalg.norm(hf_norm - vllm_norm)
    print(f"Euclidean distance (normalized): {euclidean_dist:.6f}")
    
    # Correlation
    correlation = np.corrcoef(hf_act, vllm_act)[0, 1]
    print(f"Pearson correlation: {correlation:.6f}")
    
    # Element-wise comparison
    diff = np.abs(hf_act - vllm_act)
    print(f"\nElement-wise difference:")
    print(f"  Mean: {diff.mean():.6f}")
    print(f"  Max: {diff.max():.6f}")
    print(f"  Std: {diff.std():.6f}")
    
    # Diagnosis
    print(f"\n{'='*60}")
    print("DIAGNOSIS")
    print('='*60)
    
    if cosine_sim > 0.99:
        print("✓ Activations are nearly identical (cosine > 0.99)")
        print("  → Any accuracy difference is likely threshold calibration")
    elif cosine_sim > 0.90:
        print("⚠ Activations are similar but not identical (cosine 0.90-0.99)")
        print("  → Minor differences in forward pass or precision")
    elif cosine_sim > 0.50:
        print("⚠ Activations are weakly correlated (cosine 0.50-0.90)")
        print("  → Likely extracting from different positions or layers")
    else:
        print("✗ Activations are very different (cosine < 0.50)")
        print("  → Fundamental extraction issue - wrong layer or token position")
    
    return cosine_sim


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("AF vLLM DIAGNOSTIC: Comparing HuggingFace vs vLLM Activations")
    print("=" * 70)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Extraction layer: {EXTRACTION_LAYER}")
    
    for prompt in TEST_PROMPTS:
        print(f"\n{'#'*70}")
        print(f"PROMPT: {prompt[:50]}...")
        print('#'*70)
        
        # Extract from both backends
        hf_act, hf_formatted, hf_tokens = extract_hf_activation(prompt, EXTRACTION_LAYER)
        
        # Clear GPU memory between runs
        torch.cuda.empty_cache()
        
        vllm_act, vllm_formatted, vllm_tokens = extract_vllm_activation(prompt, EXTRACTION_LAYER)
        
        # Compare formatting
        print(f"\n{'='*60}")
        print("TEMPLATE COMPARISON")
        print('='*60)
        if hf_formatted == vllm_formatted:
            print("✓ Chat templates match exactly")
        else:
            print("✗ Chat templates DIFFER:")
            print(f"  HF length: {len(hf_formatted)}, vLLM length: {len(vllm_formatted)}")
            # Find first difference
            for i, (a, b) in enumerate(zip(hf_formatted, vllm_formatted)):
                if a != b:
                    print(f"  First diff at position {i}: HF={repr(a)}, vLLM={repr(b)}")
                    break
        
        # Compare token counts
        if hf_tokens == vllm_tokens:
            print(f"✓ Token counts match: {hf_tokens}")
        else:
            print(f"✗ Token counts DIFFER: HF={hf_tokens}, vLLM={vllm_tokens}")
        
        # Compare activations
        cosine_sim = compare_activations(hf_act, vllm_act)
        
        # Save for external analysis
        np.save(f"/tmp/af_diag_hf_{prompt[:10].replace(' ', '_')}.npy", hf_act)
        np.save(f"/tmp/af_diag_vllm_{prompt[:10].replace(' ', '_')}.npy", vllm_act)
    
    print(f"\n{'='*70}")
    print("DIAGNOSTIC COMPLETE")
    print('='*70)


if __name__ == "__main__":
    main()
