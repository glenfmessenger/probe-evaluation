"""
Quick Layer Sweep for Gemma-3-4B AHD
====================================

Find the optimal layer for hallucination detection on the 4B model.
"""

import torch
import numpy as np
import json
from datetime import datetime

# Training data (same as main script)
FACTUAL = [
    "What is the capital of France?",
    "What is 7 times 8?",
    "What year did World War II end?",
    "What is the chemical symbol for water?",
    "Who wrote Romeo and Juliet?",
    "What planet is closest to the Sun?",
    "What is the largest ocean on Earth?",
    "How many continents are there?",
    "Who painted the Mona Lisa?",
    "What is the capital of Japan?",
    "What is the square root of 144?",
    "What is the capital of Italy?",
    "How many days are in a week?",
    "What color is the sky on a clear day?",
    "What is 100 divided by 4?",
]

HALLUCINATION = [
    "What is the capital of Atlantis?",
    "What year was the Treaty of Westbrook signed?",
    "Who was the 47th Emperor of Japan?",
    "What is the population of the city of El Dorado?",
    "What language is spoken in the Kingdom of Zorgon?",
    "What was Napoleon's phone number?",
    "What color was Aristotle's car?",
    "What was Shakespeare's email address?",
    "What brand of shoes did Julius Caesar wear?",
    "What was Cleopatra's Instagram username?",
    "Who won the 2030 FIFA World Cup?",
    "What was the most popular movie of 2029?",
    "Who won the 2028 US Presidential election?",
    "What company had the highest stock price in 2035?",
    "What was the population of Mars in 2050?",
]


def get_activation(model, tokenizer, layers, layer_idx, text, device):
    """Extract activation for a single layer."""
    activations = {}
    
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        activations['act'] = hidden.detach()
    
    layer = layers[layer_idx]
    hook = layer.register_forward_hook(hook_fn)
    
    formatted = f"<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    
    try:
        with torch.no_grad():
            _ = model(inputs.input_ids, attention_mask=inputs.attention_mask)
    finally:
        hook.remove()
    
    if 'act' not in activations:
        raise ValueError(f"Hook didn't capture activation. Layer {layer_idx} type: {type(layer).__name__}")
    
    raw = activations['act'][0, -1].float().cpu().numpy().astype(np.float32)
    raw = np.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return raw


def evaluate_layer(model, tokenizer, layers, layer_idx, device):
    """Evaluate AHD performance for a single layer."""
    
    # Collect activations
    factual_acts = np.stack([
        get_activation(model, tokenizer, layers, layer_idx, q, device) 
        for q in FACTUAL
    ])
    halluc_acts = np.stack([
        get_activation(model, tokenizer, layers, layer_idx, q, device) 
        for q in HALLUCINATION
    ])
    
    # Compute direction vector
    factual_mean = factual_acts.mean(axis=0)
    halluc_mean = halluc_acts.mean(axis=0)
    direction = factual_mean - halluc_mean
    direction = direction / np.linalg.norm(direction)
    
    # Compute scores
    factual_scores = factual_acts @ direction
    halluc_scores = halluc_acts @ direction
    
    # Find optimal threshold
    all_scores = np.concatenate([factual_scores, halluc_scores])
    all_labels = np.array([1]*len(factual_scores) + [0]*len(halluc_scores))
    
    # Simple AUC calculation
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(all_labels, all_scores)
    except:
        auc = 0.5
    
    # Accuracy at optimal threshold
    best_acc = 0
    for t in np.sort(all_scores):
        preds = (all_scores > t).astype(int)
        acc = (preds == all_labels).mean()
        if acc > best_acc:
            best_acc = acc
    
    # Separation
    separation = (factual_scores.mean() - halluc_scores.mean()) / max(halluc_scores.std(), 0.001)
    
    return {
        'layer': layer_idx,
        'accuracy': float(best_acc),
        'auc': float(auc),
        'separation_sigma': float(separation),
        'factual_mean': float(factual_scores.mean()),
        'halluc_mean': float(halluc_scores.mean()),
    }


def main():
    print("=" * 70)
    print("LAYER SWEEP FOR GEMMA-3-4B AHD")
    print("=" * 70)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "google/gemma-3-4b-it"
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Find layers - handle Gemma3ForConditionalGeneration architecture
    layers = None
    
    # Debug: print model structure
    print(f"\nModel type: {type(model).__name__}")
    
    # For Gemma3ForConditionalGeneration, we want the TEXT model layers, not vision
    if hasattr(model, 'language_model'):
        print(f"  model.language_model: {type(model.language_model).__name__}")
        # Gemma3TextModel has model.layers
        if hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'layers'):
            layers = model.language_model.model.layers
            print(f"  Found TEXT layers at model.language_model.model.layers")
        elif hasattr(model.language_model, 'layers'):
            layers = model.language_model.layers
            print(f"  Found TEXT layers at model.language_model.layers")
    
    # Fallback for non-multimodal models
    if layers is None and hasattr(model, 'model'):
        print(f"  model.model: {type(model.model).__name__}")
        if hasattr(model.model, 'layers'):
            layers = model.model.layers
            print(f"  Found layers at model.model.layers")
        elif hasattr(model.model, 'model') and hasattr(model.model.model, 'layers'):
            print(f"  model.model.model: {type(model.model.model).__name__}")
            layers = model.model.model.layers
            print(f"  Found layers at model.model.model.layers")
    
    if layers is None:
        raise ValueError("Cannot find text model layers - please check model architecture")
    
    num_layers = len(layers)
    device = next(model.parameters()).device
    print(f"Model has {num_layers} layers")
    
    print(f"\nTesting hook on layer 0...")
    test_activations = {}
    def test_hook(module, input, output):
        print(f"  Hook fired! Output type: {type(output)}")
        if isinstance(output, tuple):
            print(f"  Tuple length: {len(output)}, first element shape: {output[0].shape if hasattr(output[0], 'shape') else 'no shape'}")
            test_activations['act'] = output[0].detach()
        else:
            print(f"  Direct tensor shape: {output.shape if hasattr(output, 'shape') else 'no shape'}")
            test_activations['act'] = output.detach()
    
    test_layer = layers[0]
    print(f"  Layer type: {type(test_layer).__name__}")
    hook = test_layer.register_forward_hook(test_hook)
    
    test_input = tokenizer("<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n", return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model(test_input.input_ids, attention_mask=test_input.attention_mask)
    hook.remove()
    
    if 'act' not in test_activations:
        print("  ERROR: Hook did not fire!")
        print("  This model may use a different forward path.")
        print("  Checking model's forward method...")
        return
    else:
        print(f"  Success! Activation shape: {test_activations['act'].shape}")
    
    # Test layers at different depths
    test_layers = [
        int(num_layers * 0.20),  # 20%
        int(num_layers * 0.30),  # 30%
        int(num_layers * 0.38),  # 38% (current)
        int(num_layers * 0.50),  # 50%
        int(num_layers * 0.65),  # 65%
        int(num_layers * 0.75),  # 75%
        int(num_layers * 0.85),  # 85%
    ]
    
    print(f"\nTesting layers: {test_layers}")
    print()
    
    results = []
    for layer_idx in test_layers:
        print(f"Evaluating layer {layer_idx} ({layer_idx/num_layers*100:.0f}%)...", end=" ")
        result = evaluate_layer(model, tokenizer, layers, layer_idx, device)
        results.append(result)
        print(f"AUC: {result['auc']:.3f}, Acc: {result['accuracy']*100:.1f}%, Sep: {result['separation_sigma']:.2f}σ")
    
    # Find best
    best = max(results, key=lambda x: x['auc'])
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Layer':>8} {'Depth':>8} {'AUC':>8} {'Accuracy':>10} {'Separation':>12}")
    print("-" * 50)
    for r in sorted(results, key=lambda x: x['auc'], reverse=True):
        depth_pct = r['layer'] / num_layers * 100
        marker = " <-- BEST" if r['layer'] == best['layer'] else ""
        print(f"{r['layer']:>8} {depth_pct:>7.0f}% {r['auc']:>8.3f} {r['accuracy']*100:>9.1f}% {r['separation_sigma']:>11.2f}σ{marker}")
    
    print(f"\n✓ Optimal layer for Gemma-3-4B: Layer {best['layer']} ({best['layer']/num_layers*100:.0f}% depth)")
    print(f"  AUC: {best['auc']:.3f}, Separation: {best['separation_sigma']:.2f}σ")
    
    # Save results
    output = {
        'model': model_name,
        'num_layers': num_layers,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'best_layer': best['layer'],
        'best_layer_pct': best['layer'] / num_layers,
        'results': results,
    }
    
    with open('ahd_layer_sweep_4b_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to ahd_layer_sweep_4b_results.json")


if __name__ == "__main__":
    main()
