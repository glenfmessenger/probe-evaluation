#!/usr/bin/env python3
"""
Activation-based Hallucination Detection: Layer Sweep Methodology Demo
======================================================================

This script demonstrates finding the optimal layer for hallucination detection
by sweeping through all layers of Gemma-3-1B-IT and evaluating classification
accuracy using a contrastive direction vector approach.

Methodology:
1. Create contrastive dataset: factual statements vs fabricated statements
2. For each layer, extract hidden states at the last token position
3. Compute a "hallucination direction" vector using mean difference
4. Evaluate binary classification accuracy using dot product with direction
5. Identify optimal layer by highest accuracy/AUC

Author: Glen Junor
Date: January 2025
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "google/gemma-3-1b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# CONTRASTIVE DATASET
# ============================================================================
# Factual statements (high confidence - model should "know" these)
# vs Fabricated statements (low confidence - model should be uncertain)

FACTUAL_STATEMENTS = [
    # Geography - well-established facts
    "The capital of France is Paris.",
    "The capital of Japan is Tokyo.",
    "The capital of Australia is Canberra.",
    "The Amazon River is located in South America.",
    "Mount Everest is the tallest mountain on Earth.",
    
    # Science - basic facts
    "Water freezes at zero degrees Celsius.",
    "The Earth orbits around the Sun.",
    "Humans have 23 pairs of chromosomes.",
    "The speed of light is approximately 300,000 kilometers per second.",
    "Oxygen is required for human respiration.",
    
    # History - well-known events
    "World War II ended in 1945.",
    "The first Moon landing occurred in 1969.",
    "The Berlin Wall fell in 1989.",
    "The United States declared independence in 1776.",
    "The French Revolution began in 1789.",
    
    # Math/Logic - definitional truths
    "Two plus two equals four.",
    "A triangle has three sides.",
    "Pi is approximately 3.14159.",
    "The square root of 16 is 4.",
    "A decade consists of ten years.",
]

FABRICATED_STATEMENTS = [
    # Geography - plausible but false
    "The capital of France is Lyon.",
    "The capital of Japan is Osaka.",
    "The capital of Australia is Sydney.",
    "The Amazon River is located in Central Africa.",
    "Mount Kilimanjaro is the tallest mountain on Earth.",
    
    # Science - plausible but false
    "Water freezes at ten degrees Celsius.",
    "The Earth orbits around Mars.",
    "Humans have 32 pairs of chromosomes.",
    "The speed of light is approximately 500,000 kilometers per second.",
    "Nitrogen is required for human respiration.",
    
    # History - plausible but false
    "World War II ended in 1943.",
    "The first Moon landing occurred in 1972.",
    "The Berlin Wall fell in 1991.",
    "The United States declared independence in 1782.",
    "The French Revolution began in 1795.",
    
    # Math/Logic - plausible but false
    "Two plus two equals five.",
    "A triangle has four sides.",
    "Pi is approximately 2.71828.",
    "The square root of 16 is 5.",
    "A decade consists of twelve years.",
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    layer_idx: int
) -> torch.Tensor:
    """
    Extract hidden states from a specific layer at the last token position.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text to process
        layer_idx: Which layer to extract from (0-indexed)
    
    Returns:
        Hidden state tensor of shape (hidden_dim,)
    """
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Forward pass with hidden states output
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
    
    # Extract hidden states from specified layer
    # hidden_states is tuple of (n_layers + 1) tensors, each (batch, seq_len, hidden_dim)
    # Index 0 is embeddings, index 1 is layer 0, etc.
    hidden_states = outputs.hidden_states[layer_idx + 1]  # +1 to skip embedding layer
    
    # Get last token's hidden state
    last_token_hidden = hidden_states[0, -1, :]  # Shape: (hidden_dim,)
    
    return last_token_hidden.cpu()


def compute_direction_vector(
    factual_hiddens: List[torch.Tensor],
    fabricated_hiddens: List[torch.Tensor]
) -> torch.Tensor:
    """
    Compute the "hallucination direction" as mean(factual) - mean(fabricated).
    
    This direction points from "uncertain/fabricated" toward "confident/factual".
    Projecting new activations onto this direction gives a confidence score.
    
    Args:
        factual_hiddens: List of hidden states for factual statements
        fabricated_hiddens: List of hidden states for fabricated statements
    
    Returns:
        Normalized direction vector
    """
    # Stack into tensors
    factual_stack = torch.stack(factual_hiddens)      # (n_factual, hidden_dim)
    fabricated_stack = torch.stack(fabricated_hiddens)  # (n_fabricated, hidden_dim)
    
    # Compute means
    factual_mean = factual_stack.mean(dim=0)
    fabricated_mean = fabricated_stack.mean(dim=0)
    
    # Direction: from fabricated toward factual
    direction = factual_mean - fabricated_mean
    
    # Normalize
    direction = direction / (direction.norm() + 1e-8)
    
    return direction


def evaluate_layer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layer_idx: int,
    factual_texts: List[str],
    fabricated_texts: List[str],
    train_ratio: float = 0.6
) -> Dict[str, float]:
    """
    Evaluate hallucination detection accuracy at a specific layer.
    
    Uses train/test split: train set computes direction vector, test set evaluates.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        layer_idx: Which layer to evaluate
        factual_texts: List of factual statements
        fabricated_texts: List of fabricated statements
        train_ratio: Fraction of data to use for computing direction vector
    
    Returns:
        Dictionary with accuracy, AUC, and other metrics
    """
    # Split data into train/test
    n_train = int(len(factual_texts) * train_ratio)
    
    train_factual = factual_texts[:n_train]
    train_fabricated = fabricated_texts[:n_train]
    test_factual = factual_texts[n_train:]
    test_fabricated = fabricated_texts[n_train:]
    
    # Extract hidden states for training set
    train_factual_hiddens = [
        get_hidden_states(model, tokenizer, text, layer_idx)
        for text in train_factual
    ]
    train_fabricated_hiddens = [
        get_hidden_states(model, tokenizer, text, layer_idx)
        for text in train_fabricated
    ]
    
    # Compute direction vector from training set
    direction = compute_direction_vector(train_factual_hiddens, train_fabricated_hiddens)
    
    # Extract hidden states for test set
    test_factual_hiddens = [
        get_hidden_states(model, tokenizer, text, layer_idx)
        for text in test_factual
    ]
    test_fabricated_hiddens = [
        get_hidden_states(model, tokenizer, text, layer_idx)
        for text in test_fabricated
    ]
    
    # Compute scores (dot product with direction)
    # Higher score = more "factual-like"
    test_factual_scores = [h.dot(direction).item() for h in test_factual_hiddens]
    test_fabricated_scores = [h.dot(direction).item() for h in test_fabricated_hiddens]
    
    # Prepare labels and scores for evaluation
    all_scores = test_factual_scores + test_fabricated_scores
    all_labels = [1] * len(test_factual_scores) + [0] * len(test_fabricated_scores)
    
    # Compute threshold (median of all scores)
    threshold = np.median(all_scores)
    
    # Predict: score > threshold = factual (1), else fabricated (0)
    predictions = [1 if s > threshold else 0 for s in all_scores]
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, predictions)
    
    # AUC (handle edge cases)
    try:
        auc = roc_auc_score(all_labels, all_scores)
    except ValueError:
        auc = 0.5  # Default if only one class present
    
    # Compute separation (mean difference / pooled std)
    factual_mean = np.mean(test_factual_scores)
    fabricated_mean = np.mean(test_fabricated_scores)
    pooled_std = np.std(all_scores)
    separation = (factual_mean - fabricated_mean) / (pooled_std + 1e-8)
    
    return {
        "accuracy": accuracy,
        "auc": auc,
        "separation": separation,
        "factual_mean": factual_mean,
        "fabricated_mean": fabricated_mean,
        "threshold": threshold
    }


def run_layer_sweep(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    n_layers: int,
    factual_texts: List[str],
    fabricated_texts: List[str]
) -> List[Dict]:
    """
    Sweep through all layers and evaluate each one.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        n_layers: Total number of layers in the model
        factual_texts: List of factual statements
        fabricated_texts: List of fabricated statements
    
    Returns:
        List of results dictionaries, one per layer
    """
    results = []
    
    print(f"\n{'='*70}")
    print(f"LAYER SWEEP: Evaluating {n_layers} layers")
    print(f"{'='*70}\n")
    
    for layer_idx in range(n_layers):
        depth_pct = (layer_idx + 1) / n_layers * 100
        
        # Evaluate this layer
        metrics = evaluate_layer(
            model, tokenizer, layer_idx,
            factual_texts, fabricated_texts
        )
        
        metrics["layer"] = layer_idx
        metrics["depth_pct"] = depth_pct
        results.append(metrics)
        
        # Progress output
        print(f"Layer {layer_idx:2d} ({depth_pct:5.1f}%): "
              f"AUC={metrics['auc']:.3f} | "
              f"Acc={metrics['accuracy']:.3f} | "
              f"Sep={metrics['separation']:+.2f}σ")
    
    return results


def print_results_table(results: List[Dict], n_layers: int):
    """Print a formatted results table."""
    
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}\n")
    
    # Header
    print(f"{'Layer':>6} | {'Depth':>7} | {'AUC':>6} | {'Accuracy':>8} | {'Separation':>10}")
    print("-" * 50)
    
    # Find best layer by AUC
    best_idx = max(range(len(results)), key=lambda i: results[i]["auc"])
    
    for r in results:
        marker = " <-- BEST" if r["layer"] == results[best_idx]["layer"] else ""
        print(f"{r['layer']:>6} | {r['depth_pct']:>6.1f}% | {r['auc']:>6.3f} | "
              f"{r['accuracy']:>8.1%} | {r['separation']:>+9.2f}σ{marker}")
    
    # Summary
    best = results[best_idx]
    print(f"\n{'='*70}")
    print(f"OPTIMAL LAYER: {best['layer']} ({best['depth_pct']:.1f}% depth)")
    print(f"  - AUC:        {best['auc']:.3f}")
    print(f"  - Accuracy:   {best['accuracy']:.1%}")
    print(f"  - Separation: {best['separation']:+.2f}σ")
    print(f"{'='*70}\n")
    
    return best


def plot_results(results: List[Dict], save_path: str = "layer_sweep_results.png"):
    """Generate and save a plot of results by layer."""
    
    try:
        import matplotlib.pyplot as plt
        
        layers = [r["layer"] for r in results]
        depths = [r["depth_pct"] for r in results]
        aucs = [r["auc"] for r in results]
        accuracies = [r["accuracy"] for r in results]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot AUC
        color1 = 'tab:blue'
        ax1.set_xlabel('Layer (Depth %)')
        ax1.set_ylabel('AUC', color=color1)
        ax1.plot(layers, aucs, 'o-', color=color1, label='AUC', linewidth=2, markersize=6)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim(0.4, 1.0)
        
        # Add depth percentage labels on x-axis
        ax1.set_xticks(layers[::2])  # Every other layer
        ax1.set_xticklabels([f"{l}\n({d:.0f}%)" for l, d in zip(layers[::2], depths[::2])])
        
        # Plot Accuracy on secondary y-axis
        ax2 = ax1.twinx()
        color2 = 'tab:orange'
        ax2.set_ylabel('Accuracy', color=color2)
        ax2.plot(layers, accuracies, 's--', color=color2, label='Accuracy', linewidth=2, markersize=6)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(0.4, 1.0)
        
        # Mark best layer
        best_idx = max(range(len(results)), key=lambda i: results[i]["auc"])
        best_layer = results[best_idx]["layer"]
        best_auc = results[best_idx]["auc"]
        ax1.axvline(x=best_layer, color='green', linestyle=':', alpha=0.7)
        ax1.annotate(f'Best: Layer {best_layer}', 
                     xy=(best_layer, best_auc),
                     xytext=(best_layer + 2, best_auc - 0.05),
                     fontsize=10,
                     arrowprops=dict(arrowstyle='->', color='green'))
        
        # Title and legend
        plt.title('Hallucination Detection Performance by Layer\n(Gemma-3-1B-IT)', fontsize=12)
        
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"Plot saved to: {save_path}")
        
    except ImportError:
        print("matplotlib not available - skipping plot generation")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"\n{'='*70}")
    print("ACTIVATION-BASED HALLUCINATION DETECTION: LAYER SWEEP DEMO")
    print(f"{'='*70}")
    print(f"\nModel: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Factual examples: {len(FACTUAL_STATEMENTS)}")
    print(f"Fabricated examples: {len(FABRICATED_STATEMENTS)}")
    
    # Load model and tokenizer
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    # Determine number of layers
    # For Gemma-3, layers are in model.model.layers (or model.language_model.layers for multimodal)
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        n_layers = len(model.model.layers)
    elif hasattr(model, 'language_model'):
        n_layers = len(model.language_model.model.layers)
    else:
        # Fallback: count from config
        n_layers = model.config.num_hidden_layers
    
    print(f"Model has {n_layers} layers")
    
    # Run layer sweep
    results = run_layer_sweep(
        model, tokenizer, n_layers,
        FACTUAL_STATEMENTS, FABRICATED_STATEMENTS
    )
    
    # Print results table
    best = print_results_table(results, n_layers)
    
    # Generate plot
    plot_results(results)
    
    # Return best layer info for programmatic use
    return {
        "optimal_layer": best["layer"],
        "optimal_depth_pct": best["depth_pct"],
        "optimal_auc": best["auc"],
        "all_results": results
    }


if __name__ == "__main__":
    main()
