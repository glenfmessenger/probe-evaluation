#!/usr/bin/env python3
"""
AHD Scale vs Architecture Test: Gemma-2-9B-IT
==============================================

Disentangles whether AHD's failure on Gemma-1B is due to:
- Scale (1B too small) → If Gemma-9B works, scale is the issue
- Architecture (Gemma vs Llama) → If Gemma-9B fails, architecture is the issue

Results so far:
- Gemma-3-1B-IT:  AUC 0.638, +0.45σ (weak)
- Llama-3.1-8B:   AUC 0.985, +2.75σ (strong)

This test: Gemma-2-9B-IT (similar scale to Llama-8B, same family as Gemma-1B)

Author: Glen Junor
Date: January 2025
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "google/gemma-2-9b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_RATIO = 0.7

# ============================================================================
# Q&A DATASET (Same 76 pairs as all other tests)
# ============================================================================

QA_DATASET = [
    # === GEOGRAPHY ===
    ("What is the capital of France?", "Paris", True),
    ("What is the capital of France?", "Lyon", False),
    ("What is the capital of France?", "Marseille", False),
    ("What is the capital of Japan?", "Tokyo", True),
    ("What is the capital of Japan?", "Osaka", False),
    ("What is the capital of Japan?", "Kyoto", False),
    ("What is the capital of Germany?", "Berlin", True),
    ("What is the capital of Germany?", "Munich", False),
    ("What is the capital of Italy?", "Rome", True),
    ("What is the capital of Italy?", "Milan", False),
    ("What is the capital of Spain?", "Madrid", True),
    ("What is the capital of Spain?", "Barcelona", False),
    ("What is the capital of Australia?", "Canberra", True),
    ("What is the capital of Australia?", "Sydney", False),
    ("What is the capital of Canada?", "Ottawa", True),
    ("What is the capital of Canada?", "Toronto", False),
    ("What is the capital of Brazil?", "Brasilia", True),
    ("What is the capital of Brazil?", "Rio de Janeiro", False),
    ("What is the capital of China?", "Beijing", True),
    ("What is the capital of China?", "Shanghai", False),
    
    # === MATH ===
    ("What is 2 + 2?", "4", True),
    ("What is 2 + 2?", "5", False),
    ("What is 3 x 3?", "9", True),
    ("What is 3 x 3?", "6", False),
    ("What is 10 / 2?", "5", True),
    ("What is 10 / 2?", "4", False),
    ("What is the square root of 16?", "4", True),
    ("What is the square root of 16?", "8", False),
    ("What is the square root of 25?", "5", True),
    ("What is the square root of 25?", "6", False),
    ("What is 7 x 8?", "56", True),
    ("What is 7 x 8?", "54", False),
    ("What is 12 x 12?", "144", True),
    ("What is 12 x 12?", "124", False),
    ("What is 15% of 100?", "15", True),
    ("What is 15% of 100?", "20", False),
    
    # === SCIENCE ===
    ("What is the chemical symbol for water?", "H2O", True),
    ("What is the chemical symbol for water?", "CO2", False),
    ("What is the chemical symbol for gold?", "Au", True),
    ("What is the chemical symbol for gold?", "Ag", False),
    ("How many planets are in our solar system?", "8", True),
    ("How many planets are in our solar system?", "9", False),
    ("What is the largest planet in our solar system?", "Jupiter", True),
    ("What is the largest planet in our solar system?", "Saturn", False),
    ("What is the speed of light in a vacuum?", "299,792 kilometers per second", True),
    ("What is the speed of light in a vacuum?", "150,000 kilometers per second", False),
    ("What is the atomic number of carbon?", "6", True),
    ("What is the atomic number of carbon?", "12", False),
    ("What is the freezing point of water in Celsius?", "0 degrees", True),
    ("What is the freezing point of water in Celsius?", "32 degrees", False),
    ("What is the boiling point of water at sea level in Celsius?", "100 degrees", True),
    ("What is the boiling point of water at sea level in Celsius?", "212 degrees", False),
    
    # === HISTORY ===
    ("In what year did World War II end?", "1945", True),
    ("In what year did World War II end?", "1943", False),
    ("In what year did the Berlin Wall fall?", "1989", True),
    ("In what year did the Berlin Wall fall?", "1991", False),
    ("Who was the first person to walk on the Moon?", "Neil Armstrong", True),
    ("Who was the first person to walk on the Moon?", "Buzz Aldrin", False),
    ("In what year was the Declaration of Independence signed?", "1776", True),
    ("In what year was the Declaration of Independence signed?", "1781", False),
    ("Who wrote Romeo and Juliet?", "William Shakespeare", True),
    ("Who wrote Romeo and Juliet?", "Charles Dickens", False),
    ("What year did the Titanic sink?", "1912", True),
    ("What year did the Titanic sink?", "1915", False),
    
    # === LANGUAGE/CULTURE ===
    ("What language is primarily spoken in Brazil?", "Portuguese", True),
    ("What language is primarily spoken in Brazil?", "Spanish", False),
    ("What is the currency of Japan?", "Yen", True),
    ("What is the currency of Japan?", "Yuan", False),
    ("What is the currency of the United Kingdom?", "Pound Sterling", True),
    ("What is the currency of the United Kingdom?", "Euro", False),
    ("How many continents are there?", "7", True),
    ("How many continents are there?", "6", False),
    ("What is the largest ocean on Earth?", "Pacific Ocean", True),
    ("What is the largest ocean on Earth?", "Atlantic Ocean", False),
    ("What is the longest river in the world?", "Nile", True),
    ("What is the longest river in the world?", "Amazon", False),
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_qa(question: str, answer: str) -> str:
    """Format Q&A in simple style (NOT chat template)."""
    return f"Question: {question}\nAnswer: {answer}"


def get_hidden_state(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    layer_idx: int
) -> torch.Tensor:
    """Extract hidden state from specified layer at last token position."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
    
    hidden_states = outputs.hidden_states[layer_idx + 1]
    last_token_hidden = hidden_states[0, -1, :].float().cpu()
    
    return last_token_hidden


def compute_direction_vector(
    correct_hiddens: List[torch.Tensor],
    incorrect_hiddens: List[torch.Tensor]
) -> torch.Tensor:
    """Compute normalized direction vector."""
    correct_stack = torch.stack(correct_hiddens)
    incorrect_stack = torch.stack(incorrect_hiddens)
    
    direction = correct_stack.mean(dim=0) - incorrect_stack.mean(dim=0)
    direction = direction / (direction.norm() + 1e-8)
    
    return direction


def evaluate_layer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layer_idx: int,
    dataset: List[Tuple[str, str, bool]],
    train_ratio: float = 0.7
) -> Dict[str, float]:
    """Evaluate AHD at specified layer."""
    
    np.random.seed(42)
    indices = np.random.permutation(len(dataset))
    shuffled = [dataset[i] for i in indices]
    
    n_train = int(len(shuffled) * train_ratio)
    train_data = shuffled[:n_train]
    test_data = shuffled[n_train:]
    
    # Training
    train_correct_hiddens = []
    train_incorrect_hiddens = []
    
    for question, answer, is_correct in train_data:
        text = format_qa(question, answer)
        hidden = get_hidden_state(model, tokenizer, text, layer_idx)
        
        if is_correct:
            train_correct_hiddens.append(hidden)
        else:
            train_incorrect_hiddens.append(hidden)
    
    if len(train_correct_hiddens) == 0 or len(train_incorrect_hiddens) == 0:
        return {"accuracy": 0.5, "auc": 0.5, "separation": 0.0}
    
    direction = compute_direction_vector(train_correct_hiddens, train_incorrect_hiddens)
    
    # Testing
    test_scores = []
    test_labels = []
    
    for question, answer, is_correct in test_data:
        text = format_qa(question, answer)
        hidden = get_hidden_state(model, tokenizer, text, layer_idx)
        
        score = hidden.dot(direction).item()
        test_scores.append(score)
        test_labels.append(1 if is_correct else 0)
    
    test_scores = np.array(test_scores)
    test_labels = np.array(test_labels)
    
    # Metrics
    try:
        auc = roc_auc_score(test_labels, test_scores)
    except ValueError:
        auc = 0.5
    
    threshold = np.median(test_scores)
    predictions = (test_scores > threshold).astype(int)
    accuracy = accuracy_score(test_labels, predictions)
    
    correct_scores = test_scores[test_labels == 1]
    incorrect_scores = test_scores[test_labels == 0]
    
    if len(correct_scores) > 0 and len(incorrect_scores) > 0:
        pooled_std = np.sqrt((np.var(correct_scores) + np.var(incorrect_scores)) / 2)
        if pooled_std > 1e-8:
            separation = (correct_scores.mean() - incorrect_scores.mean()) / pooled_std
        else:
            separation = 0.0
    else:
        separation = 0.0
    
    return {
        "accuracy": accuracy,
        "auc": auc,
        "separation": separation
    }


def run_layer_sweep(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    n_layers: int,
    dataset: List[Tuple[str, str, bool]]
) -> List[Dict]:
    """Sweep through all layers."""
    
    results = []
    
    print(f"\n{'='*70}")
    print(f"LAYER SWEEP: {n_layers} layers | {len(dataset)} Q&A pairs")
    print(f"{'='*70}\n")
    
    print(f"{'Layer':>5} | {'Depth%':>6} | {'Accuracy':>8} | {'AUC':>6} | {'Separation':>10}")
    print("-" * 55)
    
    for layer_idx in range(n_layers):
        depth_pct = (layer_idx + 1) / n_layers * 100
        
        metrics = evaluate_layer(model, tokenizer, layer_idx, dataset, TRAIN_RATIO)
        metrics["layer"] = layer_idx
        metrics["depth_pct"] = depth_pct
        results.append(metrics)
        
        print(f"{layer_idx:>5} | {depth_pct:>5.1f}% | {metrics['accuracy']:>7.1%} | "
              f"{metrics['auc']:>6.3f} | {metrics['separation']:>+9.2f}σ")
    
    return results


def print_summary(results: List[Dict]):
    """Print summary with cross-model comparison."""
    
    best_idx = max(range(len(results)), key=lambda i: results[i]["auc"])
    best = results[best_idx]
    
    print(f"\n{'='*70}")
    print("SUMMARY: Gemma-2-9B-IT")
    print(f"{'='*70}")
    
    print(f"\n🎯 OPTIMAL LAYER: {best['layer']} ({best['depth_pct']:.1f}% depth)")
    print(f"   Accuracy:   {best['accuracy']:.1%}")
    print(f"   AUC:        {best['auc']:.3f}")
    print(f"   Separation: {best['separation']:+.2f}σ")
    
    # Top 5
    print(f"\nTop 5 Layers (by AUC):")
    sorted_results = sorted(results, key=lambda x: x["auc"], reverse=True)[:5]
    for i, r in enumerate(sorted_results, 1):
        print(f"  {i}. Layer {r['layer']:2d} ({r['depth_pct']:5.1f}%): "
              f"AUC={r['auc']:.3f}, Sep={r['separation']:+.2f}σ")
    
    # Cross-model comparison
    print(f"\n{'='*70}")
    print("CROSS-MODEL COMPARISON (Scale vs Architecture):")
    print(f"{'='*70}")
    print(f"\n{'Model':<25} | {'Size':>6} | {'Best Layer':>10} | {'AUC':>6} | {'Separation':>10}")
    print("-" * 75)
    print(f"{'Gemma-3-1B-IT':<25} | {'1B':>6} | {'9 (38.5%)':>10} | {'0.638':>6} | {'+0.45σ':>10}")
    print(f"{'Gemma-2-9B-IT':<25} | {'9B':>6} | {f'{best[\"layer\"]} ({best[\"depth_pct\"]:.0f}%)':>10} | {best['auc']:>6.3f} | {best['separation']:>+9.2f}σ")
    print(f"{'Llama-3.1-8B-Instruct':<25} | {'8B':>6} | {'30 (96.9%)':>10} | {'0.985':>6} | {'+2.75σ':>10}")
    print("-" * 75)
    
    # Interpretation
    print(f"\n{'='*70}")
    print("📊 INTERPRETATION:")
    print(f"{'='*70}")
    
    if best['auc'] > 0.85:
        print("\n  ✅ SCALE EFFECT CONFIRMED")
        print("     Gemma-9B shows strong signal like Llama-8B")
        print("     → AHD failure on Gemma-1B was due to insufficient scale")
        print("     → Larger models encode factual confidence more clearly")
        conclusion = "scale"
    elif best['auc'] > 0.70:
        print("\n  ⚠️  MIXED RESULT")
        print("     Gemma-9B shows moderate signal (better than 1B, worse than Llama)")
        print("     → Both scale AND architecture may contribute")
        conclusion = "mixed"
    else:
        print("\n  ❌ ARCHITECTURE EFFECT CONFIRMED")
        print("     Gemma-9B still shows weak signal despite similar scale to Llama-8B")
        print("     → Gemma architecture encodes factual confidence differently")
        print("     → Llama's success is architecture-specific, not just scale")
        conclusion = "architecture"
    
    # Paper statement
    print(f"\n{'='*70}")
    print("PAPER STATEMENT:")
    print(f"{'='*70}")
    
    if conclusion == "scale":
        print(f'''
  "AHD performance scales with model size. On Gemma-1B (0.64 AUC), linear
   probes show weak separation, while Gemma-9B ({best['auc']:.2f} AUC) and 
   Llama-8B (0.98 AUC) achieve strong classification. This suggests that
   larger models develop more linearly-separable representations of
   epistemic confidence."
''')
    elif conclusion == "architecture":
        print(f'''
  "AHD performance is architecture-dependent. Despite similar scale,
   Llama-8B (0.98 AUC, +2.75σ) significantly outperforms Gemma-9B 
   ({best['auc']:.2f} AUC, {best['separation']:+.2f}σ). This suggests that Llama's
   architecture encodes factual confidence in a more linearly-separable
   manner than Gemma, independent of model scale."
''')
    else:
        print(f'''
  "AHD performance depends on both scale and architecture. Gemma-9B
   ({best['auc']:.2f} AUC) outperforms Gemma-1B (0.64 AUC) but underperforms
   Llama-8B (0.98 AUC), suggesting that while larger models encode
   factual confidence more clearly, architectural differences also
   play a significant role."
''')
    
    print(f"{'='*70}\n")
    
    return best, conclusion


def plot_results(results: List[Dict], save_path: str = "ahd_layer_sweep_gemma9b.png"):
    """Generate comparison plot."""
    try:
        import matplotlib.pyplot as plt
        
        layers = [r["layer"] for r in results]
        aucs = [r["auc"] for r in results]
        separations = [r["separation"] for r in results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: AUC comparison
        ax1.plot(layers, aucs, 'o-', linewidth=2, markersize=5, color='tab:green', label='Gemma-2-9B')
        ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random')
        ax1.axhline(y=0.638, color='tab:blue', linestyle='--', alpha=0.7, label='Gemma-1B (0.638)')
        ax1.axhline(y=0.985, color='tab:red', linestyle='--', alpha=0.7, label='Llama-8B (0.985)')
        
        best_idx = max(range(len(results)), key=lambda i: results[i]["auc"])
        ax1.axvline(x=results[best_idx]["layer"], color='darkgreen', linestyle=':', alpha=0.7)
        
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('AUC')
        ax1.set_title('AHD Performance: Gemma-2-9B-IT')
        ax1.set_ylim(0.3, 1.05)
        ax1.legend(loc='lower right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Separation
        ax2.bar(layers, separations, color='tab:green', alpha=0.7)
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax2.axhline(y=0.45, color='tab:blue', linestyle='--', alpha=0.7, label='Gemma-1B (+0.45σ)')
        ax2.axhline(y=2.75, color='tab:red', linestyle='--', alpha=0.7, label='Llama-8B (+2.75σ)')
        
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Separation (σ)')
        ax2.set_title('Class Separation by Layer')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('AHD Scale vs Architecture Test: Gemma-2-9B-IT', fontsize=11)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"Plot saved: {save_path}")
        
    except ImportError:
        print("matplotlib not available")


def main():
    print(f"\n{'='*70}")
    print("AHD SCALE VS ARCHITECTURE TEST: Gemma-2-9B-IT")
    print(f"{'='*70}")
    print(f"\nHypothesis Test:")
    print(f"  - If Gemma-9B works → Scale effect (1B too small)")
    print(f"  - If Gemma-9B fails → Architecture effect (Gemma ≠ Llama)")
    print(f"\nModel: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    
    n_correct = sum(1 for _, _, c in QA_DATASET if c)
    n_incorrect = sum(1 for _, _, c in QA_DATASET if not c)
    print(f"Dataset: {len(QA_DATASET)} Q&A pairs ({n_correct} correct, {n_incorrect} incorrect)")
    print(f"Format: Simple Q&A (NOT chat template)")
    print(f"Train/Test: {TRAIN_RATIO:.0%} / {1-TRAIN_RATIO:.0%}")
    
    # Load model
    print(f"\nLoading model (this may take a minute)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    # Get layer count
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        n_layers = len(model.model.layers)
    else:
        n_layers = model.config.num_hidden_layers
    
    print(f"Layers: {n_layers}")
    
    # Run sweep
    results = run_layer_sweep(model, tokenizer, n_layers, QA_DATASET)
    
    # Summary
    best, conclusion = print_summary(results)
    
    # Plot
    plot_results(results)
    
    # Save results
    import json
    output = {
        "model": MODEL_NAME,
        "n_layers": n_layers,
        "dataset_size": len(QA_DATASET),
        "best_layer": best["layer"],
        "best_depth_pct": best["depth_pct"],
        "best_auc": best["auc"],
        "best_separation": best["separation"],
        "conclusion": conclusion,
        "all_results": results
    }
    
    with open("ahd_gemma9b_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Results saved to: ahd_gemma9b_results.json")
    
    return results, conclusion


if __name__ == "__main__":
    main()
