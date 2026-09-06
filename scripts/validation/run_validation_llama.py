#!/usr/bin/env python3
"""
AHD Validation - LLAMA ONLY
Author: Glen Messenger | January 2026

Run this AFTER restarting your Python environment or in a fresh session.
Aggressive memory management to avoid OOM.
"""

import gc
import torch
import numpy as np
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Aggressive memory cleanup
# =============================================================================

def clear_memory():
    """Aggressively clear GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

def print_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

# =============================================================================
# Configuration - LLAMA ONLY
# =============================================================================

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_SHORT = "Llama-3.1-8B"

# =============================================================================
# Calibration Data
# =============================================================================

CALIBRATION_TRIPLETS = [
    ("What is the capital of France?", "Paris", "Lyon"),
    ("What is the capital of Japan?", "Tokyo", "Osaka"),
    ("What is the capital of Australia?", "Canberra", "Sydney"),
    ("What is the capital of Canada?", "Ottawa", "Toronto"),
    ("What is the capital of Brazil?", "Brasilia", "Rio de Janeiro"),
    ("What is the capital of Germany?", "Berlin", "Munich"),
    ("What is the capital of Italy?", "Rome", "Milan"),
    ("What is the capital of China?", "Beijing", "Shanghai"),
    ("What is the capital of India?", "New Delhi", "Mumbai"),
    ("What is the largest country by area?", "Russia", "Canada"),
    ("What is the capital of Spain?", "Madrid", "Barcelona"),
    ("What is the capital of Russia?", "Moscow", "St. Petersburg"),
    ("What continent is Brazil in?", "South America", "Africa"),
    ("What ocean is between the US and Europe?", "Atlantic Ocean", "Pacific Ocean"),
    ("What is the smallest continent?", "Australia", "Europe"),
    ("What is the chemical formula for water?", "H2O", "CO2"),
    ("What is the chemical symbol for gold?", "Au", "Ag"),
    ("What is the chemical symbol for silver?", "Ag", "Au"),
    ("What is the chemical symbol for iron?", "Fe", "Ir"),
    ("What is the atomic number of hydrogen?", "1", "2"),
    ("What is the atomic number of carbon?", "6", "12"),
    ("What is the atomic number of oxygen?", "8", "16"),
    ("What gas do plants absorb from air?", "Carbon dioxide", "Oxygen"),
    ("What gas do plants release?", "Oxygen", "Nitrogen"),
    ("What is the chemical formula for table salt?", "NaCl", "KCl"),
    ("What planet is closest to the Sun?", "Mercury", "Venus"),
    ("What is the largest planet in our solar system?", "Jupiter", "Saturn"),
    ("How many planets are in our solar system?", "Eight", "Nine"),
    ("What does Earth orbit around?", "The Sun", "The Moon"),
    ("What force keeps planets in orbit around the Sun?", "Gravity", "Magnetism"),
    ("What is the closest star to Earth?", "The Sun", "Proxima Centauri"),
    ("What is the freezing point of water in Celsius?", "0 degrees", "32 degrees"),
    ("What is the boiling point of water in Celsius?", "100 degrees", "212 degrees"),
    ("What particle has a positive charge?", "Proton", "Electron"),
    ("What particle has a negative charge?", "Electron", "Proton"),
    ("What year did World War II end?", "1945", "1943"),
    ("What year did World War I begin?", "1914", "1916"),
    ("What year was the US Declaration of Independence signed?", "1776", "1774"),
    ("What year did the Berlin Wall fall?", "1989", "1991"),
    ("What year did humans first land on the Moon?", "1969", "1972"),
    ("Who was the first President of the United States?", "George Washington", "Thomas Jefferson"),
    ("Who wrote the US Declaration of Independence?", "Thomas Jefferson", "Benjamin Franklin"),
    ("Who was the first person to walk on the Moon?", "Neil Armstrong", "Buzz Aldrin"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci", "Michelangelo"),
    ("What empire built the Colosseum in Rome?", "The Roman Empire", "The Greek Empire"),
    ("What is 2 + 2?", "4", "5"),
    ("What is 7 multiplied by 8?", "56", "54"),
    ("What is 100 divided by 5?", "20", "25"),
    ("What is the square root of 144?", "12", "14"),
    ("What is 3 squared?", "9", "6"),
    ("What is the value of pi to two decimal places?", "3.14", "3.41"),
    ("How many degrees are in a circle?", "360", "180"),
    ("How many degrees are in a right angle?", "90", "45"),
    ("How many sides does a hexagon have?", "6", "8"),
    ("How many sides does an octagon have?", "8", "6"),
    ("Who wrote Romeo and Juliet?", "William Shakespeare", "Charles Dickens"),
    ("Who wrote Hamlet?", "William Shakespeare", "Christopher Marlowe"),
    ("Who wrote Pride and Prejudice?", "Jane Austen", "Charlotte Bronte"),
    ("Who wrote 1984?", "George Orwell", "Aldous Huxley"),
    ("Who wrote Brave New World?", "Aldous Huxley", "George Orwell"),
    ("Who wrote The Great Gatsby?", "F. Scott Fitzgerald", "Ernest Hemingway"),
    ("Who wrote War and Peace?", "Leo Tolstoy", "Fyodor Dostoevsky"),
    ("Who wrote Harry Potter?", "J.K. Rowling", "J.R.R. Tolkien"),
    ("Who wrote The Lord of the Rings?", "J.R.R. Tolkien", "C.S. Lewis"),
    ("Who wrote Don Quixote?", "Miguel de Cervantes", "Gabriel Garcia Marquez"),
]

EVAL_QUESTIONS = [
    {"q": "Who is Dr. Marcus Wellby, the Stanford professor who won the 2019 Nobel Prize in Economics?", "category": "fake_entity"},
    {"q": "What is the plot of the 2023 novel 'The Chromatic Paradox' by Jennifer Ashworth?", "category": "fake_entity"},
    {"q": "When was Nextera Dynamics founded and what do they manufacture?", "category": "fake_entity"},
    {"q": "What theorem did mathematician Heinrich Volkov prove in 1987?", "category": "fake_entity"},
    {"q": "What year did Netflix acquire Blockbuster?", "category": "false_premise"},
    {"q": "What was the name of the third person to walk on the Moon during Apollo 11?", "category": "false_premise"},
    {"q": "What was the exact population of Tokyo on January 1, 2023?", "category": "precise_number"},
    {"q": "What was Tesla's exact stock price at market close on March 15, 2023?", "category": "precise_number"},
    {"q": "What is the national bird of Lesotho?", "category": "obscure_fact"},
    {"q": "Who was the Prime Minister of Malta in 1987?", "category": "obscure_fact"},
    {"q": "What is the capital of France?", "category": "control"},
    {"q": "Who wrote Pride and Prejudice?", "category": "control"},
    {"q": "What is the chemical formula for water?", "category": "control"},
    {"q": "In what year did World War II end?", "category": "control"},
    {"q": "What planet is closest to the Sun?", "category": "control"},
    {"q": "What is 7 multiplied by 8?", "category": "control"},
    {"q": "Who painted the Mona Lisa?", "category": "control"},
    {"q": "What is the largest planet in our solar system?", "category": "control"},
]

# =============================================================================
# Core Functions  
# =============================================================================

@dataclass
class LayerResult:
    layer: int
    depth_pct: float
    auc: float
    mean_factual: float
    mean_fabricated: float
    separation_sigma: float

def load_model():
    """Load Llama with memory-efficient settings."""
    print(f"Loading {MODEL_NAME}...")
    clear_memory()
    print_memory()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load with more aggressive memory settings
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    print_memory()
    return model, tokenizer

def get_n_layers(model) -> int:
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return len(model.model.layers)
    return model.config.num_hidden_layers

def format_as_chat(tokenizer, question: str, answer: str) -> str:
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]
    if hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(messages, tokenize=False)
    return f"Question: {question}\nAnswer: {answer}"

def get_hidden_state_for_answer(model, tokenizer, question: str, answer: str, layer: int) -> torch.Tensor:
    text = format_as_chat(tokenizer, question, answer)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden = outputs.hidden_states[layer + 1][0, -1, :].float().cpu()
    
    # Explicit cleanup
    del outputs, inputs
    
    return hidden

def compute_direction_vector(model, tokenizer, layer: int, 
                            triplets: List[Tuple[str, str, str]]) -> torch.Tensor:
    factual_hiddens = []
    fabricated_hiddens = []
    
    for i, (question, correct, wrong) in enumerate(triplets):
        h_correct = get_hidden_state_for_answer(model, tokenizer, question, correct, layer)
        h_wrong = get_hidden_state_for_answer(model, tokenizer, question, wrong, layer)
        factual_hiddens.append(h_correct)
        fabricated_hiddens.append(h_wrong)
        
        # Periodic cleanup
        if i % 20 == 0:
            clear_memory()
    
    factual_mean = torch.stack(factual_hiddens).mean(dim=0)
    fabricated_mean = torch.stack(fabricated_hiddens).mean(dim=0)
    
    direction = factual_mean - fabricated_mean
    return direction / (direction.norm() + 1e-8)

def layer_sweep(model, tokenizer, n_layers: int, 
                triplets: List[Tuple[str, str, str]]) -> List[LayerResult]:
    n_train = int(len(triplets) * 0.7)
    train_triplets = triplets[:n_train]
    val_triplets = triplets[n_train:]
    
    print(f"Train: {len(train_triplets)} triplets, Val: {len(val_triplets)} triplets")
    
    results = []
    for layer in range(n_layers):
        direction = compute_direction_vector(model, tokenizer, layer, train_triplets)
        
        scores = []
        labels = []
        
        for question, correct, wrong in val_triplets:
            h_correct = get_hidden_state_for_answer(model, tokenizer, question, correct, layer)
            h_wrong = get_hidden_state_for_answer(model, tokenizer, question, wrong, layer)
            
            score_correct = h_correct.dot(direction).item()
            score_wrong = h_wrong.dot(direction).item()
            
            scores.extend([score_correct, score_wrong])
            labels.extend([1, 0])
        
        scores = np.array(scores)
        labels = np.array(labels)
        
        auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.5
        mean_f = scores[labels == 1].mean()
        mean_fab = scores[labels == 0].mean()
        sep = (mean_f - mean_fab) / (scores.std() + 1e-8)
        
        results.append(LayerResult(layer, layer/n_layers*100, auc, mean_f, mean_fab, sep))
        
        if layer % 5 == 0:
            print(f"  Layer {layer}/{n_layers}: AUC={auc:.3f}, sep={sep:+.2f}σ")
            print_memory()
        
        # Cleanup after each layer
        clear_memory()
    
    return results

def evaluate_generation_one_by_one(model, tokenizer, direction: torch.Tensor, 
                                    layer: int, questions: List[Dict]) -> List[Dict]:
    """Evaluate one question at a time with memory cleanup between each."""
    results = []
    
    for i, q_data in enumerate(questions):
        print(f"\n  [{i+1}/{len(questions)}] Evaluating: {q_data['category']}")
        clear_memory()
        print_memory()
        
        question = q_data["q"]
        
        messages = [{"role": "user", "content": question}]
        if hasattr(tokenizer, 'apply_chat_template'):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"Question: {question}\nAnswer:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs.input_ids.shape[1]
        
        # Generate with minimal tokens to reduce memory
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,  # Reduced from 100
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,  # Deterministic
            )
        
        generated_ids = outputs.sequences[0, prompt_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Get AHD score
        last_hidden = outputs.hidden_states[-1][layer + 1][0, -1, :].float().cpu()
        ahd_score = last_hidden.dot(direction).item()
        
        # Get logprob - do this separately to manage memory
        del outputs
        clear_memory()
        
        # Recompute logits for logprob (simpler approach)
        full_text = prompt + response
        full_inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            logits = model(**full_inputs).logits
        
        # Approximate mean logprob
        log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
        token_ids = full_inputs.input_ids[0, 1:]
        token_lps = log_probs.gather(1, token_ids.unsqueeze(1)).squeeze()
        
        # Only consider generated portion
        gen_lps = token_lps[prompt_len-1:]
        mean_lp = gen_lps.mean().item() if gen_lps.numel() > 0 else 0.0
        
        del logits, log_probs, full_inputs
        clear_memory()
        
        uncertainty_phrases = [
            "i don't know", "i'm not sure", "i cannot", "i can't", 
            "don't have", "no information", "not aware", "unclear",
            "couldn't find", "doesn't appear", "not certain",
            "unfortunately", "cannot provide", "i apologize"
        ]
        shows_uncertainty = any(p in response.lower() for p in uncertainty_phrases)
        
        results.append({
            "question": question,
            "category": q_data["category"],
            "response": response[:300],
            "ahd_score": ahd_score,
            "mean_logprob": mean_lp,
            "shows_uncertainty": shows_uncertainty
        })
        
        print(f"    AHD={ahd_score:+7.1f} | LP={mean_lp:.2f} | unc={shows_uncertainty}")
    
    return results

def compute_metrics(eval_results: List[Dict]) -> Dict:
    labels = np.array([1 if r["category"] == "control" else 0 for r in eval_results])
    ahd = np.array([r["ahd_score"] for r in eval_results])
    lp = np.array([r["mean_logprob"] for r in eval_results])
    
    ahd_auc = roc_auc_score(labels, ahd) if len(set(labels)) > 1 else 0.5
    lp_auc = roc_auc_score(labels, lp) if len(set(labels)) > 1 else 0.5
    
    return {
        "ahd_auc": float(ahd_auc),
        "logprob_auc": float(lp_auc),
        "mean_ahd_control": float(ahd[labels == 1].mean()),
        "mean_ahd_other": float(ahd[labels == 0].mean()),
        "mean_lp_control": float(lp[labels == 1].mean()),
        "mean_lp_other": float(lp[labels == 0].mean()),
    }

# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print("AHD VALIDATION - LLAMA ONLY")
    print("Author: Glen Messenger | January 2026")
    print("="*70)
    
    # Initial cleanup
    print("\nClearing memory before start...")
    clear_memory()
    print_memory()
    
    # Load model
    model, tokenizer = load_model()
    n_layers = get_n_layers(model)
    hidden_dim = model.config.hidden_size
    print(f"Layers: {n_layers}, Hidden dim: {hidden_dim}")
    
    # Layer sweep
    print(f"\nLayer sweep ({len(CALIBRATION_TRIPLETS)} triplets)...")
    layer_results = layer_sweep(model, tokenizer, n_layers, CALIBRATION_TRIPLETS)
    
    optimal = max(layer_results, key=lambda x: x.auc)
    print(f"\n*** Optimal: layer {optimal.layer} ({optimal.depth_pct:.1f}%), AUC={optimal.auc:.3f} ***")
    
    # Compute direction at optimal layer
    print("\nComputing final direction vector...")
    clear_memory()
    direction = compute_direction_vector(model, tokenizer, optimal.layer, CALIBRATION_TRIPLETS)
    
    vec_path = f"direction_{MODEL_SHORT.replace('-','_').replace('.','_').lower()}.pt"
    torch.save(direction, vec_path)
    print(f"Saved: {vec_path}")
    
    # Evaluate one by one
    print(f"\nEvaluating on {len(EVAL_QUESTIONS)} questions (one at a time)...")
    clear_memory()
    eval_results = evaluate_generation_one_by_one(model, tokenizer, direction, optimal.layer, EVAL_QUESTIONS)
    
    # Compute metrics
    metrics = compute_metrics(eval_results)
    print(f"\n*** AHD AUC: {metrics['ahd_auc']:.3f} | Logprob AUC: {metrics['logprob_auc']:.3f} ***")
    print(f"    Mean AHD (control): {metrics['mean_ahd_control']:+.1f}")
    print(f"    Mean AHD (other):   {metrics['mean_ahd_other']:+.1f}")
    
    # Save results
    result = {
        "model": MODEL_SHORT,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "optimal_layer": optimal.layer,
        "optimal_depth_pct": optimal.depth_pct,
        "optimal_auc": optimal.auc,
        "metrics": metrics,
        "eval_results": eval_results,
        "layer_results": [asdict(r) for r in layer_results]
    }
    
    with open("validation_results_llama.json", "w") as f:
        json.dump({
            "metadata": {
                "author": "Glen Messenger",
                "date": "January 2026",
                "model": MODEL_NAME,
                "n_calibration_triplets": len(CALIBRATION_TRIPLETS),
                "n_eval_questions": len(EVAL_QUESTIONS)
            },
            "result": result
        }, f, indent=2)
    print("\nSaved: validation_results_llama.json")
    
    # Plot layer sweep
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    layers = [r.layer for r in layer_results]
    axes[0].plot(layers, [r.auc for r in layer_results], 'b-', lw=2)
    axes[0].scatter([optimal.layer], [optimal.auc], c='red', s=100, zorder=5)
    axes[0].set_xlabel('Layer'); axes[0].set_ylabel('AUC')
    axes[0].set_title(f'{MODEL_SHORT}: Layer-wise AUC'); axes[0].grid(alpha=0.3)
    axes[0].axhline(0.5, color='gray', ls='--', alpha=0.5)
    
    axes[1].plot(layers, [r.separation_sigma for r in layer_results], 'b-', lw=2)
    axes[1].set_xlabel('Layer'); axes[1].set_ylabel('Separation (σ)')
    axes[1].set_title(f'{MODEL_SHORT}: Mean Separation'); axes[1].grid(alpha=0.3)
    axes[1].axhline(0, color='gray', ls='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("figure_layer_sweep_llama.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: figure_layer_sweep_llama.png")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Model:         {MODEL_SHORT}")
    print(f"Layers:        {n_layers}")
    print(f"Optimal Layer: {optimal.layer} ({optimal.depth_pct:.1f}%)")
    print(f"Optimal AUC:   {optimal.auc:.3f}")
    print(f"Eval AHD AUC:  {metrics['ahd_auc']:.3f}")
    print(f"Eval LP AUC:   {metrics['logprob_auc']:.3f}")
    print("\nDone!")

if __name__ == "__main__":
    main()
