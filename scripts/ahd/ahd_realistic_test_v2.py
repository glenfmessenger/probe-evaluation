#!/usr/bin/env python3
"""
Realistic Hallucination Detection: AHD vs Logprobs
===================================================

Tests whether AHD can detect hallucinations in realistic scenarios:
1. Ask Gemma questions (mix of knowable and obscure)
2. Let model GENERATE answers (not pre-written)
3. Score with AHD activation probe AND logprobs
4. Compare which better predicts correctness

Phase 2: Demonstrate "selective abstention" - improve accuracy by declining low-confidence answers

Author: Glen Junor
Date: January 2025
"""

import torch
import numpy as np
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "google/gemma-2-9b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AHD_LAYER = 22  # Optimal layer from previous experiments

# ============================================================================
# TEST DATASET: Mix of knowable facts and obscure/recent information
# ============================================================================

# Format: (question, correct_answer, difficulty, category)
# difficulty: "easy" (model should know), "hard" (model might not know), "trap" (common misconception)

TEST_QUESTIONS = [
    # === EASY: Model should definitely know these ===
    ("What is the capital of France?", "Paris", "easy", "geography"),
    ("What is 7 x 8?", "56", "easy", "math"),
    ("Who wrote Romeo and Juliet?", "William Shakespeare", "easy", "literature"),
    ("What is the chemical symbol for gold?", "Au", "easy", "science"),
    ("What planet is known as the Red Planet?", "Mars", "easy", "science"),
    ("What is the largest ocean on Earth?", "Pacific Ocean", "easy", "geography"),
    ("How many days are in a leap year?", "366", "easy", "general"),
    ("What is the square root of 144?", "12", "easy", "math"),
    ("What language is primarily spoken in Brazil?", "Portuguese", "easy", "geography"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci", "easy", "art"),
    
    # === MEDIUM: Model probably knows but might be uncertain ===
    ("What year did the Berlin Wall fall?", "1989", "medium", "history"),
    ("What is the atomic number of carbon?", "6", "medium", "science"),
    ("Who was the first person to walk on the Moon?", "Neil Armstrong", "medium", "history"),
    ("What is the capital of Australia?", "Canberra", "medium", "geography"),
    ("What is the speed of light in km/s (approximately)?", "300000", "medium", "science"),
    ("Who wrote '1984'?", "George Orwell", "medium", "literature"),
    ("What is the largest mammal?", "Blue whale", "medium", "science"),
    ("In what year did World War I begin?", "1914", "medium", "history"),
    ("What is the currency of Japan?", "Yen", "medium", "geography"),
    ("Who discovered penicillin?", "Alexander Fleming", "medium", "science"),
    
    # === HARD: Obscure facts model might hallucinate ===
    ("What is the capital of Myanmar?", "Naypyidaw", "hard", "geography"),
    ("Who was the 13th President of the United States?", "Millard Fillmore", "hard", "history"),
    ("What year was the Treaty of Westphalia signed?", "1648", "hard", "history"),
    ("What is the atomic number of Selenium?", "34", "hard", "science"),
    ("Who wrote 'The Crying of Lot 49'?", "Thomas Pynchon", "hard", "literature"),
    ("What is the capital of Bhutan?", "Thimphu", "hard", "geography"),
    ("In what year was the Magna Carta signed?", "1215", "hard", "history"),
    ("What is the chemical formula for sulfuric acid?", "H2SO4", "hard", "science"),
    ("Who composed 'The Rite of Spring'?", "Igor Stravinsky", "hard", "music"),
    ("What is the tallest mountain in Africa?", "Mount Kilimanjaro", "hard", "geography"),
    
    # === TRAP: Common misconceptions ===
    ("What is the largest desert in the world?", "Antarctica", "trap", "geography"),  # Not Sahara
    ("How many states does the US have?", "50", "trap", "geography"),  # Sometimes people say 52
    ("What is the closest planet to the Sun?", "Mercury", "trap", "science"),  # Not Venus
    ("Who invented the telephone?", "Alexander Graham Bell", "trap", "history"),  # Contested but standard answer
    ("What is the hardest natural substance?", "Diamond", "trap", "science"),
    
    # === NUMERICAL: Easy to verify, hard to hallucinate correctly ===
    ("How many bones are in the adult human body?", "206", "hard", "science"),
    ("What is the boiling point of water in Fahrenheit?", "212", "medium", "science"),
    ("How many chromosomes do humans have?", "46", "medium", "science"),
    ("What year was the United Nations founded?", "1945", "medium", "history"),
    ("How many planets are in our solar system?", "8", "easy", "science"),
]


@dataclass
class GenerationResult:
    """Stores results from a single generation"""
    question: str
    correct_answer: str
    generated_answer: str
    is_correct: bool
    difficulty: str
    category: str
    ahd_score: float
    mean_logprob: float
    min_logprob: float
    answer_perplexity: float


def load_model():
    """Load Gemma-2-9B with logprobs capability"""
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    return model, tokenizer


def compute_ahd_direction(model, tokenizer, layer_idx: int) -> torch.Tensor:
    """
    Compute AHD direction vector from calibration set.
    Uses a small set of obviously correct/incorrect statements.
    """
    calibration_data = [
        ("The capital of France is Paris", True),
        ("The capital of France is London", False),
        ("Water is H2O", True),
        ("Water is CO2", False),
        ("2 + 2 equals 4", True),
        ("2 + 2 equals 5", False),
        ("The Earth orbits the Sun", True),
        ("The Sun orbits the Earth", False),
        ("Humans have 206 bones", True),
        ("Humans have 500 bones", False),
        ("Shakespeare wrote Hamlet", True),
        ("Shakespeare wrote Harry Potter", False),
        ("Tokyo is in Japan", True),
        ("Tokyo is in China", False),
        ("Gold's symbol is Au", True),
        ("Gold's symbol is Ag", False),
    ]
    
    correct_hiddens = []
    incorrect_hiddens = []
    
    for statement, is_correct in calibration_data:
        inputs = tokenizer(statement, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[layer_idx + 1][0, -1, :].float().cpu()
        
        if is_correct:
            correct_hiddens.append(hidden)
        else:
            incorrect_hiddens.append(hidden)
    
    correct_mean = torch.stack(correct_hiddens).mean(dim=0)
    incorrect_mean = torch.stack(incorrect_hiddens).mean(dim=0)
    
    direction = correct_mean - incorrect_mean
    direction = direction / (direction.norm() + 1e-8)
    
    return direction


def generate_with_scores(
    model,
    tokenizer,
    question: str,
    ahd_direction: torch.Tensor,
    layer_idx: int,
    max_new_tokens: int = 50
) -> Tuple[str, float, float, float, float]:
    """
    Generate answer and compute both AHD score and logprobs.
    
    Returns: (generated_text, ahd_score, mean_logprob, min_logprob, perplexity)
    """
    # Format as chat
    messages = [
        {"role": "user", "content": f"Answer in 1-5 words only. {question}"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_length = inputs.input_ids.shape[1]
    
    # Generate with output scores
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for reproducibility
            output_scores=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Extract generated tokens
    generated_ids = outputs.sequences[0, prompt_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # Compute logprobs from scores
    logprobs = []
    for i, scores in enumerate(outputs.scores):
        probs = torch.softmax(scores[0], dim=-1)
        token_id = generated_ids[i]
        if token_id < len(probs):
            token_prob = probs[token_id].item()
            if token_prob > 0:
                logprobs.append(np.log(token_prob))
    
    if logprobs:
        mean_logprob = np.mean(logprobs)
        min_logprob = np.min(logprobs)
        perplexity = np.exp(-mean_logprob)
    else:
        mean_logprob = -10.0
        min_logprob = -10.0
        perplexity = 1000.0
    
    # Compute AHD score on the full response
    full_text = f"Question: {question}\nAnswer: {generated_text}"
    inputs_ahd = tokenizer(full_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs_ahd = model(**inputs_ahd, output_hidden_states=True)
    
    hidden = outputs_ahd.hidden_states[layer_idx + 1][0, -1, :].float().cpu()
    ahd_score = hidden.dot(ahd_direction).item()
    
    return generated_text, ahd_score, mean_logprob, min_logprob, perplexity


def check_answer(generated: str, correct: str) -> bool:
    """
    Check if generated answer matches correct answer.
    Uses fuzzy matching for robustness.
    """
    gen_clean = generated.lower().strip()
    correct_clean = correct.lower().strip()
    
    # Normalize unicode subscripts/superscripts
    unicode_map = {
        '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
        '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
        '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
        '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
    }
    for uni, ascii_char in unicode_map.items():
        gen_clean = gen_clean.replace(uni, ascii_char)
    
    # Remove commas from numbers
    gen_clean = re.sub(r'(\d),(\d)', r'\1\2', gen_clean)
    correct_clean = re.sub(r'(\d),(\d)', r'\1\2', correct_clean)
    
    # Direct containment
    if correct_clean in gen_clean or gen_clean in correct_clean:
        return True
    
    # Word-form numbers mapping
    word_to_num = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
        'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
        'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
        'thirty-four': '34', 'forty': '40', 'forty-six': '46', 'fifty': '50',
        'sixty': '60', 'seventy': '70', 'eighty': '80', 'ninety': '90',
        'hundred': '100', 'two hundred six': '206', 'two hundred twelve': '212',
        'three hundred thousand': '300000',
    }
    
    # Check if generated contains a word-form number that matches correct
    for word, num in word_to_num.items():
        if word in gen_clean and num == correct_clean:
            return True
        if word in gen_clean and num in correct_clean:
            return True
    
    # Handle numbers with extraction
    gen_numbers = re.findall(r'\d+', gen_clean)
    correct_numbers = re.findall(r'\d+', correct_clean)
    if gen_numbers and correct_numbers:
        if gen_numbers[0] == correct_numbers[0]:
            return True
    
    # If correct is a number, check if any extracted number matches
    if correct_clean.isdigit() and gen_numbers:
        if correct_clean in gen_numbers:
            return True
    
    # Handle common variations
    variations = {
        "leonardo da vinci": ["da vinci", "leonardo"],
        "william shakespeare": ["shakespeare"],
        "alexander graham bell": ["bell", "graham bell"],
        "mount kilimanjaro": ["kilimanjaro"],
        "blue whale": ["blue whale"],
        "pacific ocean": ["pacific"],
        "neil armstrong": ["armstrong"],
        "george orwell": ["orwell"],
        "alexander fleming": ["fleming"],
        "igor stravinsky": ["stravinsky"],
        "thomas pynchon": ["pynchon"],
        "millard fillmore": ["fillmore"],
        "antarctica": ["antarctic", "antarctic polar desert"],
        "h2so4": ["h2so4", "h₂so₄"],
    }
    
    for canonical, alts in variations.items():
        if correct_clean == canonical:
            for alt in alts:
                if alt in gen_clean:
                    return True
    
    return False


def run_experiment(model, tokenizer):
    """Run the full experiment"""
    print("\n" + "="*70)
    print("PHASE 1: Realistic Hallucination Detection")
    print("="*70)
    
    # Compute AHD direction
    print("\nComputing AHD direction vector...")
    ahd_direction = compute_ahd_direction(model, tokenizer, AHD_LAYER)
    
    results: List[GenerationResult] = []
    
    print(f"\nTesting {len(TEST_QUESTIONS)} questions...")
    print(f"{'Q#':>3} | {'Difficulty':<8} | {'Correct?':<8} | {'AHD':>7} | {'LogP':>7} | Question")
    print("-" * 90)
    
    for i, (question, correct, difficulty, category) in enumerate(TEST_QUESTIONS):
        generated, ahd_score, mean_logprob, min_logprob, perplexity = generate_with_scores(
            model, tokenizer, question, ahd_direction, AHD_LAYER
        )
        
        is_correct = check_answer(generated, correct)
        
        result = GenerationResult(
            question=question,
            correct_answer=correct,
            generated_answer=generated,
            is_correct=is_correct,
            difficulty=difficulty,
            category=category,
            ahd_score=ahd_score,
            mean_logprob=mean_logprob,
            min_logprob=min_logprob,
            answer_perplexity=perplexity
        )
        results.append(result)
        
        correct_str = "✓" if is_correct else "✗"
        print(f"{i+1:>3} | {difficulty:<8} | {correct_str:<8} | {ahd_score:>+7.2f} | {mean_logprob:>7.2f} | {question[:40]}")
    
    return results


def analyze_results(results: List[GenerationResult]):
    """Analyze and compare AHD vs logprobs"""
    
    print("\n" + "="*70)
    print("ANALYSIS: AHD vs Logprobs for Hallucination Detection")
    print("="*70)
    
    # Extract scores
    labels = [1 if r.is_correct else 0 for r in results]
    ahd_scores = [r.ahd_score for r in results]
    logprob_scores = [r.mean_logprob for r in results]  # Higher = more confident
    perplexity_scores = [-r.answer_perplexity for r in results]  # Negate so higher = better
    
    # Compute AUCs
    try:
        ahd_auc = roc_auc_score(labels, ahd_scores)
    except:
        ahd_auc = 0.5
    
    try:
        logprob_auc = roc_auc_score(labels, logprob_scores)
    except:
        logprob_auc = 0.5
    
    try:
        perplexity_auc = roc_auc_score(labels, perplexity_scores)
    except:
        perplexity_auc = 0.5
    
    # Compute separations
    correct_ahd = [s for s, l in zip(ahd_scores, labels) if l == 1]
    incorrect_ahd = [s for s, l in zip(ahd_scores, labels) if l == 0]
    
    correct_logprob = [s for s, l in zip(logprob_scores, labels) if l == 1]
    incorrect_logprob = [s for s, l in zip(logprob_scores, labels) if l == 0]
    
    def compute_separation(correct, incorrect):
        if len(correct) == 0 or len(incorrect) == 0:
            return 0.0
        pooled_std = np.sqrt((np.var(correct) + np.var(incorrect)) / 2)
        if pooled_std < 1e-8:
            return 0.0
        return (np.mean(correct) - np.mean(incorrect)) / pooled_std
    
    ahd_sep = compute_separation(correct_ahd, incorrect_ahd)
    logprob_sep = compute_separation(correct_logprob, incorrect_logprob)
    
    # Overall accuracy
    total_correct = sum(labels)
    total = len(labels)
    baseline_accuracy = total_correct / total
    
    print(f"\n📊 OVERALL RESULTS:")
    print(f"   Total questions: {total}")
    print(f"   Correct answers: {total_correct} ({baseline_accuracy:.1%})")
    print(f"   Hallucinations:  {total - total_correct} ({1-baseline_accuracy:.1%})")
    
    print(f"\n📈 DETECTION PERFORMANCE (AUC):")
    print(f"   {'Method':<20} | {'AUC':>6} | {'Separation':>10} | {'Winner'}")
    print(f"   {'-'*55}")
    
    winner_auc = "AHD" if ahd_auc > logprob_auc else "Logprobs"
    print(f"   {'AHD (activations)':<20} | {ahd_auc:>6.3f} | {ahd_sep:>+9.2f}σ | {'← BETTER' if winner_auc == 'AHD' else ''}")
    print(f"   {'Logprobs (mean)':<20} | {logprob_auc:>6.3f} | {logprob_sep:>+9.2f}σ | {'← BETTER' if winner_auc == 'Logprobs' else ''}")
    print(f"   {'Perplexity':<20} | {perplexity_auc:>6.3f} | {'N/A':>10} |")
    
    # Breakdown by difficulty
    print(f"\n📋 BREAKDOWN BY DIFFICULTY:")
    for diff in ["easy", "medium", "hard", "trap"]:
        diff_results = [r for r in results if r.difficulty == diff]
        if not diff_results:
            continue
        diff_correct = sum(1 for r in diff_results if r.is_correct)
        diff_total = len(diff_results)
        print(f"   {diff.capitalize():<8}: {diff_correct}/{diff_total} correct ({diff_correct/diff_total:.0%})")
    
    return {
        "ahd_auc": ahd_auc,
        "logprob_auc": logprob_auc,
        "perplexity_auc": perplexity_auc,
        "ahd_separation": ahd_sep,
        "logprob_separation": logprob_sep,
        "baseline_accuracy": baseline_accuracy,
        "winner": winner_auc
    }


def demonstrate_selective_abstention(results: List[GenerationResult], analysis: Dict):
    """
    PHASE 2: Show that abstaining on low-confidence responses improves accuracy
    """
    print("\n" + "="*70)
    print("PHASE 2: Selective Abstention Demo")
    print("="*70)
    print("\nIdea: Decline to answer when confidence is low → higher accuracy on answered questions")
    
    # Sort by AHD score (higher = more confident)
    sorted_by_ahd = sorted(results, key=lambda r: r.ahd_score, reverse=True)
    
    # Sort by logprob (higher = more confident)
    sorted_by_logprob = sorted(results, key=lambda r: r.mean_logprob, reverse=True)
    
    print(f"\n{'Abstain %':>10} | {'AHD Acc':>8} | {'LogP Acc':>8} | {'Baseline':>8}")
    print("-" * 50)
    
    abstention_results = []
    
    for abstain_pct in [0, 10, 20, 30, 40, 50]:
        n_keep = int(len(results) * (100 - abstain_pct) / 100)
        
        # AHD: keep top n_keep by AHD score
        ahd_kept = sorted_by_ahd[:n_keep]
        ahd_correct = sum(1 for r in ahd_kept if r.is_correct)
        ahd_acc = ahd_correct / n_keep if n_keep > 0 else 0
        
        # Logprob: keep top n_keep by logprob
        logprob_kept = sorted_by_logprob[:n_keep]
        logprob_correct = sum(1 for r in logprob_kept if r.is_correct)
        logprob_acc = logprob_correct / n_keep if n_keep > 0 else 0
        
        baseline = analysis["baseline_accuracy"]
        
        print(f"{abstain_pct:>9}% | {ahd_acc:>7.1%} | {logprob_acc:>7.1%} | {baseline:>7.1%}")
        
        abstention_results.append({
            "abstain_pct": abstain_pct,
            "ahd_accuracy": ahd_acc,
            "logprob_accuracy": logprob_acc,
            "n_answered": n_keep
        })
    
    # Find optimal abstention point for AHD
    best_ahd = max(abstention_results, key=lambda x: x["ahd_accuracy"])
    best_logprob = max(abstention_results, key=lambda x: x["logprob_accuracy"])
    
    print(f"\n🎯 OPTIMAL ABSTENTION:")
    print(f"   AHD:     Abstain {best_ahd['abstain_pct']}% → {best_ahd['ahd_accuracy']:.1%} accuracy")
    print(f"   Logprob: Abstain {best_logprob['abstain_pct']}% → {best_logprob['logprob_accuracy']:.1%} accuracy")
    print(f"   Baseline (no abstention): {analysis['baseline_accuracy']:.1%}")
    
    return abstention_results


def generate_pitch_statement(analysis: Dict, abstention: List[Dict]):
    """Generate the pitch statement based on results"""
    
    print("\n" + "="*70)
    print("📝 PITCH STATEMENT")
    print("="*70)
    
    winner = analysis["winner"]
    ahd_auc = analysis["ahd_auc"]
    logprob_auc = analysis["logprob_auc"]
    baseline = analysis["baseline_accuracy"]
    
    # Find best abstention result for AHD
    best = max(abstention, key=lambda x: x["ahd_accuracy"])
    improvement = best["ahd_accuracy"] - baseline
    
    if winner == "AHD":
        print(f'''
FINDING: AHD outperforms logprobs for hallucination detection

On realistic factual Q&A with Gemma-9B:
- AHD achieves {ahd_auc:.3f} AUC vs logprobs' {logprob_auc:.3f} AUC
- AHD separation: {analysis["ahd_separation"]:+.2f}σ vs logprobs: {analysis["logprob_separation"]:+.2f}σ

PRODUCT VALUE: Selective Abstention
- Baseline accuracy: {baseline:.1%}
- With {best["abstain_pct"]}% abstention (AHD-guided): {best["ahd_accuracy"]:.1%}
- Accuracy improvement: +{improvement:.1%} points

PITCH TO GEMMA TEAM:
"By integrating activation-based confidence scoring at layer 22, Gemma can 
identify when it's likely to hallucinate and either abstain or trigger 
retrieval. In testing, this improved factual accuracy from {baseline:.0%} to 
{best["ahd_accuracy"]:.0%} by declining to answer {best["abstain_pct"]}% of lowest-confidence 
queries. This outperforms logprob-based confidence ({ahd_auc:.2f} vs {logprob_auc:.2f} AUC)."
''')
    else:
        print(f'''
FINDING: Logprobs outperform AHD for hallucination detection on this dataset

- Logprobs: {logprob_auc:.3f} AUC
- AHD: {ahd_auc:.3f} AUC

This suggests AHD may not generalize from simple Q&A to realistic generation,
or that logprobs are sufficient for this use case.

NEXT STEPS:
- Test on harder hallucination scenarios (citations, dates, statistics)
- Consider combining AHD + logprobs
- Investigate why AHD underperformed
''')


def save_results(results: List[GenerationResult], analysis: Dict, abstention: List[Dict]):
    """Save all results to JSON"""
    output = {
        "model": MODEL_NAME,
        "ahd_layer": AHD_LAYER,
        "analysis": analysis,
        "abstention_results": abstention,
        "detailed_results": [
            {
                "question": r.question,
                "correct_answer": r.correct_answer,
                "generated_answer": r.generated_answer,
                "is_correct": r.is_correct,
                "difficulty": r.difficulty,
                "category": r.category,
                "ahd_score": r.ahd_score,
                "mean_logprob": r.mean_logprob,
                "min_logprob": r.min_logprob,
                "perplexity": r.answer_perplexity
            }
            for r in results
        ]
    }
    
    with open("ahd_realistic_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to: ahd_realistic_results.json")


def main():
    print("="*70)
    print("REALISTIC HALLUCINATION DETECTION: AHD vs LOGPROBS")
    print("="*70)
    print(f"\nModel: {MODEL_NAME}")
    print(f"AHD Layer: {AHD_LAYER}")
    print(f"Test Questions: {len(TEST_QUESTIONS)}")
    
    # Load model
    model, tokenizer = load_model()
    
    # Run experiment
    results = run_experiment(model, tokenizer)
    
    # Analyze
    analysis = analyze_results(results)
    
    # Demonstrate selective abstention
    abstention = demonstrate_selective_abstention(results, analysis)
    
    # Generate pitch
    generate_pitch_statement(analysis, abstention)
    
    # Save
    save_results(results, analysis, abstention)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
