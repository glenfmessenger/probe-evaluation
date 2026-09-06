"""
SimpleQA Benchmark for AHD Evaluation
=====================================

This script evaluates AHD's ability to predict hallucinations on the industry-standard
SimpleQA benchmark (OpenAI, 2024).

Experiment Design:
1. Run model on SimpleQA questions (baseline - measure actual hallucination rate)
2. For each question, capture AHD confidence score BEFORE seeing the answer
3. Compare AHD predictions against actual correct/incorrect outcomes
4. Measure: Can AHD predict which answers will be wrong?

Key Metrics:
- Baseline accuracy: How often does the model get SimpleQA questions right?
- AHD AUROC: How well does AHD score correlate with correctness?
- AHD@threshold: If we block low-confidence answers, how much do we reduce errors?

Usage:
    python simpleqa_ahd_benchmark.py --n-samples 100
    python simpleqa_ahd_benchmark.py --n-samples 500 --model google/gemma-3-4b-it

Author: Glen Messenger
"""

import torch
import numpy as np
import json
import argparse
import time
import re
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SampleResult:
    """Result for a single SimpleQA sample."""
    question: str
    gold_answer: str
    model_answer: str
    is_correct: bool
    ahd_score: float
    ahd_confidence: float  # Normalized 0-1
    ahd_prediction: str    # "grounded" or "hallucination"
    latency_ms: float


@dataclass 
class BenchmarkResults:
    """Overall benchmark results."""
    model: str
    n_samples: int
    timestamp: str
    
    # Baseline (model without AHD)
    baseline_correct: int
    baseline_incorrect: int
    baseline_not_attempted: int
    baseline_accuracy: float
    
    # AHD performance
    ahd_layer: int
    ahd_layer_pct: float
    ahd_threshold: float
    ahd_auroc: float
    
    # AHD as filter (if we reject low-confidence answers)
    ahd_filtered_correct: int
    ahd_filtered_incorrect: int  
    ahd_filtered_rejected: int
    ahd_filtered_accuracy: float  # Accuracy on attempted answers
    ahd_error_reduction_pct: float  # How much did we reduce errors?
    
    # Detailed results
    samples: List[Dict] = field(default_factory=list)


# =============================================================================
# SIMPLE ANSWER GRADING (following OpenAI's approach)
# =============================================================================

def normalize_answer(text: str) -> str:
    """Normalize answer for comparison."""
    # Lowercase
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Collapse whitespace
    text = ' '.join(text.split())
    return text


def check_answer_correct(model_answer: str, gold_answer: str) -> bool:
    """
    Check if model answer is correct.
    Uses simple containment check - if gold answer appears in model answer.
    This is a simplified version of OpenAI's GPT-4 grader.
    """
    model_norm = normalize_answer(model_answer)
    gold_norm = normalize_answer(gold_answer)
    
    # Check if gold answer is contained in model answer
    if gold_norm in model_norm:
        return True
    
    # Check word overlap for short answers
    gold_words = set(gold_norm.split())
    model_words = set(model_norm.split())
    
    if len(gold_words) <= 3:
        # For short answers, require high overlap
        overlap = len(gold_words & model_words) / len(gold_words) if gold_words else 0
        return overlap >= 0.8
    
    return False


def is_not_attempted(model_answer: str) -> bool:
    """Check if model declined to answer."""
    decline_phrases = [
        "i don't know",
        "i do not know", 
        "i'm not sure",
        "i am not sure",
        "i cannot",
        "i can't",
        "unable to",
        "don't have information",
        "no information",
        "cannot provide",
    ]
    answer_lower = model_answer.lower()
    return any(phrase in answer_lower for phrase in decline_phrases)


# =============================================================================
# AHD DETECTOR
# =============================================================================

class AHDDetector:
    """Activation-based Hallucination Detection."""
    
    def __init__(self, model, tokenizer, layer_idx: int = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Find layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            self.layers = model.model.layers
        elif hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
            self.layers = model.language_model.model.layers
        else:
            raise ValueError(f"Unknown architecture: {type(model)}")
        
        self.num_layers = len(self.layers)
        
        # Default to 38% depth (optimal from validation)
        if layer_idx is None:
            layer_idx = int(self.num_layers * 0.38)
        self.layer_idx = layer_idx
        
        if hasattr(model.config, 'hidden_size'):
            self.hidden_dim = model.config.hidden_size
        elif hasattr(model.config, 'text_config'):
            self.hidden_dim = model.config.text_config.hidden_size
        else:
            raise ValueError("Cannot determine hidden_size")
        
        self.activations = {}
        self.hooks = []
        
        # Will be set during training
        self.direction_vector = None
        self.threshold = 0.0
        self.factual_mean = 0.0
        self.factual_std = 1.0
        self.hallucination_mean = 0.0
        self.hallucination_std = 1.0
    
    def _hook_fn(self, layer_idx: int):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.activations[layer_idx] = hidden.detach()
        return hook
    
    def _register_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        h = self.layers[self.layer_idx].register_forward_hook(
            self._hook_fn(self.layer_idx)
        )
        self.hooks.append(h)
    
    def _get_activation(self, text: str) -> np.ndarray:
        """Extract activation for input text."""
        formatted = f"<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"
        self._register_hooks()
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        self.activations = {}
        
        with torch.no_grad():
            _ = self.model(inputs.input_ids, attention_mask=inputs.attention_mask)
        
        raw = self.activations[self.layer_idx][0, -1]
        activation = raw.float().cpu().numpy().astype(np.float32)
        
        if np.isnan(activation).any() or np.isinf(activation).any():
            activation = np.nan_to_num(activation, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return activation
    
    def train(self, factual_questions: List[str], hallucination_questions: List[str]):
        """Train the AHD detector on contrastive examples."""
        print(f"  Training AHD on {len(factual_questions)} factual + {len(hallucination_questions)} hallucination examples...")
        
        # Collect activations
        factual_acts = np.stack([self._get_activation(q) for q in factual_questions])
        halluc_acts = np.stack([self._get_activation(q) for q in hallucination_questions])
        
        # Compute direction vector
        factual_mean = factual_acts.mean(axis=0)
        halluc_mean = halluc_acts.mean(axis=0)
        direction = factual_mean - halluc_mean
        direction = direction / np.linalg.norm(direction)
        self.direction_vector = direction
        
        # Compute scores and statistics
        factual_scores = factual_acts @ direction
        halluc_scores = halluc_acts @ direction
        
        self.factual_mean = float(factual_scores.mean())
        self.factual_std = float(factual_scores.std())
        self.hallucination_mean = float(halluc_scores.mean())
        self.hallucination_std = float(halluc_scores.std())
        
        # Find optimal threshold
        all_scores = np.concatenate([factual_scores, halluc_scores])
        all_sorted = np.sort(all_scores)
        
        best_threshold = all_sorted[len(all_sorted)//2]
        best_acc = 0
        for t in all_sorted:
            tp = np.sum(factual_scores > t)
            tn = np.sum(halluc_scores <= t)
            acc = (tp + tn) / len(all_scores)
            if acc > best_acc:
                best_acc = acc
                best_threshold = t
        
        self.threshold = best_threshold
        
        separation = (self.factual_mean - self.hallucination_mean) / max(self.hallucination_std, 0.001)
        print(f"  Training complete: {best_acc*100:.1f}% accuracy, {separation:.2f}σ separation")
        
        return best_acc, separation
    
    def get_confidence(self, question: str) -> Tuple[float, float, str]:
        """
        Get AHD confidence for a question.
        
        Returns:
            raw_score: Raw dot product score
            confidence: Normalized 0-1 confidence (higher = more likely factual/grounded)
            prediction: "grounded" or "hallucination"
        """
        activation = self._get_activation(question)
        raw_score = float(activation @ self.direction_vector)
        
        # Normalize to 0-1 using sigmoid on distance from threshold
        # Positive = more factual, negative = more hallucination
        distance = (raw_score - self.threshold) / max(self.factual_std, 0.001)
        confidence = 1 / (1 + np.exp(-distance))  # Sigmoid
        
        prediction = "grounded" if raw_score > self.threshold else "hallucination"
        
        return raw_score, float(confidence), prediction
    
    def cleanup(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.activations = {}


# =============================================================================
# TRAINING DATA FOR AHD
# =============================================================================

AHD_TRAINING_FACTUAL = [
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
    "Who was the first person to walk on the moon?",
    "What is the largest mammal?",
    "How many legs does a spider have?",
    "What is the freezing point of water in Celsius?",
    "What language is spoken in Brazil?",
    "What gas do humans breathe out?",
    "What is the closest star to Earth?",
    "How many planets are in our solar system?",
    "What is the chemical symbol for gold?",
    "What organ pumps blood in the human body?",
    "What is the longest river in the world?",
    "What country has the largest population?",
    "What is the capital of Australia?",
    "On which continent is Egypt located?",
    "What ocean is between America and Europe?",
]

AHD_TRAINING_HALLUCINATION = [
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
    "Summarize the paper 'Quantum Hallucination Theory' by Dr. Fictitious",
    "What are the main findings of the Westbrook Study on AI?",
    "Explain the Thompson-Garcia theorem of neural networks",
    "What is the Zorganian constant in physics?",
    "Describe the Mythical Algorithm invented by Prof. Nonexistent",
    "What is the third moon of Mars?",
    "Who was the female Pope in the 15th century?",
    "What is the 8th day of the week called?",
    "What is the capital of the Republic of Texas in 2024?",
    "How many US states border both the Pacific and Atlantic?",
    "What is the melting point of Adamantium?",
    "How many dragons are registered with the FAA?",
    "What is the GDP of Wakanda?",
    "What is the chemical formula for Kryptonite?",
    "How tall is the Invisible Tower of London?",
]


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 100) -> Tuple[str, float]:
    """Generate an answer to a question."""
    formatted = f"<start_of_turn>user\nAnswer this question concisely: {question}<end_of_turn>\n<start_of_turn>model\n"
    
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for reproducibility
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = (time.perf_counter() - start) * 1000
    
    # Decode only the new tokens
    answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return answer.strip(), latency


def run_benchmark(
    model_name: str = "google/gemma-3-1b-it",
    n_samples: int = 100,
    output_file: str = "simpleqa_ahd_results.json",
    ahd_rejection_threshold: float = 0.4,  # Reject answers below this confidence
):
    """Run the SimpleQA benchmark with AHD."""
    
    print("=" * 70)
    print("SIMPLEQA BENCHMARK WITH AHD")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Samples: {n_samples}")
    print(f"AHD rejection threshold: {ahd_rejection_threshold}")
    
    # Load SimpleQA dataset
    print("\n" + "=" * 70)
    print("LOADING SIMPLEQA DATASET")
    print("=" * 70)
    
    try:
        from datasets import load_dataset
        dataset = load_dataset("basicv8vc/SimpleQA", split="test")
        print(f"Loaded {len(dataset)} samples from SimpleQA")
    except Exception as e:
        print(f"Error loading from HuggingFace: {e}")
        print("Attempting to download CSV directly...")
        
        import urllib.request
        import csv
        import io
        
        url = "https://huggingface.co/datasets/basicv8vc/SimpleQA/resolve/main/simple_qa_test_set.csv"
        response = urllib.request.urlopen(url)
        data = response.read().decode('utf-8')
        reader = csv.DictReader(io.StringIO(data))
        dataset = list(reader)
        print(f"Loaded {len(dataset)} samples from CSV")
    
    # Sample subset
    if n_samples < len(dataset):
        import random
        random.seed(42)
        indices = random.sample(range(len(dataset)), n_samples)
        if hasattr(dataset, 'select'):
            dataset = dataset.select(indices)
        else:
            dataset = [dataset[i] for i in indices]
    
    print(f"Using {len(dataset)} samples for evaluation")
    
    # Load model
    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Determine layer count
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        num_layers = len(model.model.layers)
    else:
        num_layers = 26  # Default
    
    ahd_layer = int(num_layers * 0.38)  # 38% depth - optimal from validation
    print(f"Model loaded. Using layer {ahd_layer}/{num_layers} for AHD")
    
    # Initialize and train AHD
    print("\n" + "=" * 70)
    print("TRAINING AHD DETECTOR")
    print("=" * 70)
    
    ahd = AHDDetector(model, tokenizer, layer_idx=ahd_layer)
    ahd_accuracy, ahd_separation = ahd.train(AHD_TRAINING_FACTUAL, AHD_TRAINING_HALLUCINATION)
    
    # Run benchmark
    print("\n" + "=" * 70)
    print("RUNNING BENCHMARK")
    print("=" * 70)
    
    results = []
    
    for i, sample in enumerate(dataset):
        # Handle both dict and HF dataset formats
        if hasattr(sample, 'keys'):
            question = sample.get('problem', sample.get('question', ''))
            gold_answer = sample.get('answer', sample.get('gold_answer', ''))
        else:
            question = sample['problem'] if 'problem' in sample else sample['question']
            gold_answer = sample['answer'] if 'answer' in sample else sample['gold_answer']
        
        # Get AHD confidence BEFORE generating answer
        ahd_score, ahd_confidence, ahd_prediction = ahd.get_confidence(question)
        
        # Generate answer
        model_answer, latency = generate_answer(model, tokenizer, question)
        
        # Check correctness
        if is_not_attempted(model_answer):
            is_correct = None  # Not attempted
        else:
            is_correct = check_answer_correct(model_answer, gold_answer)
        
        result = SampleResult(
            question=question,
            gold_answer=gold_answer,
            model_answer=model_answer,
            is_correct=is_correct,
            ahd_score=ahd_score,
            ahd_confidence=ahd_confidence,
            ahd_prediction=ahd_prediction,
            latency_ms=latency,
        )
        results.append(result)
        
        # Progress
        if (i + 1) % 10 == 0 or i == 0:
            correct_so_far = sum(1 for r in results if r.is_correct == True)
            incorrect_so_far = sum(1 for r in results if r.is_correct == False)
            print(f"  [{i+1}/{len(dataset)}] Correct: {correct_so_far}, Incorrect: {incorrect_so_far}, Conf: {ahd_confidence:.2f}")
    
    # Compute metrics
    print("\n" + "=" * 70)
    print("COMPUTING METRICS")
    print("=" * 70)
    
    # Baseline metrics
    correct = [r for r in results if r.is_correct == True]
    incorrect = [r for r in results if r.is_correct == False]
    not_attempted = [r for r in results if r.is_correct is None]
    
    baseline_accuracy = len(correct) / (len(correct) + len(incorrect)) if (len(correct) + len(incorrect)) > 0 else 0
    
    print(f"\nBASELINE (Model without AHD filtering):")
    print(f"  Correct: {len(correct)}")
    print(f"  Incorrect: {len(incorrect)}")
    print(f"  Not attempted: {len(not_attempted)}")
    print(f"  Accuracy (on attempted): {baseline_accuracy*100:.1f}%")
    
    # AHD correlation with correctness
    # For AUROC, we need scores and binary labels
    attempted = [r for r in results if r.is_correct is not None]
    if len(attempted) > 0:
        scores = [r.ahd_confidence for r in attempted]
        labels = [1 if r.is_correct else 0 for r in attempted]
        
        try:
            from sklearn.metrics import roc_auc_score
            auroc = roc_auc_score(labels, scores)
        except:
            auroc = 0.0
        
        print(f"\nAHD PERFORMANCE:")
        print(f"  AUROC: {auroc:.3f}")
        
        # Score distribution
        correct_scores = [r.ahd_confidence for r in correct]
        incorrect_scores = [r.ahd_confidence for r in incorrect]
        
        print(f"  Correct answers - mean confidence: {np.mean(correct_scores):.3f} (std: {np.std(correct_scores):.3f})")
        print(f"  Incorrect answers - mean confidence: {np.mean(incorrect_scores):.3f} (std: {np.std(incorrect_scores):.3f})")
    else:
        auroc = 0.0
    
    # AHD as filter
    print(f"\nAHD AS FILTER (rejection threshold: {ahd_rejection_threshold}):")
    
    accepted = [r for r in attempted if r.ahd_confidence >= ahd_rejection_threshold]
    rejected = [r for r in attempted if r.ahd_confidence < ahd_rejection_threshold]
    
    accepted_correct = sum(1 for r in accepted if r.is_correct)
    accepted_incorrect = sum(1 for r in accepted if not r.is_correct)
    
    filtered_accuracy = accepted_correct / len(accepted) if len(accepted) > 0 else 0
    
    # Error reduction
    baseline_errors = len(incorrect)
    filtered_errors = accepted_incorrect
    error_reduction = (baseline_errors - filtered_errors) / baseline_errors if baseline_errors > 0 else 0
    
    print(f"  Accepted: {len(accepted)} ({len(accepted)/len(attempted)*100:.1f}%)")
    print(f"  Rejected: {len(rejected)} ({len(rejected)/len(attempted)*100:.1f}%)")
    print(f"  Accuracy on accepted: {filtered_accuracy*100:.1f}%")
    print(f"  Errors reduced: {baseline_errors} → {filtered_errors} ({error_reduction*100:.1f}% reduction)")
    
    # What did we reject?
    rejected_correct = sum(1 for r in rejected if r.is_correct)
    rejected_incorrect = sum(1 for r in rejected if not r.is_correct)
    print(f"  Rejected breakdown: {rejected_correct} correct, {rejected_incorrect} incorrect")
    if len(rejected) > 0:
        rejection_precision = rejected_incorrect / len(rejected)
        print(f"  Rejection precision (% of rejected that were actually wrong): {rejection_precision*100:.1f}%")
    
    # Compile final results
    benchmark_results = BenchmarkResults(
        model=model_name,
        n_samples=len(results),
        timestamp=datetime.utcnow().isoformat() + 'Z',
        baseline_correct=len(correct),
        baseline_incorrect=len(incorrect),
        baseline_not_attempted=len(not_attempted),
        baseline_accuracy=baseline_accuracy,
        ahd_layer=ahd_layer,
        ahd_layer_pct=ahd_layer / num_layers,
        ahd_threshold=ahd.threshold,
        ahd_auroc=auroc,
        ahd_filtered_correct=accepted_correct,
        ahd_filtered_incorrect=accepted_incorrect,
        ahd_filtered_rejected=len(rejected),
        ahd_filtered_accuracy=filtered_accuracy,
        ahd_error_reduction_pct=error_reduction,
        samples=[asdict(r) for r in results],
    )
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(asdict(benchmark_results), f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"SimpleQA samples: {len(results)}")
    print(f"\nBaseline accuracy: {baseline_accuracy*100:.1f}%")
    print(f"AHD AUROC: {auroc:.3f}")
    print(f"With AHD filtering ({ahd_rejection_threshold} threshold):")
    print(f"  - Accept rate: {len(accepted)/len(attempted)*100:.1f}%")
    print(f"  - Filtered accuracy: {filtered_accuracy*100:.1f}%")
    print(f"  - Error reduction: {error_reduction*100:.1f}%")
    
    # Cleanup
    ahd.cleanup()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return benchmark_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SimpleQA Benchmark with AHD")
    parser.add_argument("--model", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of SimpleQA samples to evaluate")
    parser.add_argument("--output", type=str, default="simpleqa_ahd_results.json")
    parser.add_argument("--ahd-threshold", type=float, default=0.4, help="AHD confidence threshold for filtering")
    args = parser.parse_args()
    
    results = run_benchmark(
        model_name=args.model,
        n_samples=args.n_samples,
        output_file=args.output,
        ahd_rejection_threshold=args.ahd_threshold,
    )


if __name__ == "__main__":
    main()
