"""
AHD (Activation-based Hallucination Detection) Proof-of-Concept v2
===================================================================
Fixed version with consistent prompt formatting and adaptive thresholding.

Key fixes from v1:
1. Consistent prompt format between training and test
2. Adaptive threshold based on score distributions
3. Better test set design (no format mismatch)
4. Added calibration step

Author: Glen Messenger
Version: 0.2.0
"""

import torch
import numpy as np
import json
import argparse
import time
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

try:
    from safetensors.numpy import save_file as save_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

# =============================================================================
# TRAINING DATA v2 - Cleaner separation
# =============================================================================

# FACTUAL: Simple questions the model definitely knows
# Format: Just the question (we extract activation BEFORE the answer)
FACTUAL_QUESTIONS = [
    # Unambiguous facts
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
    
    # More basic facts
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
    
    # Science basics
    "What gas do humans breathe out?",
    "What is the closest star to Earth?",
    "How many planets are in our solar system?",
    "What is the chemical symbol for gold?",
    "What organ pumps blood in the human body?",
    
    # Geography
    "What is the longest river in the world?",
    "What country has the largest population?",
    "What is the capital of Australia?",
    "On which continent is Egypt located?",
    "What ocean is between America and Europe?",
]

# HALLUCINATION TRIGGERS: Questions that will cause confabulation
# These are questions about things that don't exist or can't be known
HALLUCINATION_QUESTIONS = [
    # Fictional entities
    "What is the capital of Atlantis?",
    "What year was the Treaty of Westbrook signed?",
    "Who was the 47th Emperor of Japan?",
    "What is the population of the city of El Dorado?",
    "What language is spoken in the Kingdom of Zorgon?",
    
    # Impossible specifics
    "What was Napoleon's phone number?",
    "What color was Aristotle's car?",
    "What was Shakespeare's email address?",
    "What brand of shoes did Julius Caesar wear?",
    "What was Cleopatra's Instagram username?",
    
    # Future events
    "Who won the 2030 FIFA World Cup?",
    "What was the most popular movie of 2029?",
    "Who won the 2028 US Presidential election?",
    "What company had the highest stock price in 2035?",
    "What was the population of Mars in 2050?",
    
    # Made-up academic content
    "Summarize the paper 'Quantum Hallucination Theory' by Dr. Fictitious",
    "What are the main findings of the Westbrook Study on AI?",
    "Explain the Thompson-Garcia theorem of neural networks",
    "What is the Zorganian constant in physics?",
    "Describe the Mythical Algorithm invented by Prof. Nonexistent",
    
    # Entity confusion
    "What is the third moon of Mars?",  # Mars has only 2 moons
    "Who was the female Pope in the 15th century?",
    "What is the 8th day of the week called?",
    "What is the capital of the Republic of Texas in 2024?",  # Not a country
    "How many US states border the Pacific Ocean and the Atlantic Ocean?",  # None do
    
    # Absurd but plausible-sounding
    "What is the melting point of Adamantium?",
    "How many dragons are registered with the FAA?",
    "What is the GDP of Wakanda?",
    "What is the chemical formula for Kryptonite?",
    "How tall is the Invisible Tower of London?",
]

# CALIBRATION SET: For finding optimal threshold
CALIBRATION_FACTUAL = [
    "What is the capital of Germany?",
    "What is 12 times 12?",
    "Who wrote Hamlet?",
    "What is the largest planet in our solar system?",
    "What year did World War I begin?",
]

CALIBRATION_HALLUCINATION = [
    "What was Socrates' favorite video game?",
    "What is the capital of the Republic of Zogland?",
    "Who was the 100th President of the United States?",
    "What year was the internet invented by Benjamin Franklin?",
    "What is the phone number of the Eiffel Tower?",
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class AHDStats:
    """Statistics for hallucination detection."""
    separation: float
    threshold: float
    accuracy: float
    fpr: float = 0.0
    fnr: float = 0.0
    auc: float = 0.0
    factual_mean: float = 0.0
    factual_std: float = 0.0
    hallucination_mean: float = 0.0
    hallucination_std: float = 0.0
    calibrated_threshold: float = 0.0


@dataclass
class AHDResult:
    """Result of hallucination detection."""
    is_hallucination: bool
    confidence_score: float
    normalized_score: float  # 0-1 scale
    latency_ms: float
    threshold: float


# =============================================================================
# AHD TRAINER v2
# =============================================================================

class AHDTrainerV2:
    """
    Improved hallucination detection trainer with:
    - Consistent prompt formatting
    - Adaptive thresholding
    - Calibration step
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        layer_pct: float = 0.65,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Find layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            self.layers = model.model.layers
            print(f"  Model type: standard (model.model.layers)")
        elif hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
            self.layers = model.language_model.model.layers
            print(f"  Model type: VLM (language_model.model.layers)")
        else:
            raise ValueError(f"Unknown model architecture: {type(model)}")
        
        self.num_layers = len(self.layers)
        
        if hasattr(model.config, 'hidden_size'):
            self.hidden_dim = model.config.hidden_size
        elif hasattr(model.config, 'text_config'):
            self.hidden_dim = model.config.text_config.hidden_size
        else:
            raise ValueError("Cannot determine hidden_size")
        
        self.extraction_layer = int(self.num_layers * layer_pct)
        
        print(f"  Layers: {self.num_layers}, Hidden dim: {self.hidden_dim}")
        print(f"  Extraction layer: {self.extraction_layer} ({layer_pct*100:.0f}%)")
        
        self.activations = {}
        self.hooks = []
        
        # Trained components
        self.direction_vector = None
        self.threshold = 0.0
        self.calibrated_threshold = 0.0
        self.stats = None
        
        # For normalization
        self.score_mean = 0.0
        self.score_std = 1.0
    
    def _hook_fn(self, layer_idx: int):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.activations[layer_idx] = hidden.detach()
        return hook
    
    def _register_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        h = self.layers[self.extraction_layer].register_forward_hook(
            self._hook_fn(self.extraction_layer)
        )
        self.hooks.append(h)
    
    def _format_prompt(self, question: str) -> str:
        """Consistent prompt formatting."""
        return f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
    
    def _get_activation(self, question: str) -> np.ndarray:
        """Extract activation for a question."""
        formatted = self._format_prompt(question)
        
        self._register_hooks()
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        self.activations = {}
        
        with torch.no_grad():
            _ = self.model(inputs.input_ids, attention_mask=inputs.attention_mask)
        
        if self.extraction_layer not in self.activations:
            raise ValueError(f"Layer {self.extraction_layer} not captured")
        
        raw = self.activations[self.extraction_layer][0, -1]
        activation = raw.float().cpu().numpy().astype(np.float32)
        
        if np.isnan(activation).any() or np.isinf(activation).any():
            activation = np.nan_to_num(activation, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return activation
    
    def train(self) -> AHDStats:
        """Train the hallucination detection vector."""
        print("\n  Training hallucination detection vector...")
        
        # Collect factual activations
        print(f"  Collecting factual activations ({len(FACTUAL_QUESTIONS)} samples)...")
        factual_activations = []
        for i, q in enumerate(FACTUAL_QUESTIONS):
            if i % 10 == 0:
                print(f"    Factual {i+1}/{len(FACTUAL_QUESTIONS)}")
            factual_activations.append(self._get_activation(q))
        factual_activations = np.stack(factual_activations)
        
        # Collect hallucination activations
        print(f"  Collecting hallucination activations ({len(HALLUCINATION_QUESTIONS)} samples)...")
        hallucination_activations = []
        for i, q in enumerate(HALLUCINATION_QUESTIONS):
            if i % 10 == 0:
                print(f"    Hallucination {i+1}/{len(HALLUCINATION_QUESTIONS)}")
            hallucination_activations.append(self._get_activation(q))
        hallucination_activations = np.stack(hallucination_activations)
        
        # Compute direction vector
        print("  Computing direction vector...")
        factual_mean = factual_activations.mean(axis=0)
        hallucination_mean = hallucination_activations.mean(axis=0)
        
        direction = factual_mean - hallucination_mean
        direction = direction / np.linalg.norm(direction)
        self.direction_vector = direction
        
        # Score all samples
        factual_scores = factual_activations @ direction
        hallucination_scores = hallucination_activations @ direction
        
        # Store for normalization
        all_scores = np.concatenate([factual_scores, hallucination_scores])
        self.score_mean = all_scores.mean()
        self.score_std = all_scores.std()
        
        # Compute threshold: Find optimal point between distributions
        # Use the point that minimizes classification error
        all_sorted = np.sort(all_scores)
        best_threshold = all_sorted[len(all_sorted)//2]
        best_accuracy = 0
        
        for t in all_sorted:
            tp = np.sum(factual_scores > t)
            tn = np.sum(hallucination_scores <= t)
            acc = (tp + tn) / len(all_scores)
            if acc > best_accuracy:
                best_accuracy = acc
                best_threshold = t
        
        self.threshold = best_threshold
        
        # Compute statistics
        separation = (factual_scores.mean() - hallucination_scores.mean()) / max(hallucination_scores.std(), 0.001)
        
        tp = np.sum(factual_scores > self.threshold)
        tn = np.sum(hallucination_scores <= self.threshold)
        fp = np.sum(hallucination_scores > self.threshold)
        fn = np.sum(factual_scores <= self.threshold)
        
        accuracy = (tp + tn) / len(all_scores)
        fpr = fp / len(hallucination_scores) if len(hallucination_scores) > 0 else 0
        fnr = fn / len(factual_scores) if len(factual_scores) > 0 else 0
        
        # AUC
        try:
            from sklearn.metrics import roc_auc_score
            labels = [1] * len(factual_scores) + [0] * len(hallucination_scores)
            auc = roc_auc_score(labels, all_scores)
        except:
            auc = 0.0
        
        self.stats = AHDStats(
            separation=float(separation),
            threshold=float(self.threshold),
            accuracy=float(accuracy),
            fpr=float(fpr),
            fnr=float(fnr),
            auc=float(auc),
            factual_mean=float(factual_scores.mean()),
            factual_std=float(factual_scores.std()),
            hallucination_mean=float(hallucination_scores.mean()),
            hallucination_std=float(hallucination_scores.std()),
        )
        
        print(f"\n  Training Results:")
        print(f"    Separation: {separation:.2f} σ")
        print(f"    Optimal Threshold: {self.threshold:.2f}")
        print(f"    Accuracy: {accuracy*100:.1f}%")
        print(f"    FPR: {fpr*100:.1f}%, FNR: {fnr*100:.1f}%")
        print(f"    AUC: {auc:.3f}")
        print(f"    Factual scores: {factual_scores.mean():.2f} ± {factual_scores.std():.2f}")
        print(f"    Hallucination scores: {hallucination_scores.mean():.2f} ± {hallucination_scores.std():.2f}")
        
        return self.stats
    
    def calibrate(self) -> float:
        """Calibrate threshold on held-out data."""
        print("\n  Calibrating threshold...")
        
        # Score calibration set
        cal_factual_scores = [float(self._get_activation(q) @ self.direction_vector) 
                             for q in CALIBRATION_FACTUAL]
        cal_halluc_scores = [float(self._get_activation(q) @ self.direction_vector) 
                            for q in CALIBRATION_HALLUCINATION]
        
        print(f"    Calibration factual scores: {np.mean(cal_factual_scores):.2f} ± {np.std(cal_factual_scores):.2f}")
        print(f"    Calibration halluc scores: {np.mean(cal_halluc_scores):.2f} ± {np.std(cal_halluc_scores):.2f}")
        
        # Find threshold that separates calibration set
        all_cal = cal_factual_scores + cal_halluc_scores
        all_cal_sorted = sorted(all_cal)
        
        best_threshold = self.threshold
        best_acc = 0
        
        for t in all_cal_sorted:
            tp = sum(1 for s in cal_factual_scores if s > t)
            tn = sum(1 for s in cal_halluc_scores if s <= t)
            acc = (tp + tn) / len(all_cal)
            if acc > best_acc:
                best_acc = acc
                best_threshold = t
        
        # Use midpoint between distributions
        midpoint = (np.mean(cal_factual_scores) + np.mean(cal_halluc_scores)) / 2
        
        self.calibrated_threshold = midpoint
        self.stats.calibrated_threshold = midpoint
        
        print(f"    Calibrated threshold: {self.calibrated_threshold:.2f}")
        print(f"    (Training threshold was: {self.threshold:.2f})")
        
        return self.calibrated_threshold
    
    def classify(self, question: str, use_calibrated: bool = True) -> AHDResult:
        """Classify a question."""
        start = time.perf_counter()
        
        activation = self._get_activation(question)
        score = float(activation @ self.direction_vector)
        
        threshold = self.calibrated_threshold if use_calibrated else self.threshold
        
        # Normalize to 0-1 scale (0 = definitely hallucination, 1 = definitely factual)
        normalized = (score - self.stats.hallucination_mean) / (
            self.stats.factual_mean - self.stats.hallucination_mean
        )
        normalized = max(0, min(1, normalized))
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        return AHDResult(
            is_hallucination=(score < threshold),
            confidence_score=score,
            normalized_score=normalized,
            latency_ms=elapsed_ms,
            threshold=threshold,
        )
    
    def evaluate(self, test_questions: List[Tuple[str, bool]], use_calibrated: bool = True) -> Dict:
        """Evaluate on test set. test_questions: List of (question, is_factual)"""
        results = []
        for question, is_factual in test_questions:
            result = self.classify(question, use_calibrated=use_calibrated)
            predicted_factual = not result.is_hallucination
            correct = predicted_factual == is_factual
            results.append({
                'question': question[:50] + '...' if len(question) > 50 else question,
                'is_factual': is_factual,
                'predicted_factual': predicted_factual,
                'correct': correct,
                'score': result.confidence_score,
                'normalized': result.normalized_score,
            })
        
        accuracy = sum(1 for r in results if r['correct']) / len(results)
        
        # Compute per-class metrics
        factual_correct = sum(1 for r in results if r['is_factual'] and r['correct'])
        factual_total = sum(1 for r in results if r['is_factual'])
        halluc_correct = sum(1 for r in results if not r['is_factual'] and r['correct'])
        halluc_total = sum(1 for r in results if not r['is_factual'])
        
        return {
            'accuracy': accuracy,
            'factual_accuracy': factual_correct / factual_total if factual_total > 0 else 0,
            'hallucination_accuracy': halluc_correct / halluc_total if halluc_total > 0 else 0,
            'results': results,
        }
    
    def save_vector(self, output_path: str) -> Dict[str, str]:
        """Save trained vector."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        tensors = {
            'vector_hallucination': self.direction_vector.astype(np.float32),
            'threshold': np.array([self.threshold], dtype=np.float32),
            'calibrated_threshold': np.array([self.calibrated_threshold], dtype=np.float32),
            'score_mean': np.array([self.score_mean], dtype=np.float32),
            'score_std': np.array([self.score_std], dtype=np.float32),
        }
        
        metadata = {
            'version': '0.2.0',
            'component': 'AHD',
            'model_name': getattr(self.model.config, '_name_or_path', 'unknown').split('/')[-1],
            'hidden_dimension': int(self.hidden_dim),
            'extraction_layer': int(self.extraction_layer),
            'extraction_layer_pct': float(self.extraction_layer / self.num_layers),
            'threshold': float(self.threshold),
            'calibrated_threshold': float(self.calibrated_threshold),
            'stats': {
                'separation': self.stats.separation,
                'accuracy': self.stats.accuracy,
                'auc': self.stats.auc,
                'factual_mean': self.stats.factual_mean,
                'factual_std': self.stats.factual_std,
                'hallucination_mean': self.stats.hallucination_mean,
                'hallucination_std': self.stats.hallucination_std,
            },
            'created_at': datetime.utcnow().isoformat() + 'Z',
        }
        
        created_files = {}
        
        if HAS_SAFETENSORS:
            vectors_file = output_path.with_suffix('.safetensors')
            save_safetensors(tensors, str(vectors_file))
            created_files['vectors'] = str(vectors_file)
        else:
            vectors_file = output_path.with_suffix('.npz')
            np.savez_compressed(str(vectors_file), **tensors)
            created_files['vectors'] = str(vectors_file)
        
        metadata_file = output_path.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        created_files['metadata'] = str(metadata_file)
        
        print(f"\n  Saved vector ({self.direction_vector.nbytes / 1024:.1f} KB)")
        
        return created_files
    
    def cleanup(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.activations = {}
        gc.collect()
        torch.cuda.empty_cache()


# =============================================================================
# TEST SET
# =============================================================================

TEST_QUESTIONS = [
    # Factual (model should know these)
    ("What is the capital of Spain?", True),
    ("What is 15 plus 27?", True),
    ("Who invented the telephone?", True),
    ("What is the chemical symbol for sodium?", True),
    ("How many sides does a triangle have?", True),
    ("What planet is known as the Red Planet?", True),
    ("What is the largest country by area?", True),
    ("Who wrote '1984'?", True),
    ("What is the boiling point of water in Fahrenheit?", True),
    ("What year did the Berlin Wall fall?", True),
    
    # Hallucination triggers (model will confabulate)
    ("What is the capital of the Kingdom of Narnia?", False),
    ("What was Einstein's favorite pizza topping?", False),
    ("Who was the first female President of the USA?", False),
    ("What year did humans land on Jupiter?", False),
    ("What is the phone number of the White House?", False),
    ("How many Olympic gold medals did Shakespeare win?", False),
    ("What is the population of Hogwarts?", False),
    ("Who invented the flux capacitor?", False),
    ("What is the GDP of Mordor?", False),
    ("When did the Fourth World War end?", False),
]


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="AHD PoC v2")
    parser.add_argument("--model", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--layer-pct", type=float, default=0.65)
    parser.add_argument("--output-dir", type=str, default="ahd_vectors")
    args = parser.parse_args()
    
    print("=" * 60)
    print("AHD (Activation-based Hallucination Detection) PoC v2")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Layer %: {args.layer_pct}")
    
    # Load model
    print("\n[1/5] Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    model = None
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        try:
            print(f"  Trying {dtype}...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            test_input = tokenizer("Hello", return_tensors="pt").to(model.device)
            with torch.no_grad():
                test_out = model(test_input.input_ids)
                if torch.isnan(test_out.logits).any():
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                    model = None
                    continue
            print(f"  Loaded with {dtype}")
            break
        except Exception as e:
            print(f"    {dtype} failed: {e}")
    
    if model is None:
        raise ValueError("Failed to load model")
    
    # Create trainer
    print("\n[2/5] Creating trainer...")
    trainer = AHDTrainerV2(
        model=model,
        tokenizer=tokenizer,
        layer_pct=args.layer_pct,
    )
    
    # Train
    print("\n[3/5] Training...")
    start = time.time()
    stats = trainer.train()
    train_time = time.time() - start
    print(f"\n  Training completed in {train_time:.1f}s")
    
    # Calibrate
    print("\n[4/5] Calibrating...")
    trainer.calibrate()
    
    # Evaluate
    print("\n[5/5] Evaluating on test set...")
    eval_results = trainer.evaluate(TEST_QUESTIONS, use_calibrated=True)
    
    print(f"\n  Test Results:")
    print(f"    Overall Accuracy: {eval_results['accuracy']*100:.1f}%")
    print(f"    Factual Accuracy: {eval_results['factual_accuracy']*100:.1f}%")
    print(f"    Hallucination Detection: {eval_results['hallucination_accuracy']*100:.1f}%")
    
    print("\n  Detailed Results:")
    for r in eval_results['results']:
        status = "✓" if r['correct'] else "✗"
        pred = "fact" if r['predicted_factual'] else "hall"
        actual = "fact" if r['is_factual'] else "hall"
        norm = r['normalized']
        print(f"    {status} [{pred}/{actual}] score={r['score']:.1f} norm={norm:.2f} | {r['question']}")
    
    # Save
    model_name = args.model.split('/')[-1]
    output_path = Path(args.output_dir) / f"ahd_v2_{model_name}"
    trainer.save_vector(str(output_path))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Training Accuracy: {stats.accuracy*100:.1f}%")
    print(f"Test Accuracy: {eval_results['accuracy']*100:.1f}%")
    print(f"AUC: {stats.auc:.3f}")
    print(f"Separation: {stats.separation:.2f}σ")
    
    # Viability
    print("\n" + "-" * 60)
    print("VIABILITY ASSESSMENT")
    print("-" * 60)
    
    test_acc = eval_results['accuracy']
    if test_acc >= 0.80 and stats.auc >= 0.85:
        print("🟢 GREEN LIGHT: Strong results!")
        print("   Recommend proceeding with IDF filing.")
    elif test_acc >= 0.70 and stats.auc >= 0.75:
        print("🟡 YELLOW LIGHT: Promising but needs refinement.")
        print("   Consider: more training data, different layer %")
    else:
        print("🔴 RED LIGHT: Weak separation on test set.")
        print("   Hypothesis may need revision.")
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive Testing (Ctrl+C to exit)")
    print("=" * 60)
    print(f"Threshold: {trainer.calibrated_threshold:.2f}")
    print("Score interpretation: higher = more likely factual")
    
    try:
        while True:
            print()
            q = input("Question: ").strip()
            if not q:
                continue
            
            result = trainer.classify(q, use_calibrated=True)
            status = "🔴 LIKELY HALLUCINATION" if result.is_hallucination else "🟢 LIKELY FACTUAL"
            print(f"  {status}")
            print(f"  Score: {result.confidence_score:.2f} | Normalized: {result.normalized_score:.2f}")
            print(f"  Threshold: {result.threshold:.2f} | Latency: {result.latency_ms:.2f}ms")
    except KeyboardInterrupt:
        print("\n\nExiting...")
    
    trainer.cleanup()


if __name__ == "__main__":
    main()
