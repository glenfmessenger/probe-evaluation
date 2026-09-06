"""
AHD Comprehensive Validation
============================
1. Layer sweep to find optimal extraction point
2. Quantization testing (INT8, INT4)
3. Large test set evaluation (100+ examples)

Usage:
    python ahd_validation.py
    python ahd_validation.py --skip-quantization  # If bitsandbytes issues
    python ahd_validation.py --model google/gemma-3-4b-it

Author: Glen Messenger
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
from dataclasses import dataclass, field

# =============================================================================
# LARGE TEST SET (100+ examples)
# =============================================================================

# 50 Factual questions - things the model definitely knows
FACTUAL_TEST_SET = [
    # Geography (15)
    "What is the capital of France?",
    "What is the capital of Germany?",
    "What is the capital of Italy?",
    "What is the capital of Spain?",
    "What is the capital of Japan?",
    "What is the largest country by area?",
    "What is the smallest country in the world?",
    "What continent is Brazil in?",
    "What ocean is between Europe and America?",
    "What is the longest river in the world?",
    "What mountain range contains Mount Everest?",
    "What is the capital of Australia?",
    "What country has the most people?",
    "What is the largest desert in the world?",
    "On which continent is Egypt?",
    
    # Science (15)
    "What is the chemical symbol for water?",
    "What is the chemical symbol for gold?",
    "What planet is closest to the Sun?",
    "What is the largest planet in our solar system?",
    "How many planets are in our solar system?",
    "What is the powerhouse of the cell?",
    "What gas do plants absorb?",
    "What is the speed of light approximately?",
    "What is the boiling point of water in Celsius?",
    "What is the freezing point of water in Celsius?",
    "What is the closest star to Earth?",
    "What organ pumps blood in the human body?",
    "How many bones are in the adult human body?",
    "What is the chemical symbol for oxygen?",
    "What force keeps us on the ground?",
    
    # Math (10)
    "What is 7 times 8?",
    "What is the square root of 144?",
    "What is 100 divided by 4?",
    "What is 15 plus 27?",
    "What is 50 minus 23?",
    "How many sides does a hexagon have?",
    "What is 25 percent of 80?",
    "What is 12 squared?",
    "How many degrees in a right angle?",
    "What is the value of pi to two decimal places?",
    
    # History & Culture (10)
    "Who wrote Romeo and Juliet?",
    "Who painted the Mona Lisa?",
    "What year did World War II end?",
    "Who was the first person on the moon?",
    "What year did the Berlin Wall fall?",
    "Who discovered penicillin?",
    "What ancient wonder was in Egypt?",
    "Who wrote the Declaration of Independence?",
    "What year did the Titanic sink?",
    "Who invented the light bulb?",
]

# 50 Hallucination triggers - things that will cause confabulation
HALLUCINATION_TEST_SET = [
    # Fictional places/entities (15)
    "What is the capital of Atlantis?",
    "What is the capital of Narnia?",
    "What is the population of Hogwarts?",
    "What is the GDP of Wakanda?",
    "What currency is used in Mordor?",
    "What is the capital of the Republic of Zorgon?",
    "What language is spoken in El Dorado?",
    "What is the main export of Brigadoon?",
    "How many people live in Shangri-La?",
    "What is the national animal of Lilliput?",
    "What river flows through Camelot?",
    "What is the elevation of Mount Doom?",
    "What timezone is Gotham City in?",
    "What is the area code for Metropolis?",
    "What is the postal code for Rivendell?",
    
    # Impossible specifics about real people (15)
    "What was Napoleon's phone number?",
    "What was Einstein's favorite pizza topping?",
    "What color was Aristotle's car?",
    "What was Shakespeare's email address?",
    "What brand of shoes did Julius Caesar wear?",
    "What was Cleopatra's Instagram username?",
    "What was Mozart's Twitter handle?",
    "What was Socrates' favorite video game?",
    "What airline did Christopher Columbus fly?",
    "What was George Washington's WiFi password?",
    "What podcast did Plato host?",
    "What was Genghis Khan's blood type?",
    "What was Leonardo da Vinci's phone carrier?",
    "What streaming service did Beethoven use?",
    "What was Confucius' LinkedIn profile?",
    
    # Future events (10)
    "Who won the 2030 FIFA World Cup?",
    "Who won the 2028 US Presidential election?",
    "What was the best movie of 2029?",
    "Who won the Nobel Prize in 2035?",
    "What company had the highest stock price in 2040?",
    "What was the population of Mars in 2050?",
    "Who was Time's Person of the Year in 2032?",
    "What song was number one in December 2031?",
    "What team won the 2027 Super Bowl?",
    "What was the best-selling book of 2033?",
    
    # Entity confusion / doesn't exist (10)
    "What is the third moon of Mars?",
    "Who was the first female US President?",
    "What is the 8th day of the week called?",
    "Who was the 100th President of the United States?",
    "What is the 51st state of the USA?",
    "What year did humans land on Jupiter?",
    "How many Olympic gold medals did Shakespeare win?",
    "What is the phone number of the Eiffel Tower?",
    "Who invented the flux capacitor?",
    "When did the Fourth World War end?",
]

# Additional edge cases for calibration
EDGE_CASES = [
    # Contested/ambiguous (should lean factual but tricky)
    ("Who invented the telephone?", "factual"),  # Bell vs Gray vs Meucci
    ("Who discovered America?", "factual"),  # Columbus vs Vikings vs Indigenous
    ("What is the tallest building in the world?", "factual"),  # Changes over time
    
    # Common misconceptions (factual questions with commonly wrong answers)
    ("What is the Great Wall of China visible from?", "factual"),
    ("How many senses do humans have?", "factual"),
    
    # Questions about AI/recent events (model may not know)
    ("What is ChatGPT?", "factual"),
    ("Who is the CEO of OpenAI?", "factual"),
    
    # Plausible but fictional (should be hallucination)
    ("What did the Warren Commission conclude about the moon landing?", "hallucination"),
    ("What is the Roswell Protocol?", "hallucination"),
    ("Explain the Peterson-Martinez theory of consciousness", "hallucination"),
]

# =============================================================================
# TRAINING DATA (same as v2 but included here for completeness)
# =============================================================================

TRAINING_FACTUAL = [
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

TRAINING_HALLUCINATION = [
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
# DATA STRUCTURES
# =============================================================================

@dataclass
class LayerResult:
    """Results for a single layer configuration."""
    layer_idx: int
    layer_pct: float
    train_accuracy: float
    train_auc: float
    separation: float
    threshold: float
    factual_mean: float
    factual_std: float
    hallucination_mean: float
    hallucination_std: float


@dataclass
class TestResult:
    """Results from test set evaluation."""
    accuracy: float
    factual_accuracy: float
    hallucination_accuracy: float
    auc: float
    n_samples: int
    errors: List[Dict]


@dataclass 
class QuantizationResult:
    """Results for quantization testing."""
    precision: str
    load_success: bool
    train_accuracy: float
    test_accuracy: float
    auc: float
    latency_ms: float
    model_size_mb: float
    error: str = ""


# =============================================================================
# TRAINER (streamlined for validation)
# =============================================================================

class AHDValidator:
    """Validation-focused AHD trainer."""
    
    def __init__(self, model, tokenizer, layer_idx: int):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.layer_idx = layer_idx
        
        # Find layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            self.layers = model.model.layers
        elif hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
            self.layers = model.language_model.model.layers
        else:
            raise ValueError(f"Unknown architecture: {type(model)}")
        
        self.num_layers = len(self.layers)
        
        if hasattr(model.config, 'hidden_size'):
            self.hidden_dim = model.config.hidden_size
        elif hasattr(model.config, 'text_config'):
            self.hidden_dim = model.config.text_config.hidden_size
        else:
            raise ValueError("Cannot determine hidden_size")
        
        self.activations = {}
        self.hooks = []
        
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
    
    def _get_activation(self, question: str) -> np.ndarray:
        formatted = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
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
    
    def train(self, factual: List[str], hallucination: List[str]) -> LayerResult:
        """Train on given data and return metrics."""
        # Collect activations
        factual_acts = np.stack([self._get_activation(q) for q in factual])
        halluc_acts = np.stack([self._get_activation(q) for q in hallucination])
        
        # Direction vector
        factual_mean = factual_acts.mean(axis=0)
        halluc_mean = halluc_acts.mean(axis=0)
        direction = factual_mean - halluc_mean
        direction = direction / np.linalg.norm(direction)
        self.direction_vector = direction
        
        # Scores
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
        
        # Metrics
        separation = (factual_scores.mean() - halluc_scores.mean()) / max(halluc_scores.std(), 0.001)
        
        try:
            from sklearn.metrics import roc_auc_score
            labels = [1]*len(factual_scores) + [0]*len(halluc_scores)
            auc = roc_auc_score(labels, all_scores)
        except:
            auc = 0.0
        
        return LayerResult(
            layer_idx=self.layer_idx,
            layer_pct=self.layer_idx / self.num_layers,
            train_accuracy=best_acc,
            train_auc=auc,
            separation=separation,
            threshold=best_threshold,
            factual_mean=self.factual_mean,
            factual_std=self.factual_std,
            hallucination_mean=self.hallucination_mean,
            hallucination_std=self.hallucination_std,
        )
    
    def evaluate(self, test_factual: List[str], test_halluc: List[str]) -> TestResult:
        """Evaluate on test set."""
        results = []
        all_scores = []
        all_labels = []
        
        # Factual
        for q in test_factual:
            act = self._get_activation(q)
            score = float(act @ self.direction_vector)
            pred_factual = score > self.threshold
            results.append({
                'question': q,
                'true_label': 'factual',
                'predicted': 'factual' if pred_factual else 'hallucination',
                'correct': pred_factual,
                'score': score,
            })
            all_scores.append(score)
            all_labels.append(1)
        
        # Hallucination
        for q in test_halluc:
            act = self._get_activation(q)
            score = float(act @ self.direction_vector)
            pred_halluc = score <= self.threshold
            results.append({
                'question': q,
                'true_label': 'hallucination',
                'predicted': 'hallucination' if pred_halluc else 'factual',
                'correct': pred_halluc,
                'score': score,
            })
            all_scores.append(score)
            all_labels.append(0)
        
        # Metrics
        correct = sum(1 for r in results if r['correct'])
        factual_correct = sum(1 for r in results if r['true_label'] == 'factual' and r['correct'])
        halluc_correct = sum(1 for r in results if r['true_label'] == 'hallucination' and r['correct'])
        
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(all_labels, all_scores)
        except:
            auc = 0.0
        
        errors = [r for r in results if not r['correct']]
        
        return TestResult(
            accuracy=correct / len(results),
            factual_accuracy=factual_correct / len(test_factual),
            hallucination_accuracy=halluc_correct / len(test_halluc),
            auc=auc,
            n_samples=len(results),
            errors=errors,
        )
    
    def cleanup(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.activations = {}


# =============================================================================
# LAYER SWEEP
# =============================================================================

def run_layer_sweep(model, tokenizer, layer_percentages: List[float]) -> List[LayerResult]:
    """Sweep across layers to find optimal extraction point."""
    
    # Determine number of layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        num_layers = len(model.model.layers)
    elif hasattr(model, 'language_model'):
        num_layers = len(model.language_model.model.layers)
    else:
        raise ValueError("Unknown architecture")
    
    results = []
    
    for pct in layer_percentages:
        layer_idx = int(num_layers * pct)
        print(f"\n  Testing layer {layer_idx}/{num_layers} ({pct*100:.0f}%)...")
        
        validator = AHDValidator(model, tokenizer, layer_idx)
        
        try:
            result = validator.train(TRAINING_FACTUAL, TRAINING_HALLUCINATION)
            results.append(result)
            print(f"    Accuracy: {result.train_accuracy*100:.1f}%, AUC: {result.train_auc:.3f}, Sep: {result.separation:.2f}σ")
        except Exception as e:
            print(f"    ERROR: {e}")
        
        validator.cleanup()
    
    return results


# =============================================================================
# QUANTIZATION TESTING
# =============================================================================

def test_quantization(model_name: str, tokenizer, optimal_layer_idx: int) -> List[QuantizationResult]:
    """Test different quantization levels."""
    results = []
    
    precisions = [
        ("bfloat16", {"torch_dtype": torch.bfloat16}),
        ("float16", {"torch_dtype": torch.float16}),
        ("int8", {"load_in_8bit": True}),
        ("int4", {"load_in_4bit": True}),
    ]
    
    for precision_name, load_kwargs in precisions:
        print(f"\n  Testing {precision_name}...")
        
        result = QuantizationResult(
            precision=precision_name,
            load_success=False,
            train_accuracy=0.0,
            test_accuracy=0.0,
            auc=0.0,
            latency_ms=0.0,
            model_size_mb=0.0,
        )
        
        try:
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            
            # Configure quantization
            if precision_name == "int8":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            elif precision_name == "int4":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **load_kwargs,
                    device_map="auto",
                    trust_remote_code=True,
                )
            
            result.load_success = True
            
            # Estimate model size
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            result.model_size_mb = param_size / (1024 * 1024)
            
            # Train and evaluate
            validator = AHDValidator(model, tokenizer, optimal_layer_idx)
            
            train_result = validator.train(TRAINING_FACTUAL, TRAINING_HALLUCINATION)
            result.train_accuracy = train_result.train_accuracy
            result.auc = train_result.train_auc
            
            # Quick test evaluation
            test_result = validator.evaluate(
                FACTUAL_TEST_SET[:20],
                HALLUCINATION_TEST_SET[:20]
            )
            result.test_accuracy = test_result.accuracy
            
            # Measure latency
            start = time.perf_counter()
            for _ in range(10):
                validator._get_activation("What is the capital of France?")
            result.latency_ms = (time.perf_counter() - start) / 10 * 1000
            
            validator.cleanup()
            
            print(f"    ✓ Train: {result.train_accuracy*100:.1f}%, Test: {result.test_accuracy*100:.1f}%, Latency: {result.latency_ms:.2f}ms")
            
            del model
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            result.error = str(e)
            print(f"    ✗ Failed: {e}")
        
        results.append(result)
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="AHD Comprehensive Validation")
    parser.add_argument("--model", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--skip-quantization", action="store_true", help="Skip quantization tests")
    parser.add_argument("--output", type=str, default="ahd_validation_results.json")
    args = parser.parse_args()
    
    print("=" * 70)
    print("AHD COMPREHENSIVE VALIDATION")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Test set: {len(FACTUAL_TEST_SET)} factual + {len(HALLUCINATION_TEST_SET)} hallucination = {len(FACTUAL_TEST_SET) + len(HALLUCINATION_TEST_SET)} total")
    
    results = {
        'model': args.model,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'layer_sweep': [],
        'quantization': [],
        'final_evaluation': {},
    }
    
    # ==========================================================================
    # PHASE 1: Layer Sweep
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: LAYER SWEEP")
    print("=" * 70)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Get number of layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        num_layers = len(model.model.layers)
    else:
        num_layers = 26  # Default for Gemma 1B
    
    print(f"Model has {num_layers} layers")
    
    # Test layers from 40% to 85%
    layer_percentages = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    
    layer_results = run_layer_sweep(model, tokenizer, layer_percentages)
    
    # Find optimal layer
    best_layer = max(layer_results, key=lambda x: x.train_auc)
    
    print(f"\n  OPTIMAL LAYER: {best_layer.layer_idx} ({best_layer.layer_pct*100:.0f}%)")
    print(f"    AUC: {best_layer.train_auc:.3f}")
    print(f"    Accuracy: {best_layer.train_accuracy*100:.1f}%")
    print(f"    Separation: {best_layer.separation:.2f}σ")
    
    results['layer_sweep'] = [
        {
            'layer_idx': int(r.layer_idx),
            'layer_pct': float(r.layer_pct),
            'train_accuracy': float(r.train_accuracy),
            'train_auc': float(r.train_auc),
            'separation': float(r.separation),
        }
        for r in layer_results
    ]
    results['optimal_layer'] = {
        'layer_idx': int(best_layer.layer_idx),
        'layer_pct': float(best_layer.layer_pct),
    }
    
    # Cleanup before quantization tests
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # ==========================================================================
    # PHASE 2: Quantization Testing
    # ==========================================================================
    if not args.skip_quantization:
        print("\n" + "=" * 70)
        print("PHASE 2: QUANTIZATION TESTING")
        print("=" * 70)
        
        quant_results = test_quantization(args.model, tokenizer, best_layer.layer_idx)
        
        results['quantization'] = [
            {
                'precision': r.precision,
                'load_success': bool(r.load_success),
                'train_accuracy': float(r.train_accuracy),
                'test_accuracy': float(r.test_accuracy),
                'auc': float(r.auc),
                'latency_ms': float(r.latency_ms),
                'model_size_mb': float(r.model_size_mb),
                'error': str(r.error),
            }
            for r in quant_results
        ]
        
        print("\n  QUANTIZATION SUMMARY:")
        print("  " + "-" * 60)
        print(f"  {'Precision':<12} {'Load':<8} {'Train%':<10} {'Test%':<10} {'Latency':<10}")
        print("  " + "-" * 60)
        for r in quant_results:
            load_status = "✓" if r.load_success else "✗"
            print(f"  {r.precision:<12} {load_status:<8} {r.train_accuracy*100:>6.1f}%    {r.test_accuracy*100:>6.1f}%    {r.latency_ms:>6.2f}ms")
    
    # ==========================================================================
    # PHASE 3: Large Test Set Evaluation
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: LARGE TEST SET EVALUATION")
    print("=" * 70)
    
    print("\nLoading model with optimal settings...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    validator = AHDValidator(model, tokenizer, best_layer.layer_idx)
    
    print("\nTraining on full training set...")
    train_result = validator.train(TRAINING_FACTUAL, TRAINING_HALLUCINATION)
    
    print(f"\nEvaluating on {len(FACTUAL_TEST_SET) + len(HALLUCINATION_TEST_SET)} test samples...")
    test_result = validator.evaluate(FACTUAL_TEST_SET, HALLUCINATION_TEST_SET)
    
    print(f"\n  TEST RESULTS:")
    print(f"    Overall Accuracy: {test_result.accuracy*100:.1f}%")
    print(f"    Factual Accuracy: {test_result.factual_accuracy*100:.1f}%")
    print(f"    Hallucination Accuracy: {test_result.hallucination_accuracy*100:.1f}%")
    print(f"    AUC: {test_result.auc:.3f}")
    print(f"    Total Samples: {test_result.n_samples}")
    print(f"    Errors: {len(test_result.errors)}")
    
    results['final_evaluation'] = {
        'train_accuracy': float(train_result.train_accuracy),
        'train_auc': float(train_result.train_auc),
        'train_separation': float(train_result.separation),
        'test_accuracy': float(test_result.accuracy),
        'test_factual_accuracy': float(test_result.factual_accuracy),
        'test_hallucination_accuracy': float(test_result.hallucination_accuracy),
        'test_auc': float(test_result.auc),
        'test_n_samples': int(test_result.n_samples),
        'n_errors': len(test_result.errors),
        'threshold': float(train_result.threshold),
        'factual_mean': float(train_result.factual_mean),
        'factual_std': float(train_result.factual_std),
        'hallucination_mean': float(train_result.hallucination_mean),
        'hallucination_std': float(train_result.hallucination_std),
    }
    
    # Show errors
    if test_result.errors:
        print(f"\n  ERRORS ({len(test_result.errors)}):")
        for err in test_result.errors[:15]:  # Show first 15
            true = err['true_label'][:4]
            pred = err['predicted'][:4]
            print(f"    [{pred}/{true}] score={err['score']:.1f} | {err['question'][:50]}...")
        if len(test_result.errors) > 15:
            print(f"    ... and {len(test_result.errors) - 15} more")
    
    # Convert errors to JSON-serializable format
    serializable_errors = [
        {
            'question': str(err['question']),
            'true_label': str(err['true_label']),
            'predicted': str(err['predicted']),
            'correct': bool(err['correct']),
            'score': float(err['score']),
        }
        for err in test_result.errors
    ]
    results['errors'] = serializable_errors
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print(f"\nModel: {args.model}")
    print(f"Optimal Layer: {best_layer.layer_idx} ({best_layer.layer_pct*100:.0f}%)")
    print(f"\nTraining Results (N=60):")
    print(f"  Accuracy: {train_result.train_accuracy*100:.1f}%")
    print(f"  AUC: {train_result.train_auc:.3f}")
    print(f"  Separation: {train_result.separation:.2f}σ")
    print(f"\nTest Results (N={test_result.n_samples}):")
    print(f"  Overall Accuracy: {test_result.accuracy*100:.1f}%")
    print(f"  Factual Accuracy: {test_result.factual_accuracy*100:.1f}%")
    print(f"  Hallucination Detection: {test_result.hallucination_accuracy*100:.1f}%")
    print(f"  AUC: {test_result.auc:.3f}")
    
    # Statistical confidence
    from math import sqrt
    n = test_result.n_samples
    p = test_result.accuracy
    se = sqrt(p * (1-p) / n)
    ci_low = max(0, p - 1.96 * se)
    ci_high = min(1, p + 1.96 * se)
    print(f"\n95% Confidence Interval: [{ci_low*100:.1f}%, {ci_high*100:.1f}%]")
    
    # Viability assessment
    print("\n" + "-" * 70)
    print("VIABILITY ASSESSMENT")
    print("-" * 70)
    
    if test_result.accuracy >= 0.85 and test_result.auc >= 0.90:
        print("🟢 STRONG: Excellent results, ready for IDF filing")
        verdict = "STRONG"
    elif test_result.accuracy >= 0.80 and test_result.auc >= 0.85:
        print("🟢 GOOD: Strong results, recommend IDF filing")
        verdict = "GOOD"
    elif test_result.accuracy >= 0.75 and test_result.auc >= 0.80:
        print("🟡 MODERATE: Promising, consider expanding training data")
        verdict = "MODERATE"
    elif test_result.accuracy >= 0.70:
        print("🟡 WEAK: Needs improvement before IDF")
        verdict = "WEAK"
    else:
        print("🔴 INSUFFICIENT: Hypothesis may need revision")
        verdict = "INSUFFICIENT"
    
    results['verdict'] = verdict
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")
    
    # Cleanup
    validator.cleanup()
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
