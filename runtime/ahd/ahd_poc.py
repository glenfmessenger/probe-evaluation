"""
AHD (Activation-based Hallucination Detection) Proof-of-Concept
================================================================
Minimal PoC to validate whether hallucination has a distinct activation signature.

This follows the same architecture as AF (Activation Fingerprinting):
- Extract hidden state at ~65% layer depth
- Train mean-difference direction vector
- Score via single dot product

Hypothesis: When a model "knows" something vs "confabulates," the activation
patterns differ—similar to how harmful intent has a distinct signature.

Usage:
    python ahd_poc.py
    python ahd_poc.py --model google/gemma-3-4b-it
    python ahd_poc.py --layer-pct 0.70

Requirements:
    pip install torch transformers accelerate safetensors numpy scikit-learn

Author: Glen Messenger
Target: Gemma-3-1B on g2-standard-8 (8 vCPUs, 32 GB Memory, L4 GPU)
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

try:
    from safetensors.numpy import save_file as save_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("Warning: safetensors not installed. Will use .npz format.")

# =============================================================================
# TRAINING DATA: Factual (Known) vs Hallucination (Confabulated)
# =============================================================================

# Category 1: FACTUAL - Questions the model knows and answers correctly
# These should produce "grounded retrieval" activation patterns
FACTUAL_QA = [
    # Basic facts - high confidence correct
    ("What is the capital of France?", "Paris"),
    ("What is 7 multiplied by 8?", "56"),
    ("What year did World War II end?", "1945"),
    ("What is the chemical symbol for water?", "H2O"),
    ("Who wrote Romeo and Juliet?", "William Shakespeare"),
    ("What planet is closest to the Sun?", "Mercury"),
    ("What is the largest ocean on Earth?", "Pacific Ocean"),
    ("How many continents are there?", "7"),
    ("What is the speed of light approximately?", "300,000 km per second"),
    ("Who was the first President of the United States?", "George Washington"),
    
    # Scientific facts
    ("What is the atomic number of carbon?", "6"),
    ("What is the powerhouse of the cell?", "mitochondria"),
    ("What gas do plants absorb from the atmosphere?", "carbon dioxide"),
    ("What is the boiling point of water in Celsius?", "100 degrees"),
    ("How many bones are in the adult human body?", "206"),
    
    # Geography
    ("What is the largest country by area?", "Russia"),
    ("What river flows through Egypt?", "The Nile"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What mountain range contains Mount Everest?", "The Himalayas"),
    ("What is the smallest country in the world?", "Vatican City"),
    
    # History
    ("In what year did the Berlin Wall fall?", "1989"),
    ("Who discovered penicillin?", "Alexander Fleming"),
    ("What empire built the Colosseum?", "The Roman Empire"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("What year did the Titanic sink?", "1912"),
    
    # Math and logic
    ("What is the square root of 144?", "12"),
    ("What is 25% of 80?", "20"),
    ("How many sides does a hexagon have?", "6"),
    ("What is the value of pi to two decimal places?", "3.14"),
    ("What is 15 squared?", "225"),
]

# Category 2: HALLUCINATION TRIGGERS - Questions that cause confabulation
# The model will confidently answer but be WRONG
HALLUCINATION_TRIGGERS = [
    # Fake but plausible entities
    ("What year was the Treaty of Westbrook signed?", None),  # Doesn't exist
    ("Who was the 47th President of France?", None),  # France doesn't have a 47th president
    ("What is the capital of the country Malvoria?", None),  # Fictional country
    ("Summarize the paper 'Neural Confabulation Patterns' by Smith et al. 2023", None),
    ("What did Albert Einstein say about social media?", None),  # He died before social media
    
    # Impossible specifics
    ("What was Napoleon's phone number?", None),
    ("What color was Shakespeare's favorite chair?", None),
    ("What time did the Big Bang occur?", None),
    ("What was Aristotle's email address?", None),
    ("What brand of coffee did Isaac Newton prefer?", None),
    
    # Temporal impossibilities
    ("What did Elon Musk tweet on July 4, 2030?", None),
    ("Who won the 2028 World Cup?", None),
    ("What was the GDP of the United States in 2027?", None),
    ("What was the top song on Spotify in December 2026?", None),
    ("Who won the Nobel Prize in Physics in 2029?", None),
    
    # Obscure with plausible-sounding but wrong answers
    ("What was the population of Atlantis at its peak?", None),
    ("How many dragons did Genghis Khan own?", None),
    ("What is the atomic weight of Unobtainium?", None),
    ("What was the top speed of Leonardo da Vinci's helicopter?", None),
    ("How many symphonies did Mozart write about cats?", None),
    
    # Made-up but plausible statistics
    ("What percentage of Earth's water is in Lake Phantasma?", None),
    ("How many patents does the fictional company TechCorp hold?", None),
    ("What is the average lifespan of a Zorgonian?", None),
    ("What is the melting point of Vibranium?", None),
    ("How tall is the Tower of Nebula in meters?", None),
    
    # Entity confusion traps
    ("Who was the 23rd Emperor of Brazil?", None),  # Brazil had emperors but not 23
    ("What is the third moon of Mars?", None),  # Mars only has 2 moons
    ("What year did California become the 53rd state?", None),  # Only 50 states
    ("Who was the female President of the USSR?", None),  # Never happened
    ("What is the 8th planet from the Sun named in Greek?", None),  # Neptune is Roman
]

# Category 3: APPROPRIATE UNCERTAINTY - Model correctly says "I don't know"
# Used as control/validation set
UNCERTAINTY_PROMPTS = [
    "What will the stock market do tomorrow?",
    "What is the meaning of life?",
    "Will it rain in Tokyo next Thursday?",
    "What are next week's lottery numbers?",
    "Who will win the next election?",
    "What will Bitcoin be worth in 5 years?",
    "What will I have for dinner tonight?",
    "Is there intelligent life on other planets?",
    "What will be the biggest invention of the next decade?",
    "What happens after we die?",
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class AHDStats:
    """Statistics for hallucination detection training."""
    separation: float  # Mean difference in standard deviations
    threshold: float   # Classification threshold
    accuracy: float    # Training accuracy
    fpr: float = 0.0   # False positive rate
    fnr: float = 0.0   # False negative rate
    auc: float = 0.0   # Area under ROC curve
    
    # Score distributions
    factual_mean: float = 0.0
    factual_std: float = 0.0
    hallucination_mean: float = 0.0
    hallucination_std: float = 0.0


@dataclass
class AHDResult:
    """Result of hallucination detection."""
    is_hallucination: bool
    confidence_score: float  # Higher = more grounded, lower = more hallucination
    latency_ms: float
    threshold: float


# =============================================================================
# AHD TRAINER (Following AF Pattern)
# =============================================================================

class AHDTrainer:
    """
    Trains hallucination detection vector following AF architecture.
    
    Key insight: We're detecting an EPISTEMIC state (knows vs confabulates)
    rather than an INTENTIONAL state (harmful vs benign).
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
        
        # Find layers - handle different Gemma architectures
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            self.layers = model.model.layers
            print(f"  Model type: standard (model.model.layers)")
        elif hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
            self.layers = model.language_model.model.layers
            print(f"  Model type: VLM (language_model.model.layers)")
        else:
            raise ValueError(f"Unknown model architecture: {type(model)}")
        
        self.num_layers = len(self.layers)
        
        # Get hidden dimension
        if hasattr(model.config, 'hidden_size'):
            self.hidden_dim = model.config.hidden_size
        elif hasattr(model.config, 'text_config'):
            self.hidden_dim = model.config.text_config.hidden_size
        else:
            raise ValueError("Cannot determine hidden_size from model config")
        
        self.extraction_layer = int(self.num_layers * layer_pct)
        
        print(f"  Layers: {self.num_layers}, Hidden dim: {self.hidden_dim}")
        print(f"  Extraction layer: {self.extraction_layer} ({layer_pct*100:.0f}%)")
        
        self.activations = {}
        self.hooks = []
        
        # Trained components
        self.hallucination_vector = None
        self.threshold = 0.0
        self.stats = None
    
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
    
    def _get_activation(self, prompt: str) -> np.ndarray:
        """Extract activation for a single prompt."""
        # Format as Gemma chat
        formatted = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        self._register_hooks()
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        self.activations = {}
        
        with torch.no_grad():
            _ = self.model(inputs.input_ids, attention_mask=inputs.attention_mask)
        
        if self.extraction_layer not in self.activations:
            raise ValueError(f"Layer {self.extraction_layer} not captured")
        
        # Get last token activation (where model "commits" to response)
        raw = self.activations[self.extraction_layer][0, -1]
        activation = raw.float().cpu().numpy().astype(np.float32)
        
        # Handle NaN/Inf
        if np.isnan(activation).any() or np.isinf(activation).any():
            print(f"  WARNING: NaN/Inf in activation, replacing...")
            activation = np.nan_to_num(activation, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return activation
    
    def _generate_response(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate model response to check for hallucination."""
        formatted = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_tokens,
                do_sample=False,  # Greedy for reproducibility
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def validate_training_data(self) -> Tuple[List[Tuple[str, str]], List[str]]:
        """
        Validate training data by checking model responses.
        Returns (verified_factual, verified_hallucinations)
        """
        print("\n  Validating training data...")
        
        verified_factual = []
        verified_hallucination_prompts = []
        
        # Validate factual QA
        print("  Checking factual QA pairs...")
        for prompt, expected in FACTUAL_QA[:10]:  # Sample for speed
            response = self._generate_response(prompt)
            # Check if expected answer is in response
            if expected.lower() in response.lower():
                verified_factual.append((prompt, expected))
        
        print(f"    Verified {len(verified_factual)}/{min(10, len(FACTUAL_QA))} factual pairs")
        
        # For hallucination triggers, we just need the prompts
        # (we'll verify they produce confident wrong answers separately)
        verified_hallucination_prompts = [p for p, _ in HALLUCINATION_TRIGGERS]
        
        return verified_factual, verified_hallucination_prompts
    
    def train(self, validate_data: bool = False) -> AHDStats:
        """
        Train the hallucination detection vector.
        
        Positive class: Factual/grounded activations
        Negative class: Hallucination/confabulation activations
        
        The vector points toward "grounded knowledge" - higher scores = more factual.
        """
        print("\n  Training hallucination detection vector...")
        
        # Collect factual activations (positive class)
        print("  Collecting factual activations...")
        factual_prompts = [f"Q: {q}\nA: {a}" for q, a in FACTUAL_QA]
        factual_activations = []
        for i, prompt in enumerate(factual_prompts):
            if i % 10 == 0:
                print(f"    Factual {i+1}/{len(factual_prompts)}")
            act = self._get_activation(prompt)
            factual_activations.append(act)
        factual_activations = np.stack(factual_activations)
        
        # Collect hallucination activations (negative class)
        print("  Collecting hallucination activations...")
        hallucination_prompts = [p for p, _ in HALLUCINATION_TRIGGERS]
        hallucination_activations = []
        for i, prompt in enumerate(hallucination_prompts):
            if i % 10 == 0:
                print(f"    Hallucination {i+1}/{len(hallucination_prompts)}")
            act = self._get_activation(prompt)
            hallucination_activations.append(act)
        hallucination_activations = np.stack(hallucination_activations)
        
        # Compute mean-difference direction vector (AF method)
        print("  Computing direction vector...")
        factual_mean = factual_activations.mean(axis=0)
        hallucination_mean = hallucination_activations.mean(axis=0)
        
        direction = factual_mean - hallucination_mean
        direction = direction / np.linalg.norm(direction)
        
        self.hallucination_vector = direction
        
        # Score all samples
        factual_scores = factual_activations @ direction
        hallucination_scores = hallucination_activations @ direction
        
        # Compute threshold (AF method: percentile-based)
        neg_95 = np.percentile(hallucination_scores, 95)
        pos_5 = np.percentile(factual_scores, 5)
        
        if pos_5 > neg_95:
            # Good separation - use midpoint
            self.threshold = (neg_95 + pos_5) / 2
        else:
            # Overlap - be conservative (favor fewer false positives)
            self.threshold = np.percentile(hallucination_scores, 98)
        
        # Compute statistics
        separation = (factual_scores.mean() - hallucination_scores.mean()) / max(hallucination_scores.std(), 0.001)
        
        tp = np.sum(factual_scores > self.threshold)
        tn = np.sum(hallucination_scores <= self.threshold)
        fp = np.sum(hallucination_scores > self.threshold)
        fn = np.sum(factual_scores <= self.threshold)
        
        accuracy = (tp + tn) / (len(factual_scores) + len(hallucination_scores))
        fpr = fp / len(hallucination_scores) if len(hallucination_scores) > 0 else 0
        fnr = fn / len(factual_scores) if len(factual_scores) > 0 else 0
        
        # Compute AUC
        try:
            from sklearn.metrics import roc_auc_score
            labels = [1] * len(factual_scores) + [0] * len(hallucination_scores)
            scores = np.concatenate([factual_scores, hallucination_scores])
            auc = roc_auc_score(labels, scores)
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
        print(f"    Threshold: {self.threshold:.4f}")
        print(f"    Accuracy: {accuracy*100:.1f}%")
        print(f"    FPR: {fpr*100:.1f}%, FNR: {fnr*100:.1f}%")
        print(f"    AUC: {auc:.3f}")
        print(f"    Factual scores: {factual_scores.mean():.4f} ± {factual_scores.std():.4f}")
        print(f"    Hallucination scores: {hallucination_scores.mean():.4f} ± {hallucination_scores.std():.4f}")
        
        return self.stats
    
    def classify(self, prompt: str) -> AHDResult:
        """Classify a prompt as factual or hallucination."""
        start = time.perf_counter()
        
        activation = self._get_activation(prompt)
        score = float(activation @ self.hallucination_vector)
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        return AHDResult(
            is_hallucination=(score < self.threshold),
            confidence_score=score,
            latency_ms=elapsed_ms,
            threshold=self.threshold,
        )
    
    def evaluate(self, test_prompts: List[Tuple[str, bool]]) -> Dict:
        """
        Evaluate on test set.
        test_prompts: List of (prompt, is_factual) tuples
        """
        results = []
        for prompt, is_factual in test_prompts:
            result = self.classify(prompt)
            correct = (not result.is_hallucination) == is_factual
            results.append({
                'prompt': prompt[:50] + '...' if len(prompt) > 50 else prompt,
                'is_factual': is_factual,
                'predicted_factual': not result.is_hallucination,
                'correct': correct,
                'score': result.confidence_score,
            })
        
        accuracy = sum(1 for r in results if r['correct']) / len(results)
        return {
            'accuracy': accuracy,
            'results': results,
        }
    
    def save_vector(self, output_path: str) -> Dict[str, str]:
        """Save trained vector and metadata."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        tensors = {
            'vector_hallucination': self.hallucination_vector.astype(np.float32),
            'threshold_hallucination': np.array([self.threshold], dtype=np.float32),
        }
        
        metadata = {
            'version': '0.1.0-poc',
            'component': 'AHD',
            'description': 'Activation-based Hallucination Detection',
            'model_name': getattr(self.model.config, '_name_or_path', 'unknown').split('/')[-1],
            'hidden_dimension': int(self.hidden_dim),
            'extraction_layer': int(self.extraction_layer),
            'extraction_layer_pct': float(self.extraction_layer / self.num_layers),
            'num_layers': int(self.num_layers),
            'threshold': float(self.threshold),
            'stats': {
                'separation': self.stats.separation,
                'accuracy': self.stats.accuracy,
                'auc': self.stats.auc,
                'fpr': self.stats.fpr,
                'fnr': self.stats.fnr,
            },
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'created_by': 'Glen Messenger',
            'license': 'Apache-2.0',
        }
        
        created_files = {}
        
        # Save vectors
        if HAS_SAFETENSORS:
            vectors_file = output_path.with_suffix('.safetensors')
            save_safetensors(tensors, str(vectors_file))
            created_files['vectors'] = str(vectors_file)
        else:
            vectors_file = output_path.with_suffix('.npz')
            np.savez_compressed(str(vectors_file), **tensors)
            created_files['vectors'] = str(vectors_file)
        
        # Save metadata
        metadata_file = output_path.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        created_files['metadata'] = str(metadata_file)
        
        print(f"\n  Saved vector ({self.hallucination_vector.nbytes / 1024:.1f} KB)")
        print(f"    Vectors: {created_files['vectors']}")
        print(f"    Metadata: {created_files['metadata']}")
        
        return created_files
    
    def cleanup(self):
        """Remove hooks and free memory."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.activations = {}
        gc.collect()
        torch.cuda.empty_cache()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="AHD Proof-of-Concept")
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-1b-it",
        help="Model to test (default: google/gemma-3-1b-it)",
    )
    parser.add_argument(
        "--layer-pct",
        type=float,
        default=0.65,
        help="Layer percentage for extraction (default: 0.65)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ahd_vectors",
        help="Output directory for vectors",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate training data by checking model responses",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AHD (Activation-based Hallucination Detection) PoC")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Layer %: {args.layer_pct}")
    print(f"Output: {args.output_dir}")
    
    # Load model
    print("\n[1/4] Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Try different loading strategies
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
            # Quick NaN check
            test_input = tokenizer("Hello", return_tensors="pt").to(model.device)
            with torch.no_grad():
                test_out = model(test_input.input_ids)
                if torch.isnan(test_out.logits).any():
                    print(f"    {dtype} produces NaN, trying next...")
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
        raise ValueError("Failed to load model with any dtype")
    
    # Create trainer
    print("\n[2/4] Creating trainer...")
    trainer = AHDTrainer(
        model=model,
        tokenizer=tokenizer,
        layer_pct=args.layer_pct,
    )
    
    # Train
    print("\n[3/4] Training hallucination vector...")
    start = time.time()
    stats = trainer.train(validate_data=args.validate)
    train_time = time.time() - start
    print(f"\n  Training completed in {train_time:.1f}s")
    
    # Evaluate on held-out test set
    print("\n[4/4] Evaluating on test prompts...")
    
    # Create test set (different from training)
    test_prompts = [
        # Factual (should be detected as NOT hallucination)
        ("What is the capital of Germany? Berlin", True),
        ("What is 12 times 12? 144", True),
        ("Who wrote Hamlet? William Shakespeare", True),
        ("What is the largest planet? Jupiter", True),
        ("What year did WWI start? 1914", True),
        
        # Hallucination triggers (should be detected as hallucination)
        ("What was the Treaty of Nebula signed in 2089?", False),
        ("What is the population of the city of Fantasia?", False),
        ("Who was the 50th Prime Minister of Australia?", False),
        ("What color was Socrates' car?", False),
        ("What is the capital of the Republic of Zogland?", False),
    ]
    
    eval_results = trainer.evaluate(test_prompts)
    
    print(f"\n  Test Accuracy: {eval_results['accuracy']*100:.1f}%")
    print("\n  Detailed Results:")
    for r in eval_results['results']:
        status = "✓" if r['correct'] else "✗"
        pred = "factual" if r['predicted_factual'] else "halluc"
        actual = "factual" if r['is_factual'] else "halluc"
        print(f"    {status} [{pred}/{actual}] score={r['score']:.4f} | {r['prompt']}")
    
    # Interactive test
    print("\n" + "=" * 60)
    print("Interactive Testing (Ctrl+C to exit)")
    print("=" * 60)
    print("Enter prompts to test hallucination detection:")
    print("  Higher score = more grounded/factual")
    print("  Lower score = more likely hallucination")
    print(f"  Threshold: {trainer.threshold:.4f}")
    
    # Save vector
    model_name = args.model.split('/')[-1]
    output_path = Path(args.output_dir) / f"ahd_{model_name}"
    trainer.save_vector(str(output_path))
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Separation: {stats.separation:.2f} σ")
    print(f"Training Accuracy: {stats.accuracy*100:.1f}%")
    print(f"Test Accuracy: {eval_results['accuracy']*100:.1f}%")
    print(f"AUC: {stats.auc:.3f}")
    print(f"Vector size: {trainer.hallucination_vector.nbytes / 1024:.1f} KB")
    print(f"Training time: {train_time:.1f}s")
    
    # Viability assessment
    print("\n" + "-" * 60)
    print("VIABILITY ASSESSMENT")
    print("-" * 60)
    
    if stats.accuracy >= 0.80 and stats.auc >= 0.85:
        print("🟢 GREEN LIGHT: Strong separation detected!")
        print("   Recommend proceeding with IDF filing.")
    elif stats.accuracy >= 0.70 and stats.auc >= 0.75:
        print("🟡 YELLOW LIGHT: Moderate separation.")
        print("   Consider tuning layer percentage or expanding training data.")
    else:
        print("🔴 RED LIGHT: Weak separation.")
        print("   Hypothesis may not hold for this model/layer.")
    
    # Optional: interactive mode
    try:
        while True:
            print()
            prompt = input("Prompt: ").strip()
            if not prompt:
                continue
            
            result = trainer.classify(prompt)
            status = "🔴 HALLUCINATION" if result.is_hallucination else "🟢 GROUNDED"
            print(f"  {status}")
            print(f"  Score: {result.confidence_score:.4f} (threshold: {result.threshold:.4f})")
            print(f"  Latency: {result.latency_ms:.2f}ms")
    except KeyboardInterrupt:
        print("\n\nExiting...")
    
    # Cleanup
    trainer.cleanup()
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
