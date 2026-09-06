"""
Activation Fingerprinting (AF) Runtime
======================================
Universal runtime for AF safety classification.
Works with any compatible model + vector file.

This file contains the code logic only - no vectors or policy.
Vectors and policy are loaded from separate files, enabling:
- Vector updates without code changes
- Policy tuning without retraining
- Multi-model support via vector swapping

Author: Glen Messenger
Version: 1.0.0
License: Apache-2.0
"""

import json
import numpy as np
import torch
import time
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path

try:
    from safetensors.numpy import load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ClassificationResult:
    """Result of AF classification."""
    blocked: bool
    triggered_categories: List[str]
    scores: Dict[str, float]
    latency_ms: float
    
    def to_dict(self) -> Dict:
        return {
            'blocked': self.blocked,
            'triggered_categories': self.triggered_categories,
            'scores': self.scores,
            'latency_ms': self.latency_ms,
        }


@dataclass 
class CategoryConfig:
    """Configuration for a single category."""
    enabled: bool = True
    threshold: float = 0.0
    action: str = "block"  # block, warn, log


@dataclass
class PolicyConfig:
    """Loaded policy configuration."""
    version: str
    default_extraction_layer_pct: float
    model_overrides: Dict[str, Dict]
    obfuscation_enabled: bool
    obfuscation_require_confirmation: bool
    obfuscation_high_confidence_enabled: bool
    categories: Dict[str, CategoryConfig]
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PolicyConfig':
        categories = {}
        for cat_name, cat_data in data.get('categories', {}).items():
            categories[cat_name] = CategoryConfig(
                enabled=cat_data.get('enabled', True),
                threshold=cat_data.get('threshold_override', 0.0),  # 0 means use vector's threshold
                action=cat_data.get('action', 'block'),
            )
        
        obf = data.get('obfuscation', {})
        
        return cls(
            version=data.get('version', '1.0'),
            default_extraction_layer_pct=data.get('default_extraction_layer_pct', 0.65),
            model_overrides=data.get('model_overrides', {}),
            obfuscation_enabled=obf.get('enabled', True),
            obfuscation_require_confirmation=obf.get('require_category_confirmation', True),
            obfuscation_high_confidence_enabled=obf.get('high_confidence_standalone', True),
            categories=categories,
        )


# =============================================================================
# VECTOR STORE
# =============================================================================

class VectorStore:
    """Loads and manages direction vectors from file."""
    
    def __init__(self, vectors_path: Union[str, Path]):
        self.vectors_path = Path(vectors_path)
        self.vectors: Dict[str, np.ndarray] = {}
        self.thresholds: Dict[str, float] = {}
        self.metadata: Dict = {}
        
        self._load()
    
    def _load(self):
        """Load vectors from file."""
        if self.vectors_path.suffix == '.safetensors':
            self._load_safetensors()
        elif self.vectors_path.suffix == '.npz':
            self._load_npz()
        else:
            raise ValueError(f"Unsupported vector format: {self.vectors_path.suffix}")
    
    def _load_safetensors(self):
        """Load from safetensors format."""
        if not HAS_SAFETENSORS:
            raise ImportError("safetensors not installed. Run: pip install safetensors")
        
        data = load_safetensors(str(self.vectors_path))
        
        for key, value in data.items():
            if key.startswith('vector_'):
                category = key[7:]  # Remove 'vector_' prefix
                self.vectors[category] = value
            elif key.startswith('threshold_'):
                category = key[10:]  # Remove 'threshold_' prefix
                # Handle both scalar and array threshold values
                self.thresholds[category] = float(value.item() if hasattr(value, 'item') else value)
        
        # Load metadata from sidecar JSON if exists
        metadata_path = self.vectors_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
    
    def _load_npz(self):
        """Load from numpy npz format."""
        data = np.load(str(self.vectors_path), allow_pickle=True)
        
        for key in data.files:
            if key.startswith('vector_'):
                category = key[7:]
                self.vectors[category] = data[key]
            elif key.startswith('threshold_'):
                category = key[10:]
                self.thresholds[category] = float(data[key])
            elif key == 'metadata':
                self.metadata = json.loads(str(data[key]))
    
    @property
    def model_name(self) -> str:
        return self.metadata.get('model_name', 'unknown')
    
    @property
    def hidden_dim(self) -> int:
        return self.metadata.get('hidden_dimension', 0)
    
    @property
    def extraction_layer_pct(self) -> float:
        return self.metadata.get('extraction_layer_pct', 0.65)
    
    @property
    def categories(self) -> List[str]:
        return [c for c in self.vectors.keys() if c != 'obfuscation']
    
    @property
    def has_obfuscation(self) -> bool:
        return 'obfuscation' in self.vectors


# =============================================================================
# ACTIVATION EXTRACTOR
# =============================================================================

class ActivationExtractor:
    """Extracts activations from transformer models."""
    
    def __init__(self, model, tokenizer, layer_pct: float = 0.65):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Find layers - handle different model architectures
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # Gemma 3 1B, 270M, Llama, etc.
            self.layers = model.model.layers
        elif hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
            # Some Gemma variants
            self.layers = model.model.decoder.layers
        elif hasattr(model, 'language_model') and hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'layers'):
            # Gemma3ForConditionalGeneration (vision-language variant) - path 1
            self.layers = model.language_model.model.layers
        elif hasattr(model, 'language_model') and hasattr(model.language_model, 'layers'):
            # Gemma3ForConditionalGeneration - direct layers
            self.layers = model.language_model.layers
        elif hasattr(model, 'text_model') and hasattr(model.text_model, 'layers'):
            # Some VLMs use text_model
            self.layers = model.text_model.layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # GPT-2 style
            self.layers = model.transformer.h
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
            # Some other transformers
            self.layers = model.transformer.layers
        else:
            raise ValueError("Unsupported model architecture")
        
        self.num_layers = len(self.layers)
        self.extraction_layer = int(self.num_layers * layer_pct)
        
        self.activations = {}
        self.hooks = []
    
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
    
    def extract(self, prompt: str, chat_template: str = "gemma") -> np.ndarray:
        """Extract activation for a prompt."""
        # Format prompt based on model
        if chat_template == "gemma":
            formatted = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        elif chat_template == "llama":
            formatted = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        elif chat_template == "chatml":
            formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            formatted = prompt
        
        self._register_hooks()
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        self.activations = {}
        
        with torch.no_grad():
            _ = self.model(inputs.input_ids, attention_mask=inputs.attention_mask)
        
        # Get last token activation, convert to float32 first (handles bfloat16)
        raw_activation = self.activations[self.extraction_layer][0, -1]
        activation = raw_activation.float().cpu().numpy().astype(np.float32)
        return activation
    
    def cleanup(self):
        """Remove hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []


# =============================================================================
# AF SAFETY FILTER
# =============================================================================

class AFSafetyFilter:
    """
    Main AF safety filter class.
    
    Usage:
        filter = AFSafetyFilter(
            model=model,
            tokenizer=tokenizer,
            vectors_path="vectors/gemma-3-1b.safetensors",
            policy_path="policies/af_policy.json",
        )
        
        result = filter.check(prompt)
        if result.blocked:
            print(f"Blocked: {result.triggered_categories}")
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        vectors_path: Union[str, Path],
        policy_path: Union[str, Path],
        chat_template: str = "gemma",
    ):
        # Load vectors
        self.vectors = VectorStore(vectors_path)
        
        # Load policy
        with open(policy_path, 'r') as f:
            policy_data = json.load(f)
        self.policy = PolicyConfig.from_dict(policy_data)
        
        # Determine extraction layer
        model_name = self.vectors.model_name
        if model_name in self.policy.model_overrides:
            layer_pct = self.policy.model_overrides[model_name].get(
                'extraction_layer_pct', 
                self.policy.default_extraction_layer_pct
            )
        else:
            layer_pct = self.vectors.extraction_layer_pct or self.policy.default_extraction_layer_pct
        
        # Initialize extractor
        self.extractor = ActivationExtractor(model, tokenizer, layer_pct=layer_pct)
        self.chat_template = chat_template
        
        # Compute high-confidence threshold for obfuscation if needed
        self._obf_high_confidence_threshold = None
        if 'obfuscation' in self.vectors.thresholds:
            # High confidence = threshold + 1 std (approximated)
            # In practice this should be stored in the vectors file
            base_threshold = self.vectors.thresholds['obfuscation']
            self._obf_high_confidence_threshold = self.vectors.metadata.get(
                'obfuscation_high_confidence_threshold',
                base_threshold * 1.5  # Fallback approximation
            )
    
    def check(self, prompt: str) -> ClassificationResult:
        """
        Check if a prompt should be blocked.
        
        Returns ClassificationResult with:
        - blocked: True if prompt should be blocked
        - triggered_categories: List of categories that triggered
        - scores: Dict of all category scores
        - latency_ms: Processing time
        """
        start = time.perf_counter()
        
        # Extract activation
        activation = self.extractor.extract(prompt, self.chat_template)
        
        scores = {}
        triggered = []
        
        # Check obfuscation
        obf_triggered = False
        obf_high_confidence = False
        
        if self.policy.obfuscation_enabled and self.vectors.has_obfuscation:
            obf_vector = self.vectors.vectors['obfuscation']
            obf_threshold = self.vectors.thresholds['obfuscation']
            
            obf_score = float(activation @ obf_vector)
            scores['obfuscation'] = obf_score
            
            obf_triggered = obf_score > obf_threshold
            if self._obf_high_confidence_threshold is not None:
                obf_high_confidence = obf_score > self._obf_high_confidence_threshold
        
        # Check categories
        category_triggered = []
        
        for category in self.vectors.categories:
            # Check if category is enabled in policy
            cat_config = self.policy.categories.get(category)
            if cat_config and not cat_config.enabled:
                continue
            
            vector = self.vectors.vectors[category]
            threshold = self.vectors.thresholds[category]
            
            # Apply threshold override from policy if set
            if cat_config and cat_config.threshold > 0:
                threshold = cat_config.threshold
            
            score = float(activation @ vector)
            scores[category] = score
            
            if score > threshold:
                category_triggered.append(category)
        
        # Decision logic
        if self.policy.obfuscation_require_confirmation:
            if obf_triggered:
                if len(category_triggered) > 0:
                    triggered.append('obfuscation')
                    triggered.extend(category_triggered)
                elif obf_high_confidence and self.policy.obfuscation_high_confidence_enabled:
                    triggered.append('obfuscation_high_confidence')
            elif len(category_triggered) > 0:
                triggered.extend(category_triggered)
        else:
            if obf_triggered:
                triggered.append('obfuscation')
            triggered.extend(category_triggered)
        
        latency = (time.perf_counter() - start) * 1000
        
        # Determine if blocked based on action type
        blocked = False
        for cat in triggered:
            cat_name = cat.replace('_high_confidence', '')
            cat_config = self.policy.categories.get(cat_name)
            if cat_config is None or cat_config.action == 'block':
                blocked = True
                break
        
        # If obfuscation triggered, always block
        if 'obfuscation' in triggered or 'obfuscation_high_confidence' in triggered:
            blocked = True
        
        return ClassificationResult(
            blocked=blocked,
            triggered_categories=triggered,
            scores=scores,
            latency_ms=latency,
        )
    
    def is_safe(self, prompt: str) -> Tuple[bool, List[str]]:
        """
        Simple interface - returns (is_safe, triggered_categories).
        
        Usage:
            safe, triggered = filter.is_safe(prompt)
            if not safe:
                print(f"Blocked by: {triggered}")
        """
        result = self.check(prompt)
        return not result.blocked, result.triggered_categories
    
    def cleanup(self):
        """Clean up resources."""
        self.extractor.cleanup()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_filter(
    model,
    tokenizer,
    model_name: str,
    vectors_dir: Union[str, Path] = "vectors",
    policy_path: Union[str, Path] = "policies/af_policy.json",
    chat_template: str = "gemma",
) -> AFSafetyFilter:
    """
    Convenience function to load AF filter with automatic vector selection.
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer
        model_name: Name of the model (e.g., "gemma-3-1b")
        vectors_dir: Directory containing vector files
        policy_path: Path to policy JSON file
        chat_template: Chat template format ("gemma", "llama", "chatml")
    
    Returns:
        AFSafetyFilter instance
    """
    vectors_dir = Path(vectors_dir)
    
    # Try to find matching vector file
    for suffix in ['.safetensors', '.npz']:
        vectors_path = vectors_dir / f"{model_name}{suffix}"
        if vectors_path.exists():
            return AFSafetyFilter(
                model=model,
                tokenizer=tokenizer,
                vectors_path=vectors_path,
                policy_path=policy_path,
                chat_template=chat_template,
            )
    
    raise FileNotFoundError(f"No vector file found for model: {model_name}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AF Safety Filter")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--vectors", type=str, required=True, help="Path to vectors file")
    parser.add_argument("--policy", type=str, required=True, help="Path to policy file")
    parser.add_argument("--prompt", type=str, help="Single prompt to check")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    print("Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    print("Loading AF filter...")
    af_filter = AFSafetyFilter(
        model=model,
        tokenizer=tokenizer,
        vectors_path=args.vectors,
        policy_path=args.policy,
    )
    
    if args.prompt:
        result = af_filter.check(args.prompt)
        print(f"\nPrompt: {args.prompt}")
        print(f"Blocked: {result.blocked}")
        print(f"Triggered: {result.triggered_categories}")
        print(f"Latency: {result.latency_ms:.1f}ms")
    
    elif args.interactive:
        print("\nInteractive mode. Type 'quit' to exit.\n")
        while True:
            prompt = input("Prompt: ").strip()
            if prompt.lower() == 'quit':
                break
            
            result = af_filter.check(prompt)
            status = "🚫 BLOCKED" if result.blocked else "✅ SAFE"
            print(f"  {status} | Triggered: {result.triggered_categories} | {result.latency_ms:.1f}ms\n")
    
    af_filter.cleanup()
