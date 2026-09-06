"""
HuggingFace Transformers Integration for AASE

This module provides utilities for using AASE with standard
HuggingFace Transformers models.

Advantages of Transformers backend:
- Clean hidden state access via output_hidden_states=True
- No special hooks or serialization needed
- Easy debugging and inspection
- Great for training and development

Disadvantages:
- Slower than vLLM for production inference
- Higher memory usage
- No continuous batching

Recommended workflow:
1. Develop and train probes using Transformers
2. Validate on Transformers
3. Deploy with vLLM or llm-d
4. Recalibrate if needed
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import torch


class TransformersSafetyWrapper:
    """
    Safety wrapper for HuggingFace Transformers models.
    
    Usage:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model = AutoModelForCausalLM.from_pretrained("model_name")
        tokenizer = AutoTokenizer.from_pretrained("model_name")
        
        wrapper = TransformersSafetyWrapper(model, tokenizer, layer_index=16)
        
        # Check safety
        result = wrapper.check_safety("user prompt", safety_stack)
    """
    
    def __init__(
        self,
        model: Any,  # transformers.PreTrainedModel
        tokenizer: Any,  # transformers.PreTrainedTokenizer
        layer_index: int,
        device: str = "cuda",
    ):
        """
        Initialize wrapper.
        
        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            layer_index: Layer to extract activations from
            device: Device for inference
        """
        self.model = model
        self.tokenizer = tokenizer
        self.layer_index = layer_index
        self.device = device
        
        # Enable hidden state output
        self.model.config.output_hidden_states = True
        self.model.to(device)
        self.model.eval()
    
    def extract_activation(
        self,
        text: str,
        position: str = "last"
    ) -> np.ndarray:
        """
        Extract activation for text.
        
        Args:
            text: Input text
            position: Token position ("last", "first", "mean")
            
        Returns:
            Activation vector [hidden_dim]
        """
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get hidden states from specified layer (+1 for embedding layer)
        hidden_states = outputs.hidden_states[self.layer_index + 1]
        
        # Select position
        if position == "last":
            activation = hidden_states[0, -1, :]
        elif position == "first":
            activation = hidden_states[0, 0, :]
        elif position == "mean":
            activation = hidden_states[0].mean(dim=0)
        else:
            activation = hidden_states[0, int(position), :]
        
        return activation.float().cpu().numpy()
    
    def extract_batch(
        self,
        texts: List[str],
        position: str = "last"
    ) -> np.ndarray:
        """
        Extract activations for a batch of texts.
        
        Args:
            texts: List of input texts
            position: Token position
            
        Returns:
            Activations [batch, hidden_dim]
        """
        # Tokenize with padding
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get hidden states
        hidden_states = outputs.hidden_states[self.layer_index + 1]
        
        # Handle padded sequences
        attention_mask = inputs.attention_mask
        batch_size = hidden_states.shape[0]
        
        activations = []
        for i in range(batch_size):
            seq_len = attention_mask[i].sum().item()
            
            if position == "last":
                act = hidden_states[i, seq_len - 1, :]
            elif position == "first":
                act = hidden_states[i, 0, :]
            elif position == "mean":
                act = hidden_states[i, :seq_len, :].mean(dim=0)
            else:
                pos = min(int(position), seq_len - 1)
                act = hidden_states[i, pos, :]
            
            activations.append(act.float().cpu().numpy())
        
        return np.array(activations)
    
    def check_safety(
        self,
        text: str,
        safety_stack: "SafetyStack"
    ) -> "StackResult":
        """
        Check text safety using SafetyStack.
        
        Args:
            text: Input text
            safety_stack: SafetyStack for evaluation
            
        Returns:
            StackResult
        """
        activation = self.extract_activation(text)
        return safety_stack.evaluate(activation)
    
    def generate_safe(
        self,
        prompt: str,
        safety_stack: "SafetyStack",
        max_new_tokens: int = 100,
        block_unsafe: bool = True,
        **generate_kwargs
    ) -> Dict:
        """
        Generate with safety checking.
        
        Args:
            prompt: Input prompt
            safety_stack: SafetyStack
            max_new_tokens: Maximum tokens to generate
            block_unsafe: Block unsafe prompts
            **generate_kwargs: Additional generation arguments
            
        Returns:
            Dict with text and safety info
        """
        # Check safety first
        safety_result = self.check_safety(prompt, safety_stack)
        
        if not safety_result.overall_safe and block_unsafe:
            return {
                "text": "I cannot help with that request.",
                "blocked": True,
                "safety": safety_result.to_dict(),
            }
        
        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **generate_kwargs
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "text": generated_text,
            "blocked": False,
            "safety": safety_result.to_dict(),
        }
    
    @property
    def hidden_dim(self) -> int:
        return self.model.config.hidden_size
    
    @property
    def num_layers(self) -> int:
        return self.model.config.num_hidden_layers


def load_model_for_aase(
    model_name: str,
    layer_index: Optional[int] = None,
    layer_depth: float = 0.5,
    device: str = "cuda",
    torch_dtype: str = "auto",
    **kwargs
) -> TransformersSafetyWrapper:
    """
    Load a model configured for AASE.
    
    Args:
        model_name: HuggingFace model name
        layer_index: Specific layer (or use layer_depth)
        layer_depth: Layer depth as fraction (0.0-1.0)
        device: Device for inference
        torch_dtype: Torch dtype ("auto", "float16", "bfloat16")
        **kwargs: Additional model loading arguments
        
    Returns:
        Configured TransformersSafetyWrapper
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load model
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype_map.get(torch_dtype, "auto"),
        output_hidden_states=True,
        **kwargs
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine layer index
    num_layers = model.config.num_hidden_layers
    if layer_index is None:
        layer_index = int(layer_depth * num_layers)
    
    return TransformersSafetyWrapper(model, tokenizer, layer_index, device)


def train_probe_transformers(
    probe: "ActivationProbe",
    positive_texts: List[str],
    negative_texts: List[str],
    wrapper: TransformersSafetyWrapper,
    **train_kwargs
) -> "TrainingMetrics":
    """
    Train a probe using Transformers backend.
    
    Convenience function for training probes.
    
    Args:
        probe: Probe to train
        positive_texts: Positive class examples
        negative_texts: Negative class examples
        wrapper: TransformersSafetyWrapper
        **train_kwargs: Additional training arguments
        
    Returns:
        Training metrics
    """
    # Extract activations
    pos_acts = wrapper.extract_batch(positive_texts)
    neg_acts = wrapper.extract_batch(negative_texts)
    
    # Train probe
    return probe.train_from_activations(pos_acts, neg_acts)
