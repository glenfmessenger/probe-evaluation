"""
vLLM Integration for AASE

This module provides vLLM-specific utilities for activation extraction
and safety probe integration.

Key vLLM considerations:
- V1 engine runs model in separate process
- Uses apply_model() to register hooks
- Requires VLLM_ALLOW_INSECURE_SERIALIZATION=1
- Tensor shape is [total_tokens, hidden_dim] (2D)
- Must use .detach().float().cpu().numpy() for bf16 → float32
- enforce_eager=True required for hooks

Usage:
    from aase.integrations.vllm import VLLMSafetyWrapper
    
    llm = vllm.LLM(model, enforce_eager=True)
    wrapper = VLLMSafetyWrapper(llm, layer_index=18)
    
    # Safe generation
    result = wrapper.safe_generate(prompts, safety_stack)
"""

import os
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

# Ensure serialization is enabled
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


@dataclass
class SafeGenerationResult:
    """Result from safe generation."""
    outputs: List[Any]  # vLLM RequestOutput objects
    safety_results: List["StackResult"]
    blocked_indices: List[int]
    

class VLLMSafetyWrapper:
    """
    Wrapper for vLLM that adds AASE safety checking.
    
    Provides:
    - Automatic activation extraction via hooks
    - Pre-generation safety checking
    - Post-generation safety checking
    - Configurable blocking behavior
    """
    
    def __init__(
        self,
        llm: Any,  # vllm.LLM
        layer_index: int,
        cache_file: str = "/tmp/aase_vllm_activation.npy",
    ):
        """
        Initialize wrapper.
        
        Args:
            llm: vLLM LLM instance (must use enforce_eager=True)
            layer_index: Layer to extract activations from
            cache_file: File for cross-process activation transfer
        """
        self.llm = llm
        self.layer_index = layer_index
        self.cache_file = cache_file
        self._hook_registered = False
        
        # Register hook
        self._register_hook()
    
    def _register_hook(self) -> None:
        """Register activation extraction hook."""
        layer_idx = self.layer_index
        cache_file = self.cache_file
        
        def register_hook(model):
            """Runs inside vLLM worker process."""
            import numpy as np
            layers = model.model.layers
            
            def hook_fn(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                # vLLM shape: [total_tokens, hidden_dim] - take last token
                np.save(cache_file, hidden[-1, :].detach().float().cpu().numpy())
            
            layers[layer_idx].register_forward_hook(hook_fn)
            return {"success": True}
        
        result = self.llm.apply_model(register_hook)
        self._hook_registered = True
    
    def extract_activation(self, text: str, sampling_params: Any = None) -> np.ndarray:
        """
        Extract activation for a single text.
        
        Args:
            text: Input text
            sampling_params: vLLM SamplingParams (default: 1 token)
            
        Returns:
            Activation vector [hidden_dim]
        """
        from vllm import SamplingParams
        
        if sampling_params is None:
            sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
        
        # Clear previous
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        
        # Generate (triggers hook)
        self.llm.generate([text], sampling_params)
        
        # Load activation
        if os.path.exists(self.cache_file):
            return np.load(self.cache_file)
        else:
            raise RuntimeError(
                "Activation not captured. Ensure model was loaded with enforce_eager=True"
            )
    
    def safe_generate(
        self,
        prompts: List[str],
        sampling_params: Any,
        safety_stack: "SafetyStack",
        block_unsafe: bool = True,
        unsafe_response: str = "I cannot help with that request.",
    ) -> SafeGenerationResult:
        """
        Generate with safety checking.
        
        Args:
            prompts: List of prompts to generate from
            sampling_params: vLLM SamplingParams
            safety_stack: SafetyStack for evaluation
            block_unsafe: If True, block unsafe prompts
            unsafe_response: Response for blocked prompts
            
        Returns:
            SafeGenerationResult with outputs and safety info
        """
        from vllm import SamplingParams
        
        # First pass: extract activations and check safety
        safety_results = []
        blocked_indices = []
        
        check_params = SamplingParams(max_tokens=1, temperature=0.0)
        
        for i, prompt in enumerate(prompts):
            activation = self.extract_activation(prompt, check_params)
            result = safety_stack.evaluate(activation)
            safety_results.append(result)
            
            if not result.overall_safe:
                blocked_indices.append(i)
        
        # Second pass: generate for safe prompts
        if block_unsafe:
            safe_prompts = [p for i, p in enumerate(prompts) if i not in blocked_indices]
            if safe_prompts:
                safe_outputs = self.llm.generate(safe_prompts, sampling_params)
            else:
                safe_outputs = []
            
            # Reconstruct full output list
            outputs = []
            safe_idx = 0
            for i in range(len(prompts)):
                if i in blocked_indices:
                    # Create placeholder for blocked
                    outputs.append(self._create_blocked_output(prompts[i], unsafe_response))
                else:
                    outputs.append(safe_outputs[safe_idx])
                    safe_idx += 1
        else:
            # Generate all, just flag unsafe
            outputs = self.llm.generate(prompts, sampling_params)
        
        return SafeGenerationResult(
            outputs=outputs,
            safety_results=safety_results,
            blocked_indices=blocked_indices,
        )
    
    def _create_blocked_output(self, prompt: str, response: str) -> Any:
        """Create a placeholder output for blocked prompts."""
        # Return a simple dict instead of vLLM objects
        return {
            "prompt": prompt,
            "outputs": [{"text": response}],
            "blocked": True,
        }
    
    @property
    def model_config(self) -> Any:
        """Access model config."""
        return self.llm.llm_engine.model_config
    
    @property
    def hidden_dim(self) -> int:
        """Get hidden dimension."""
        return self.model_config.hf_config.hidden_size
    
    @property
    def num_layers(self) -> int:
        """Get number of layers."""
        return self.model_config.hf_config.num_hidden_layers


def create_vllm_extractor(
    model_name: str,
    layer_index: int,
    gpu_memory_utilization: float = 0.8,
    max_model_len: int = 2048,
    **kwargs
) -> Tuple[Any, "VLLMExtractor"]:
    """
    Convenience function to create vLLM model with extractor.
    
    Args:
        model_name: HuggingFace model name
        layer_index: Layer to extract from
        **kwargs: Additional vLLM.LLM arguments
        
    Returns:
        Tuple of (LLM, VLLMExtractor)
    """
    from vllm import LLM
    from aase.core.extraction import VLLMExtractor, ExtractionConfig
    
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=True,  # Required for hooks
        **kwargs
    )
    
    config = ExtractionConfig(layer_index=layer_index)
    extractor = VLLMExtractor(llm, config)
    extractor.setup_hooks()
    
    return llm, extractor


def register_aase_hooks(
    llm: Any,
    layer_index: int,
    callback: callable = None,
    cache_file: str = "/tmp/aase_activation.npy",
) -> dict:
    """
    Register AASE activation extraction hooks on a vLLM model.
    
    This is the simplest way to enable AASE on any vLLM instance.
    
    Args:
        llm: vLLM LLM instance
        layer_index: Layer to extract activations from
        callback: Optional callback(activation) called on each forward pass
        cache_file: File to cache activations (for cross-process access)
    
    Returns:
        Dict with registration info
    
    Example:
        from vllm import LLM
        from aase.integrations.vllm import register_aase_hooks
        
        llm = LLM(model="google/gemma-2-2b-it", enforce_eager=True)
        info = register_aase_hooks(llm, layer_index=13)
        
        # Now activations are saved to /tmp/aase_activation.npy on each generate()
    """
    import numpy as np
    
    def register_hook(model):
        layers = model.model.layers
        
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            activation = hidden[-1, :].detach().float().cpu().numpy()
            np.save(cache_file, activation)
            if callback:
                callback(activation)
        
        layers[layer_index].register_forward_hook(hook_fn)
        return {
            "layer": layer_index,
            "total_layers": len(layers),
            "cache_file": cache_file,
        }
    
    return llm.apply_model(register_hook)


def get_optimal_layer(model_name: str, probe_type: str) -> int:
    """
    Get optimal layer index for a model and probe type.
    
    Based on empirical testing across models.
    """
    # Model layer counts
    layer_counts = {
        "gemma-2-2b": 26,
        "gemma-2-9b": 42,
        "gemma-3-1b": 26,
        "llama-3-8b": 32,
        "llama-3.1-8b": 32,
        "mistral-7b": 32,
    }
    
    # Optimal depth percentages by probe type
    optimal_depth = {
        "af": 0.50,   # 50% depth
        "aag": 0.55,  # 55% depth  
        "apc": 0.25,  # 25% depth (earlier layers)
    }
    
    # Normalize model name
    model_lower = model_name.lower()
    for key, layers in layer_counts.items():
        if key in model_lower:
            depth = optimal_depth.get(probe_type, 0.5)
            return int(depth * layers)
    
    # Default to 50% of 32 layers
    return 16
