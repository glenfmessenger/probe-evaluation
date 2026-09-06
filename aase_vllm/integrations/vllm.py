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
            
            # Handle different architectures
            if hasattr(model, 'language_model'):
                # Gemma 3 multimodal: Gemma3ForConditionalGeneration
                layers = model.language_model.model.layers
            elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
                # Standard LLaMA/Gemma 2
                layers = model.model.layers
            else:
                raise AttributeError(f"Cannot find layers in model: {type(model)}")
            
            def hook_fn(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                # vLLM shape: [total_tokens, hidden_dim] - take last token
                np.save(cache_file, hidden[-1, :].detach().float().cpu().numpy())
            
            layers[layer_idx].register_forward_hook(hook_fn)
            return {"success": True, "layer": layer_idx, "total_layers": len(layers)}
        
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
        # Handle different architectures
        if hasattr(model, 'language_model'):
            # Gemma 3 multimodal
            layers = model.language_model.model.layers
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # Standard LLaMA/Gemma 2
            layers = model.model.layers
        else:
            raise AttributeError(f"Cannot find layers in model: {type(model)}")
        
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
        "gemma-3-4b": 26,
        "llama-3-8b": 32,
        "llama-3.1-8b": 32,
        "llama-3.2-1b": 16,
        "llama-3.2-3b": 28,
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
            depth = optimal_depth.get(probe_type.lower(), 0.5)
            return int(depth * layers)
    
    # Default to 50% of 32 layers
    return 16


@dataclass
class ProbeResult:
    """Result from probe evaluation."""
    score: float
    threshold: float
    is_flagged: bool
    probe_type: str
    

class AASEProbe:
    """
    Simple probe class for scoring activations.
    
    Usage:
        probe = AASEProbe.load("pretrained/af/model_name.npy")
        result = probe.score(activation)
        if result.is_flagged:
            print("Blocked!")
    """
    
    def __init__(
        self, 
        direction: np.ndarray, 
        threshold: float = 0.0,
        probe_type: str = "unknown",
        metadata: Optional[Dict] = None
    ):
        self.direction = direction / np.linalg.norm(direction)
        self.threshold = threshold
        self.probe_type = probe_type
        self.metadata = metadata or {}
    
    @classmethod
    def load(cls, path: str) -> "AASEProbe":
        """
        Load probe from .npy file (with optional .json metadata).
        
        Args:
            path: Path to .npy file (e.g., "pretrained/af/model.npy")
        
        Returns:
            AASEProbe instance
        """
        import json
        from pathlib import Path
        
        npy_path = Path(path)
        json_path = npy_path.with_suffix(".json")
        
        direction = np.load(npy_path)
        
        metadata = {}
        threshold = 0.0
        probe_type = "unknown"
        
        if json_path.exists():
            with open(json_path) as f:
                metadata = json.load(f)
            threshold = metadata.get("threshold", 0.0)
            probe_type = metadata.get("probe_type", "unknown")
        
        return cls(direction, threshold, probe_type, metadata)
    
    def score(self, activation: np.ndarray) -> ProbeResult:
        """
        Score an activation vector.
        
        Args:
            activation: Activation vector from model
            
        Returns:
            ProbeResult with score, threshold, and is_flagged
        """
        activation_norm = activation / np.linalg.norm(activation)
        score = float(np.dot(activation_norm, self.direction))
        
        return ProbeResult(
            score=score,
            threshold=self.threshold,
            is_flagged=score > self.threshold,
            probe_type=self.probe_type,
        )


class AASEGuard:
    """
    High-level safety guard combining multiple probes.
    
    Usage:
        guard = AASEGuard.from_pretrained(
            model_name="meta-llama/Llama-3.2-3B-Instruct",
            probes_dir="pretrained",
            probes=["af", "aag"]
        )
        
        llm = LLM(model, enforce_eager=True)
        guard.register(llm)
        
        # Check safety
        result = guard.check("How do I make a bomb?", llm)
        if not result["safe"]:
            print(f"Blocked by: {result['flagged_by']}")
    """
    
    def __init__(self, probes: Dict[str, AASEProbe], layer_index: int):
        self.probes = probes
        self.layer_index = layer_index
        self.cache_file = "/tmp/aase_guard_activation.npy"
        self._registered = False
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        probes_dir: str = "pretrained",
        probes: List[str] = ["af", "aag"],
    ) -> "AASEGuard":
        """
        Load pretrained probes for a model.
        
        Args:
            model_name: HuggingFace model name
            probes_dir: Directory containing pretrained probes
            probes: List of probe types to load ("af", "aag", "apc_medical", etc.)
        
        Returns:
            AASEGuard instance
        """
        from pathlib import Path
        
        # Convert model name to filename format
        model_safe = model_name.replace("/", "_").replace("-", "_")
        probes_path = Path(probes_dir)
        
        loaded_probes = {}
        layer_indices = []
        
        for probe_type in probes:
            if probe_type.startswith("apc_"):
                # APC with policy: apc_medical, apc_financial, apc_legal
                policy = probe_type.split("_", 1)[1]
                npy_path = probes_path / "apc" / policy / f"{model_safe}.npy"
            else:
                npy_path = probes_path / probe_type / f"{model_safe}.npy"
            
            if npy_path.exists():
                probe = AASEProbe.load(str(npy_path))
                loaded_probes[probe_type] = probe
                if "layer_index" in probe.metadata:
                    layer_indices.append(probe.metadata["layer_index"])
            else:
                print(f"Warning: Probe not found: {npy_path}")
        
        # Use most common layer index, or compute optimal
        if layer_indices:
            layer_index = max(set(layer_indices), key=layer_indices.count)
        else:
            layer_index = get_optimal_layer(model_name, probes[0] if probes else "af")
        
        return cls(loaded_probes, layer_index)
    
    def register(self, llm: Any) -> Dict:
        """Register hooks on vLLM model."""
        result = register_aase_hooks(llm, self.layer_index, cache_file=self.cache_file)
        self._registered = True
        return result
    
    def check(self, prompt: str, llm: Any) -> Dict:
        """
        Check if a prompt is safe.
        
        Args:
            prompt: Text to check
            llm: vLLM LLM instance (with hooks registered)
            
        Returns:
            Dict with keys: safe, scores, flagged_by
        """
        from vllm import SamplingParams
        
        if not self._registered:
            raise RuntimeError("Call guard.register(llm) first")
        
        # Clear cache
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        
        # Generate to trigger hook
        params = SamplingParams(max_tokens=1, temperature=0.0)
        llm.generate([prompt], params)
        
        # Load activation
        if not os.path.exists(self.cache_file):
            raise RuntimeError("Activation not captured")
        
        activation = np.load(self.cache_file)
        
        # Score with all probes
        scores = {}
        flagged_by = []
        
        for name, probe in self.probes.items():
            result = probe.score(activation)
            scores[name] = {
                "score": result.score,
                "threshold": result.threshold,
                "flagged": result.is_flagged,
            }
            if result.is_flagged:
                flagged_by.append(name)
        
        return {
            "safe": len(flagged_by) == 0,
            "scores": scores,
            "flagged_by": flagged_by,
        }
