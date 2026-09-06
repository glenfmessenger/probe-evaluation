"""
Shared activation extraction hooks for different backends.

This module provides a unified interface for extracting hidden states
from different inference backends (vLLM, llm-d, HuggingFace Transformers).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import os


@dataclass
class ExtractionConfig:
    """Configuration for activation extraction."""
    layer_index: int
    position: str = "last"  # "last", "first", "mean", or int index
    normalize: bool = False
    dtype: str = "float32"
    cache_file: Optional[str] = None  # For cross-process communication


class ActivationExtractor(ABC):
    """
    Base class for extracting activations from LLM inference backends.
    
    Each backend (vLLM, llm-d, Transformers) has different APIs for
    accessing hidden states. This class provides a unified interface.
    """
    
    BACKEND_NAME: str = "base"
    
    def __init__(
        self,
        model: Any,
        config: ExtractionConfig,
        **kwargs
    ):
        """
        Initialize extractor.
        
        Args:
            model: The loaded model object
            config: Extraction configuration
            **kwargs: Backend-specific options
        """
        self.model = model
        self.config = config
        self._hooks: List[Any] = []
        self._last_activation: Optional[np.ndarray] = None
        
    @abstractmethod
    def setup_hooks(self) -> None:
        """Register forward hooks to capture activations."""
        pass
    
    @abstractmethod
    def remove_hooks(self) -> None:
        """Remove registered hooks."""
        pass
    
    @abstractmethod
    def extract(self, text: str) -> np.ndarray:
        """
        Extract activation for a single text input.
        
        Args:
            text: Input text to process
            
        Returns:
            Activation vector [hidden_dim]
        """
        pass
    
    @abstractmethod
    def extract_batch(self, texts: List[str]) -> np.ndarray:
        """
        Extract activations for a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Activations [batch, hidden_dim]
        """
        pass
    
    @property
    def hidden_dim(self) -> int:
        """Get the hidden dimension of the model."""
        raise NotImplementedError
    
    @property
    def num_layers(self) -> int:
        """Get the number of layers in the model."""
        raise NotImplementedError
    
    def _select_position(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Select activation at specified position from sequence.
        
        Args:
            hidden_states: Shape [seq_len, hidden_dim] or [batch, seq_len, hidden_dim]
            
        Returns:
            Selected activation(s)
        """
        position = self.config.position
        
        if isinstance(position, int):
            return hidden_states[..., position, :]
        elif position == "last":
            return hidden_states[..., -1, :]
        elif position == "first":
            return hidden_states[..., 0, :]
        elif position == "mean":
            return hidden_states.mean(axis=-2)
        else:
            raise ValueError(f"Unknown position: {position}")
    
    def __enter__(self):
        self.setup_hooks()
        return self
    
    def __exit__(self, *args):
        self.remove_hooks()


class VLLMExtractor(ActivationExtractor):
    """
    Activation extractor for vLLM inference.
    
    Key vLLM considerations:
    - V1 engine runs model in separate process
    - Uses apply_model() to register hooks
    - Activations saved to file for cross-process communication
    - Tensor shape is [total_tokens, hidden_dim] (2D, not batched)
    - Must use .detach().float().cpu().numpy() for bf16 → float32
    
    Usage:
        llm = vllm.LLM(model_name, enforce_eager=True)
        extractor = VLLMExtractor(llm, config)
        activation = extractor.extract("Hello world")
    """
    
    BACKEND_NAME = "vllm"
    
    def __init__(
        self,
        model: Any,  # vllm.LLM
        config: ExtractionConfig,
        sampling_params: Optional[Any] = None,  # vllm.SamplingParams
    ):
        super().__init__(model, config)
        
        # Set default cache file if not specified
        if self.config.cache_file is None:
            self.config.cache_file = f"/tmp/aase_vllm_activation_{id(self)}.npy"
        
        # Create sampling params for minimal generation
        if sampling_params is None:
            from vllm import SamplingParams
            self.sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
        else:
            self.sampling_params = sampling_params
    
    def setup_hooks(self) -> None:
        """Register forward hook via vLLM's apply_model."""
        layer_idx = self.config.layer_index
        cache_file = self.config.cache_file
        
        def register_hook(model):
            """Function to run inside vLLM worker process."""
            import numpy as np
            layers = model.model.layers
            
            def hook_fn(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                # vLLM shape: [total_tokens, hidden_dim] - take last token
                np.save(cache_file, hidden[-1, :].detach().float().cpu().numpy())
            
            layers[layer_idx].register_forward_hook(hook_fn)
            return {"success": True}
        
        self.model.apply_model(register_hook)
    
    def remove_hooks(self) -> None:
        """
        Remove hooks - currently not supported in vLLM.
        Hooks persist for lifetime of model.
        """
        # vLLM doesn't provide a way to remove hooks
        # Model must be reloaded to clear hooks
        pass
    
    def extract(self, text: str) -> np.ndarray:
        """Extract activation for a single text."""
        # Clear previous activation file
        if os.path.exists(self.config.cache_file):
            os.remove(self.config.cache_file)
        
        # Run inference (triggers hook)
        self.model.generate([text], self.sampling_params)
        
        # Load captured activation
        if os.path.exists(self.config.cache_file):
            self._last_activation = np.load(self.config.cache_file)
            return self._last_activation
        else:
            raise RuntimeError(
                "Activation file not created - hook may have failed. "
                "Ensure enforce_eager=True when loading vLLM model."
            )
    
    def extract_batch(self, texts: List[str]) -> np.ndarray:
        """
        Extract activations for batch - processes sequentially.
        
        Note: vLLM hook captures last token per inference call,
        so batched extraction processes one at a time.
        """
        activations = []
        for text in texts:
            act = self.extract(text)
            activations.append(act)
        return np.array(activations)
    
    @property
    def hidden_dim(self) -> int:
        """Get hidden dimension from model config."""
        return self.model.llm_engine.model_config.hf_config.hidden_size
    
    @property
    def num_layers(self) -> int:
        """Get number of layers from model config."""
        return self.model.llm_engine.model_config.hf_config.num_hidden_layers


class TransformersExtractor(ActivationExtractor):
    """
    Activation extractor for HuggingFace Transformers.
    
    Uses output_hidden_states=True for clean extraction.
    """
    
    BACKEND_NAME = "transformers"
    
    def __init__(
        self,
        model: Any,  # transformers.PreTrainedModel
        tokenizer: Any,  # transformers.PreTrainedTokenizer
        config: ExtractionConfig,
        device: str = "cuda",
    ):
        super().__init__(model, config)
        self.tokenizer = tokenizer
        self.device = device
        
        # Enable hidden state output
        self.model.config.output_hidden_states = True
    
    def setup_hooks(self) -> None:
        """No hooks needed - uses output_hidden_states."""
        pass
    
    def remove_hooks(self) -> None:
        """No hooks to remove."""
        pass
    
    def extract(self, text: str) -> np.ndarray:
        """Extract activation for a single text."""
        import torch
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get hidden states from specified layer
        hidden_states = outputs.hidden_states[self.config.layer_index + 1]  # +1 for embedding layer
        
        # Convert to numpy
        hidden_np = hidden_states.float().cpu().numpy()[0]  # Remove batch dim
        
        # Select position
        self._last_activation = self._select_position(hidden_np)
        return self._last_activation
    
    def extract_batch(self, texts: List[str]) -> np.ndarray:
        """Extract activations for a batch of texts."""
        import torch
        
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
        hidden_states = outputs.hidden_states[self.config.layer_index + 1]
        hidden_np = hidden_states.float().cpu().numpy()
        
        # For padded inputs, we need attention mask to find last real token
        attention_mask = inputs.attention_mask.cpu().numpy()
        
        # Extract last real token for each sequence
        batch_size = hidden_np.shape[0]
        activations = []
        for i in range(batch_size):
            seq_len = attention_mask[i].sum()
            if self.config.position == "last":
                activations.append(hidden_np[i, seq_len - 1, :])
            else:
                activations.append(self._select_position(hidden_np[i, :seq_len, :]))
        
        return np.array(activations)
    
    @property
    def hidden_dim(self) -> int:
        return self.model.config.hidden_size
    
    @property
    def num_layers(self) -> int:
        return self.model.config.num_hidden_layers


class LLMDExtractor(ActivationExtractor):
    """
    Activation extractor for llm-d (GKE inference stack).
    
    llm-d is based on vLLM but runs as a Kubernetes service.
    This extractor handles the specific APIs and considerations.
    
    KEY CONSIDERATIONS FOR llm-d:
    
    1. HIDDEN STATE ACCESS:
       - llm-d exposes hidden states via the /v1/completions endpoint
       - Requires model deployed with hidden_states=True in config
       - Returns hidden states in response JSON under "hidden_states" key
       
    2. DIRECTION VECTOR TRANSFERABILITY:
       - Direction vectors trained on HuggingFace Transformers may need
         recalibration on llm-d due to:
         * Different attention implementations
         * Quantization (INT4/INT8)
         * Different layer normalization
       - Recommend: Train fresh on llm-d or apply recalibration
       
    3. LATENCY:
       - Network overhead adds ~1-5ms per request
       - Batch requests where possible
       - Consider caching for repeated inputs
       
    4. DEPLOYMENT:
       - Requires custom llm-d image or sidecar for hidden state extraction
       - May need to modify llm-d serving config
    """
    
    BACKEND_NAME = "llm-d"
    
    def __init__(
        self,
        endpoint_url: str,
        config: ExtractionConfig,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        use_grpc: bool = False,
    ):
        """
        Initialize llm-d extractor.
        
        Args:
            endpoint_url: llm-d service URL (e.g., "http://llmd-service:8000")
            config: Extraction configuration
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            use_grpc: Use gRPC instead of REST (lower latency)
        """
        super().__init__(model=None, config=config)
        self.endpoint_url = endpoint_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.use_grpc = use_grpc
        self._model_info: Optional[Dict] = None
        
    def setup_hooks(self) -> None:
        """
        No local hooks - llm-d handles extraction server-side.
        
        Instead, we verify the endpoint is configured correctly.
        """
        import requests
        
        # Check endpoint health and hidden state support
        try:
            response = requests.get(
                f"{self.endpoint_url}/health",
                timeout=self.timeout
            )
            response.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"llm-d endpoint not healthy: {e}")
        
        # Get model info
        try:
            response = requests.get(
                f"{self.endpoint_url}/v1/models",
                headers=self._get_headers(),
                timeout=self.timeout
            )
            self._model_info = response.json()
        except Exception as e:
            print(f"Warning: Could not fetch model info: {e}")
    
    def remove_hooks(self) -> None:
        """No hooks to remove."""
        pass
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers including auth."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def extract(self, text: str) -> np.ndarray:
        """
        Extract activation via llm-d API.
        
        NOTE: This requires llm-d to be configured to return hidden states.
        The standard llm-d deployment does NOT expose hidden states.
        
        Options:
        1. Custom llm-d build with hidden state endpoint
        2. Sidecar that hooks into vLLM internals
        3. Custom model wrapper that returns hidden states
        """
        if self.use_grpc:
            return self._extract_grpc(text)
        else:
            return self._extract_rest(text)
    
    def _extract_rest(self, text: str) -> np.ndarray:
        """Extract via REST API."""
        import requests
        
        # Standard completion request with hidden state flag
        payload = {
            "prompt": text,
            "max_tokens": 1,
            "temperature": 0,
            "return_hidden_states": True,  # Custom extension
            "hidden_state_layer": self.config.layer_index,
        }
        
        response = requests.post(
            f"{self.endpoint_url}/v1/completions",
            headers=self._get_headers(),
            json=payload,
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            # Check if hidden states are not supported
            if "hidden_states" in response.text.lower():
                raise RuntimeError(
                    "llm-d endpoint does not support hidden state extraction. "
                    "See LLMDExtractor docstring for options."
                )
            raise RuntimeError(f"llm-d request failed: {response.text}")
        
        result = response.json()
        
        # Extract hidden states from response
        if "hidden_states" not in result:
            raise RuntimeError(
                "Response does not contain hidden_states. "
                "Ensure llm-d is configured with return_hidden_states=True"
            )
        
        hidden_states = np.array(result["hidden_states"])
        self._last_activation = hidden_states[-1]  # Last token
        return self._last_activation
    
    def _extract_grpc(self, text: str) -> np.ndarray:
        """Extract via gRPC (lower latency)."""
        raise NotImplementedError(
            "gRPC extraction not yet implemented. Use REST API."
        )
    
    def extract_batch(self, texts: List[str]) -> np.ndarray:
        """Extract activations for batch."""
        # llm-d can batch internally, but we process sequentially for now
        activations = []
        for text in texts:
            act = self.extract(text)
            activations.append(act)
        return np.array(activations)
    
    @property
    def hidden_dim(self) -> int:
        if self._model_info:
            return self._model_info.get("hidden_size", 2048)
        return 2048  # Common default
    
    @property  
    def num_layers(self) -> int:
        if self._model_info:
            return self._model_info.get("num_layers", 32)
        return 32  # Common default


# Factory function for creating extractors
def create_extractor(
    backend: str,
    model: Any,
    layer_index: int,
    **kwargs
) -> ActivationExtractor:
    """
    Factory function to create appropriate extractor.
    
    Args:
        backend: "vllm", "transformers", or "llm-d"
        model: Model object (or endpoint URL for llm-d)
        layer_index: Layer to extract from
        **kwargs: Backend-specific options
        
    Returns:
        Configured ActivationExtractor
    """
    config = ExtractionConfig(
        layer_index=layer_index,
        position=kwargs.pop("position", "last"),
        normalize=kwargs.pop("normalize", False),
    )
    
    if backend == "vllm":
        return VLLMExtractor(model, config, **kwargs)
    elif backend == "transformers":
        tokenizer = kwargs.pop("tokenizer")
        return TransformersExtractor(model, tokenizer, config, **kwargs)
    elif backend in ("llm-d", "llmd"):
        return LLMDExtractor(model, config, **kwargs)  # model is URL
    else:
        raise ValueError(f"Unknown backend: {backend}")
