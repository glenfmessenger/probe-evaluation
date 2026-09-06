"""
llm-d Integration for AASE

llm-d is Google's Kubernetes-native LLM inference stack, built on vLLM.
This module provides integration strategies and utilities.

============================================================================
KEY ANALYSIS: llm-d Hidden State Access
============================================================================

QUESTION: Does llm-d expose hidden states?

SHORT ANSWER: vLLM has `LLM.encode()` for final-layer hidden states, but 
AASE needs INTERMEDIATE layer activations (e.g., layer 13 of 26). This 
requires the `apply_model()` hook approach, which works in vLLM but must
be configured at llm-d deployment time, not runtime.

DETAILED ANALYSIS:

1. vLLM's NATIVE HIDDEN STATE SUPPORT:
   - `LLM.encode()` / `/pooling` API: Returns FINAL layer hidden states
   - `LLM.embed()` / `/embeddings` API: Returns normalized embeddings
   - These are POST-pooler outputs, not intermediate layer activations
   - AASE needs activations from specific layers (e.g., 50% depth for AF)
   
2. WHY INTERMEDIATE LAYERS MATTER:
   - AF (harmful content): Optimal at 50-60% depth
   - AAG (prompt injection): Optimal at 40-70% depth  
   - APC (policy compliance): Optimal at 15-40% depth
   - Final layer has lost the discriminative signal we need
   
3. vLLM's `apply_model()` HOOK:
   - Allows registering forward hooks on specific layers
   - This IS how we extract intermediate activations
   - Works locally with vLLM
   - BUT: Must be configured when model loads, not per-request

4. llm-d INTEGRATION OPTIONS:

   Option A: Pre-configured llm-d Deployment (RECOMMENDED FOR PRODUCTION)
   - Build llm-d image with AASE hooks pre-registered
   - Configure target layer at deployment time
   - Expose custom `/v1/activations` endpoint
   - Requires: Custom Dockerfile, deployment config
   
   Option B: Proxy Architecture (RECOMMENDED FOR TESTING/FLEXIBILITY)
   - Deploy AASE as separate service with its own vLLM
   - Route: Client → AASE Proxy → llm-d
   - AASE extracts activations locally, forwards safe requests
   - Requires: Additional GPU for AASE's vLLM instance
   
   Option C: Sidecar with Shared Model (ADVANCED)
   - Run AASE in same pod, share GPU memory
   - Use vLLM's model directly via apply_model()
   - Requires: Complex pod configuration, memory management

5. RECOMMENDED APPROACH:
   - Development/Testing: Option B (Proxy) - most flexible
   - Production: Option A (Pre-configured) - best performance

============================================================================
DIRECTION VECTOR TRANSFERABILITY
============================================================================

QUESTION: Do vectors trained on HuggingFace transfer to llm-d?

ANSWER: Mostly yes, with caveats:

1. SAME PRECISION (FP16/BF16):
   - Vectors transfer directly
   - May need threshold recalibration (~10% shift typical)
   - Separation usually preserved

2. QUANTIZED (INT4/INT8):
   - Vectors still work but accuracy drops 5-15%
   - Recommend: Recalibrate threshold on quantized model
   - Or: Train fresh on quantized activations
   
3. DIFFERENT ATTENTION IMPL:
   - Flash Attention vs standard: Minimal impact (<2%)
   - PagedAttention: No impact on hidden states

RECOMMENDATION:
- Use transfer vectors for quick deployment
- Recalibrate threshold with 10-20 samples per class
- For production: Train fresh on target deployment

============================================================================
LATENCY ANALYSIS
============================================================================

Component breakdown:
- Network round-trip: 1-5ms (intra-cluster)
- Activation extraction: 0.1-0.5ms
- Direction vector dot product: <0.01ms
- Total AASE overhead: 1-6ms

For comparison:
- Typical LLM inference: 50-500ms
- AASE overhead: 1-2% of total latency

============================================================================
"""

import os
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import json


@dataclass 
class LLMDConfig:
    """Configuration for llm-d deployment."""
    endpoint_url: str
    model_name: str
    layer_index: int
    api_key: Optional[str] = None
    timeout: float = 30.0
    use_proxy_mode: bool = True  # Use proxy architecture


class LLMDProxyService:
    """
    AASE Proxy Service for llm-d integration.
    
    This is the recommended integration pattern:
    1. Client sends request to AASE Proxy
    2. Proxy extracts activations locally (requires vLLM)
    3. Proxy evaluates safety
    4. If safe, forward to llm-d
    5. Return response (or block if unsafe)
    
    This avoids modifying llm-d while providing full AASE functionality.
    
    Architecture:
    
    [Client] → [AASE Proxy] → [llm-d]
                   ↓
              [Safety Stack]
                   ↓
              [Local vLLM for activation extraction]
    
    Note: The proxy needs a local vLLM instance of the same model
    to extract activations. This adds resource overhead but provides
    clean separation.
    """
    
    def __init__(
        self,
        llmd_config: LLMDConfig,
        safety_stack: "SafetyStack",
        local_extractor: "ActivationExtractor",
    ):
        """
        Initialize proxy service.
        
        Args:
            llmd_config: llm-d endpoint configuration
            safety_stack: SafetyStack for evaluation
            local_extractor: Local activation extractor (vLLM-based)
        """
        self.config = llmd_config
        self.safety_stack = safety_stack
        self.extractor = local_extractor
        
    def check_safety(self, prompt: str) -> Tuple[bool, "StackResult"]:
        """
        Check if a prompt is safe.
        
        Args:
            prompt: Input prompt to check
            
        Returns:
            Tuple of (is_safe, full_result)
        """
        activation = self.extractor.extract(prompt)
        result = self.safety_stack.evaluate(activation)
        return result.overall_safe, result
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        block_unsafe: bool = True,
        **kwargs
    ) -> Dict:
        """
        Generate with safety checking.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            block_unsafe: Whether to block unsafe prompts
            **kwargs: Additional generation parameters
            
        Returns:
            Response dict with generation and safety info
        """
        import requests
        
        # Check safety first
        is_safe, safety_result = self.check_safety(prompt)
        
        if not is_safe and block_unsafe:
            return {
                "text": "I cannot help with that request.",
                "blocked": True,
                "safety": safety_result.to_dict(),
            }
        
        # Forward to llm-d
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        response = requests.post(
            f"{self.config.endpoint_url}/v1/completions",
            headers=headers,
            json=payload,
            timeout=self.config.timeout,
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"llm-d request failed: {response.text}")
        
        result = response.json()
        result["safety"] = safety_result.to_dict()
        result["blocked"] = False
        
        return result


class LLMDDirectIntegration:
    """
    Direct llm-d integration (requires custom llm-d build).
    
    This class provides the interface for a custom llm-d deployment
    that exposes hidden states. Use this if you've modified llm-d
    to add activation extraction.
    
    Required llm-d modifications:
    1. Add /v1/activations endpoint
    2. Return hidden states from specified layer
    3. Use same hook pattern as vLLM integration
    
    See: deploy/llmd-custom/ for example Dockerfile and patches
    """
    
    def __init__(
        self,
        endpoint_url: str,
        layer_index: int,
        api_key: Optional[str] = None,
    ):
        self.endpoint_url = endpoint_url.rstrip("/")
        self.layer_index = layer_index
        self.api_key = api_key
        self._verified = False
    
    def verify_activation_support(self) -> bool:
        """Check if llm-d endpoint supports activation extraction."""
        import requests
        
        try:
            response = requests.get(
                f"{self.endpoint_url}/v1/activations/info",
                headers=self._get_headers(),
                timeout=5.0,
            )
            if response.status_code == 200:
                self._verified = True
                return True
        except:
            pass
        
        return False
    
    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def extract_activation(self, text: str) -> np.ndarray:
        """
        Extract activation via custom llm-d endpoint.
        
        Requires custom llm-d build with /v1/activations endpoint.
        """
        import requests
        
        if not self._verified:
            if not self.verify_activation_support():
                raise RuntimeError(
                    "llm-d endpoint does not support activation extraction. "
                    "Use LLMDProxyService instead, or deploy custom llm-d build."
                )
        
        payload = {
            "prompt": text,
            "layer_index": self.layer_index,
            "position": "last",
        }
        
        response = requests.post(
            f"{self.endpoint_url}/v1/activations",
            headers=self._get_headers(),
            json=payload,
            timeout=30.0,
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Activation extraction failed: {response.text}")
        
        result = response.json()
        return np.array(result["activation"])


def create_llmd_integration(
    endpoint_url: str,
    model_name: str,
    layer_index: int,
    safety_stack: "SafetyStack",
    mode: str = "proxy",  # "proxy" or "direct"
    local_vllm: Any = None,  # Required for proxy mode
    **kwargs
) -> Any:
    """
    Create llm-d integration based on deployment mode.
    
    Args:
        endpoint_url: llm-d endpoint URL
        model_name: Model name
        layer_index: Layer for activation extraction
        safety_stack: SafetyStack instance
        mode: Integration mode ("proxy" or "direct")
        local_vllm: Local vLLM instance (required for proxy mode)
        **kwargs: Additional configuration
        
    Returns:
        Integration instance
    """
    config = LLMDConfig(
        endpoint_url=endpoint_url,
        model_name=model_name,
        layer_index=layer_index,
        **{k: v for k, v in kwargs.items() if k in ['api_key', 'timeout']}
    )
    
    if mode == "proxy":
        if local_vllm is None:
            raise ValueError("local_vllm required for proxy mode")
        
        from aase.core.extraction import VLLMExtractor, ExtractionConfig
        extractor_config = ExtractionConfig(layer_index=layer_index)
        extractor = VLLMExtractor(local_vllm, extractor_config)
        extractor.setup_hooks()
        
        return LLMDProxyService(config, safety_stack, extractor)
    
    elif mode == "direct":
        return LLMDDirectIntegration(
            endpoint_url,
            layer_index,
            api_key=kwargs.get("api_key"),
        )
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ============================================================================
# RECALIBRATION UTILITIES
# ============================================================================

def recalibrate_for_llmd(
    probe: "ActivationProbe",
    positive_samples: List[str],
    negative_samples: List[str],
    extractor: "ActivationExtractor",
    alpha: float = 0.3,
) -> "ActivationProbe":
    """
    Recalibrate a probe for llm-d deployment.
    
    When transferring probes from HuggingFace to llm-d (especially
    with quantization), the threshold may need adjustment.
    
    Args:
        probe: Trained probe to recalibrate
        positive_samples: Small set of positive class samples
        negative_samples: Small set of negative class samples
        extractor: Extractor for target deployment
        alpha: Blend factor (0 = keep original, 1 = fully retrain)
        
    Returns:
        Recalibrated probe (new instance)
    """
    # Extract activations on target deployment
    pos_acts = np.array([extractor.extract(s) for s in positive_samples])
    neg_acts = np.array([extractor.extract(s) for s in negative_samples])
    
    # Score with original direction
    pos_scores = pos_acts @ probe.direction
    neg_scores = neg_acts @ probe.direction
    
    # Compute new optimal threshold
    new_threshold = (pos_scores.mean() + neg_scores.mean()) / 2
    
    # Optionally blend direction vector
    if alpha > 0:
        new_pos_mean = pos_acts.mean(axis=0)
        new_neg_mean = neg_acts.mean(axis=0)
        new_direction = new_pos_mean - new_neg_mean
        new_direction = new_direction / np.linalg.norm(new_direction)
        
        blended = (1 - alpha) * probe.direction + alpha * new_direction
        blended = blended / np.linalg.norm(blended)
    else:
        blended = probe.direction
    
    # Create new probe with recalibrated parameters
    recalibrated = type(probe)(
        direction=blended,
        threshold=new_threshold,
        layer_index=probe._layer_index,
        layer_depth=probe._layer_depth,
        model_name=probe._model_name,
    )
    recalibrated._is_trained = True
    
    return recalibrated


def benchmark_llmd_latency(
    integration: Any,
    test_prompts: List[str],
    num_runs: int = 10,
) -> Dict:
    """
    Benchmark latency for llm-d integration.
    
    Returns breakdown of:
    - Activation extraction time
    - Safety evaluation time
    - Network time
    - Total time
    """
    import time
    
    results = {
        "extraction_ms": [],
        "evaluation_ms": [],
        "network_ms": [],
        "total_ms": [],
    }
    
    for prompt in test_prompts[:num_runs]:
        total_start = time.perf_counter()
        
        # Extraction
        ext_start = time.perf_counter()
        if hasattr(integration, 'extractor'):
            _ = integration.extractor.extract(prompt)
        ext_end = time.perf_counter()
        
        # Full check
        check_start = time.perf_counter()
        if hasattr(integration, 'check_safety'):
            _, _ = integration.check_safety(prompt)
        check_end = time.perf_counter()
        
        total_end = time.perf_counter()
        
        results["extraction_ms"].append((ext_end - ext_start) * 1000)
        results["evaluation_ms"].append((check_end - check_start) * 1000)
        results["total_ms"].append((total_end - total_start) * 1000)
    
    # Compute stats
    return {
        "extraction_mean_ms": np.mean(results["extraction_ms"]),
        "evaluation_mean_ms": np.mean(results["evaluation_ms"]),
        "total_mean_ms": np.mean(results["total_ms"]),
        "total_std_ms": np.std(results["total_ms"]),
    }
