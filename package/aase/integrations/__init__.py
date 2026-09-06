"""
AASE Integrations Module

Backend-specific integrations for different inference engines.
"""

from typing import List

# Lazy imports to avoid requiring all backends
_AVAILABLE_BACKENDS = []

def get_backend(name: str):
    """
    Get integration module for a backend.
    
    Args:
        name: Backend name ("vllm", "llmd", "transformers")
        
    Returns:
        Integration module
    """
    if name == "vllm":
        from aase.integrations import vllm
        return vllm
    elif name in ("llmd", "llm-d"):
        from aase.integrations import llmd
        return llmd
    elif name == "transformers":
        from aase.integrations import transformers
        return transformers
    else:
        raise ValueError(f"Unknown backend: {name}")


def list_backends() -> List[str]:
    """List available backend integrations."""
    return ["vllm", "llmd", "transformers"]


# Try to import backends and track availability
try:
    from aase.integrations.vllm import (
        VLLMSafetyWrapper,
        create_vllm_extractor,
        get_optimal_layer,
        register_aase_hooks,
    )
    _AVAILABLE_BACKENDS.append("vllm")
except ImportError:
    pass

try:
    from aase.integrations.llmd import (
        LLMDConfig,
        LLMDProxyService,
        LLMDDirectIntegration,
        create_llmd_integration,
        recalibrate_for_llmd,
        benchmark_llmd_latency,
    )
    _AVAILABLE_BACKENDS.append("llmd")
except ImportError:
    pass

try:
    from aase.integrations.transformers import (
        TransformersSafetyWrapper,
        load_model_for_aase,
        train_probe_transformers,
    )
    _AVAILABLE_BACKENDS.append("transformers")
except ImportError:
    pass


__all__ = [
    "get_backend",
    "list_backends",
    # vLLM
    "VLLMSafetyWrapper",
    "create_vllm_extractor", 
    "get_optimal_layer",
    "register_aase_hooks",
    # llm-d
    "LLMDConfig",
    "LLMDProxyService",
    "LLMDDirectIntegration",
    "create_llmd_integration",
    "recalibrate_for_llmd",
    "benchmark_llmd_latency",
    # Transformers
    "TransformersSafetyWrapper",
    "load_model_for_aase",
    "train_probe_transformers",
]
