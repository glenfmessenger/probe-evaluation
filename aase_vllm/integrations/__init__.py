"""
AASE Integrations Module

vLLM integration for activation-based safety probes.
"""

from aase.integrations.vllm import (
    VLLMSafetyWrapper,
    create_vllm_extractor,
    get_optimal_layer,
    register_aase_hooks,
)

__all__ = [
    "VLLMSafetyWrapper",
    "create_vllm_extractor", 
    "get_optimal_layer",
    "register_aase_hooks",
]
