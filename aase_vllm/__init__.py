"""
AASE - Activation-based AI Safety Enforcement
==============================================

A lightweight safety layer for vLLM using activation probes.

Probes:
    - AF (Activation Fingerprinting): Harmful content detection
    - AAG (Agent Action Gating): Prompt injection detection
    - APC (Activation Policy Compliance): Enterprise policy enforcement

Usage:
    from aase import AASEGuard
    
    guard = AASEGuard.from_pretrained(model_name, probes=["af", "aag"])
    guard.register(llm)
    
    result = guard.check("user prompt", llm)
    if not result["safe"]:
        print(f"Blocked by: {result['flagged_by']}")

Author: Glen Messenger
License: Apache 2.0
"""

__version__ = "0.1.0"
__author__ = "Glen Messenger"

# vLLM integration
from aase.integrations.vllm import (
    VLLMSafetyWrapper,
    register_aase_hooks,
    create_vllm_extractor,
    get_optimal_layer,
    AASEProbe,
    AASEGuard,
    ProbeResult,
)

__all__ = [
    "__version__",
    "__author__",
    "VLLMSafetyWrapper",
    "register_aase_hooks",
    "create_vllm_extractor",
    "get_optimal_layer",
    "AASEProbe",
    "AASEGuard",
    "ProbeResult",
]
