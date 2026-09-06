"""
AASE - Activation-based AI Safety Enforcement
==============================================

A unified package for activation-based safety probes that detect harmful content,
prompt injections, policy violations, and hallucinations in LLM outputs.

Core Components:
    - AF (Activation Fingerprinting): Harmful content detection
    - AAG (Agent Action Gating): Prompt injection detection  
    - APC (Activation Policy Compliance): Enterprise policy enforcement
    - AHD (Activation Hallucination Detection): Hallucination detection [experimental]

Usage:
    from aase import AF, AAG, APC, SafetyStack
    
    # Individual probes
    af = AF.load("path/to/direction.npy")
    score = af.score(activations)
    
    # Combined safety stack
    stack = SafetyStack(probes=[af, aag, apc])
    results = stack.evaluate(activations)

Author: Glen Messenger
License: Apache 2.0
"""

__version__ = "0.1.0"
__author__ = "Glen Messenger"

# Core components
from aase.core.probe import ActivationProbe
from aase.core.extraction import ActivationExtractor
from aase.core.vector import DirectionVector

# Probes
from aase.probes.af import AF
from aase.probes.aag import AAG
from aase.probes.apc import APC
from aase.probes.ahd import AHD

# Safety Stack
from aase.stack import SafetyStack

# Backend integrations
from aase.integrations import get_backend, list_backends

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Core
    "ActivationProbe",
    "ActivationExtractor", 
    "DirectionVector",
    # Probes
    "AF",
    "AAG",
    "APC",
    "AHD",
    # Stack
    "SafetyStack",
    # Utils
    "get_backend",
    "list_backends",
]
