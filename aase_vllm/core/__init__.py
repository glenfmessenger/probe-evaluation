"""
AASE Core Module

Contains base classes and utilities used by all probes.
"""

from aase.core.probe import ActivationProbe, ProbeResult, TrainingMetrics
from aase.core.extraction import (
    ActivationExtractor,
    ExtractionConfig,
    VLLMExtractor,
    TransformersExtractor,
    LLMDExtractor,
    create_extractor,
)
from aase.core.vector import (
    DirectionVector,
    VectorCollection,
    compute_direction_mean_diff,
    compute_direction_contrastive,
)

__all__ = [
    # Probe base
    "ActivationProbe",
    "ProbeResult", 
    "TrainingMetrics",
    # Extraction
    "ActivationExtractor",
    "ExtractionConfig",
    "VLLMExtractor",
    "TransformersExtractor",
    "LLMDExtractor",
    "create_extractor",
    # Vector utilities
    "DirectionVector",
    "VectorCollection",
    "compute_direction_mean_diff",
    "compute_direction_contrastive",
]
