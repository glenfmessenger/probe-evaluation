"""
Activation Hallucination Detection (AHD) - PLACEHOLDER

STATUS: Experimental / Not Implemented

This is a placeholder for future hallucination detection capabilities.
"""

from typing import List, Optional
import numpy as np

from aase.core.probe import ActivationProbe, TrainingMetrics
from aase.core.extraction import ActivationExtractor


class AHD(ActivationProbe):
    """
    Activation Hallucination Detection [PLACEHOLDER].
    
    Not yet implemented. This is a placeholder for future development.
    """
    
    PROBE_TYPE = "ahd"
    PROBE_NAME = "ActivationHallucinationDetection"
    DEFAULT_LAYER_DEPTH = 0.70
    POSITIVE_LABEL = "HALLUCINATION"
    NEGATIVE_LABEL = "FACTUAL"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import warnings
        warnings.warn(
            "AHD is a placeholder and not yet implemented.",
            UserWarning
        )
    
    def train(
        self,
        positive_examples: List[str],
        negative_examples: List[str],
        extractor: ActivationExtractor,
        **kwargs
    ) -> TrainingMetrics:
        raise NotImplementedError("AHD is not yet implemented")
