"""
SafetyStack - Unified multi-probe evaluation.

Runs multiple AASE probes and combines results for comprehensive safety assessment.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import numpy as np
import time

from aase.core.probe import ActivationProbe, ProbeResult


@dataclass
class StackResult:
    """Combined result from all probes in the stack."""
    
    results: Dict[str, ProbeResult]
    overall_safe: bool
    overall_score: float  # Aggregated risk score
    flagged_probes: List[str]
    total_latency_ms: float
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "overall_safe": self.overall_safe,
            "overall_score": float(self.overall_score),
            "flagged_probes": self.flagged_probes,
            "total_latency_ms": float(self.total_latency_ms),
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "metadata": self.metadata,
        }
    
    def __repr__(self) -> str:
        status = "SAFE" if self.overall_safe else f"FLAGGED ({', '.join(self.flagged_probes)})"
        return f"StackResult({status}, score={self.overall_score:.3f}, latency={self.total_latency_ms:.1f}ms)"


class SafetyStack:
    """
    Run multiple safety probes and combine results.
    
    The SafetyStack provides:
    - Unified evaluation across AF, AAG, APC probes
    - Configurable aggregation (any-flagged, majority, weighted)
    - Combined risk scoring
    - Latency tracking
    
    Usage:
        from aase import AF, AAG, APC, SafetyStack
        
        # Create probes
        af = AF.load("path/to/af")
        aag = AAG.load("path/to/aag")
        apc = APC.load("path/to/apc")
        
        # Create stack
        stack = SafetyStack(probes=[af, aag, apc])
        
        # Evaluate
        result = stack.evaluate(activation)
        if not result.overall_safe:
            print(f"Flagged by: {result.flagged_probes}")
    """
    
    def __init__(
        self,
        probes: Optional[List[ActivationProbe]] = None,
        aggregation: str = "any",  # "any", "majority", "weighted"
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.5,  # For weighted aggregation
    ):
        """
        Initialize SafetyStack.
        
        Args:
            probes: List of trained probes to run
            aggregation: How to combine results
                - "any": Flag if ANY probe flags (most conservative)
                - "majority": Flag if majority of probes flag
                - "weighted": Flag if weighted score exceeds threshold
            weights: Weights per probe type for weighted aggregation
            threshold: Threshold for weighted aggregation
        """
        self.probes: Dict[str, ActivationProbe] = {}
        self.aggregation = aggregation
        self.weights = weights or {}
        self.threshold = threshold
        
        if probes:
            for probe in probes:
                self.add_probe(probe)
    
    def add_probe(self, probe: ActivationProbe) -> None:
        """Add a probe to the stack."""
        if not probe.is_trained:
            raise ValueError(f"Probe {probe.PROBE_NAME} is not trained")
        self.probes[probe.PROBE_TYPE] = probe
    
    def remove_probe(self, probe_type: str) -> None:
        """Remove a probe from the stack."""
        if probe_type in self.probes:
            del self.probes[probe_type]
    
    def list_probes(self) -> List[str]:
        """List active probe types."""
        return list(self.probes.keys())
    
    def evaluate(self, activation: np.ndarray) -> StackResult:
        """
        Run all probes on an activation.
        
        Args:
            activation: Hidden state vector [hidden_dim]
            
        Returns:
            StackResult with combined evaluation
        """
        start = time.perf_counter()
        
        results = {}
        flagged_probes = []
        
        for probe_type, probe in self.probes.items():
            result = probe.evaluate(activation)
            results[probe_type] = result
            if result.is_flagged:
                flagged_probes.append(probe_type)
        
        # Aggregate results
        overall_safe, overall_score = self._aggregate(results)
        
        total_latency = (time.perf_counter() - start) * 1000
        
        return StackResult(
            results=results,
            overall_safe=overall_safe,
            overall_score=overall_score,
            flagged_probes=flagged_probes,
            total_latency_ms=total_latency,
            metadata={
                "aggregation": self.aggregation,
                "n_probes": len(self.probes),
            }
        )
    
    def evaluate_batch(self, activations: np.ndarray) -> List[StackResult]:
        """
        Run all probes on a batch of activations.
        
        Args:
            activations: Hidden states [batch, hidden_dim]
            
        Returns:
            List of StackResults
        """
        return [self.evaluate(act) for act in activations]
    
    def _aggregate(self, results: Dict[str, ProbeResult]) -> tuple:
        """Aggregate probe results into overall decision."""
        if not results:
            return True, 0.0
        
        if self.aggregation == "any":
            # Flag if ANY probe flags
            any_flagged = any(r.is_flagged for r in results.values())
            # Score is max of normalized scores
            max_score = max(r.score for r in results.values())
            return not any_flagged, max_score
        
        elif self.aggregation == "majority":
            # Flag if majority of probes flag
            n_flagged = sum(1 for r in results.values() if r.is_flagged)
            majority_flagged = n_flagged > len(results) / 2
            avg_score = np.mean([r.score for r in results.values()])
            return not majority_flagged, avg_score
        
        elif self.aggregation == "weighted":
            # Compute weighted score
            total_weight = 0
            weighted_score = 0
            
            for probe_type, result in results.items():
                weight = self.weights.get(probe_type, 1.0)
                # Normalize score to [0, 1] range using sigmoid
                normalized = 1 / (1 + np.exp(-result.score))
                weighted_score += weight * normalized
                total_weight += weight
            
            if total_weight > 0:
                weighted_score /= total_weight
            
            return weighted_score < self.threshold, weighted_score
        
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
    
    def save(self, directory: str) -> None:
        """Save all probes and stack config to directory."""
        import os
        import json
        
        os.makedirs(directory, exist_ok=True)
        
        # Save each probe
        for probe_type, probe in self.probes.items():
            probe.save(os.path.join(directory, probe_type))
        
        # Save stack config
        config = {
            "aggregation": self.aggregation,
            "weights": self.weights,
            "threshold": self.threshold,
            "probes": list(self.probes.keys()),
        }
        with open(os.path.join(directory, "stack_config.json"), "w") as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load(cls, directory: str) -> "SafetyStack":
        """Load SafetyStack from directory."""
        import os
        import json
        from aase.probes import AF, AAG, APC
        
        # Load config
        with open(os.path.join(directory, "stack_config.json")) as f:
            config = json.load(f)
        
        # Map probe types to classes
        probe_classes = {
            "af": AF,
            "aag": AAG,
            "apc": APC,
        }
        
        # Load probes
        probes = []
        for probe_type in config["probes"]:
            if probe_type in probe_classes:
                probe = probe_classes[probe_type].load(
                    os.path.join(directory, probe_type)
                )
                probes.append(probe)
        
        return cls(
            probes=probes,
            aggregation=config.get("aggregation", "any"),
            weights=config.get("weights", {}),
            threshold=config.get("threshold", 0.5),
        )
    
    def __repr__(self) -> str:
        probes_str = ", ".join(self.probes.keys())
        return f"SafetyStack([{probes_str}], aggregation={self.aggregation})"


class SafetyGate:
    """
    High-level safety gate for integration with inference pipelines.
    
    Provides a simple pass/fail interface for production use.
    
    Usage:
        gate = SafetyGate(stack, extractor)
        
        # Check if content is safe
        if gate.check(user_input):
            response = model.generate(user_input)
        else:
            response = "I cannot help with that request."
    """
    
    def __init__(
        self,
        stack: SafetyStack,
        extractor: "ActivationExtractor",
        fail_closed: bool = True,
    ):
        """
        Initialize SafetyGate.
        
        Args:
            stack: SafetyStack to use for evaluation
            extractor: Activation extractor for the model
            fail_closed: If True, failures default to unsafe
        """
        self.stack = stack
        self.extractor = extractor
        self.fail_closed = fail_closed
    
    def check(self, text: str) -> bool:
        """
        Check if text passes safety gate.
        
        Args:
            text: Text to evaluate
            
        Returns:
            True if safe, False if flagged
        """
        try:
            activation = self.extractor.extract(text)
            result = self.stack.evaluate(activation)
            return result.overall_safe
        except Exception as e:
            if self.fail_closed:
                return False
            raise
    
    def check_with_details(self, text: str) -> StackResult:
        """
        Check with full result details.
        
        Args:
            text: Text to evaluate
            
        Returns:
            Full StackResult
        """
        activation = self.extractor.extract(text)
        return self.stack.evaluate(activation)
