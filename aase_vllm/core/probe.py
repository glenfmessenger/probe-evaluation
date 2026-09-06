"""
Base ActivationProbe class for all AASE safety probes.

All probes (AF, AAG, APC, AHD) inherit from this base class and implement
the same interface for training, scoring, saving, and loading.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path
import json
import time


@dataclass
class ProbeResult:
    """Result from a single probe evaluation."""
    probe_name: str
    score: float
    prediction: str  # "SAFE", "FLAGGED", etc.
    threshold: float
    confidence: float  # How far from threshold (in std devs)
    latency_ms: float = 0.0
    metadata: Dict = field(default_factory=dict)
    
    @property
    def is_flagged(self) -> bool:
        return self.prediction != "SAFE"
    
    def to_dict(self) -> Dict:
        return {
            "probe_name": self.probe_name,
            "score": float(self.score),
            "prediction": self.prediction,
            "threshold": float(self.threshold),
            "confidence": float(self.confidence),
            "latency_ms": float(self.latency_ms),
            "is_flagged": self.is_flagged,
            "metadata": self.metadata,
        }


@dataclass
class TrainingMetrics:
    """Metrics from probe training."""
    separation: float  # Class separation in standard deviations
    train_accuracy: float
    test_accuracy: Optional[float] = None
    auc_roc: Optional[float] = None
    false_positive_rate: Optional[float] = None
    false_negative_rate: Optional[float] = None
    n_positive: int = 0
    n_negative: int = 0
    optimal_threshold: float = 0.0
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class ActivationProbe(ABC):
    """
    Base class for activation-based safety probes.
    
    All probes use the same core pattern:
    1. Extract activations at a specific layer
    2. Project onto a direction vector
    3. Compare to threshold
    
    The direction vector encodes the concept being detected (harmful content,
    injection attempts, policy violations, etc.)
    """
    
    # Class attributes to be overridden by subclasses
    PROBE_TYPE: str = "base"
    PROBE_NAME: str = "BaseProbe"
    DEFAULT_LAYER_DEPTH: float = 0.5  # 50% depth by default
    POSITIVE_LABEL: str = "FLAGGED"
    NEGATIVE_LABEL: str = "SAFE"
    
    def __init__(
        self,
        direction: Optional[np.ndarray] = None,
        threshold: float = 0.0,
        layer_index: Optional[int] = None,
        layer_depth: Optional[float] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize probe with optional pre-trained direction vector.
        
        Args:
            direction: Pre-trained direction vector (normalized)
            threshold: Classification threshold
            layer_index: Specific layer to extract from
            layer_depth: Layer depth as fraction (0.0-1.0), used if layer_index not set
            model_name: Name of model this probe was trained for
        """
        self._direction = direction
        self._threshold = threshold
        self._layer_index = layer_index
        self._layer_depth = layer_depth or self.DEFAULT_LAYER_DEPTH
        self._model_name = model_name
        self._is_trained = direction is not None
        self._training_metrics: Optional[TrainingMetrics] = None
        
    @property
    def direction(self) -> Optional[np.ndarray]:
        """Get the direction vector."""
        return self._direction
    
    @property
    def threshold(self) -> float:
        """Get the classification threshold."""
        return self._threshold
    
    @property
    def is_trained(self) -> bool:
        """Check if probe has been trained."""
        return self._is_trained
    
    @property
    def hidden_dim(self) -> Optional[int]:
        """Get hidden dimension of direction vector."""
        return self._direction.shape[0] if self._direction is not None else None
    
    def get_layer_index(self, num_layers: int) -> int:
        """
        Get the layer index to extract from.
        
        Args:
            num_layers: Total number of layers in the model
            
        Returns:
            Layer index (0-indexed)
        """
        if self._layer_index is not None:
            return self._layer_index
        return int(self._layer_depth * num_layers)
    
    @abstractmethod
    def train(
        self,
        positive_examples: List[str],
        negative_examples: List[str],
        extractor: "ActivationExtractor",
        **kwargs
    ) -> TrainingMetrics:
        """
        Train the direction vector from labeled examples.
        
        Args:
            positive_examples: Examples that should be flagged
            negative_examples: Examples that should be safe
            extractor: Activation extractor for the target backend
            **kwargs: Probe-specific training options
            
        Returns:
            Training metrics
        """
        pass
    
    def train_from_activations(
        self,
        positive_activations: np.ndarray,
        negative_activations: np.ndarray,
        method: str = "mean_diff"
    ) -> TrainingMetrics:
        """
        Train direction vector from pre-extracted activations.
        
        This is the core training algorithm shared by all probes.
        
        Args:
            positive_activations: Shape [n_pos, hidden_dim]
            negative_activations: Shape [n_neg, hidden_dim]
            method: Training method - "mean_diff" or "contrastive"
            
        Returns:
            Training metrics
        """
        if method == "mean_diff":
            return self._train_mean_diff(positive_activations, negative_activations)
        elif method == "contrastive":
            return self._train_contrastive(positive_activations, negative_activations)
        else:
            raise ValueError(f"Unknown training method: {method}")
    
    def _train_mean_diff(
        self,
        positive_acts: np.ndarray,
        negative_acts: np.ndarray
    ) -> TrainingMetrics:
        """
        Standard mean difference training.
        
        direction = normalize(mean(positive) - mean(negative))
        """
        pos_mean = positive_acts.mean(axis=0)
        neg_mean = negative_acts.mean(axis=0)
        
        direction = pos_mean - neg_mean
        direction = direction / np.linalg.norm(direction)
        
        self._direction = direction
        
        # Compute scores and metrics
        pos_scores = positive_acts @ direction
        neg_scores = negative_acts @ direction
        
        # Optimal threshold is midpoint between means
        self._threshold = (pos_scores.mean() + neg_scores.mean()) / 2
        
        # Compute separation
        pooled_std = np.sqrt((pos_scores.var() + neg_scores.var()) / 2)
        separation = (pos_scores.mean() - neg_scores.mean()) / max(pooled_std, 1e-6)
        
        # Compute accuracy
        all_scores = np.concatenate([pos_scores, neg_scores])
        all_labels = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
        predictions = (all_scores > self._threshold).astype(int)
        accuracy = np.mean(predictions == all_labels)
        
        # Confusion matrix for FPR/FNR
        tp = np.sum((predictions == 1) & (all_labels == 1))
        fp = np.sum((predictions == 1) & (all_labels == 0))
        tn = np.sum((predictions == 0) & (all_labels == 0))
        fn = np.sum((predictions == 0) & (all_labels == 1))
        
        self._is_trained = True
        self._training_metrics = TrainingMetrics(
            separation=float(separation),
            train_accuracy=float(accuracy),
            false_positive_rate=fp / (fp + tn) if (fp + tn) > 0 else 0,
            false_negative_rate=fn / (fn + tp) if (fn + tp) > 0 else 0,
            n_positive=len(positive_acts),
            n_negative=len(negative_acts),
            optimal_threshold=float(self._threshold),
        )
        
        return self._training_metrics
    
    def _train_contrastive(
        self,
        positive_acts: np.ndarray,
        negative_acts: np.ndarray
    ) -> TrainingMetrics:
        """
        Contrastive pair training (used by APC).
        
        direction = normalize(mean(positive_i - negative_i))
        
        Requires paired examples (same length arrays).
        """
        if len(positive_acts) != len(negative_acts):
            raise ValueError(
                f"Contrastive training requires paired examples. "
                f"Got {len(positive_acts)} positive and {len(negative_acts)} negative."
            )
        
        # Compute pairwise differences
        pair_diffs = positive_acts - negative_acts
        
        # Direction is mean of differences, normalized
        direction = pair_diffs.mean(axis=0)
        direction = direction / np.linalg.norm(direction)
        
        self._direction = direction
        
        # Compute scores
        pos_scores = positive_acts @ direction
        neg_scores = negative_acts @ direction
        
        # Optimal threshold
        self._threshold = (pos_scores.mean() + neg_scores.mean()) / 2
        
        # Separation
        separation = (pos_scores.mean() - neg_scores.mean()) / (
            np.sqrt(pos_scores.var() + neg_scores.var()) + 1e-8
        )
        
        # Accuracy
        all_scores = np.concatenate([pos_scores, neg_scores])
        all_labels = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
        predictions = (all_scores > self._threshold).astype(int)
        accuracy = np.mean(predictions == all_labels)
        
        self._is_trained = True
        self._training_metrics = TrainingMetrics(
            separation=float(separation),
            train_accuracy=float(accuracy),
            n_positive=len(positive_acts),
            n_negative=len(negative_acts),
            optimal_threshold=float(self._threshold),
        )
        
        return self._training_metrics
    
    def score(self, activation: np.ndarray) -> float:
        """
        Compute safety score for an activation.
        
        Args:
            activation: Hidden state vector [hidden_dim]
            
        Returns:
            Score (higher = more likely to be flagged)
        """
        if not self._is_trained:
            raise RuntimeError("Probe not trained. Call train() or load() first.")
        
        # Normalize activation for stability
        activation_norm = activation / np.linalg.norm(activation)
        
        return float(np.dot(activation_norm, self._direction))
    
    def score_batch(self, activations: np.ndarray) -> np.ndarray:
        """
        Compute safety scores for a batch of activations.
        
        Args:
            activations: Hidden states [batch, hidden_dim]
            
        Returns:
            Scores [batch]
        """
        if not self._is_trained:
            raise RuntimeError("Probe not trained. Call train() or load() first.")
        
        # Normalize each activation
        norms = np.linalg.norm(activations, axis=1, keepdims=True)
        activations_norm = activations / np.maximum(norms, 1e-8)
        
        return activations_norm @ self._direction
    
    def classify(self, activation: np.ndarray) -> str:
        """
        Classify an activation as safe or flagged.
        
        Args:
            activation: Hidden state vector [hidden_dim]
            
        Returns:
            Classification label
        """
        score = self.score(activation)
        return self.POSITIVE_LABEL if score > self._threshold else self.NEGATIVE_LABEL
    
    def evaluate(self, activation: np.ndarray) -> ProbeResult:
        """
        Full evaluation with score, classification, and confidence.
        
        Args:
            activation: Hidden state vector [hidden_dim]
            
        Returns:
            ProbeResult with all evaluation data
        """
        start = time.perf_counter()
        
        score = self.score(activation)
        prediction = self.POSITIVE_LABEL if score > self._threshold else self.NEGATIVE_LABEL
        
        # Confidence: how many std devs from threshold
        if self._training_metrics:
            # Use pooled std from training
            std = 1.0 / self._training_metrics.separation if self._training_metrics.separation > 0 else 1.0
        else:
            std = 1.0
        confidence = abs(score - self._threshold) / std
        
        latency_ms = (time.perf_counter() - start) * 1000
        
        return ProbeResult(
            probe_name=self.PROBE_NAME,
            score=score,
            prediction=prediction,
            threshold=self._threshold,
            confidence=confidence,
            latency_ms=latency_ms,
            metadata={
                "probe_type": self.PROBE_TYPE,
                "layer_index": self._layer_index,
                "layer_depth": self._layer_depth,
                "model_name": self._model_name,
            }
        )
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save probe to disk.
        
        Creates two files:
        - {path}.npy: Direction vector
        - {path}.json: Metadata and training config
        """
        path = Path(path)
        
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained probe")
        
        # Save direction vector
        np.save(f"{path}.npy", self._direction)
        
        # Save metadata
        metadata = {
            "probe_type": self.PROBE_TYPE,
            "probe_name": self.PROBE_NAME,
            "version": "1.0",
            "threshold": float(self._threshold),
            "layer_index": self._layer_index,
            "layer_depth": float(self._layer_depth) if self._layer_depth else None,
            "model_name": self._model_name,
            "hidden_dim": int(self._direction.shape[0]),
            "training_metrics": self._training_metrics.to_dict() if self._training_metrics else None,
        }
        
        with open(f"{path}.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "ActivationProbe":
        """
        Load probe from disk.
        
        Args:
            path: Path without extension (will load .npy and .json)
            
        Returns:
            Loaded probe instance
        """
        path = Path(path)
        
        # Handle case where full filename is provided
        if path.suffix == ".npy":
            path = path.with_suffix("")
        elif path.suffix == ".json":
            path = path.with_suffix("")
        
        # Load direction vector
        direction = np.load(f"{path}.npy")
        
        # Load metadata
        meta_path = Path(f"{path}.json")
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Create probe instance
        probe = cls(
            direction=direction,
            threshold=metadata.get("threshold", 0.0),
            layer_index=metadata.get("layer_index"),
            layer_depth=metadata.get("layer_depth"),
            model_name=metadata.get("model_name"),
        )
        probe._is_trained = True
        
        # Restore training metrics if available
        if metadata.get("training_metrics"):
            probe._training_metrics = TrainingMetrics(**metadata["training_metrics"])
        
        return probe
    
    def __repr__(self) -> str:
        status = "trained" if self._is_trained else "untrained"
        dim = self.hidden_dim or "?"
        return f"{self.PROBE_NAME}({status}, dim={dim}, threshold={self._threshold:.4f})"
