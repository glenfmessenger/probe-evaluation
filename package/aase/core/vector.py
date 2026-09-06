"""
Direction vector utilities for saving, loading, and manipulating safety vectors.

Direction vectors are the core of AASE - they encode learned safety concepts
in a single vector that can be used for fast classification.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path
import json


@dataclass
class DirectionVector:
    """
    Container for a trained direction vector with metadata.
    
    The direction vector is a normalized vector in the model's hidden space
    that points from "safe" to "flagged" activations. Projecting an activation
    onto this vector gives a score: higher = more likely to be flagged.
    """
    
    vector: np.ndarray
    probe_type: str  # "af", "aag", "apc", "ahd"
    model_name: str
    layer_index: int
    threshold: float = 0.0
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Normalize vector on creation."""
        self.vector = self.vector / np.linalg.norm(self.vector)
    
    @property
    def dim(self) -> int:
        """Get vector dimension."""
        return self.vector.shape[0]
    
    def score(self, activation: np.ndarray) -> float:
        """
        Score an activation against this direction.
        
        Args:
            activation: Hidden state [hidden_dim]
            
        Returns:
            Score (higher = more likely flagged)
        """
        activation_norm = activation / np.linalg.norm(activation)
        return float(np.dot(activation_norm, self.vector))
    
    def classify(self, activation: np.ndarray) -> bool:
        """Classify as flagged (True) or safe (False)."""
        return self.score(activation) > self.threshold
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save direction vector to disk.
        
        Args:
            path: Output path (without extension)
        """
        path = Path(path)
        
        # Save vector
        np.save(f"{path}.npy", self.vector)
        
        # Save metadata
        meta = {
            "probe_type": self.probe_type,
            "model_name": self.model_name,
            "layer_index": self.layer_index,
            "threshold": float(self.threshold),
            "dim": int(self.dim),
            **self.metadata
        }
        with open(f"{path}.json", "w") as f:
            json.dump(meta, f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "DirectionVector":
        """
        Load direction vector from disk.
        
        Args:
            path: Path to vector (with or without extension)
            
        Returns:
            Loaded DirectionVector
        """
        path = Path(path)
        if path.suffix == ".npy":
            path = path.with_suffix("")
        elif path.suffix == ".json":
            path = path.with_suffix("")
        
        # Load vector
        vector = np.load(f"{path}.npy")
        
        # Load metadata
        with open(f"{path}.json") as f:
            meta = json.load(f)
        
        return cls(
            vector=vector,
            probe_type=meta.get("probe_type", "unknown"),
            model_name=meta.get("model_name", "unknown"),
            layer_index=meta.get("layer_index", 0),
            threshold=meta.get("threshold", 0.0),
            metadata={k: v for k, v in meta.items() 
                     if k not in ["probe_type", "model_name", "layer_index", "threshold", "dim"]}
        )
    
    def recalibrate(
        self,
        new_positive_acts: np.ndarray,
        new_negative_acts: np.ndarray,
        alpha: float = 0.3
    ) -> "DirectionVector":
        """
        Recalibrate vector for new backend/quantization.
        
        When transferring from one backend to another (e.g., HuggingFace → llm-d),
        the direction vector may need adjustment. This method performs soft
        recalibration by blending the original direction with a new one
        computed from the target backend.
        
        Args:
            new_positive_acts: Positive class activations from new backend
            new_negative_acts: Negative class activations from new backend
            alpha: Blending factor (0 = keep original, 1 = use new)
            
        Returns:
            New recalibrated DirectionVector
        """
        # Compute new direction from target backend
        new_pos_mean = new_positive_acts.mean(axis=0)
        new_neg_mean = new_negative_acts.mean(axis=0)
        new_direction = new_pos_mean - new_neg_mean
        new_direction = new_direction / np.linalg.norm(new_direction)
        
        # Blend original and new directions
        blended = (1 - alpha) * self.vector + alpha * new_direction
        blended = blended / np.linalg.norm(blended)
        
        # Compute new threshold
        new_pos_scores = new_positive_acts @ blended
        new_neg_scores = new_negative_acts @ blended
        new_threshold = (new_pos_scores.mean() + new_neg_scores.mean()) / 2
        
        return DirectionVector(
            vector=blended,
            probe_type=self.probe_type,
            model_name=self.model_name,
            layer_index=self.layer_index,
            threshold=new_threshold,
            metadata={
                **self.metadata,
                "recalibrated": True,
                "recalibration_alpha": alpha,
            }
        )
    
    def __repr__(self) -> str:
        return (
            f"DirectionVector("
            f"type={self.probe_type}, "
            f"model={self.model_name}, "
            f"layer={self.layer_index}, "
            f"dim={self.dim})"
        )


class VectorCollection:
    """
    Collection of direction vectors for a model.
    
    Manages multiple probes (AF, AAG, APC, etc.) for a single model.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._vectors: Dict[str, DirectionVector] = {}
    
    def add(self, vector: DirectionVector) -> None:
        """Add a vector to the collection."""
        self._vectors[vector.probe_type] = vector
    
    def get(self, probe_type: str) -> Optional[DirectionVector]:
        """Get vector by probe type."""
        return self._vectors.get(probe_type)
    
    def list_probes(self) -> List[str]:
        """List available probe types."""
        return list(self._vectors.keys())
    
    def save(self, directory: Union[str, Path]) -> None:
        """Save all vectors to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        for probe_type, vector in self._vectors.items():
            vector.save(directory / f"{self.model_name}_{probe_type}")
        
        # Save collection metadata
        meta = {
            "model_name": self.model_name,
            "probes": list(self._vectors.keys()),
        }
        with open(directory / "collection.json", "w") as f:
            json.dump(meta, f, indent=2)
    
    @classmethod
    def load(cls, directory: Union[str, Path]) -> "VectorCollection":
        """Load collection from directory."""
        directory = Path(directory)
        
        with open(directory / "collection.json") as f:
            meta = json.load(f)
        
        collection = cls(meta["model_name"])
        
        for probe_type in meta["probes"]:
            vector = DirectionVector.load(
                directory / f"{meta['model_name']}_{probe_type}"
            )
            collection.add(vector)
        
        return collection


def compute_direction_mean_diff(
    positive_acts: np.ndarray,
    negative_acts: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """
    Compute direction vector using mean difference method.
    
    Args:
        positive_acts: Positive class activations [n_pos, hidden_dim]
        negative_acts: Negative class activations [n_neg, hidden_dim]
        
    Returns:
        Tuple of (direction, threshold, separation)
    """
    pos_mean = positive_acts.mean(axis=0)
    neg_mean = negative_acts.mean(axis=0)
    
    direction = pos_mean - neg_mean
    direction = direction / np.linalg.norm(direction)
    
    pos_scores = positive_acts @ direction
    neg_scores = negative_acts @ direction
    
    threshold = (pos_scores.mean() + neg_scores.mean()) / 2
    
    pooled_std = np.sqrt((pos_scores.var() + neg_scores.var()) / 2)
    separation = (pos_scores.mean() - neg_scores.mean()) / max(pooled_std, 1e-6)
    
    return direction, threshold, separation


def compute_direction_contrastive(
    positive_acts: np.ndarray,
    negative_acts: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """
    Compute direction vector using contrastive pair method.
    
    Requires paired examples (same length).
    
    Args:
        positive_acts: Positive class activations [n, hidden_dim]
        negative_acts: Negative class activations [n, hidden_dim]
        
    Returns:
        Tuple of (direction, threshold, separation)
    """
    if len(positive_acts) != len(negative_acts):
        raise ValueError("Contrastive method requires paired examples")
    
    # Pairwise differences
    diffs = positive_acts - negative_acts
    
    direction = diffs.mean(axis=0)
    direction = direction / np.linalg.norm(direction)
    
    pos_scores = positive_acts @ direction
    neg_scores = negative_acts @ direction
    
    threshold = (pos_scores.mean() + neg_scores.mean()) / 2
    
    separation = (pos_scores.mean() - neg_scores.mean()) / (
        np.sqrt(pos_scores.var() + neg_scores.var()) + 1e-8
    )
    
    return direction, threshold, separation
