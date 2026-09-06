"""
AF Vector Packaging for Distribution
=====================================
Creates a clean, distributable artifact containing all AF direction vectors
for integration with Gemma model releases.

Output formats:
1. .safetensors (HuggingFace standard - recommended)
2. .npz (NumPy native - simple)
3. JSON metadata + binary vectors (portable)

Package Contents:
- 16 direction vectors (1 obfuscation + 15 categories)
- Per-vector thresholds
- Metadata (model, layer, dimensions, training date)
- Simple loader function

Author: Glen Messenger
Date: December 2024
"""

import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict


# =============================================================================
# METADATA SCHEMA
# =============================================================================

@dataclass
class VectorMetadata:
    """Metadata for a single direction vector."""
    category: str
    description: str
    threshold: float
    separation_sigma: float
    train_accuracy: float
    num_positive_examples: int
    num_negative_examples: int


@dataclass
class PackageMetadata:
    """Metadata for the entire AF package."""
    version: str
    model_name: str
    model_family: str
    model_size: str
    hidden_dimension: int
    num_layers: int
    extraction_layer: int
    extraction_layer_pct: float
    num_vectors: int
    total_size_bytes: int
    created_at: str
    created_by: str
    license: str
    vectors: Dict[str, VectorMetadata]


# =============================================================================
# CATEGORY DEFINITIONS
# =============================================================================

CATEGORY_DEFINITIONS = {
    # Meta-vector
    "obfuscation": {
        "description": "Detects encoded/obfuscated attacks (Base64, ROT13, leetspeak)",
        "hazard_code": None,
        "ailuminate_mapping": None,
    },
    
    # AILuminate hazard categories
    "vcr": {
        "description": "Violent Crimes - requests for violence against persons",
        "hazard_code": "VCR",
        "ailuminate_mapping": "vcr",
    },
    "src": {
        "description": "Sex-Related Crimes - sexual assault, trafficking, exploitation",
        "hazard_code": "SRC",
        "ailuminate_mapping": "src",
    },
    "cse": {
        "description": "Child Sexual Exploitation - CSAM and grooming content",
        "hazard_code": "CSE",
        "ailuminate_mapping": "cse",
    },
    "ssh": {
        "description": "Suicide & Self-Harm - methods, encouragement, pro-suicide content",
        "hazard_code": "SSH",
        "ailuminate_mapping": "ssh",
    },
    "iwp": {
        "description": "Indiscriminate Weapons - CBRNE, explosives, mass casualty",
        "hazard_code": "IWP",
        "ailuminate_mapping": "iwp",
    },
    "ipv": {
        "description": "Intellectual Property Violations - copyright infringement requests",
        "hazard_code": "IPV",
        "ailuminate_mapping": "ipv",
    },
    "dfm": {
        "description": "Defamation - false statements, libel, reputation attacks",
        "hazard_code": "DFM",
        "ailuminate_mapping": "dfm",
    },
    "ncr": {
        "description": "Non-Violent Crimes - fraud, theft, hacking, illegal activities",
        "hazard_code": "NCR",
        "ailuminate_mapping": "ncr",
    },
    "hte": {
        "description": "Hate - discrimination, slurs, dehumanization",
        "hazard_code": "HTE",
        "ailuminate_mapping": "hte",
    },
    "prv": {
        "description": "Privacy Violations - doxxing, surveillance, data theft",
        "hazard_code": "PRV",
        "ailuminate_mapping": "prv",
    },
    "spc_ele": {
        "description": "Specialized Advice: Elections - voting guidance, election interference",
        "hazard_code": "SPC_ELE",
        "ailuminate_mapping": "spc_ele",
    },
    "spc_fin": {
        "description": "Specialized Advice: Financial - investment advice, market manipulation",
        "hazard_code": "SPC_FIN",
        "ailuminate_mapping": "spc_fin",
    },
    "spc_hlt": {
        "description": "Specialized Advice: Health - medical diagnosis, treatment advice",
        "hazard_code": "SPC_HLT",
        "ailuminate_mapping": "spc_hlt",
    },
    "spc_lgl": {
        "description": "Specialized Advice: Legal - legal advice, case strategy",
        "hazard_code": "SPC_LGL",
        "ailuminate_mapping": "spc_lgl",
    },
    "sxc_prn": {
        "description": "Sexual Content: Pornographic - explicit sexual material",
        "hazard_code": "SXC_PRN",
        "ailuminate_mapping": "sxc_prn",
    },
}


# =============================================================================
# PACKAGER CLASS
# =============================================================================

class AFVectorPackager:
    """Package AF vectors for distribution."""
    
    def __init__(
        self,
        vectors: Dict[str, np.ndarray],
        thresholds: Dict[str, float],
        stats: Dict[str, dict],
        model_name: str,
        model_family: str = "gemma",
        model_size: str = "1B",
        hidden_dim: int = 1536,
        num_layers: int = 26,
        extraction_layer: int = 17,
        layer_pct: float = 0.65,
    ):
        self.vectors = vectors
        self.thresholds = thresholds
        self.stats = stats
        self.model_name = model_name
        self.model_family = model_family
        self.model_size = model_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.extraction_layer = extraction_layer
        self.layer_pct = layer_pct
        
        # Validate
        self._validate()
    
    def _validate(self):
        """Validate inputs."""
        # Check all vectors have same dimension
        dims = set(v.shape[0] for v in self.vectors.values())
        if len(dims) != 1:
            raise ValueError(f"Inconsistent vector dimensions: {dims}")
        
        if list(dims)[0] != self.hidden_dim:
            raise ValueError(f"Vector dim {list(dims)[0]} != hidden_dim {self.hidden_dim}")
        
        # Check all vectors have thresholds
        for cat in self.vectors.keys():
            if cat not in self.thresholds:
                raise ValueError(f"Missing threshold for {cat}")
        
        print(f"Validated: {len(self.vectors)} vectors, {self.hidden_dim}-dim")
    
    def _build_metadata(self) -> PackageMetadata:
        """Build package metadata."""
        
        vector_metadata = {}
        for cat, vector in self.vectors.items():
            stats = self.stats.get(cat, {})
            defn = CATEGORY_DEFINITIONS.get(cat, {})
            
            vector_metadata[cat] = VectorMetadata(
                category=cat,
                description=defn.get("description", f"Category: {cat}"),
                threshold=float(self.thresholds[cat]),
                separation_sigma=float(stats.get("separation", 0)),
                train_accuracy=float(stats.get("train_accuracy", 0)),
                num_positive_examples=int(stats.get("num_positive", 0)),
                num_negative_examples=int(stats.get("num_negative", 0)),
            )
        
        total_size = sum(v.nbytes for v in self.vectors.values())
        
        return PackageMetadata(
            version="2.0.0",
            model_name=self.model_name,
            model_family=self.model_family,
            model_size=self.model_size,
            hidden_dimension=self.hidden_dim,
            num_layers=self.num_layers,
            extraction_layer=self.extraction_layer,
            extraction_layer_pct=self.layer_pct,
            num_vectors=len(self.vectors),
            total_size_bytes=total_size,
            created_at=datetime.utcnow().isoformat() + "Z",
            created_by="Glen Messenger (gkelley@google.com)",
            license="Apache-2.0",
            vectors={k: asdict(v) for k, v in vector_metadata.items()},
        )
    
    def save_safetensors(self, output_path: str):
        """
        Save as .safetensors (HuggingFace standard).
        
        This is the recommended format for integration with HuggingFace models.
        """
        try:
            from safetensors.numpy import save_file
        except ImportError:
            raise ImportError("Install safetensors: pip install safetensors")
        
        # Prepare tensors dict
        tensors = {}
        for cat, vector in self.vectors.items():
            tensors[f"vector_{cat}"] = vector.astype(np.float32)
            tensors[f"threshold_{cat}"] = np.array([self.thresholds[cat]], dtype=np.float32)
        
        # Save tensors
        save_file(tensors, output_path)
        
        # Save metadata as JSON sidecar
        metadata = self._build_metadata()
        metadata_path = output_path.replace('.safetensors', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        print(f"Saved: {output_path} ({os.path.getsize(output_path) / 1024:.1f}KB)")
        print(f"Saved: {metadata_path}")
        
        return output_path, metadata_path
    
    def save_npz(self, output_path: str):
        """
        Save as .npz (NumPy native).
        
        Simple format, good for quick prototyping.
        """
        # Prepare arrays
        data = {}
        for cat, vector in self.vectors.items():
            data[f"vector_{cat}"] = vector.astype(np.float32)
            data[f"threshold_{cat}"] = np.array([self.thresholds[cat]], dtype=np.float32)
        
        # Add metadata as JSON string
        metadata = self._build_metadata()
        data["metadata"] = np.array([json.dumps(asdict(metadata))], dtype=str)
        
        # Save
        np.savez_compressed(output_path, **data)
        
        print(f"Saved: {output_path} ({os.path.getsize(output_path) / 1024:.1f}KB)")
        
        return output_path
    
    def save_json_binary(self, output_dir: str):
        """
        Save as JSON metadata + binary vector files.
        
        Most portable format, works with any language.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each vector as binary
        for cat, vector in self.vectors.items():
            vector_path = os.path.join(output_dir, f"vector_{cat}.bin")
            vector.astype(np.float32).tofile(vector_path)
        
        # Build manifest
        metadata = self._build_metadata()
        manifest = {
            "metadata": asdict(metadata),
            "files": {
                cat: {
                    "vector_file": f"vector_{cat}.bin",
                    "threshold": float(self.thresholds[cat]),
                    "dtype": "float32",
                    "shape": list(self.vectors[cat].shape),
                }
                for cat in self.vectors.keys()
            },
        }
        
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Calculate total size
        total_size = sum(
            os.path.getsize(os.path.join(output_dir, f"vector_{cat}.bin"))
            for cat in self.vectors.keys()
        )
        
        print(f"Saved to: {output_dir}/")
        print(f"  - manifest.json")
        print(f"  - {len(self.vectors)} vector files")
        print(f"  - Total: {total_size / 1024:.1f}KB")
        
        return output_dir


# =============================================================================
# LOADER CLASS
# =============================================================================

class AFVectorLoader:
    """Load AF vectors from packaged files."""
    
    @staticmethod
    def load_safetensors(path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, float], dict]:
        """Load from .safetensors format."""
        try:
            from safetensors.numpy import load_file
        except ImportError:
            raise ImportError("Install safetensors: pip install safetensors")
        
        data = load_file(path)
        
        vectors = {}
        thresholds = {}
        
        for key, value in data.items():
            if key.startswith("vector_"):
                cat = key.replace("vector_", "")
                vectors[cat] = value
            elif key.startswith("threshold_"):
                cat = key.replace("threshold_", "")
                thresholds[cat] = float(value[0])
        
        # Load metadata
        metadata_path = path.replace('.safetensors', '_metadata.json')
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return vectors, thresholds, metadata
    
    @staticmethod
    def load_npz(path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, float], dict]:
        """Load from .npz format."""
        data = np.load(path, allow_pickle=True)
        
        vectors = {}
        thresholds = {}
        metadata = {}
        
        for key in data.files:
            if key.startswith("vector_"):
                cat = key.replace("vector_", "")
                vectors[cat] = data[key]
            elif key.startswith("threshold_"):
                cat = key.replace("threshold_", "")
                thresholds[cat] = float(data[key][0])
            elif key == "metadata":
                metadata = json.loads(str(data[key][0]))
        
        return vectors, thresholds, metadata
    
    @staticmethod
    def load_json_binary(manifest_path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, float], dict]:
        """Load from JSON + binary format."""
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        base_dir = os.path.dirname(manifest_path)
        
        vectors = {}
        thresholds = {}
        
        for cat, info in manifest["files"].items():
            vector_path = os.path.join(base_dir, info["vector_file"])
            vectors[cat] = np.fromfile(vector_path, dtype=np.float32)
            thresholds[cat] = info["threshold"]
        
        return vectors, thresholds, manifest.get("metadata", {})


# =============================================================================
# SIMPLE CLASSIFIER USING LOADED VECTORS
# =============================================================================

class AFClassifier:
    """Simple classifier using loaded AF vectors."""
    
    def __init__(
        self,
        vectors: Dict[str, np.ndarray],
        thresholds: Dict[str, float],
        metadata: dict = None,
    ):
        self.vectors = vectors
        self.thresholds = thresholds
        self.metadata = metadata or {}
        
        # Separate obfuscation vector
        self.obfuscation_vector = vectors.get("obfuscation")
        self.obfuscation_threshold = thresholds.get("obfuscation", 0)
        
        # Category vectors
        self.category_vectors = {
            k: v for k, v in vectors.items() if k != "obfuscation"
        }
        self.category_thresholds = {
            k: v for k, v in thresholds.items() if k != "obfuscation"
        }
    
    @classmethod
    def from_safetensors(cls, path: str) -> "AFClassifier":
        vectors, thresholds, metadata = AFVectorLoader.load_safetensors(path)
        return cls(vectors, thresholds, metadata)
    
    @classmethod
    def from_npz(cls, path: str) -> "AFClassifier":
        vectors, thresholds, metadata = AFVectorLoader.load_npz(path)
        return cls(vectors, thresholds, metadata)
    
    def classify(
        self,
        activation: np.ndarray,
        enabled_categories: List[str] = None,
        require_category_confirmation: bool = True,
    ) -> Tuple[bool, List[str], Dict[str, float]]:
        """
        Classify an activation vector.
        
        Args:
            activation: Hidden state activation from model (shape: [hidden_dim])
            enabled_categories: List of categories to check (None = all)
            require_category_confirmation: If True, obfuscation only blocks when
                                          a category also triggers (reduces FP)
        
        Returns:
            (is_blocked, triggered_categories, all_scores)
        """
        scores = {}
        triggered = []
        
        # Stage 1: Obfuscation check
        obf_triggered = False
        if self.obfuscation_vector is not None:
            obf_score = float(activation @ self.obfuscation_vector)
            scores["obfuscation"] = obf_score
            obf_triggered = obf_score > self.obfuscation_threshold
        
        # Stage 2: Category checks
        category_triggered = []
        for cat, vector in self.category_vectors.items():
            if enabled_categories is not None and cat not in enabled_categories:
                continue
            
            score = float(activation @ vector)
            scores[cat] = score
            
            if score > self.category_thresholds[cat]:
                category_triggered.append(cat)
        
        # Decision logic
        if require_category_confirmation:
            if obf_triggered and len(category_triggered) > 0:
                triggered.append("obfuscation")
                triggered.extend(category_triggered)
            elif len(category_triggered) > 0:
                triggered.extend(category_triggered)
            # Obfuscation alone doesn't block
        else:
            if obf_triggered:
                triggered.append("obfuscation")
            triggered.extend(category_triggered)
        
        return len(triggered) > 0, triggered, scores


# =============================================================================
# PACKAGING SCRIPT
# =============================================================================

def package_from_classifier(classifier, output_dir: str = "af_vectors"):
    """
    Package vectors from a trained AFCategoryClassifier.
    
    Usage:
        from af_benchmark_category import AFCategoryClassifier
        
        # ... train classifier ...
        
        package_from_classifier(classifier, "af_vectors_gemma3_1b")
    """
    
    # Extract vectors and thresholds
    vectors = dict(classifier.vectors)
    thresholds = dict(classifier.thresholds)
    
    # Add obfuscation vector
    if classifier.obfuscation_vector is not None:
        vectors["obfuscation"] = classifier.obfuscation_vector
        thresholds["obfuscation"] = classifier.obfuscation_threshold
    
    # Build stats dict
    stats = {cat: asdict(s) if hasattr(s, '__dict__') else s 
             for cat, s in classifier.category_stats.items()}
    
    if classifier.obfuscation_stats:
        stats["obfuscation"] = (asdict(classifier.obfuscation_stats) 
                               if hasattr(classifier.obfuscation_stats, '__dict__') 
                               else classifier.obfuscation_stats)
    
    # Create packager
    packager = AFVectorPackager(
        vectors=vectors,
        thresholds=thresholds,
        stats=stats,
        model_name=getattr(classifier, 'model_name', 'gemma-3-1b'),
        model_family="gemma",
        model_size="1B",
        hidden_dim=classifier.hidden_dim,
        num_layers=classifier.num_layers,
        extraction_layer=classifier.extraction_layer,
        layer_pct=classifier.extraction_layer / classifier.num_layers,
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all formats
    print("\nPackaging AF vectors...")
    print("=" * 50)
    
    packager.save_safetensors(os.path.join(output_dir, "af_vectors.safetensors"))
    packager.save_npz(os.path.join(output_dir, "af_vectors.npz"))
    packager.save_json_binary(os.path.join(output_dir, "json_binary"))
    
    print("\n" + "=" * 50)
    print(f"All formats saved to: {output_dir}/")
    print("=" * 50)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

EXAMPLE_USAGE = '''
# Example: Package vectors from trained classifier

from af_benchmark_category import AFCategoryClassifier
from af_vector_packaging import package_from_classifier, AFClassifier

# 1. Train classifier (or load trained one)
classifier = AFCategoryClassifier(model, tokenizer, layer_pct=0.65)
classifier.train_all_categories()

# 2. Package for distribution
package_from_classifier(classifier, "af_vectors_gemma3_1b")

# 3. Load and use packaged vectors
af = AFClassifier.from_safetensors("af_vectors_gemma3_1b/af_vectors.safetensors")

# 4. Classify (assuming you have activation extraction)
activation = extract_activation(model, prompt, layer=17)
blocked, triggered, scores = af.classify(activation)

if blocked:
    print(f"BLOCKED: {triggered}")
'''


if __name__ == "__main__":
    print("AF Vector Packaging Module")
    print("=" * 50)
    print("\nThis module provides:")
    print("1. AFVectorPackager - Package vectors in multiple formats")
    print("2. AFVectorLoader - Load vectors from any format")
    print("3. AFClassifier - Simple classifier using loaded vectors")
    print("4. package_from_classifier() - Quick packaging from trained classifier")
    print("\nSupported formats:")
    print("  - .safetensors (HuggingFace standard - recommended)")
    print("  - .npz (NumPy native)")
    print("  - JSON + binary (portable)")
    print("\n" + EXAMPLE_USAGE)
