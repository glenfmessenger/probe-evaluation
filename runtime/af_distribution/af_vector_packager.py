"""
AF Vector Packager
==================
Exports trained vectors from AFCategoryClassifier to distribution format.

Usage:
    # After training with af_benchmark_category_v2.py:
    from af_vector_packager import package_vectors
    package_vectors(classifier, "vectors/gemma-3-1b")

Output:
    vectors/gemma-3-1b.safetensors  - Vector data
    vectors/gemma-3-1b.json         - Metadata

Author: Glen Messenger
Version: 1.0.0
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

try:
    from safetensors.numpy import save_file as save_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


def package_vectors(
    classifier,
    output_path: str,
    model_name: str = None,
    author: str = "Glen Messenger",
    license: str = "Apache-2.0",
    format: str = "safetensors",
) -> Dict[str, str]:
    """
    Package trained vectors for distribution.
    
    Args:
        classifier: Trained AFCategoryClassifier instance
        output_path: Output path without extension (e.g., "vectors/gemma-3-1b")
        model_name: Override model name (default: extracted from classifier)
        author: Author attribution
        license: License identifier
        format: Output format ("safetensors" or "npz")
    
    Returns:
        Dict with paths to created files
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract model info
    if model_name is None:
        model_name = getattr(classifier.model.config, '_name_or_path', 'unknown')
        model_name = model_name.split('/')[-1]  # Remove org prefix
    
    # Build tensors dict
    tensors = {}
    
    # Add category vectors
    for category, vector in classifier.vectors.items():
        tensors[f'vector_{category}'] = vector.astype(np.float32)
        tensors[f'threshold_{category}'] = np.array([classifier.thresholds[category]], dtype=np.float32)
    
    # Add obfuscation vector
    if classifier.obfuscation_vector is not None:
        tensors['vector_obfuscation'] = classifier.obfuscation_vector.astype(np.float32)
        tensors['threshold_obfuscation'] = np.array([classifier.obfuscation_threshold], dtype=np.float32)
        
        # Add high-confidence threshold if available
        if hasattr(classifier, 'obfuscation_high_confidence_threshold'):
            tensors['threshold_obfuscation_high_confidence'] = np.array(
                [classifier.obfuscation_high_confidence_threshold], dtype=np.float32
            )
    
    # Build metadata
    metadata = {
        'version': '1.0.0',
        'model_name': model_name,
        'model_family': model_name.split('-')[0] if '-' in model_name else model_name,
        'hidden_dimension': int(classifier.hidden_dim),
        'extraction_layer': int(classifier.extraction_layer),
        'extraction_layer_pct': float(classifier.extraction_layer / classifier.num_layers),
        'num_layers': int(classifier.num_layers),
        'num_vectors': len(classifier.vectors) + (1 if classifier.obfuscation_vector is not None else 0),
        'categories': list(classifier.vectors.keys()),
        'has_obfuscation': classifier.obfuscation_vector is not None,
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'created_by': author,
        'license': license,
    }
    
    # Add obfuscation config
    if classifier.obfuscation_vector is not None:
        metadata['obfuscation_threshold'] = float(classifier.obfuscation_threshold)
        if hasattr(classifier, 'obfuscation_high_confidence_threshold'):
            metadata['obfuscation_high_confidence_threshold'] = float(
                classifier.obfuscation_high_confidence_threshold
            )
        if hasattr(classifier, 'obf_sensitivity'):
            metadata['obfuscation_sensitivity'] = float(classifier.obf_sensitivity)
    
    # Add category stats
    if hasattr(classifier, 'category_stats'):
        metadata['category_stats'] = {}
        for cat, stats in classifier.category_stats.items():
            metadata['category_stats'][cat] = {
                'separation': float(stats.separation),
                'threshold': float(stats.threshold),
                'train_accuracy': float(stats.train_accuracy),
            }
            if hasattr(stats, 'false_positive_rate'):
                metadata['category_stats'][cat]['false_positive_rate'] = float(stats.false_positive_rate)
            if hasattr(stats, 'false_negative_rate'):
                metadata['category_stats'][cat]['false_negative_rate'] = float(stats.false_negative_rate)
    
    # Calculate total size
    total_bytes = sum(v.nbytes for v in tensors.values())
    metadata['total_size_bytes'] = total_bytes
    
    # Save files
    created_files = {}
    
    if format == "safetensors":
        if not HAS_SAFETENSORS:
            raise ImportError("safetensors not installed. Run: pip install safetensors")
        
        vectors_file = output_path.with_suffix('.safetensors')
        save_safetensors(tensors, str(vectors_file))
        created_files['vectors'] = str(vectors_file)
        
    elif format == "npz":
        vectors_file = output_path.with_suffix('.npz')
        np.savez_compressed(str(vectors_file), **tensors)
        created_files['vectors'] = str(vectors_file)
        
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # Save metadata
    metadata_file = output_path.with_suffix('.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    created_files['metadata'] = str(metadata_file)
    
    print(f"Packaged {metadata['num_vectors']} vectors ({total_bytes / 1024:.1f} KB)")
    print(f"  Vectors: {created_files['vectors']}")
    print(f"  Metadata: {created_files['metadata']}")
    
    return created_files


def package_from_benchmark(
    benchmark_results_path: str,
    classifier,
    output_dir: str = "vectors",
) -> Dict[str, str]:
    """
    Package vectors using benchmark results for metadata enrichment.
    
    Args:
        benchmark_results_path: Path to benchmark JSON results
        classifier: Trained AFCategoryClassifier
        output_dir: Output directory
    
    Returns:
        Dict with paths to created files
    """
    with open(benchmark_results_path, 'r') as f:
        results = json.load(f)
    
    model_name = results['config']['model'].split('/')[-1]
    output_path = Path(output_dir) / model_name
    
    return package_vectors(
        classifier=classifier,
        output_path=str(output_path),
        model_name=model_name,
    )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("AF Vector Packager")
    print("==================")
    print("This script is meant to be imported and used after training.")
    print()
    print("Usage:")
    print("  from af_vector_packager import package_vectors")
    print("  package_vectors(classifier, 'vectors/gemma-3-1b')")
