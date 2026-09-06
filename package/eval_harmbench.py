#!/usr/bin/env python3
"""
Evaluate AASE AF probe on HarmBench Benchmark

HarmBench: 400+ harmful behaviors across categories:
- Chemical/Biological weapons
- Cybercrime
- Harassment/Discrimination
- Illegal activities
- Misinformation
- Copyright violations
- Malware

Usage:
    # First clone HarmBench
    git clone https://github.com/centerforaisafety/HarmBench.git
    
    # Or use JailbreakBench (smaller, 100 behaviors)
    pip install jailbreakbench
    
    # Then run evaluation
    python eval_harmbench.py --model google/gemma-2-2b-it --probes-dir pretrained/
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass
import csv

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# The hand-written HarmBench-style / benign lists that used to live here (56 + 31 prompts)
# were moved to aase_eval/legacy_authored_eval_set.json. They are loadable ONLY with
# --use-legacy-authored-set and are never used as a fallback for missing HarmBench data.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aase_eval import DataLoadError, load_config, resolve_path
from aase_eval.harmbench import load_harmbench_behaviors, load_benign_set, harmbench_breakdown
from aase_eval.legacy import load_legacy_authored_set


def load_harmbench_data(harmbench_dir: str, subset: str = "all",
                        functional_categories=("standard", "contextual")) -> Dict[str, List[str]]:
    """Load real HarmBench behaviors grouped by SemanticCategory. Raises DataLoadError; no fallback."""
    records = load_harmbench_behaviors(harmbench_dir, subset, tuple(functional_categories))
    by_cat: Dict[str, List[str]] = {}
    for r in records:
        by_cat.setdefault(r["semantic_category"], []).append(r["prompt"])
    bd = harmbench_breakdown(records)
    print(f"HarmBench loaded: {bd['n']} behaviors; functional={bd['by_functional_category']}; semantic={bd['by_semantic_category']}")
    return by_cat


def try_load_jailbreakbench() -> Tuple[List[str], List[str]]:
    """Load JailbreakBench JBB-Behaviors (harmful + benign) from the pinned CSVs in third_party/. Raises; no fallback."""
    from aase_eval.harmbench import load_jbb_benign, _read_csv, JBB_COLUMNS
    cfg = load_config()
    harm_rows = _read_csv(resolve_path(cfg["datasets"]["jbb_benign_csv"]).with_name("jbb_harmful_behaviors.csv"), JBB_COLUMNS, "JBB harmful")
    harmful = [r["Goal"].strip() for r in harm_rows]
    benign = [r["prompt"] for r in load_jbb_benign(resolve_path(cfg["datasets"]["jbb_benign_csv"]))]
    return harmful, benign


@dataclass
class CategoryResult:
    category: str
    total: int
    detected: int
    detection_rate: float
    mean_score: float
    std_score: float
    scores: List[float]


@dataclass 
class EvalResults:
    harmful_results: Dict[str, CategoryResult]
    benign_result: CategoryResult
    overall_detection_rate: float
    overall_fpr: float
    auc: float


class AFEvaluator:
    """Evaluate AF probe on HarmBench."""
    
    def __init__(self, model_name: str, probes_dir: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading {model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        dtype = torch.bfloat16 if "gemma-3" in model_name.lower() else torch.float16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        
        # Load AF probe
        model_safe = model_name.replace("/", "_").replace("-", "_")
        af_path = Path(probes_dir) / "af" / f"{model_safe}"
        
        if not (af_path.with_suffix(".npy")).exists():
            raise FileNotFoundError(f"AF probe not found: {af_path}.npy")
        
        self.direction = np.load(f"{af_path}.npy")
        self.direction = self.direction / np.linalg.norm(self.direction)
        
        with open(f"{af_path}.json") as f:
            meta = json.load(f)
        
        self.threshold = meta.get("threshold", 0.0)
        self.layer = meta.get("layer_index", 13)
        
        print(f"Loaded AF probe: layer={self.layer}, threshold={self.threshold:.4f}")
    
    def get_score(self, text: str) -> float:
        """Get AF score for a text."""
        import torch
        
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        hidden = outputs.hidden_states[self.layer]
        last_pos = inputs.attention_mask.sum(dim=1) - 1
        act = hidden[0, last_pos[0], :].float().cpu().numpy()
        act = act / np.linalg.norm(act)
        
        return float(np.dot(act, self.direction))
    
    def is_harmful(self, text: str) -> Tuple[bool, float]:
        """Check if text is detected as harmful."""
        score = self.get_score(text)
        return score > self.threshold, score


def compute_auc(harmful_scores: List[float], benign_scores: List[float]) -> float:
    """Compute AUC-ROC."""
    try:
        from sklearn.metrics import roc_auc_score
        y_true = [1] * len(harmful_scores) + [0] * len(benign_scores)
        y_scores = harmful_scores + benign_scores
        return roc_auc_score(y_true, y_scores)
    except:
        # Simple approximation
        harmful_mean = np.mean(harmful_scores)
        benign_mean = np.mean(benign_scores)
        return 0.5 + 0.5 * np.sign(harmful_mean - benign_mean)


def evaluate_harmbench(
    evaluator: AFEvaluator,
    harmful_behaviors: Dict[str, List[str]],
    benign_behaviors: List[str],
) -> EvalResults:
    """Run full HarmBench evaluation."""
    
    harmful_results = {}
    all_harmful_scores = []
    
    # Evaluate harmful behaviors by category
    for category, behaviors in harmful_behaviors.items():
        print(f"\nEvaluating {category} ({len(behaviors)} behaviors)...")
        
        scores = []
        detected = 0
        
        for behavior in behaviors:
            is_harm, score = evaluator.is_harmful(behavior)
            scores.append(score)
            if is_harm:
                detected += 1
        
        all_harmful_scores.extend(scores)
        
        harmful_results[category] = CategoryResult(
            category=category,
            total=len(behaviors),
            detected=detected,
            detection_rate=detected / len(behaviors) if behaviors else 0,
            mean_score=float(np.mean(scores)),
            std_score=float(np.std(scores)),
            scores=scores,
        )
        
        print(f"  Detection: {detected}/{len(behaviors)} ({detected/len(behaviors)*100:.1f}%)")
    
    # Evaluate benign behaviors
    print(f"\nEvaluating benign behaviors ({len(benign_behaviors)})...")
    
    benign_scores = []
    false_positives = 0
    
    for behavior in benign_behaviors:
        is_harm, score = evaluator.is_harmful(behavior)
        benign_scores.append(score)
        if is_harm:
            false_positives += 1
    
    benign_result = CategoryResult(
        category="benign",
        total=len(benign_behaviors),
        detected=false_positives,
        detection_rate=false_positives / len(benign_behaviors) if benign_behaviors else 0,
        mean_score=float(np.mean(benign_scores)),
        std_score=float(np.std(benign_scores)),
        scores=benign_scores,
    )
    
    print(f"  FPR: {false_positives}/{len(benign_behaviors)} ({false_positives/len(benign_behaviors)*100:.1f}%)")
    
    # Compute overall metrics
    total_harmful = sum(r.total for r in harmful_results.values())
    total_detected = sum(r.detected for r in harmful_results.values())
    overall_detection = total_detected / total_harmful if total_harmful else 0
    overall_fpr = benign_result.detection_rate
    
    auc = compute_auc(all_harmful_scores, benign_scores)
    
    return EvalResults(
        harmful_results=harmful_results,
        benign_result=benign_result,
        overall_detection_rate=overall_detection,
        overall_fpr=overall_fpr,
        auc=auc,
    )


def print_results(results: EvalResults, model_name: str):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print(" HARMBENCH EVALUATION RESULTS")
    print("=" * 70)
    print(f"Model: {model_name}")
    
    print(f"\n{'Category':<30} {'Detection':>12} {'Mean Score':>12}")
    print("-" * 56)
    
    for category, result in sorted(results.harmful_results.items()):
        det_str = f"{result.detected}/{result.total} ({result.detection_rate*100:.0f}%)"
        print(f"{category:<30} {det_str:>12} {result.mean_score:>12.4f}")
    
    print("-" * 56)
    total_harmful = sum(r.total for r in results.harmful_results.values())
    total_detected = sum(r.detected for r in results.harmful_results.values())
    print(f"{'OVERALL HARMFUL':<30} {total_detected}/{total_harmful} ({results.overall_detection_rate*100:.1f}%)")
    
    print(f"\n{'Benign (FPR)':<30} {results.benign_result.detected}/{results.benign_result.total} ({results.overall_fpr*100:.1f}%)")
    
    print(f"\n{'Metric':<30} {'Value':>12}")
    print("-" * 44)
    print(f"{'Overall Detection Rate':<30} {results.overall_detection_rate*100:>11.1f}%")
    print(f"{'False Positive Rate':<30} {results.overall_fpr*100:>11.1f}%")
    print(f"{'AUC-ROC':<30} {results.auc:>12.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--probes-dir", default="pretrained")
    parser.add_argument("--harmbench-dir", default=None, help="HarmBench clone (default: eval_config.yaml datasets.harmbench_dir)")
    parser.add_argument("--config", default=None, help="eval_config.yaml")
    parser.add_argument("--benign-set", default=None, choices=["jbb_benign", "xstest_safe"], help="override harmbench.benign_set")
    parser.add_argument("--use-legacy-authored-set", action="store_true",
                        help="score the old hand-written 56+31 list instead of HarmBench (explicit; never a fallback)")
    parser.add_argument("--use-jailbreakbench", action="store_true", 
                        help="Use JailbreakBench dataset instead")
    parser.add_argument("--output", default="harmbench_results.json")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(" AASE AF Probe Evaluation on HarmBench")
    print("=" * 70)
    
    cfg = load_config(args.config)
    # Load data — every branch raises DataLoadError on failure; nothing falls back.
    if args.use_legacy_authored_set:
        legacy = load_legacy_authored_set(allow_legacy=True)
        print("\nWARNING: scoring the LEGACY AUTHORED 56+31 list, not HarmBench. Report it as such.")
        harmful_behaviors = {}
        for r in legacy["harmful"]:
            harmful_behaviors.setdefault(r["semantic_category"], []).append(r["prompt"])
        benign_behaviors = [r["prompt"] for r in legacy["benign"]]
        data_source = "legacy_authored"
    elif args.use_jailbreakbench:
        print("\nLoading JailbreakBench dataset...")
        harmful, benign = try_load_jailbreakbench()
        harmful_behaviors = {"jailbreakbench": harmful}
        benign_behaviors = benign
        data_source = "jailbreakbench"
    else:
        print("\nLoading HarmBench data...")
        hb_dir = args.harmbench_dir or str(resolve_path(cfg["datasets"]["harmbench_dir"]))
        harmful_behaviors = load_harmbench_data(hb_dir, cfg["harmbench"].get("subset", "all"),
                                                cfg["harmbench"].get("functional_categories", ("standard", "contextual")))
        benign_names = [args.benign_set] if args.benign_set else list(cfg["harmbench"].get("benign_sets") or [cfg["harmbench"]["benign_set"]])
        benign_behaviors = [r["prompt"] for name in benign_names for r in load_benign_set(name, cfg["datasets"])]
        benign_name = "+".join(benign_names)
        data_source = f"harmbench:{cfg['harmbench'].get('subset', 'all')} + benign:{benign_name}"
    print(f"Data source: {data_source}")
    total_harmful = sum(len(v) for v in harmful_behaviors.values())
    print(f"Loaded {total_harmful} harmful behaviors across {len(harmful_behaviors)} categories")
    print(f"Loaded {len(benign_behaviors)} benign behaviors")
    
    # Initialize evaluator
    evaluator = AFEvaluator(args.model, args.probes_dir)
    
    # Run evaluation
    results = evaluate_harmbench(evaluator, harmful_behaviors, benign_behaviors)
    
    # Print results
    print_results(results, args.model)
    
    # Save results
    output = {
        "model": args.model,
        "threshold": evaluator.threshold,
        "harmful_results": {
            cat: {
                "total": r.total,
                "detected": r.detected,
                "detection_rate": r.detection_rate,
                "mean_score": r.mean_score,
                "std_score": r.std_score,
            }
            for cat, r in results.harmful_results.items()
        },
        "benign_results": {
            "total": results.benign_result.total,
            "false_positives": results.benign_result.detected,
            "fpr": results.benign_result.detection_rate,
            "mean_score": results.benign_result.mean_score,
        },
        "overall": {
            "detection_rate": results.overall_detection_rate,
            "fpr": results.overall_fpr,
            "auc": results.auc,
        },
    }
    
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
