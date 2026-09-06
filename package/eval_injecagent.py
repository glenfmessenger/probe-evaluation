#!/usr/bin/env python3
"""
Evaluate AASE on InjecAgent Benchmark

InjecAgent: 1,054 test cases for indirect prompt injection
https://github.com/uiuc-kang-lab/InjecAgent

Usage:
    # First clone InjecAgent
    git clone https://github.com/uiuc-kang-lab/InjecAgent.git
    
    # Then run evaluation
    python eval_injecagent.py --model google/gemma-2-2b-it --probes-dir pretrained/
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
import time
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aase_eval import DataLoadError

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class EvalResult:
    total: int
    detected: int
    missed: int
    detection_rate: float
    scores: List[float]
    

def load_injecagent_data(injecagent_dir: str, setting: str = "base") -> Tuple[List[dict], List[dict]]:
    """Load the 1,054 real paired InjecAgent test cases via aase_eval (raises on any problem; no fallback)."""
    from aase_eval.injecagent import load_test_cases
    cases = load_test_cases(injecagent_dir, setting)
    dh_cases = [c for c in cases if c["split"] == "dh"]
    ds_cases = [c for c in cases if c["split"] == "ds"]
    print(f"Loaded {len(dh_cases)} direct harm cases and {len(ds_cases)} data stealing cases ({setting})")
    if not dh_cases or not ds_cases:
        raise DataLoadError(f"InjecAgent {setting}: dh={len(dh_cases)} ds={len(ds_cases)} — one split is empty")
    return dh_cases, ds_cases


def format_injection_prompt(case: dict, setting: str = "base") -> str:
    """Real user instruction + the real injected tool response (the 'enhanced' setting is already
    baked into the Tool Response by upstream, so no prefix is added here)."""
    from aase_eval.injecagent import format_prompt
    if not case.get("attacker_instruction"):
        raise DataLoadError(f"case {case.get('case_id')} has no attacker instruction; refusing to score an empty injection")
    return format_prompt(case["user_instruction"], case["tool_response"])


def load_benign_prompts(injecagent_dir: str, n: int = None) -> List[str]:
    """Real benign prompts: the InjecAgent user cases with the injection slot removed (17 exist)."""
    from aase_eval.injecagent import load_user_cases, build_benign_prompts
    recs = build_benign_prompts(load_user_cases(injecagent_dir))
    if n is not None:
        recs = recs[:n]
    if not recs:
        raise DataLoadError("no benign InjecAgent user cases loaded")
    return [r["prompt"] for r in recs]


class AASEEvaluator:
    """Evaluate AASE probes."""
    
    def __init__(self, model_name: str, probes_dir: str, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading {model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Auto-select dtype
        if "gemma-3" in model_name.lower():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self.device = device
        
        # Get num layers
        config = self.model.config
        if hasattr(config, 'text_config'):
            self.num_layers = config.text_config.num_hidden_layers
        else:
            self.num_layers = config.num_hidden_layers
        
        # Load AAG probe
        model_safe = model_name.replace("/", "_").replace("-", "_")
        aag_path = Path(probes_dir) / "aag" / f"{model_safe}.npy"
        aag_json = aag_path.with_suffix(".json")
        
        if not aag_path.exists():
            raise FileNotFoundError(f"AAG probe not found: {aag_path}")
        
        self.aag_direction = np.load(aag_path)
        self.aag_direction = self.aag_direction / np.linalg.norm(self.aag_direction)
        
        with open(aag_json) as f:
            meta = json.load(f)
        self.aag_threshold = meta.get("threshold", 0.0)
        self.aag_layer = meta.get("layer_index", int(0.55 * self.num_layers))
        
        print(f"Loaded AAG probe: layer={self.aag_layer}, threshold={self.aag_threshold:.4f}")
    
    def extract_activation(self, text: str, layer: int) -> np.ndarray:
        """Extract activation from specified layer."""
        import torch
        
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        hidden = outputs.hidden_states[layer]
        last_pos = inputs.attention_mask.sum(dim=1) - 1
        return hidden[0, last_pos[0], :].float().cpu().numpy()
    
    def score_prompt(self, prompt: str) -> float:
        """Get AAG score for a prompt."""
        act = self.extract_activation(prompt, self.aag_layer)
        act_norm = act / np.linalg.norm(act)
        return float(np.dot(act_norm, self.aag_direction))
    
    def is_injection(self, prompt: str) -> Tuple[bool, float]:
        """Check if prompt is detected as injection."""
        score = self.score_prompt(prompt)
        return score > self.aag_threshold, score


def evaluate_injecagent(
    evaluator: AASEEvaluator,
    dh_cases: List[dict],
    ds_cases: List[dict],
    setting: str = "base",
    max_cases: int = None,
) -> Dict[str, EvalResult]:
    """Evaluate on InjecAgent cases."""
    
    results = {}
    
    for attack_type, cases in [("direct_harm", dh_cases), ("data_stealing", ds_cases)]:
        if max_cases:
            cases = cases[:max_cases]
        
        detected = 0
        scores = []
        
        print(f"\nEvaluating {attack_type} ({len(cases)} cases)...")
        
        for i, case in enumerate(cases):
            prompt = format_injection_prompt(case, setting)
            is_inj, score = evaluator.is_injection(prompt)
            
            scores.append(score)
            if is_inj:
                detected += 1
            
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(cases)} - Detection rate: {detected/(i+1):.1%}")
        
        results[attack_type] = EvalResult(
            total=len(cases),
            detected=detected,
            missed=len(cases) - detected,
            detection_rate=detected / len(cases) if cases else 0,
            scores=scores,
        )
    
    return results


def evaluate_benign(evaluator: AASEEvaluator, benign_prompts: List[str]) -> EvalResult:
    """Evaluate false positive rate on benign prompts."""
    
    false_positives = 0
    scores = []
    
    print(f"\nEvaluating benign prompts ({len(benign_prompts)} cases)...")
    
    for i, prompt in enumerate(benign_prompts):
        is_inj, score = evaluator.is_injection(prompt)
        scores.append(score)
        if is_inj:
            false_positives += 1
        
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(benign_prompts)} - FPR: {false_positives/(i+1):.1%}")
    
    return EvalResult(
        total=len(benign_prompts),
        detected=false_positives,  # False positives in this case
        missed=len(benign_prompts) - false_positives,
        detection_rate=false_positives / len(benign_prompts),  # This is FPR
        scores=scores,
    )


def compute_metrics_at_thresholds(
    injection_scores: List[float],
    benign_scores: List[float],
    thresholds: List[float] = None,
) -> Dict[str, List[float]]:
    """Compute TPR/FPR at various thresholds for ROC curve."""
    
    if thresholds is None:
        all_scores = injection_scores + benign_scores
        thresholds = np.linspace(min(all_scores), max(all_scores), 100)
    
    tprs = []
    fprs = []
    
    for thresh in thresholds:
        tp = sum(1 for s in injection_scores if s > thresh)
        fp = sum(1 for s in benign_scores if s > thresh)
        
        tpr = tp / len(injection_scores) if injection_scores else 0
        fpr = fp / len(benign_scores) if benign_scores else 0
        
        tprs.append(tpr)
        fprs.append(fpr)
    
    # Compute AUC
    auc = 0
    for i in range(1, len(fprs)):
        auc += (fprs[i-1] - fprs[i]) * (tprs[i-1] + tprs[i]) / 2
    
    return {
        "thresholds": list(thresholds),
        "tpr": tprs,
        "fpr": fprs,
        "auc": auc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--probes-dir", default="pretrained")
    parser.add_argument("--injecagent-dir", default=None, help="InjecAgent clone (default: eval_config.yaml datasets.injecagent_dir)")
    parser.add_argument("--setting", choices=["base", "enhanced"], default="base")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit cases per split for testing")
    parser.add_argument("--max-benign", type=int, default=None, help="Limit benign user cases (default: all 17)")
    parser.add_argument("--output", default="injecagent_results.json")
    
    args = parser.parse_args()
    if args.injecagent_dir is None:
        from aase_eval import load_config, resolve_path
        args.injecagent_dir = str(resolve_path(load_config()["datasets"]["injecagent_dir"]))
    print("=" * 70)
    print(" AASE Evaluation on InjecAgent Benchmark")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Setting: {args.setting}")
    
    # Load InjecAgent data
    dh_cases, ds_cases = load_injecagent_data(args.injecagent_dir, args.setting)  # raises DataLoadError if anything is wrong
    benign_prompts = load_benign_prompts(args.injecagent_dir, args.max_benign)
    
    # Initialize evaluator
    evaluator = AASEEvaluator(args.model, args.probes_dir)
    
    # Evaluate on injection cases
    inj_results = evaluate_injecagent(
        evaluator, dh_cases, ds_cases, args.setting, args.max_cases
    )
    
    # Evaluate on benign cases
    benign_result = evaluate_benign(evaluator, benign_prompts)
    
    # Compute ROC metrics
    all_injection_scores = []
    for r in inj_results.values():
        all_injection_scores.extend(r.scores)
    
    roc_metrics = compute_metrics_at_thresholds(
        all_injection_scores, benign_result.scores
    )
    
    # Print results
    print("\n" + "=" * 70)
    print(" RESULTS")
    print("=" * 70)
    
    print(f"\nInjection Detection (setting={args.setting}):")
    for attack_type, result in inj_results.items():
        print(f"  {attack_type}:")
        print(f"    Detection Rate: {result.detection_rate:.1%} ({result.detected}/{result.total})")
        print(f"    Mean Score: {np.mean(result.scores):.4f}")
    
    total_detected = sum(r.detected for r in inj_results.values())
    total_cases = sum(r.total for r in inj_results.values())
    overall_detection = total_detected / total_cases if total_cases else 0
    
    print(f"\n  Overall Detection Rate: {overall_detection:.1%} ({total_detected}/{total_cases})")
    
    print(f"\nFalse Positive Rate:")
    print(f"  FPR: {benign_result.detection_rate:.1%} ({benign_result.detected}/{benign_result.total})")
    print(f"  Mean Score: {np.mean(benign_result.scores):.4f}")
    
    print(f"\nROC Metrics:")
    print(f"  AUC: {roc_metrics['auc']:.4f}")
    
    # Save results
    output = {
        "model": args.model,
        "setting": args.setting,
        "threshold": evaluator.aag_threshold,
        "injection_results": {
            k: {
                "total": v.total,
                "detected": v.detected,
                "detection_rate": v.detection_rate,
                "mean_score": float(np.mean(v.scores)),
                "std_score": float(np.std(v.scores)),
            }
            for k, v in inj_results.items()
        },
        "benign_results": {
            "total": benign_result.total,
            "false_positives": benign_result.detected,
            "fpr": benign_result.detection_rate,
            "mean_score": float(np.mean(benign_result.scores)),
        },
        "overall": {
            "detection_rate": overall_detection,
            "fpr": benign_result.detection_rate,
            "auc": roc_metrics["auc"],
        },
        "data": {"injecagent_dir": args.injecagent_dir, "n_injections": total_cases, "n_benign": benign_result.total,
                 "benign_source": "InjecAgent user_cases.jsonl (injection slot removed)"},
        "per_case_scores": {k: v.scores for k, v in inj_results.items()},
        "benign_scores": benign_result.scores,
    }
    
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
