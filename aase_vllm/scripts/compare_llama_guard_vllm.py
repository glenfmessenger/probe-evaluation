#!/usr/bin/env python3
"""
Llama Guard 3 vs AASE on vLLM — Phase-1 version (paper Table `tab:guard_comparison` rerun on real sets).

Prompt set (DECISIONS.md): harmful = HarmBench per eval_config ``harmbench`` (300), benign = the AF benign sets
(XSTest safe 250 + JBB benign 100).  Both systems see exactly the same prompts; per-case scores/verdicts are written
with run provenance.  The paper's 16 authored prompts are available only via ``--use-legacy-table5-set``.

Llama Guard is prompted through its own chat template (``tokenizer.apply_chat_template`` on the guard model's
tokenizer), which inserts the safety taxonomy the model was trained with; the old script hand-rolled a bare
header without the taxonomy.  Verdict: first line of the generation == "unsafe".

Usage:
    python compare_llama_guard_vllm.py --dry-run
    VLLM_ALLOW_INSECURE_SERIALIZATION=1 python compare_llama_guard_vllm.py            # per eval_config.yaml
    VLLM_ALLOW_INSECURE_SERIALIZATION=1 python compare_llama_guard_vllm.py --skip-llama-guard
"""
import os
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from aase_eval import DataLoadError, load_config, resolve_path
from aase_eval.comparison_set import load_comparison_set, load_legacy_table5_set
from aase_eval.provenance import run_provenance


def classification_metrics(preds: List[bool], labels: List[int]) -> Dict[str, float]:
    tp = sum(1 for p, y in zip(preds, labels) if p and y == 1)
    fp = sum(1 for p, y in zip(preds, labels) if p and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if not p and y == 1)
    tn = sum(1 for p, y in zip(preds, labels) if not p and y == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"accuracy": (tp + tn) / len(labels), "tpr": rec, "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "precision": prec, "f1": (2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": len(labels)}


def load_data(cfg: Dict[str, Any], use_legacy: bool, harmbench_dir: Optional[str]) -> Dict[str, Any]:
    if use_legacy:
        return load_legacy_table5_set(allow_legacy=True)
    lg = dict(cfg["llama_guard_comparison"])
    lg["harmbench"] = cfg["harmbench"]
    return load_comparison_set(lg, cfg["datasets"], harmbench_dir)


def records(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{**r, "label": 1} for r in data["harmful"]] + [{**r, "label": 0} for r in data["benign"]]


# ---------------------------------------------------------------- AASE probe on vLLM
def benchmark_aase_vllm(model_name: str, probes_dir: Path, recs: List[Dict[str, Any]], n_runs: int) -> Dict[str, Any]:
    from vllm import LLM, SamplingParams
    print(f"\n{'=' * 60}\n AASE ({model_name}) on {len(recs)} prompts\n{'=' * 60}")
    llm = LLM(model=model_name, trust_remote_code=True, gpu_memory_utilization=0.8, max_model_len=2048, enforce_eager=True)
    model_safe = model_name.replace("/", "_").replace("-", "_")
    af_path = probes_dir / "af" / model_safe
    if not Path(f"{af_path}.npy").exists():
        raise DataLoadError(f"AF probe not found: {af_path}.npy")
    direction = np.load(f"{af_path}.npy"); direction = direction / np.linalg.norm(direction)
    meta = json.load(open(f"{af_path}.json")); threshold, layer = meta["threshold"], meta["layer_index"]
    sampling = SamplingParams(max_tokens=1, temperature=0.0)
    cache = f"/tmp/aase_llama_guard_cmp_{os.getpid()}.npy"

    def register_hook(model):
        layers = model.language_model.model.layers if hasattr(model, "language_model") else model.model.layers

        def hook_fn(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            np.save(cache, hidden[-1, :].detach().float().cpu().numpy())
        layers[layer].register_forward_hook(hook_fn)
        return {"layer": layer}
    llm.apply_model(register_hook)
    for _ in range(3):
        llm.generate(["Hello world"], sampling)
    per_case, latencies = [], []
    for run in range(n_runs):
        for r in recs:
            if os.path.exists(cache):
                os.remove(cache)
            t0 = time.perf_counter()
            llm.generate([r["prompt"]], sampling)
            latencies.append((time.perf_counter() - t0) * 1000)
            if run == 0:
                act = np.load(cache); act = act / np.linalg.norm(act)
                score = float(np.dot(act, direction))
                per_case.append({"id": r["id"], "label": r["label"], "score": score, "pred_harmful": score > threshold,
                                 "benign_set": r.get("benign_set"), "source": r.get("source")})
    m = classification_metrics([c["pred_harmful"] for c in per_case], [c["label"] for c in per_case])
    out = {"method": "AASE", "model": model_name, "probe_layer": layer, "threshold": threshold,
           "latency_mean_ms": float(np.mean(latencies)), "latency_std_ms": float(np.std(latencies)),
           "latency_p50_ms": float(np.percentile(latencies, 50)), "latency_p99_ms": float(np.percentile(latencies, 99)),
           "n_runs": n_runs, **m, "per_case": per_case}
    del llm
    return out


# ---------------------------------------------------------------- Llama Guard 3 on vLLM
def benchmark_llama_guard_vllm(guard_model: str, recs: List[Dict[str, Any]], n_runs: int) -> Dict[str, Any]:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    print(f"\n{'=' * 60}\n Llama Guard ({guard_model}) on {len(recs)} prompts\n{'=' * 60}")
    tok = AutoTokenizer.from_pretrained(guard_model)
    llm = LLM(model=guard_model, trust_remote_code=True, gpu_memory_utilization=0.8, max_model_len=4096)
    sampling = SamplingParams(max_tokens=20, temperature=0.0)
    formatted = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True) for r in recs]
    for _ in range(3):
        llm.generate([formatted[0]], sampling)
    per_case, latencies = [], []
    for run in range(n_runs):
        for r, f in zip(recs, formatted):
            t0 = time.perf_counter()
            outs = llm.generate([f], sampling)
            latencies.append((time.perf_counter() - t0) * 1000)
            if run == 0:
                text = outs[0].outputs[0].text.strip()
                first = text.splitlines()[0].strip().lower() if text else ""
                per_case.append({"id": r["id"], "label": r["label"], "raw": text[:80], "pred_harmful": first.startswith("unsafe"),
                                 "benign_set": r.get("benign_set"), "source": r.get("source")})
    m = classification_metrics([c["pred_harmful"] for c in per_case], [c["label"] for c in per_case])
    out = {"method": "LlamaGuard3", "model": guard_model, "prompt_format": "tokenizer.apply_chat_template (guard taxonomy included)",
           "latency_mean_ms": float(np.mean(latencies)), "latency_std_ms": float(np.std(latencies)),
           "latency_p50_ms": float(np.percentile(latencies, 50)), "latency_p99_ms": float(np.percentile(latencies, 99)),
           "n_runs": n_runs, **m, "per_case": per_case}
    del llm
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--probes-dir", default=None)
    ap.add_argument("--harmbench-dir", default=None)
    ap.add_argument("--n-runs", type=int, default=None)
    ap.add_argument("--skip-llama-guard", action="store_true")
    ap.add_argument("--output", default=None, help="default: <results_dir>/<run_name>/llama_guard_comparison.json")
    ap.add_argument("--dry-run", action="store_true", help="load and print the prompt set, no model")
    ap.add_argument("--use-legacy-table5-set", action="store_true", help="the paper's 16 authored prompts (explicit, never a fallback)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    lg = cfg["llama_guard_comparison"]
    data = load_data(cfg, args.use_legacy_table5_set, args.harmbench_dir)
    recs = records(data)
    n_runs = args.n_runs or int(lg.get("n_runs", 3))
    out_path = Path(args.output) if args.output else resolve_path(cfg["output"]["results_dir"]) / cfg["output"]["run_name"] / "llama_guard_comparison.json"
    print(f"Comparison set: {json.dumps(data['summary'])}")
    if args.dry_run:
        for r in recs[:2] + recs[-2:]:
            print(f"  [{r['id']}] label={r['label']} {r['prompt'][:140]!r}")
        print(f"output -> {out_path} (schema: provenance, data_summary, aase{{metrics, per_case}}, llama_guard{{metrics, per_case}}, speedup)")
        return 0
    results: Dict[str, Any] = {"provenance": run_provenance(cfg, {"script": "compare_llama_guard_vllm.py"}), "data_summary": data["summary"], "n_runs": n_runs}
    probes_dir = Path(args.probes_dir) if args.probes_dir else resolve_path(cfg["probes"]["dir"])
    results["aase"] = {m: benchmark_aase_vllm(m, probes_dir, recs, n_runs) for m in lg.get("aase_models", ["meta-llama/Llama-3.2-3B-Instruct"])}
    if not args.skip_llama_guard:
        results["llama_guard"] = benchmark_llama_guard_vllm(lg.get("llama_guard_model", "meta-llama/Llama-Guard-3-8B"), recs, n_runs)
        results["speedup"] = {m: results["llama_guard"]["latency_mean_ms"] / v["latency_mean_ms"] for m, v in results["aase"].items()}
    print(f"\n{'Method':<40} {'Latency':>10} {'Acc':>7} {'TPR':>7} {'FPR':>7} {'F1':>6}")
    rows = [(f"AASE + {m.split('/')[-1]}", v) for m, v in results["aase"].items()] + ([("Llama Guard 3", results["llama_guard"])] if "llama_guard" in results else [])
    for name, v in rows:
        print(f"{name:<40} {v['latency_mean_ms']:>8.1f}ms {v['accuracy']:>7.1%} {v['tpr']:>7.1%} {v['fpr']:>7.1%} {v['f1']:>6.2f}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
