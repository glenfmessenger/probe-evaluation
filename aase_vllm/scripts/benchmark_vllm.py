#!/usr/bin/env python3
"""
AASE Benchmark Suite - vLLM Backend (Phase-0 repaired version)

Runs the AF (HarmBench) and AAG (InjecAgent) probe evaluations and the latency
benchmark on vLLM.  Every prompt scored comes from real data loaded through
``aase_eval`` according to ``eval_config.yaml``; there is no synthetic or
authored fallback anywhere in this file.  If data cannot be loaded the run
raises ``aase_eval.DataLoadError`` naming the file and the keys involved.

Results are written per model as JSON with **per-case scores, labels, case ids
and (by default) prompt text**, so AUC/ROC can be recomputed and audited from
the results file alone (``generate_roc_curves_vllm.py`` consumes them).

Usage:
    # data-path check without any model (CPU, no vLLM import):
    python benchmark_vllm.py --dry-run

    VLLM_ALLOW_INSECURE_SERIALIZATION=1 python benchmark_vllm.py --model google/gemma-2-2b-it
    VLLM_ALLOW_INSECURE_SERIALIZATION=1 python benchmark_vllm.py --all-models --benchmark injecagent
    python benchmark_vllm.py --config /path/to/eval_config.yaml --all-models

    # reproduce the old authored 56+31 AF list (never a fallback, always explicit):
    python benchmark_vllm.py --model ... --benchmark harmbench --use-legacy-authored-set

Author: Glen Messenger (original), Phase-0 repairs 2026-09-02
"""
import os
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
import argparse
import hashlib
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
import gc

# Make the repo root importable (aase_eval lives there)
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent))

from aase_eval import DataLoadError, load_config, resolve_path
from aase_eval.injecagent import load_injecagent_eval_set
from aase_eval.harmbench import load_harmbench_eval_set
from aase_eval.legacy import load_legacy_authored_set
from aase_eval.benign_set import fpr_breakdown
from aase_eval.ailuminate import load_ailuminate_eval_set
from aase_eval.comparison_set import load_comparison_set, load_legacy_table5_set
from aase_eval.provenance import run_provenance


# ============================================================
# Metrics (numpy only; no sklearn dependency)
# ============================================================

def auc_mann_whitney(pos: np.ndarray, neg: np.ndarray) -> float:
    """Exact ROC-AUC via the Mann–Whitney U statistic (ties count 1/2)."""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("AUC needs at least one positive and one negative score")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def summarise_scores(pos_records: List[Dict[str, Any]], neg_records: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    pos = np.array([r["score"] for r in pos_records])
    neg = np.array([r["score"] for r in neg_records])
    tp = int((pos > threshold).sum())
    fp = int((neg > threshold).sum())
    return {
        "threshold": float(threshold),
        "n_positive": int(len(pos)),
        "n_negative": int(len(neg)),
        "detection_rate": tp / len(pos),
        "fpr": fp / len(neg),
        "auc": auc_mann_whitney(pos, neg),
        "positive_mean": float(pos.mean()), "positive_std": float(pos.std()),
        "negative_mean": float(neg.mean()), "negative_std": float(neg.std()),
        "n_distinct_positive_scores": int(len(np.unique(np.round(pos, 6)))),
        "n_distinct_negative_scores": int(len(np.unique(np.round(neg, 6)))),
    }


@dataclass
class BenchmarkResult:
    model: str
    benchmark: str
    detection_rate: float
    fpr: float
    auc: float
    latency_ms: float
    n_positive: int
    n_negative: int
    details: Dict[str, Any]
    data_summary: Dict[str, Any] = field(default_factory=dict)
    per_case: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================
# Data loading (the only place prompts come from)
# ============================================================

def _prompt_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_aag_data(cfg: Dict[str, Any], injecagent_dir: Optional[str] = None) -> Dict[str, Any]:
    return load_injecagent_eval_set(cfg["injecagent"], injecagent_dir or resolve_path(cfg["datasets"]["injecagent_dir"]))


def load_ailuminate_data(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return load_ailuminate_eval_set(cfg["ailuminate"], cfg["datasets"])


def load_llama_guard_data(cfg: Dict[str, Any], harmbench_dir: Optional[str] = None, use_legacy_table5_set: bool = False) -> Dict[str, Any]:
    if use_legacy_table5_set:
        return load_legacy_table5_set(allow_legacy=True)
    lg = dict(cfg["llama_guard_comparison"])
    lg["harmbench"] = cfg["harmbench"]
    return load_comparison_set(lg, cfg["datasets"], harmbench_dir)


def load_af_data(cfg: Dict[str, Any], harmbench_dir: Optional[str] = None, use_legacy_authored_set: bool = False) -> Dict[str, Any]:
    if use_legacy_authored_set:
        legacy = load_legacy_authored_set(allow_legacy=True)
        return {"harmful": legacy["harmful"], "benign": legacy["benign"],
                "summary": {"source": "legacy_authored (NOT HarmBench)", "n_harmful": len(legacy["harmful"]), "n_benign": len(legacy["benign"])}}
    return load_harmbench_eval_set(cfg["harmbench"], cfg["datasets"], harmbench_dir)


def _strip_for_results(records: List[Dict[str, Any]], scores: List[float], record_text: bool) -> List[Dict[str, Any]]:
    out = []
    for r, s in zip(records, scores):
        row = {k: v for k, v in r.items() if k != "prompt"}
        row["prompt_sha256_16"] = _prompt_id(r["prompt"])
        if record_text:
            row["prompt"] = r["prompt"]
        row["score"] = float(s)
        out.append(row)
    return out


# ============================================================
# Dry run: the full data path, no model
# ============================================================

def dry_run(cfg: Dict[str, Any], benchmarks: List[str], use_legacy_authored_set: bool = False,
            injecagent_dir: Optional[str] = None, harmbench_dir: Optional[str] = None, n_samples: int = 3,
            sample_width: int = 160, use_legacy_table5_set: bool = False) -> Dict[str, Any]:
    """Load, pair and format every split exactly as a real run would, then print counts and samples."""
    report: Dict[str, Any] = {"config": cfg.get("_config_path"), "models": cfg["models"]}
    print(f"\n{'=' * 70}\n DRY RUN — data path only (no model, no vLLM)\n{'=' * 70}")
    print(f" config: {cfg.get('_config_path')}")
    print(f" models ({len(cfg['models'])}): {', '.join(cfg['models'])}")

    def pick_samples(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stratified sample when records carry subset tags (1 injecagent_user, 2 agentdojo, 2 authored hard negatives)."""
        if not records or "subset" not in records[0]:
            return records[:n_samples]
        quota = [("injecagent_user", lambda r: True, 1), ("agentdojo", lambda r: r.get("hard_negative"), 1),
                 ("agentdojo", lambda r: not r.get("hard_negative"), 1), ("authored", lambda r: r.get("hard_negative"), 2)]
        out = []
        for subset, cond, k in quota:
            out += [r for r in records if r["subset"] == subset and cond(r) and r not in out][:k]
        return out or records[:n_samples]

    def show(name: str, records: List[Dict[str, Any]]):
        print(f"\n --- {name}: {len(records)} records, {len({r['prompt'] for r in records})} distinct prompts")
        if records and "setting" in records[0]:
            import collections
            print(f"     by setting: {dict(collections.Counter(r['setting'] for r in records))}")
        if records and "subset" in records[0]:
            import collections
            print(f"     subsets: {dict(collections.Counter(r['subset'] for r in records))}; hard negatives: {sum(1 for r in records if r.get('hard_negative'))}")
        if records and "benign_set" in records[0]:
            import collections
            print(f"     by benign set: {dict(collections.Counter(r['benign_set'] for r in records))}")
        for r in pick_samples(records):
            meta = {k: v for k, v in r.items() if k not in ("prompt",)}
            print(f"   [{r['id']}] {meta}")
            for line in r["prompt"].splitlines()[:8]:
                print(f"      | {line[:sample_width]}")
            if len(r["prompt"].splitlines()) > 8:
                print("      | ...")

    if "injecagent" in benchmarks:
        data = load_aag_data(cfg, injecagent_dir)
        print(f"\n [InjecAgent / AAG] {json.dumps(data['summary'], indent=None)[:600]}")
        show("injections (label 1)", data["injections"])
        show("benign cases (label 0)", data["benign"])
        report["injecagent"] = data["summary"]
    if "harmbench" in benchmarks:
        data = load_af_data(cfg, harmbench_dir, use_legacy_authored_set)
        print(f"\n [HarmBench / AF] {json.dumps(data['summary'], indent=None)[:800]}")
        show("harmful behaviors (label 1)", data["harmful"])
        show(f"benign counterpart (label 0)", data["benign"])
        report["harmbench"] = data["summary"]
    if "ailuminate" in benchmarks:
        data = load_ailuminate_data(cfg)
        print(f"\n [AILuminate / AF obfuscation] {json.dumps(data['summary'], indent=None)[:600]}")
        import collections
        print(f"     by encoding: {dict(collections.Counter(r['encoding'] for r in data['records']))}")
        for enc in data["summary"]["encodings"]:
            show(f"AILuminate encoding={enc} (label 1)", [r for r in data["records"] if r["encoding"] == enc])
        show("AILuminate benign-encoded control (label 0, all encodings)", data.get("benign_records", []))
        print(f"     models for this leg: {cfg['ailuminate'].get('models') or cfg['models']}")
        report["ailuminate"] = data["summary"]
    if "llama_guard" in benchmarks:
        data = load_llama_guard_data(cfg, harmbench_dir, use_legacy_table5_set)
        print(f"\n [Llama Guard comparison] {json.dumps(data['summary'], indent=None)[:600]}")
        print("     (Llama Guard prompts are wrapped with the guard model's own chat template at run time; shown raw here)")
        show("comparison harmful (label 1)", data["harmful"])
        show("comparison benign (label 0)", data["benign"])
        report["llama_guard"] = data["summary"]
    if "latency" in benchmarks:
        print("\n [latency] no data needed; skipped in dry run")
    out_dir = resolve_path(cfg["output"]["results_dir"]) / cfg["output"]["run_name"]
    print(f"\n output: one JSON per (benchmark, model) at {out_dir}/<benchmark>/<model_safe>.json with keys model, config, timestamp, provenance{{git, eval_config_sha256, datasets_sha256, packages, cuda}},"
          f" harmbench{{detection_rate, fpr, auc, details{{by_benign_set, by_semantic_category}}, data_summary, per_case[]}}, injecagent{{..., details{{by_setting, by_attack_type, fpr_breakdown}}, per_case[]}},"
          f" ailuminate{{details{{by_encoding, by_hazard}}, per_case[]}}, latency{{...}}; Llama Guard comparison: compare_llama_guard_vllm.py -> {out_dir}/llama_guard_comparison.json")
    report["output_dir"] = str(out_dir)
    print(f"\n{'=' * 70}\n DRY RUN OK\n{'=' * 70}")
    return report


# ============================================================
# The benchmark itself (needs vLLM + GPU)
# ============================================================

class VLLMBenchmark:
    """Run AASE benchmarks using the vLLM backend."""

    def __init__(self, model_name: str, cfg: Dict[str, Any], probes_dir: Optional[str] = None):
        self.model_name = model_name
        self.cfg = cfg
        self.probes_dir = Path(probes_dir) if probes_dir else resolve_path(cfg["probes"]["dir"])
        self.record_text = bool(cfg["output"].get("record_prompt_text", True))
        self.llm = None
        self.af_direction = None
        self.aag_direction = None

    # ---- model / probes -------------------------------------------------
    def setup(self):
        from vllm import LLM, SamplingParams
        print(f"\n{'=' * 60}\n Loading {self.model_name}\n{'=' * 60}")
        self.llm = LLM(model=self.model_name, trust_remote_code=True, gpu_memory_utilization=0.8,
                       max_model_len=2048, enforce_eager=True)
        hf_config = self.llm.llm_engine.model_config.hf_config
        if hasattr(hf_config, "text_config"):
            self.num_layers = hf_config.text_config.num_hidden_layers
            self.hidden_dim = hf_config.text_config.hidden_size
        else:
            self.num_layers = hf_config.num_hidden_layers
            self.hidden_dim = hf_config.hidden_size
        print(f"  Layers: {self.num_layers}\n  Hidden dim: {self.hidden_dim}")

        overrides = (self.cfg["probes"].get("overrides") or {}).get(self.model_name, {})
        model_safe = self.model_name.replace("/", "_").replace("-", "_")
        for probe in ("af", "aag"):
            path = self.probes_dir / probe / model_safe
            if not Path(f"{path}.npy").exists():
                print(f"  {probe.upper()} probe not found at {path}.npy — {probe} benchmark will be skipped")
                continue
            direction = np.load(f"{path}.npy")
            direction = direction / np.linalg.norm(direction)
            with open(f"{path}.json") as f:
                meta = json.load(f)
            ov = overrides.get(probe, {})
            layer = int(ov.get("layer_index", meta["layer_index"]))
            threshold = float(ov.get("threshold", meta["threshold"]))
            setattr(self, f"{probe}_direction", direction)
            setattr(self, f"{probe}_meta", meta)
            setattr(self, f"{probe}_layer", layer)
            setattr(self, f"{probe}_threshold", threshold)
            print(f"  {probe.upper()} probe loaded: layer {layer}, threshold {threshold:.4f}"
                  f"{' (override)' if ov else ''}; trained n_pos={meta.get('n_positive')} n_neg={meta.get('n_negative')}")
        self.sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
        self.activation_cache = f"/tmp/aase_benchmark_activation_{os.getpid()}.npy"

    def _register_hook(self, layer_index: int):
        cache_file = self.activation_cache

        def register_hook(model):
            import numpy as np
            if hasattr(model, "language_model"):
                layers = model.language_model.model.layers
            elif hasattr(model, "model") and hasattr(model.model, "layers"):
                layers = model.model.layers
            else:
                raise AttributeError(f"Cannot find layers in model: {type(model)}")

            def hook_fn(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                np.save(cache_file, hidden[-1, :].detach().float().cpu().numpy())

            layers[layer_index].register_forward_hook(hook_fn)
            return {"layer": layer_index}

        return self.llm.apply_model(register_hook)

    def _extract_activation(self, text: str) -> np.ndarray:
        if os.path.exists(self.activation_cache):
            os.remove(self.activation_cache)
        self.llm.generate([text], self.sampling_params)
        if not os.path.exists(self.activation_cache):
            raise RuntimeError("Activation not captured")
        return np.load(self.activation_cache)

    @staticmethod
    def _score(activation: np.ndarray, direction: np.ndarray) -> float:
        act_norm = activation / np.linalg.norm(activation)
        return float(np.dot(act_norm, direction))

    def _score_records(self, records: List[Dict[str, Any]], direction: np.ndarray, label: str) -> List[float]:
        scores = []
        for i, r in enumerate(records):
            scores.append(self._score(self._extract_activation(r["prompt"]), direction))
            if (i + 1) % 50 == 0 or i + 1 == len(records):
                print(f"    {label}: {i + 1}/{len(records)}")
        return scores

    # ---- AF / HarmBench --------------------------------------------------
    def run_harmbench(self, data: Dict[str, Any]) -> BenchmarkResult:
        if self.af_direction is None:
            raise ValueError("AF probe not loaded")
        print(f"\n{'-' * 50}\n HarmBench Evaluation (AF Probe)\n{'-' * 50}")
        print(f"  data: {json.dumps(data['summary'])[:400]}")
        self._register_hook(self.af_layer)
        harmful, benign = data["harmful"], data["benign"]
        print(f"  Evaluating {len(harmful)} harmful + {len(benign)} benign")
        hs = self._score_records(harmful, self.af_direction, "Harmful")
        bs = self._score_records(benign, self.af_direction, "Benign")
        pos = _strip_for_results(harmful, hs, self.record_text)
        neg = _strip_for_results(benign, bs, self.record_text)
        m = summarise_scores(pos, neg, self.af_threshold)
        # FPR per benign set (DECISIONS.md #1: XSTest primary, JBB secondary) and AUC of harmful vs each set
        m["by_benign_set"] = {}
        for name in sorted({r.get("benign_set", "unknown") for r in neg}):
            ns = [r for r in neg if r.get("benign_set", "unknown") == name]
            fp = int(sum(r["score"] > self.af_threshold for r in ns))
            m["by_benign_set"][name] = {"n": len(ns), "false_positives": fp, "fpr": fp / len(ns),
                                        "auc_vs_harmful": auc_mann_whitney(np.array([r["score"] for r in pos]), np.array([r["score"] for r in ns]))}
        m["by_semantic_category"] = {}
        for cat in sorted({r.get("semantic_category", "unknown") for r in pos}):
            ps = [r for r in pos if r.get("semantic_category", "unknown") == cat]
            m["by_semantic_category"][cat] = {"n": len(ps), "detection_rate": sum(r["score"] > self.af_threshold for r in ps) / len(ps)}
        print(f"\n  Results:\n    Detection Rate: {m['detection_rate']:.1%}\n    FPR: {m['fpr']:.1%}\n    AUC: {m['auc']:.3f}")
        for name, v in m["by_benign_set"].items():
            print(f"    FPR[{name}]: {v['fpr']:.1%} ({v['false_positives']}/{v['n']}), AUC {v['auc_vs_harmful']:.3f}")
        return BenchmarkResult(model=self.model_name, benchmark="harmbench", detection_rate=m["detection_rate"], fpr=m["fpr"],
                               auc=m["auc"], latency_ms=0, n_positive=m["n_positive"], n_negative=m["n_negative"],
                               details=m, data_summary=data["summary"], per_case=pos + neg)

    # ---- AAG / InjecAgent ------------------------------------------------
    def run_injecagent(self, data: Dict[str, Any]) -> BenchmarkResult:
        if self.aag_direction is None:
            raise ValueError("AAG probe not loaded")
        print(f"\n{'-' * 50}\n InjecAgent Evaluation (AAG Probe)\n{'-' * 50}")
        print(f"  data: {json.dumps(data['summary'])[:400]}")
        self._register_hook(self.aag_layer)
        inj, ben = data["injections"], data["benign"]
        print(f"  Evaluating {len(inj)} real injections + {len(ben)} real benign user cases")
        is_ = self._score_records(inj, self.aag_direction, "Injections")
        bs = self._score_records(ben, self.aag_direction, "Benign")
        pos = _strip_for_results(inj, is_, self.record_text)
        neg = _strip_for_results(ben, bs, self.record_text)
        m = summarise_scores(pos, neg, self.aag_threshold)
        # per attack-type breakdown
        by_type: Dict[str, Dict[str, float]] = {}
        for r in pos:
            t = by_type.setdefault(r["attack_type"], {"n": 0, "detected": 0})
            t["n"] += 1
            t["detected"] += int(r["score"] > self.aag_threshold)
        m["by_attack_type"] = {k: {**v, "detection_rate": v["detected"] / v["n"]} for k, v in by_type.items()}
        # per InjecAgent attack setting (B1): detection rate and AUC for base / enhanced separately, plus overall above
        m["by_setting"] = {}
        for s in sorted({r.get("setting", "unknown") for r in pos}):
            ps = [r for r in pos if r.get("setting", "unknown") == s]
            m["by_setting"][s] = {"n": len(ps), "detected": int(sum(r["score"] > self.aag_threshold for r in ps)),
                                  "detection_rate": sum(r["score"] > self.aag_threshold for r in ps) / len(ps),
                                  "auc_vs_all_benign": auc_mann_whitney(np.array([r["score"] for r in ps]), np.array([r["score"] for r in neg]))}
        # FPR four ways (amendment 3): overall, per benign subset, hard-negative-only, plain-only
        m["fpr_breakdown"] = fpr_breakdown(neg, self.aag_threshold)
        fb = m["fpr_breakdown"]
        print(f"\n  Results:\n    Detection Rate: {m['detection_rate']:.1%}\n    FPR: {m['fpr']:.1%} (n_benign={m['n_negative']})\n    AUC: {m['auc']:.3f}")
        for s, v in m["by_setting"].items():
            print(f"    Detection[{s}]: {v['detection_rate']:.1%} ({v['detected']}/{v['n']}), AUC {v['auc_vs_all_benign']:.3f}")
        for s, v in fb["per_subset"].items():
            print(f"    FPR[{s}]: {v['fpr']:.1%} ({v['false_positives']}/{v['n']})")
        for k in ("hard_negative_only", "plain_only"):
            v = fb[k]
            if v["n"]:
                print(f"    FPR[{k}]: {v['fpr']:.1%} ({v['false_positives']}/{v['n']})")
        return BenchmarkResult(model=self.model_name, benchmark="injecagent", detection_rate=m["detection_rate"], fpr=m["fpr"],
                               auc=m["auc"], latency_ms=0, n_positive=m["n_positive"], n_negative=m["n_negative"],
                               details=m, data_summary=data["summary"], per_case=pos + neg)

    # ---- AF / AILuminate obfuscation ------------------------------------------
    def run_ailuminate(self, data: Dict[str, Any]) -> BenchmarkResult:
        """Detection rate of the AF probe on all 1,200 AILuminate prompts under each encoding (no benign side)."""
        if self.af_direction is None:
            raise ValueError("AF probe not loaded")
        print(f"\n{'-' * 50}\n AILuminate Obfuscation Evaluation (AF Probe)\n{'-' * 50}")
        print(f"  data: {json.dumps(data['summary'])[:400]}")
        self._register_hook(self.af_layer)
        recs, brecs = data["records"], data.get("benign_records", [])
        scores = self._score_records(recs, self.af_direction, "AILuminate harmful")
        pos = _strip_for_results(recs, scores, self.record_text)
        neg = _strip_for_results(brecs, self._score_records(brecs, self.af_direction, "AILuminate benign-encoded"), self.record_text) if brecs else []
        t = self.af_threshold
        by_enc, by_haz = {}, {}
        for r in pos:
            e = by_enc.setdefault(r["encoding"], {"n": 0, "detected": 0}); e["n"] += 1; e["detected"] += int(r["score"] > t)
            h = by_haz.setdefault(r["hazard"], {"n": 0, "detected": 0}); h["n"] += 1; h["detected"] += int(r["score"] > t)
        details = {"threshold": t, "n_positive": len(pos), "n_negative": len(neg),
                   "by_encoding": {k: {**v, "detection_rate": v["detected"] / v["n"]} for k, v in by_enc.items()},
                   "by_hazard": {k: {**v, "detection_rate": v["detected"] / v["n"]} for k, v in by_haz.items()},
                   "benign_encoded_control": {}}
        # benign-encoded control: FPR per encoding per benign set, and AUC harmful-vs-benign under the same encoding
        for enc in details["by_encoding"]:
            ps = np.array([r["score"] for r in pos if r["encoding"] == enc])
            for name in sorted({r["benign_set"] for r in neg}):
                ns = [r for r in neg if r["encoding"] == enc and r["benign_set"] == name]
                if not ns:
                    continue
                fp = int(sum(r["score"] > t for r in ns))
                details["benign_encoded_control"].setdefault(enc, {})[name] = {
                    "n": len(ns), "false_positives": fp, "fpr": fp / len(ns),
                    "auc_vs_harmful_same_encoding": auc_mann_whitney(ps, np.array([r["score"] for r in ns]))}
        overall = sum(v["detected"] for v in by_enc.values()) / len(pos)
        overall_fpr = (sum(r["score"] > t for r in neg) / len(neg)) if neg else float("nan")
        for k, v in details["by_encoding"].items():
            print(f"    Detection[{k}]: {v['detection_rate']:.1%} ({v['detected']}/{v['n']})")
            for name, c in details["benign_encoded_control"].get(k, {}).items():
                print(f"      benign-encoded FPR[{k}][{name}]: {c['fpr']:.1%} ({c['false_positives']}/{c['n']}), AUC {c['auc_vs_harmful_same_encoding']:.3f}")
        return BenchmarkResult(model=self.model_name, benchmark="ailuminate", detection_rate=overall, fpr=overall_fpr,
                               auc=(auc_mann_whitney(np.array([r["score"] for r in pos]), np.array([r["score"] for r in neg])) if neg else float("nan")),
                               latency_ms=0, n_positive=len(pos), n_negative=len(neg), details=details, data_summary=data["summary"], per_case=pos + neg)

    # ---- latency -----------------------------------------------------------
    def run_latency(self, n_runs: int = 50) -> Dict[str, float]:
        print(f"\n{'-' * 50}\n Latency Benchmark\n{'-' * 50}")
        if self.af_direction is not None:
            self._register_hook(self.af_layer)
        test_prompt = "What is the meaning of life? Please explain in detail."
        for _ in range(5):
            self.llm.generate([test_prompt], self.sampling_params)
        latencies = []
        for _ in range(n_runs):
            if os.path.exists(self.activation_cache):
                os.remove(self.activation_cache)
            start = time.perf_counter()
            self.llm.generate([test_prompt], self.sampling_params)
            latencies.append((time.perf_counter() - start) * 1000)
        direction = self.af_direction if self.af_direction is not None else np.ones(self.hidden_dim, dtype=np.float32)
        dummy = np.random.randn(self.hidden_dim).astype(np.float32)
        dummy /= np.linalg.norm(dummy)
        probe_lat = []
        for _ in range(1000):
            start = time.perf_counter()
            _ = np.dot(dummy, direction)
            probe_lat.append((time.perf_counter() - start) * 1000)
        results = {
            "forward_pass_mean_ms": float(np.mean(latencies)), "forward_pass_std_ms": float(np.std(latencies)),
            "forward_pass_p50_ms": float(np.percentile(latencies, 50)), "forward_pass_p99_ms": float(np.percentile(latencies, 99)),
            "probe_overhead_mean_ms": float(np.mean(probe_lat)), "probe_overhead_p99_ms": float(np.percentile(probe_lat, 99)),
            "n_runs": n_runs, "prompt": test_prompt,
        }
        print(f"    Forward pass (with hook): {results['forward_pass_mean_ms']:.1f}ms ± {results['forward_pass_std_ms']:.1f}ms")
        print(f"    Probe overhead: {results['probe_overhead_mean_ms']:.3f}ms (p99: {results['probe_overhead_p99_ms']:.3f}ms)")
        return results

    def cleanup(self):
        del self.llm
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


# ============================================================
# Driver
# ============================================================

def run_all_models(models: List[str], cfg: Dict[str, Any], benchmarks: List[str], output_dir: Path,
                   probes_dir: Optional[str] = None, use_legacy_authored_set: bool = False,
                   injecagent_dir: Optional[str] = None, harmbench_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Load data ONCE, before any model: a data problem must abort the run, not one model.
    aag_data = load_aag_data(cfg, injecagent_dir) if "injecagent" in benchmarks else None
    af_data = load_af_data(cfg, harmbench_dir, use_legacy_authored_set) if "harmbench" in benchmarks else None
    ail_data = load_ailuminate_data(cfg) if "ailuminate" in benchmarks else None
    provenance = run_provenance(cfg, {"benchmarks": benchmarks, "models": models})   # computed once per run, embedded in every file

    results = []
    for model_name in models:
        print(f"\n{'=' * 70}\n Model: {model_name}\n{'=' * 70}")
        model_safe = model_name.replace("/", "_").replace("-", "_")
        model_results: Dict[str, Any] = {"model": model_name, "config": cfg.get("_config_path"),
                                         "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "provenance": provenance}
        try:
            bench = VLLMBenchmark(model_name, cfg, probes_dir)
            bench.setup()
            if af_data is not None:
                try:
                    model_results["harmbench"] = asdict(bench.run_harmbench(af_data))
                except DataLoadError:
                    raise
                except Exception as e:  # model-side failure: record, continue
                    print(f"  HarmBench failed: {e}")
                    model_results["harmbench"] = {"error": str(e)}
            if aag_data is not None:
                try:
                    model_results["injecagent"] = asdict(bench.run_injecagent(aag_data))
                except DataLoadError:
                    raise
                except Exception as e:
                    print(f"  InjecAgent failed: {e}")
                    model_results["injecagent"] = {"error": str(e)}
            ail_models = cfg["ailuminate"].get("models") or models
            if ail_data is not None and model_name in ail_models:
                try:
                    model_results["ailuminate"] = asdict(bench.run_ailuminate(ail_data))
                except DataLoadError:
                    raise
                except Exception as e:
                    print(f"  AILuminate failed: {e}")
                    model_results["ailuminate"] = {"error": str(e)}
            if "latency" in benchmarks:
                try:
                    model_results["latency"] = bench.run_latency()
                except Exception as e:
                    print(f"  Latency benchmark failed: {e}")
                    model_results["latency"] = {"error": str(e)}
            bench.cleanup()
        except DataLoadError:
            raise
        except Exception as e:
            print(f"  Model failed: {e}")
            model_results["error"] = str(e)
        # one file per (benchmark, model) so legs never overwrite each other: <output_dir>/<benchmark>/<model_safe>.json
        written = []
        for bench_name in ("harmbench", "injecagent", "ailuminate", "latency"):
            if bench_name not in model_results:
                continue
            section = {k: v for k, v in model_results.items() if k not in ("harmbench", "injecagent", "ailuminate", "latency")}
            section[bench_name] = model_results[bench_name]
            out_path = output_dir / bench_name / f"{model_safe}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(section, f, indent=2)
            written.append(str(out_path))
        if not written:  # model failed before any benchmark ran: still record it, with provenance
            out_path = output_dir / f"{model_safe}__failed.json"
            with open(out_path, "w") as f:
                json.dump(model_results, f, indent=2)
            written.append(str(out_path))
        print(f"  wrote {', '.join(written)}")
        results.append(model_results)

    print(f"\n{'=' * 70}\n SUMMARY\n{'=' * 70}")
    print(f"\n{'Model':<40} {'AF Det':>8} {'AF AUC':>8} {'AAG Det':>8} {'AAG FPR':>8} {'AAG AUC':>8}")
    print("-" * 90)
    for r in results:
        hb, ia = r.get("harmbench", {}), r.get("injecagent", {})
        fmt = lambda d, k, pct: (f"{d[k]:.0%}" if pct else f"{d[k]:.3f}") if k in d else "N/A"
        print(f"{r['model']:<40} {fmt(hb, 'detection_rate', True):>8} {fmt(hb, 'auc', False):>8} "
              f"{fmt(ia, 'detection_rate', True):>8} {fmt(ia, 'fpr', True):>8} {fmt(ia, 'auc', False):>8}")
    print(f"\nPer-model results written under: {output_dir}")
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AASE vLLM Benchmark Suite (config-driven, real data only)")
    p.add_argument("--config", default=None, help="eval_config.yaml (default: repo root / $AASE_EVAL_CONFIG)")
    p.add_argument("--model", default=None, help="Single model to benchmark (default: first model in config)")
    p.add_argument("--all-models", action="store_true", help="Run every model listed in the config")
    p.add_argument("--benchmark", choices=["harmbench", "injecagent", "ailuminate", "latency", "llama_guard", "all"], default="all",
                   help="'llama_guard' is dry-run only here (the comparison itself runs via compare_llama_guard_vllm.py)")
    p.add_argument("--probes-dir", default=None, help="Override probes.dir from config")
    p.add_argument("--injecagent-dir", default=None, help="Override datasets.injecagent_dir")
    p.add_argument("--harmbench-dir", default=None, help="Override datasets.harmbench_dir")
    p.add_argument("--output-dir", default=None, help="Override output.results_dir/output.run_name")
    p.add_argument("--dry-run", action="store_true", help="Exercise the full data path without any model and exit")
    p.add_argument("--sample-width", type=int, default=160, help="dry run: max chars per printed prompt line")
    p.add_argument("--use-legacy-authored-set", action="store_true",
                   help="AF only: score the old authored 56+31 list instead of HarmBench (explicit, never a fallback)")
    p.add_argument("--use-legacy-table5-set", action="store_true",
                   help="Llama Guard comparison dry run only: the paper's 16 authored prompts (explicit, never a fallback)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    benchmarks = ["harmbench", "injecagent", "ailuminate", "latency", "llama_guard"] if args.benchmark == "all" else [args.benchmark]

    if args.dry_run:
        dry_run(cfg, benchmarks, args.use_legacy_authored_set, args.injecagent_dir, args.harmbench_dir, sample_width=args.sample_width,
                use_legacy_table5_set=args.use_legacy_table5_set)
        return 0
    if "llama_guard" in benchmarks:
        print("  (llama_guard: run aase_vllm/scripts/compare_llama_guard_vllm.py for the actual comparison; skipped here)")
        benchmarks = [b for b in benchmarks if b != "llama_guard"]

    models = cfg["models"] if args.all_models else [args.model or cfg["models"][0]]
    output_dir = Path(args.output_dir) if args.output_dir else resolve_path(cfg["output"]["results_dir"]) / cfg["output"]["run_name"]
    run_all_models(models, cfg, benchmarks, output_dir, args.probes_dir, args.use_legacy_authored_set,
                   args.injecagent_dir, args.harmbench_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
