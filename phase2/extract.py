"""Phase 2 model-side utilities (HF transformers, bf16, single GPU): batched multi-block last-token extraction in one forward
pass, chat-template generation, probe training and scoring.

Backend note: Phase 1 used vLLM hooks; Phase 2 uses HF `output_hidden_states` so that all 22 blocks come from one pass and
so that generation (P1/P4) shares the model. hidden_states[k+1] is the post-block output of decoder block k — the same
tensor a forward hook on model.model.layers[k] returns (verified numerically in PREFLIGHT D2). Scores of the January probe
at block 21 in raw mode are compared with Phase 1's vLLM scores for the same prompts as a backend-consistency check
(reported in P2).
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .common import BLOCKS, format_for_mode


# ---------------------------------------------------------------- model
def load_model(model_name: str, dtype: str = "bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"                         # so position -1 is the last real token for every row
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"device_map": "cuda"}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=getattr(torch, dtype), **kw)        # transformers >= 5
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=getattr(torch, dtype), **kw)  # transformers 4.x
    model.eval()
    cfg = getattr(model.config, "text_config", model.config)
    return model, tok, {"num_layers": cfg.num_hidden_layers, "hidden": cfg.hidden_size, "dtype": dtype}


def chat_template_check(tok) -> Dict[str, Any]:
    """Does the tokenizer's own chat template reproduce the December manual string (modulo BOS)?"""
    from .common import GEMMA_TEMPLATE
    p = "Test prompt."
    try:
        t = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
    except Exception as e:
        return {"tokenizer_template": None, "error": str(e)[:200]}
    manual = GEMMA_TEMPLATE.format(prompt=p)
    return {"tokenizer_template": t, "december_manual": manual, "identical_modulo_bos": t.replace(tok.bos_token or "", "") == manual}


# ---------------------------------------------------------------- extraction
FEATURES = ("residual", "mlp_out")


def _mlp_branch_module(layer):
    """The module whose output vLLM's decoder layer returns as output[0] (what Phase 1's `output[0] if isinstance(output, tuple)`
    hook captured): Gemma-2 = post_feedforward_layernorm(mlp(.)); Llama/Qwen-2 = mlp(.). NOT the residual stream."""
    return getattr(layer, "post_feedforward_layernorm", None) or layer.mlp


def extract_last_token(model, tok, texts: Sequence[str], blocks: Sequence[int] = BLOCKS, batch_size: int = 16,
                       max_length: int = 1024, progress: Optional[str] = None, feature: str = "residual") -> np.ndarray:
    """Returns float16 array [n_texts, len(blocks), hidden] of last-token activations, one forward pass per batch.

    feature='residual': post-block residual stream = HF hidden_states[b+1] (what December's HF forward hook on layers[b]
                        returned, modulo December's one-layer offset). The "block b activation" of the Phase 2 design.
    feature='mlp_out':  the block's MLP-branch output before the residual add (Gemma-2: post_feedforward_layernorm output).
                        This is what vLLM's decoder layer returns as output[0] and therefore the tensor every vLLM-lineage
                        probe (January AF/AAG/APC, Phase 1 R1 retrain) was trained and evaluated on — see PHASE2_REPORT.md.
    """
    import torch
    if feature not in FEATURES:
        raise ValueError(f"feature must be one of {FEATURES}, got {feature!r}")
    out = None
    n = len(texts)
    t0 = time.time()
    # Run the decoder body only: the CausalLM wrapper would also materialise full-vocabulary logits (Gemma-2: 256k vocab,
    # float32 for the logit soft-capping = ~7.6 GB per batch of 16 x 1024 tokens), which OOMed a 40 GB A100 and is unused here.
    # The base model's hidden_states tuple is identical to the wrapper's (the wrapper just applies lm_head on top).
    base = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    captured: Dict[int, "torch.Tensor"] = {}
    handles = []
    if feature == "mlp_out":
        for b in blocks:
            def _hook(module, inp, output, _b=b):
                captured[_b] = (output[0] if isinstance(output, tuple) else output)[:, -1, :].detach()
            handles.append(_mlp_branch_module(base.layers[b]).register_forward_hook(_hook))
    try:
        for i in range(0, n, batch_size):
            batch = list(texts[i:i + batch_size])
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length, add_special_tokens=True).to(model.device)
            with torch.no_grad():
                res = base(**enc, output_hidden_states=(feature == "residual"), use_cache=False)
            if feature == "residual":
                hs = res.hidden_states                      # tuple len num_layers+1; hs[k+1] = post-block k
                per_block = {b: hs[b + 1][:, -1, :] for b in blocks}
            else:
                per_block = {b: captured[b] for b in blocks}
            if out is None:
                out = np.zeros((n, len(blocks), per_block[blocks[0]].shape[-1]), dtype=np.float16)
            for j, b in enumerate(blocks):
                out[i:i + len(batch), j, :] = per_block[b].float().cpu().numpy().astype(np.float16)
            if progress and ((i // batch_size) % 10 == 0 or i + batch_size >= n):
                print(f"    {progress}: {min(i + batch_size, n)}/{n} ({time.time() - t0:.0f}s)", flush=True)
    finally:
        for h in handles:
            h.remove()
    return out


def generate(model, tok, prompts: Sequence[str], max_new_tokens: int = 96, batch_size: int = 8) -> List[str]:
    """Greedy generation through the tokenizer's chat template; returns the decoded continuations."""
    import torch
    outs = []
    for i in range(0, len(prompts), batch_size):
        batch = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts[i:i + batch_size]]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        for row in gen[:, enc["input_ids"].shape[1]:]:
            outs.append(tok.decode(row, skip_special_tokens=True))
    return outs


# ---------------------------------------------------------------- probes
def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def train_direction(pos: np.ndarray, neg: np.ndarray, rule: str) -> Dict[str, Any]:
    """Difference-of-means direction (both December and January use this).
    rule='january': cosine scores, threshold = 5%-FPR quantile of benign training scores floored at the midpoint (train_probes_vllm._train_direction);
    rule='december': raw dot-product scores, threshold = midpoint of the class means (AFClassifier.train)."""
    pos = pos.astype(np.float32); neg = neg.astype(np.float32)
    d = pos.mean(0) - neg.mean(0)
    d = d / np.linalg.norm(d)
    if rule == "january":
        ps, ns = unit(pos) @ d, unit(neg) @ d
        sorted_neg = np.sort(ns)[::-1]
        idx = max(0, int(len(sorted_neg) * 0.05) - 1)
        thr = float(max(sorted_neg[idx], (ps.mean() + ns.mean()) / 2))
    elif rule == "december":
        ps, ns = pos @ d, neg @ d
        thr = float((ps.mean() + ns.mean()) / 2)
    else:
        raise ValueError(rule)
    sep = float((ps.mean() - ns.mean()) / max(ps.std(), 1e-3)) if rule == "december" else float((ps.mean() - ns.mean()) / max(np.sqrt((ps.var() + ns.var()) / 2), 1e-6))
    return {"direction": d.astype(np.float32), "threshold": thr, "rule": rule, "train_pos_mean": float(ps.mean()), "train_neg_mean": float(ns.mean()),
            "train_separation": sep, "train_auc": auc(ps, ns), "n_pos": int(len(pos)), "n_neg": int(len(neg))}


def score(acts: np.ndarray, direction: np.ndarray, rule: str) -> np.ndarray:
    acts = acts.astype(np.float32)
    return (unit(acts) @ direction) if rule == "january" else (acts @ direction)


def auc(pos, neg) -> float:
    p = np.asarray(pos, np.float64); n = np.asarray(neg, np.float64)
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    # rank-based (Mann-Whitney) to avoid the O(n*m) matrix for 1200x350 x 22 x 4 x 2 evaluations
    allv = np.concatenate([p, n]); order = allv.argsort(kind="mergesort"); ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks); ranks = sums[inv] / counts[inv]
    return float((ranks[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def dist_stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, np.float64)
    return {"n": int(len(x)), "mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()), "p50": float(np.median(x)), "max": float(x.max()),
            "distinct": int(len(np.unique(np.round(x, 6))))}


def load_january_probe(repo: Path, model_name: str = "google/gemma-2-9b-it") -> Dict[str, Any]:
    safe = model_name.replace("/", "_").replace("-", "_")
    p = repo / "aase_vllm/pretrained/af" / safe
    d = np.load(f"{p}.npy").astype(np.float32); d = d / np.linalg.norm(d)
    meta = json.load(open(f"{p}.json"))
    return {"direction": d, "threshold": float(meta["threshold"]), "layer_index": int(meta["layer_index"]), "rule": "january", "meta": meta,
            "path": str(p)}
