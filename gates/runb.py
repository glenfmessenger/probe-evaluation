#!/usr/bin/env python3
"""Stage B shared machinery: config, provenance, the two default controls (length stratification, refusal separation),
probe rules, and the matched-FPR operating point used by the Arm 1 threshold.

Extraction reuses `phase2.extract` unchanged (same model loading, same `mlp_out` hook on the MLP-branch module — the
space every probe in this repository was trained in, PHASE2_REPORT 0.1), with one addition here: `extract_at_positions`,
which returns per-position hidden states at a single layer so Arm 2 can read the tool-response span rather than only
the last token.
"""
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aase_eval.errors import DataLoadError  # noqa: E402
from aase_eval.provenance import run_provenance, sha256_file  # noqa: E402
from phase2.common import GEMMA_TEMPLATE, is_refusal  # noqa: E402
from phase2.extract import auc, dist_stats, load_model, score, unit  # noqa: E402

N_BINS = 5
MIN_PER_CLASS = 5


# ---------------------------------------------------------------- config / provenance
def load_gates_config(path: Optional[str] = None) -> Dict[str, Any]:
    p = Path(path) if path else REPO / "gates_config.yaml"
    cfg = yaml.safe_load(p.read_text())
    cfg["_config_path"] = str(p)
    return cfg


def provenance(cfg: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    from aase_eval.config import load_config
    from gates.signoff import record, verify
    base = load_config()
    bad = [n for n, (_, _, ok) in verify().items() if not ok]
    if bad:
        raise DataLoadError(f"authored data changed since sign-off: {bad} — Stage B must not run on unreviewed data")
    prov = run_provenance(base, {"phase": "gates-stage-b"})
    prov["gates_config_sha256"] = sha256_file(cfg["_config_path"])
    prov["gates_config"] = {k: v for k, v in cfg.items() if not k.startswith("_")}
    prov["signoff"] = record()
    if extra:
        prov.update(extra)
    return prov


def out_dir(cfg) -> Path:
    d = REPO / cfg["output_dir"]; d.mkdir(parents=True, exist_ok=True); return d


def act_dir(cfg) -> Path:
    """Bulk activation tensors. Gitignored by design; nothing that has to survive the run may live here alone."""
    d = REPO / cfg["activations_dir"]; d.mkdir(parents=True, exist_ok=True); return d


def scores_dir(cfg) -> Path:
    """Per-case scores, committed. Every aggregate this project reports must be rebuildable from a file written here:
    an AUC without its scores can be quoted but never re-interrogated, and tests/test_results_completeness.py fails a
    results directory that ships the one without the other."""
    d = REPO / cfg.get("scores_dir", "results/gates/scores"); d.mkdir(parents=True, exist_ok=True); return d


def dump_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1, default=float)


def safe(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_").replace(".", "_")


def load_probe(cfg, kind: str, model_name: str, policy: str = None) -> Dict[str, Any]:
    """A shipped January probe: direction, threshold and — the reason we always load it — its layer_index."""
    d = REPO / cfg[{"af": "b1", "aag": "b2", "apc": "b3"}[kind]]["probe_dir"]
    if policy:
        d = d / policy
    stem = d / model_name.replace("/", "_").replace("-", "_")
    npy, js = Path(f"{stem}.npy"), Path(f"{stem}.json")
    if not npy.exists() or not js.exists():
        raise DataLoadError(f"missing {kind} probe for {model_name}: {npy}")
    v = np.load(npy).astype(np.float32); v = v / np.linalg.norm(v)
    meta = json.load(open(js))
    return {"direction": v, "threshold": float(meta["threshold"]), "layer_index": int(meta["layer_index"]),
            "meta": meta, "path": str(stem)}


# ---------------------------------------------------------------- formatting
def format_mode(text: str, mode: str, model_name: str, tok=None) -> str:
    """raw = the bare string (the Phase 1 convention); templated = the model's own user-turn chat template."""
    if mode == "raw":
        return text
    if mode != "templated":
        raise ValueError(f"unknown mode {mode!r}")
    if tok is not None:
        try:
            return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return GEMMA_TEMPLATE.format(prompt=text)


# ---------------------------------------------------------------- probe rules
def train_rule(pos: np.ndarray, neg: np.ndarray, rule: str, pairs: bool = False) -> Dict[str, Any]:
    """paired  — v = mean_i(h_i^+ - h_i^-) over aligned pairs (the paper's eq. 1);
    standard — v = mean(h^+) - mean(h^-) over the same texts (the Phase 1 rule).
    Threshold in both cases is the January rule: cosine scores, the 5%-FPR quantile of the benign training scores,
    floored at the midpoint of the class means. Only the direction differs, so a difference in results is the method."""
    pos = np.asarray(pos, np.float32); neg = np.asarray(neg, np.float32)
    if rule == "paired":
        if not pairs or pos.shape != neg.shape:
            raise ValueError("paired rule needs aligned pairs of equal shape")
        d = (pos - neg).mean(0)
    elif rule == "standard":
        d = pos.mean(0) - neg.mean(0)
    else:
        raise ValueError(f"unknown rule {rule!r}")
    n = np.linalg.norm(d)
    if n < 1e-8:
        raise DataLoadError(f"{rule} direction is degenerate (norm {n:.2e})")
    d = (d / n).astype(np.float32)
    ps, ns = unit(pos) @ d, unit(neg) @ d
    srt = np.sort(ns)[::-1]
    thr = float(max(srt[max(0, int(len(srt) * 0.05) - 1)], (ps.mean() + ns.mean()) / 2))
    return {"direction": d, "threshold": thr, "rule": rule, "train_auc": auc(ps, ns),
            "train_separation": float((ps.mean() - ns.mean()) / max(np.sqrt((ps.var() + ns.var()) / 2), 1e-6)),
            "n_pos": int(len(pos)), "n_neg": int(len(neg))}


# ---------------------------------------------------------------- control 1: length
def stratified_auc(s: np.ndarray, y: np.ndarray, L: np.ndarray, n_bins: int = N_BINS,
                   min_per_class: int = MIN_PER_CLASS) -> Tuple[float, List[Dict]]:
    """Length-stratified AUC: quantile bins of L, per-bin Mann-Whitney AUC pooled by n_pos*n_neg. Bins with fewer than
    `min_per_class` of either class are dropped and reported, so a set whose classes barely overlap in length cannot
    hide behind a single pooled number."""
    s, y, L = np.asarray(s, float), np.asarray(y, int), np.asarray(L, float)
    edges = np.quantile(L, np.linspace(0, 1, n_bins + 1)); edges[-1] += 1
    num = den = 0.0; bins = []
    for k in range(n_bins):
        m = (L >= edges[k]) & (L < edges[k + 1])
        p, n = s[m & (y == 1)], s[m & (y == 0)]
        b = {"len_range": [int(edges[k]), int(edges[k + 1] - 1)], "n_pos": int(len(p)), "n_neg": int(len(n)), "auc": None}
        if len(p) >= min_per_class and len(n) >= min_per_class:
            b["auc"] = auc(p, n); w = len(p) * len(n); num += b["auc"] * w; den += w
        bins.append(b)
    return (num / den if den else float("nan")), bins


def length_report(s, y, L) -> Dict[str, Any]:
    s, y, L = np.asarray(s, float), np.asarray(y, int), np.asarray(L, float)
    a_str, bins = stratified_auc(s, y, L)
    return {"auc": auc(s[y == 1], s[y == 0]), "auc_length_stratified": a_str, "bins": bins,
            "length_auc": auc(L[y == 1], L[y == 0]), "rho_score_length": spearman(s, L),
            "median_len_pos": float(np.median(L[y == 1])), "median_len_neg": float(np.median(L[y == 0]))}


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


# ---------------------------------------------------------------- control 2: refusal
def refusal_table(items: Sequence[Dict[str, Any]], group_key: Callable = lambda r: r["set"]) -> Dict[str, Any]:
    """Detection conditioned on whether the model itself refused. `items` carry set, label, refusal and flagged."""
    out: Dict[str, Any] = {}
    for r in items:
        g = out.setdefault(str(group_key(r)), {}).setdefault("refused" if r["refusal"] else "not_refused",
                                                             {"n": 0, "flagged": 0})
        g["n"] += 1; g["flagged"] += int(r["flagged"])
    for g in out.values():
        for cell in g.values():
            cell["flag_rate"] = cell["flagged"] / cell["n"] if cell["n"] else float("nan")
    return out


# ---------------------------------------------------------------- operating points
def fpr_at_threshold(scores, thr) -> float:
    s = np.asarray(scores, float)
    return float((s > thr).mean()) if len(s) else float("nan")


def threshold_at_fpr(neg_scores, target_fpr: float) -> float:
    """Smallest threshold whose FPR on `neg_scores` is <= target_fpr. Used for the matched-XSTest-FPR operating point in
    GATES.md ARM 1 criterion 2, so the JBB-benign FPR comparison is not just a threshold shift."""
    s = np.sort(np.asarray(neg_scores, float))[::-1]
    if target_fpr <= 0:
        return float(s[0]) + 1e-9
    k = int(np.floor(target_fpr * len(s)))
    if k <= 0:
        return float(s[0]) + 1e-9
    if k >= len(s):
        return float(s[-1]) - 1e-9
    return float(s[k - 1])


def matched_operating_point(pos, xstest, jbb, target_xstest_fpr: float) -> Dict[str, Any]:
    thr = threshold_at_fpr(xstest, target_xstest_fpr)
    return {"target_xstest_fpr": target_xstest_fpr, "threshold": thr,
            "xstest_fpr": fpr_at_threshold(xstest, thr), "jbb_fpr": fpr_at_threshold(jbb, thr),
            "detection": fpr_at_threshold(pos, thr)}


# ---------------------------------------------------------------- degenerate-score stop rule
def degenerate(scores) -> Optional[str]:
    s = np.asarray(scores, float)
    if len(s) == 0:
        return "empty"
    if np.isnan(s).any():
        return "contains NaN"
    if np.allclose(s, s[0]):
        return "constant"
    if len(np.unique(np.round(s, 6))) < 0.5 * len(s):
        return f"only {len(np.unique(np.round(s, 6)))} distinct values in {len(s)}"
    return None


def check_scores(name: str, scores) -> Dict[str, Any]:
    d = degenerate(scores)
    st = dist_stats(np.asarray(scores, float))
    if d:
        raise DataLoadError(f"DEGENERATE SCORES [{name}]: {d} — hard stop (GATES.md rule 3); stats {st}")
    return st


# ---------------------------------------------------------------- extraction: the Arm 2 reads
def extract_reads(model, tok, texts: Sequence[str], spans: Sequence[Dict[str, Any]], layers: Sequence[int],
                  feature: str = "mlp_out", batch_size: int = 8, max_length: int = 1024,
                  directions: Optional[Dict[str, np.ndarray]] = None, progress: str = None) -> Dict[str, Any]:
    """One forward pass per batch yields all four Arm 2 reads, at every requested layer.

    `spans[i]` carries the character spans for text i: {"tool_response": (c0, c1)}, optionally {"injection": (a0, a1)}
    for an injection case, and {"benign_fraction": f} for a benign case under the oracle design. Character spans are
    mapped to token indices with the tokenizer's offset mapping; an unmappable span raises (no heuristic fallback).

    Returns, per layer: `final` / `mean` / `mean_all` / `oracle` vectors [n, d], and — when `directions` is given — `maxproj[name]`,
    the maximum cosine projection over the tool-response span, computed in the same pass so that design (c) needs no
    second extraction. Hooking several layers at once makes the Arm 2 three-layer check almost free.
    """
    import time

    import torch

    from gates.arm2_protocol import token_span
    from phase2.extract import _mlp_branch_module
    base = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    cap: Dict[int, Any] = {}
    handles = []
    for L in layers:
        mod = _mlp_branch_module(base.layers[L]) if feature == "mlp_out" else base.layers[L]

        def hook(module, inp, output, _L=L):
            cap[_L] = (output[0] if isinstance(output, tuple) else output).detach()
        handles.append(mod.register_forward_hook(hook))
    n = len(texts)
    out = {L: {"final": [], "mean": [], "mean_all": [], "oracle": [], "maxproj": {k: [] for k in (directions or {})}}
           for L in layers}
    meta = []
    t0 = time.time()
    try:
        for i0 in range(0, n, batch_size):
            batch = list(texts[i0:i0 + batch_size])
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length,
                      add_special_tokens=True, return_offsets_mapping=True)
            offs = enc.pop("offset_mapping")
            enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                base(**enc, use_cache=False)
            attn = enc["attention_mask"]
            for j in range(len(batch)):
                idx = i0 + j
                keep = attn[j].bool(); pad = int((~keep).sum())
                off = [tuple(map(int, o)) for o, k in zip(offs[j].tolist(), attn[j].tolist()) if k]
                n_tok = len(off)
                sp = spans[idx]
                t_a, t_b = token_span(off, sp["tool_response"])
                if "injection" in sp:
                    _, i_b = token_span(off, sp["injection"])
                    o_idx = i_b - 1
                else:
                    f = sp.get("benign_fraction")
                    if f is None:
                        raise DataLoadError(f"text {idx}: benign case needs benign_fraction for the oracle read")
                    o_idx = min(max(t_a + int(round(f * max(0, t_b - 1 - t_a))), t_a), t_b - 1)
                if not (0 <= t_a < t_b <= n_tok and t_a <= o_idx < t_b):
                    raise DataLoadError(f"text {idx}: bad token spans tool=({t_a},{t_b}) oracle={o_idx} n={n_tok}")
                # design (e), Amendment 7: the whole prompt, first non-special token to the final token (BOS excluded)
                real = [k for k, (s, e) in enumerate(off) if e > s]
                if not real:
                    raise DataLoadError(f"text {idx}: no non-special token for the mean_all read")
                a_all = real[0]
                for L in layers:
                    h = cap[L][j]
                    blk = h[pad + t_a: pad + t_b, :].float()
                    out[L]["final"].append(h[pad + n_tok - 1, :].float().cpu().numpy().astype(np.float16))
                    out[L]["mean"].append(blk.mean(0).cpu().numpy().astype(np.float16))
                    out[L]["mean_all"].append(h[pad + a_all: pad + n_tok, :].float().mean(0).cpu().numpy().astype(np.float16))
                    out[L]["oracle"].append(h[pad + o_idx, :].float().cpu().numpy().astype(np.float16))
                    if directions:
                        bn = blk / (blk.norm(dim=-1, keepdim=True) + 1e-8)
                        for name, d in directions.items():
                            dv = torch.as_tensor(d, dtype=bn.dtype, device=bn.device)
                            out[L]["maxproj"][name].append(float((bn @ dv).max().item()))
                meta.append({"i": idx, "n_tokens": n_tok, "tool_span": [t_a, t_b], "all_span": [a_all, n_tok],
                             "oracle_index": o_idx, "oracle_from_label": "injection" in sp})
            if progress and ((i0 // batch_size) % 20 == 0 or i0 + batch_size >= n):
                print(f"    {progress}: {min(i0 + batch_size, n)}/{n} ({time.time() - t0:.0f}s)", flush=True)
    finally:
        for h in handles:
            h.remove()
    res = {}
    for L in layers:
        res[L] = {k: np.stack(v) for k, v in out[L].items() if k != "maxproj"}
        res[L]["maxproj"] = {k: np.array(v, dtype=np.float32) for k, v in out[L]["maxproj"].items()}
    return {"by_layer": res, "meta": meta[:len(texts)]}
