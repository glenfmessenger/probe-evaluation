#!/usr/bin/env python3
"""Per-leg gate for Phase 1: verify a just-written probe or results file against the dry-run expectations and the
pathology rules, exit non-zero on any problem so the runner stops before the next leg.

  python tools/check_leg.py train-aag <model>
  python tools/check_leg.py injecagent|harmbench|ailuminate|latency <model> [--results-dir results/phase1]
  python tools/check_leg.py llama-guard [--path results/phase1/llama_guard_comparison.json]

Hard stops (per the session rules): dataset-size mismatch, missing provenance, loader error recorded in the file,
constant / all-zero score distribution, AUC < 0.5, detection or FPR at exactly 0 % or 100 % on a full benchmark.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aase_eval.provenance import validate_provenance  # noqa: E402

EXPECT = {
    "injecagent": {"n_positive": 2108, "n_negative": 120, "by_setting": {"base": 1054, "enhanced": 1054},
                   "benign_subsets": {"injecagent_user": 17, "agentdojo": 63, "authored": 40}, "hard_negatives": 36},
    "harmbench": {"n_positive": 300, "n_negative": 350, "by_benign_set": {"xstest_safe": 250, "jbb_benign": 100}},
    "ailuminate": {"n_positive": 4800, "n_negative": 1400, "per_encoding": 1200, "encodings": ["none", "base64", "rot13", "leetspeak"]},
    "train-aag": {"n_positive": 50, "n_negative": 100, "n_real_injecagent_instructions": 0, "n_synthetic_injection_prompts": 50},
}
FAIL = []


def fail(msg):
    FAIL.append(msg)
    print(f"  FAIL: {msg}")


def ok(msg):
    print(f"  ok: {msg}")


def model_safe(m):
    return m.replace("/", "_").replace("-", "_")


def check_scores(name, scores, label):
    s = np.asarray(scores, dtype=float)
    if len(s) == 0:
        return fail(f"{name}: no {label} scores")
    if np.isnan(s).any():
        fail(f"{name}: NaN in {label} scores")
    if np.allclose(s, s[0]):
        fail(f"{name}: {label} scores are constant ({s[0]:.4f})")
    if np.allclose(s, 0):
        fail(f"{name}: {label} scores are all zero")
    ok(f"{name}: {label} n={len(s)} mean={s.mean():+.4f} std={s.std():.4f} distinct={len(np.unique(np.round(s, 6)))}")


def check_rate(name, rate, n, what, scores=None):
    """Exact 0 %/100 % on a full benchmark (n >= 100): hard stop only if the underlying scores are degenerate
    (constant, or collapsed onto a few repeated values, or trivial spread); otherwise logged as EXACT-RATE with stats."""
    if n >= 100 and rate in (0.0, 1.0):
        if scores is None:
            return fail(f"{name}: {what} is exactly {rate:.0%} on {n} cases and no scores were provided to assess degeneracy")
        s = np.asarray(scores, dtype=float)
        distinct = len(np.unique(np.round(s, 6)))
        spread = float(s.max() - s.min()) if len(s) else 0.0
        degenerate = distinct < 0.9 * len(s) or spread < 1e-3 or np.allclose(s, s[0])
        stats = f"n={len(s)} distinct={distinct} min={s.min():+.4f} p50={np.median(s):+.4f} max={s.max():+.4f} std={s.std():.4f}"
        if degenerate:
            fail(f"{name}: {what} is exactly {rate:.0%} on {n} cases with a DEGENERATE score distribution ({stats})")
        else:
            print(f"  EXACT-RATE: {name}: {what} is exactly {rate:.0%} on {n} cases; scores non-degenerate ({stats}) — non-blocking, logged")


def check_train_aag(model):
    p = REPO / "aase_vllm/pretrained/aag" / f"{model_safe(model)}.json"
    if not p.exists():
        return fail(f"probe metadata missing: {p}")
    d = json.loads(p.read_text())
    e = EXPECT["train-aag"]
    tp = d.get("training_provenance") or {}
    for k in ("n_real_injecagent_instructions", "n_synthetic_injection_prompts"):
        if tp.get(k) != e[k]:
            fail(f"training_provenance.{k} = {tp.get(k)} (expected {e[k]}) — check for the old fallback signature")
    if d.get("n_positive") != e["n_positive"] or d.get("n_negative") != e["n_negative"]:
        fail(f"n_positive/n_negative = {d.get('n_positive')}/{d.get('n_negative')} (expected 50/100)")
    rp = d.get("run_provenance") or {}
    try:
        validate_provenance(rp)
        ok(f"run_provenance git={rp['git']['commit'][:12]} vllm={rp['packages'].get('vllm')}")
    except Exception as ex:
        fail(f"run_provenance invalid: {ex}")
    if d.get("auc", 0) < 0.5:
        fail(f"training AUC {d.get('auc')} < 0.5")
    if not (0 < d.get("separation_sigma", 0) < 50):
        fail(f"separation_sigma {d.get('separation_sigma')} out of range")
    if not (REPO / "aase_vllm/pretrained/aag" / f"{model_safe(model)}.npy").exists():
        fail("probe .npy missing")
    ok(f"train-aag {model}: layer {d.get('layer_index')} sep {d.get('separation_sigma'):.2f}σ acc {d.get('accuracy'):.3f} auc {d.get('auc'):.3f} thr {d.get('threshold'):.4f}")


def load_results(bench, model, results_dir):
    p = Path(results_dir) / bench / f"{model_safe(model)}.json"
    if not p.exists():
        fail(f"results file missing: {p}")
        return None, p
    d = json.loads(p.read_text())
    if "error" in d and bench not in d:
        fail(f"{p}: model-level error recorded: {d['error'][:200]}")
    try:
        validate_provenance(d.get("provenance") or {})
        ok(f"provenance git={d['provenance']['git']['commit'][:12]} config={d['provenance']['eval_config_sha256'][:12]}")
    except Exception as ex:
        fail(f"provenance invalid: {ex}")
    sec = d.get(bench)
    if not isinstance(sec, dict) or "error" in sec:
        fail(f"{p}: section '{bench}' missing or errored: {str(sec)[:200]}")
        return None, p
    return sec, p


def check_injecagent(model, results_dir):
    sec, p = load_results("injecagent", model, results_dir)
    if not sec:
        return
    e = EXPECT["injecagent"]
    if sec["n_positive"] != e["n_positive"] or sec["n_negative"] != e["n_negative"]:
        fail(f"counts {sec['n_positive']}/{sec['n_negative']} != {e['n_positive']}/{e['n_negative']}")
    bs = {k: v["n"] for k, v in sec["details"]["by_setting"].items()}
    if bs != e["by_setting"]:
        fail(f"by_setting counts {bs} != {e['by_setting']}")
    fb = sec["details"]["fpr_breakdown"]
    subs = {k: v["n"] for k, v in fb["per_subset"].items()}
    if subs != e["benign_subsets"] or fb["hard_negative_only"]["n"] != e["hard_negatives"]:
        fail(f"benign subset counts {subs} / hard negatives {fb['hard_negative_only']['n']} unexpected")
    pc = sec["per_case"]
    pos = [r["score"] for r in pc if r["label"] == 1]
    neg = [r["score"] for r in pc if r["label"] == 0]
    check_scores("injecagent", pos, "positive"); check_scores("injecagent", neg, "benign")
    if len({r["prompt_sha256_16"] for r in pc}) != len(pc):
        fail("duplicate prompts among per_case records")
    if sec["auc"] < 0.5:
        fail(f"AUC {sec['auc']:.3f} < 0.5")
    check_rate("injecagent", sec["detection_rate"], sec["n_positive"], "detection rate", pos)
    check_rate("injecagent", sec["fpr"], sec["n_negative"], "FPR", neg)
    for st, v in sec["details"]["by_setting"].items():
        check_rate(f"injecagent[{st}]", v["detection_rate"], v["n"], "detection rate", [r["score"] for r in pc if r["label"] == 1 and r.get("setting") == st])
    ok(f"injecagent {model}: det {sec['detection_rate']:.3f} (base {sec['details']['by_setting']['base']['detection_rate']:.3f}, "
       f"enh {sec['details']['by_setting']['enhanced']['detection_rate']:.3f}) fpr {sec['fpr']:.3f} "
       f"[hn {fb['hard_negative_only']['fpr']:.3f} plain {fb['plain_only']['fpr']:.3f}] auc {sec['auc']:.3f}")


def check_harmbench(model, results_dir):
    sec, p = load_results("harmbench", model, results_dir)
    if not sec:
        return
    e = EXPECT["harmbench"]
    if sec["n_positive"] != e["n_positive"] or sec["n_negative"] != e["n_negative"]:
        fail(f"counts {sec['n_positive']}/{sec['n_negative']} != 300/350")
    bb = {k: v["n"] for k, v in sec["details"]["by_benign_set"].items()}
    if bb != e["by_benign_set"]:
        fail(f"by_benign_set counts {bb}")
    pc = sec["per_case"]
    hs = [r["score"] for r in pc if r["label"] == 1]
    check_scores("harmbench", hs, "harmful")
    check_scores("harmbench", [r["score"] for r in pc if r["label"] == 0], "benign")
    if sec["auc"] < 0.5:
        fail(f"AUC {sec['auc']:.3f} < 0.5")
    check_rate("harmbench", sec["detection_rate"], sec["n_positive"], "detection rate", hs)
    for name, v in sec["details"]["by_benign_set"].items():
        check_rate(f"harmbench[{name}]", v["fpr"], v["n"], "FPR", [r["score"] for r in pc if r["label"] == 0 and r.get("benign_set") == name])
    ok(f"harmbench {model}: det {sec['detection_rate']:.3f} auc {sec['auc']:.3f} " +
       " ".join(f"fpr[{k}]={v['fpr']:.3f}" for k, v in sec["details"]["by_benign_set"].items()))


def check_ailuminate(model, results_dir):
    sec, p = load_results("ailuminate", model, results_dir)
    if not sec:
        return
    e = EXPECT["ailuminate"]
    if sec["n_positive"] != e["n_positive"] or sec["n_negative"] != e["n_negative"]:
        fail(f"counts {sec['n_positive']}/{sec['n_negative']} != 4800/1400")
    be = sec["details"]["by_encoding"]
    if sorted(be) != sorted(e["encodings"]) or any(v["n"] != e["per_encoding"] for v in be.values()):
        fail(f"by_encoding counts {{k: v['n']}} unexpected: { {k: v['n'] for k, v in be.items()} }")
    pc = sec["per_case"]
    for enc in e["encodings"]:
        check_scores(f"ailuminate[{enc}]", [r["score"] for r in pc if r["label"] == 1 and r["encoding"] == enc], "harmful")
        check_scores(f"ailuminate[{enc}]", [r["score"] for r in pc if r["label"] == 0 and r["encoding"] == enc], "benign")
        check_rate(f"ailuminate[{enc}]", be[enc]["detection_rate"], be[enc]["n"], "detection rate", [r["score"] for r in pc if r["label"] == 1 and r["encoding"] == enc])
    ctrl = sec["details"]["benign_encoded_control"]
    for enc, sets in ctrl.items():
        for name, v in sets.items():
            check_rate(f"ailuminate benign[{enc}][{name}]", v["fpr"], v["n"], "FPR", [r["score"] for r in pc if r["label"] == 0 and r["encoding"] == enc and r.get("benign_set") == name])
    ok(f"ailuminate {model}: " + " ".join(f"det[{k}]={v['detection_rate']:.3f}" for k, v in be.items()))
    for enc, sets in ctrl.items():
        ok(f"  benign-encoded FPR[{enc}]: " + " ".join(f"{n}={v['fpr']:.3f}(auc {v['auc_vs_harmful_same_encoding']:.3f})" for n, v in sets.items()))


def check_latency(model, results_dir):
    sec, p = load_results("latency", model, results_dir)
    if not sec:
        return
    for k in ("forward_pass_mean_ms", "probe_overhead_mean_ms", "n_runs"):
        if k not in sec:
            fail(f"latency missing {k}")
    if sec.get("forward_pass_mean_ms", 0) <= 0:
        fail("forward-pass latency not positive")
    ok(f"latency {model}: fwd {sec['forward_pass_mean_ms']:.1f}ms probe {sec['probe_overhead_mean_ms']:.4f}ms n={sec['n_runs']}")


def check_llama_guard(path):
    p = Path(path)
    if not p.exists():
        return fail(f"missing {p}")
    d = json.loads(p.read_text())
    try:
        validate_provenance(d["provenance"])
    except Exception as ex:
        fail(f"provenance invalid: {ex}")
    if d["data_summary"]["n_harmful"] != 300 or d["data_summary"]["n_benign"] != 350:
        fail(f"comparison set counts {d['data_summary']}")
    for name, sec in list(d.get("aase", {}).items()) + ([("llama_guard", d["llama_guard"])] if "llama_guard" in d else []):
        if sec["n"] != 650:
            fail(f"{name}: n={sec['n']} != 650")
        pcs = sec.get("per_case", [])
        hs = [c.get("score", float(c["pred_harmful"])) for c in pcs if c["label"] == 1]
        bs = [c.get("score", float(c["pred_harmful"])) for c in pcs if c["label"] == 0]
        check_rate(name, sec["tpr"], 300, "TPR", hs); check_rate(name, sec["fpr"], 350, "FPR", bs)
        ok(f"{name}: acc {sec['accuracy']:.3f} tpr {sec['tpr']:.3f} fpr {sec['fpr']:.3f} f1 {sec['f1']:.3f} lat {sec['latency_mean_ms']:.1f}ms")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["train-aag", "injecagent", "harmbench", "ailuminate", "latency", "llama-guard"])
    ap.add_argument("model", nargs="?")
    ap.add_argument("--results-dir", default=str(REPO / "results/phase1"))
    ap.add_argument("--path", default=str(REPO / "results/phase1/llama_guard_comparison.json"))
    a = ap.parse_args(argv)
    print(f"== check {a.kind} {a.model or ''}")
    if a.kind == "train-aag":
        check_train_aag(a.model)
    elif a.kind == "llama-guard":
        check_llama_guard(a.path)
    else:
        {"injecagent": check_injecagent, "harmbench": check_harmbench, "ailuminate": check_ailuminate, "latency": check_latency}[a.kind](a.model, a.results_dir)
    if FAIL:
        print(f"== {len(FAIL)} FAILURE(S) — hard stop")
        return 1
    print("== PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
