#!/usr/bin/env python3
"""
D2 — CPU smoke test of the activation-extraction path with the smallest real model (no GPU, no vLLM).

What it verifies, end to end, on ~5 real prompts from the pinned eval sets:
  1. the chat template applies (tokenizer.apply_chat_template) and what the harness actually feeds the model;
  2. layer-path resolution: the same rule the vLLM hook uses (model.language_model.model.layers for Gemma-3
     multimodal, else model.model.layers) resolves on the HF model, and the probe's layer_index is in range;
  3. hidden-state shape: a forward hook on layers[layer_index] yields [seq, hidden]; last-token vector == hidden_dim;
  4. HF `output_hidden_states[layer_index + 1]` equals the hook output (the vLLM hook is *post*-layer), i.e. the
     December HF-era scripts' `hidden_states[layer]` convention is one layer EARLIER than the vLLM probes' layer;
  5. a real pretrained probe (.npy) dots against the extracted activation and thresholds.

Usage:  python tools/smoke_cpu.py [--model meta-llama/Llama-3.2-1B-Instruct] [--n 5]
Writes results/preflight/smoke_cpu_<model_safe>.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aase_eval import load_config, resolve_path  # noqa: E402
from aase_eval.harmbench import load_harmbench_eval_set  # noqa: E402
from aase_eval.injecagent import load_injecagent_eval_set  # noqa: E402
from aase_eval.provenance import run_provenance  # noqa: E402


def resolve_layers(model):
    """Same rule as VLLMBenchmark._register_hook."""
    if hasattr(model, "language_model"):
        return model.language_model.model.layers, "model.language_model.model.layers"
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers, "model.model.layers"
    raise AttributeError(f"Cannot find layers in model: {type(model)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = load_config(args.config)
    model_safe = args.model.replace("/", "_").replace("-", "_")
    probes_dir = resolve_path(cfg["probes"]["dir"])
    report = {"model": args.model, "steps": [], "provenance": run_provenance(cfg, {"script": "tools/smoke_cpu.py"})}

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, device_map="cpu")
    model.eval()
    cfg_m = model.config
    text_cfg = getattr(cfg_m, "text_config", cfg_m)
    num_layers, hidden = text_cfg.num_hidden_layers, text_cfg.hidden_size
    report["steps"].append({"step": "load", "seconds": round(time.time() - t0, 1), "num_layers": num_layers, "hidden_dim": hidden, "dtype": "float32"})
    print(f"[load] {args.model}: {num_layers} layers, hidden {hidden}, {time.time() - t0:.0f}s")

    layers, path = resolve_layers(model)
    assert len(layers) == num_layers, (len(layers), num_layers)
    report["steps"].append({"step": "layer_path", "path": path, "n_layers_found": len(layers)})
    print(f"[layer path] {path} -> {len(layers)} layers")

    # probes for this model (AF + AAG), trained by the vLLM script (layer_index = post-layer index of the hook)
    probes = {}
    for kind in ("af", "aag"):
        p = probes_dir / kind / model_safe
        if Path(f"{p}.npy").exists():
            meta = json.load(open(f"{p}.json"))
            vec = np.load(f"{p}.npy"); vec = vec / np.linalg.norm(vec)
            assert vec.shape == (hidden,), (vec.shape, hidden)
            assert 0 <= meta["layer_index"] < num_layers
            probes[kind] = (vec, meta)
            print(f"[probe] {kind}: layer {meta['layer_index']}, threshold {meta['threshold']:.4f}, dim {vec.shape[0]}")
    if not probes:
        raise SystemExit(f"no pretrained probes for {args.model} under {probes_dir}")
    report["probes"] = {k: {"layer_index": m["layer_index"], "threshold": m["threshold"], "n_positive": m.get("n_positive")} for k, (v, m) in probes.items()}

    # prompts: 2 injections (one per setting) + 1 benign from the pinned set, 1 HarmBench, 1 XSTest
    aag = load_injecagent_eval_set(cfg["injecagent"], resolve_path(cfg["datasets"]["injecagent_dir"]))
    af = load_harmbench_eval_set(cfg["harmbench"], cfg["datasets"])
    picks = [("aag", next(r for r in aag["injections"] if r["setting"] == "base")),
             ("aag", next(r for r in aag["injections"] if r["setting"] == "enhanced")),
             ("aag", next(r for r in aag["benign"] if r["subset"] == "authored" and r["hard_negative"])),
             ("af", af["harmful"][0]), ("af", next(r for r in af["benign"] if r["benign_set"] == "xstest_safe"))][: args.n]

    captured = {}

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach()

    for kind, rec in picks:
        vec, meta = probes.get(kind) or next(iter(probes.values()))
        layer = meta["layer_index"]
        # 1. chat template: the harness scores the raw scaffold text (the vLLM path calls llm.generate on the string);
        #    show both so the difference is visible
        chat = tok.apply_chat_template([{"role": "user", "content": rec["prompt"]}], tokenize=False, add_generation_prompt=True)
        ids = tok(rec["prompt"], return_tensors="pt", truncation=True, max_length=2048)
        handle = layers[layer].register_forward_hook(hook)
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        handle.remove()
        h = captured["h"]                               # [batch, seq, hidden]
        assert h.dim() == 3 and h.shape[-1] == hidden, tuple(h.shape)
        last = h[0, -1, :].float().numpy()
        hs_post = out.hidden_states[layer + 1][0, -1, :].float().numpy()   # post-layer == hook output
        hs_hf_era = out.hidden_states[layer][0, -1, :].float().numpy()     # what the December HF scripts read at "layer"
        assert np.allclose(last, hs_post, atol=1e-4), "hook output must equal hidden_states[layer+1]"
        act = last / np.linalg.norm(last)
        score = float(np.dot(act, vec))
        score_hf_era = float(np.dot(hs_hf_era / np.linalg.norm(hs_hf_era), vec))
        row = {"probe": kind, "id": rec["id"], "label": rec["label"], "layer_index": layer, "seq_len": int(ids["input_ids"].shape[1]),
               "hidden_shape": list(h.shape), "score": score, "pred_positive": score > meta["threshold"],
               "score_if_hidden_states[layer]": score_hf_era, "chat_template_prefix": chat[:80], "raw_prefix": rec["prompt"][:80],
               "chat_template_adds_tokens": len(tok(chat)["input_ids"]) - int(ids["input_ids"].shape[1])}
        report["steps"].append(row)
        print(f"[{kind}] {rec['id']:<28} label={rec['label']} layer={layer} seq={row['seq_len']} shape={row['hidden_shape']} score={score:+.4f} "
              f"pred={row['pred_positive']} (hidden_states[layer] would give {score_hf_era:+.4f}); chat template adds {row['chat_template_adds_tokens']} tokens")

    out_dir = REPO / "results" / "preflight"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"smoke_cpu_{model_safe}.json"
    report["ok"] = True
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nSMOKE OK -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
