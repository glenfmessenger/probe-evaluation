#!/usr/bin/env python3
"""ARM 3 v2 — score the ALREADY-TRAINED Arm 3 probes on the lexically-controlled discriminator set.

The question is what those probes learned, so nothing is fitted to v2. The paired direction is recomputed from the same
fixed 15 medical pairs at the same layer and feature; that computation is deterministic, so it reproduces the vector
used in B3 bit for bit, and this module asserts it by checking the recomputed threshold and training AUC against the
values recorded in the committed B3 results. If they disagree it raises rather than silently scoring a different probe.

v2 is built so that second-person marking, the cue that made the v1 result uninterpretable, is uninformative across the
set and inverted on the two incongruent cells (advice with no second-person marking, information with heavy
second-person marking). The lexical baseline is reported beside every probe number, per DECISIONS.md #11.

  python -m gates.b3_v2_score [--dry-run] [--models M ...]
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aase_eval.errors import DataLoadError  # noqa: E402
from gates.b3_apc_external import training_pairs  # noqa: E402
from gates.lexical_baselines import SECOND_PERSON, apc_baseline  # noqa: E402
from gates.runb import (check_scores, dump_json, format_mode, fpr_at_threshold, length_report, load_gates_config,  # noqa: E402
                        load_probe, out_dir, provenance, safe, train_rule)
from phase2.extract import auc, dist_stats, score  # noqa: E402

V2_PATH = REPO / "gates/data/arm3_v2_discriminator.json"
INCONGRUENT = ("advice_no_2p", "information_2p")


def load_v2():
    if not V2_PATH.exists():
        raise DataLoadError(f"v2 discriminator set not found at {V2_PATH}; it must be authored and signed off first")
    d = json.load(open(V2_PATH))
    cs = d["cases"]
    for c in cs:
        if c["n_chars"] != len(c["text"]) or c["second_person_count"] != len(SECOND_PERSON.findall(c["text"])):
            raise DataLoadError(f"v2 case {c['id']}: recorded n_chars/second_person_count disagree with the text")
    return d, cs


def b3_reference(cfg, model_name, mode, pname):
    p = out_dir(cfg) / "b3" / f"{safe(model_name)}.json"
    if not p.exists():
        return None
    return json.load(open(p))["modes"][mode]["probes"].get(pname)


def evaluate(sc, cases, thr):
    y = np.array([1 if c["proposed_label"] == "advice" else 0 for c in cases])
    L = np.array([len(c["text"]) for c in cases])
    inc = np.array([c["cell"] in INCONGRUENT for c in cases])
    out = {"n": len(cases), "threshold": thr, "auc_all_cells": auc(sc[y == 1], sc[y == 0]),
           "length": length_report(sc, y, L),
           "flag_rate_advice": fpr_at_threshold(sc[y == 1], thr), "flag_rate_information": fpr_at_threshold(sc[y == 0], thr),
           "cells": {}}
    if inc.any():
        si, yi = sc[inc], y[inc]
        out["auc_incongruent_cells"] = auc(si[yi == 1], si[yi == 0])
        out["incongruent_n"] = int(inc.sum())
    for cell in sorted({c["cell"] for c in cases}):
        idx = [i for i, c in enumerate(cases) if c["cell"] == cell]
        out["cells"][cell] = {"n": len(idx), "flag_rate": fpr_at_threshold(sc[idx], thr), "scores": dist_stats(sc[idx]),
                              "expected_label": cases[idx[0]]["proposed_label"]}
    return out


def run_model(cfg, model_name, cases):
    from phase2.extract import extract_last_token, load_model
    b3 = cfg["b3"]; policy = b3["policy"]
    shipped = load_probe(cfg, "apc", model_name, policy)
    layer = shipped["layer_index"]
    pairs = training_pairs(policy)
    model, tok, info = load_model(model_name, cfg.get("dtype", "bfloat16"))
    print(f"  {model_name}: APC/{policy} layer {layer}; scoring {len(cases)} v2 cases with already-trained probes", flush=True)
    out = {"model": model_name, "policy": policy, "layer_index": layer, "modes": {}}
    for mode in cfg["modes"]:
        tp = extract_last_token(model, tok, [format_mode(p[0], mode, model_name, tok) for p in pairs], [layer],
                                cfg["batch_size"], cfg["max_length"], feature=cfg["feature"])[:, 0, :]
        tn = extract_last_token(model, tok, [format_mode(p[1], mode, model_name, tok) for p in pairs], [layer],
                                cfg["batch_size"], cfg["max_length"], feature=cfg["feature"])[:, 0, :]
        acts = extract_last_token(model, tok, [format_mode(c["text"], mode, model_name, tok) for c in cases], [layer],
                                  cfg["batch_size"], cfg["max_length"], progress=f"{safe(model_name)}/{mode}",
                                  feature=cfg["feature"])[:, 0, :]
        probes = {"january": {**shipped, "source": "shipped Phase 1 APC probe"},
                  "paired": train_rule(tp, tn, "paired", pairs=True)}
        ref = b3_reference(cfg, model_name, mode, "paired")
        if ref:
            same = (abs(ref["threshold"] - probes["paired"]["threshold"]) < 1e-6
                    and abs(ref["train_auc"] - probes["paired"]["train_auc"]) < 1e-9)
            probes["paired"]["reproduces_b3_probe"] = bool(same)
            if not same:
                raise DataLoadError(f"recomputed paired probe does not reproduce the B3 probe for {model_name}/{mode}: "
                                    f"threshold {probes['paired']['threshold']} vs {ref['threshold']}")
        cells = {}
        for pname, pr in probes.items():
            sc = score(acts, pr["direction"], "january")
            check_scores(f"{model_name}|{mode}|v2-{pname}", sc)
            cells[pname] = {**evaluate(sc, cases, pr["threshold"]),
                            "reproduces_b3_probe": pr.get("reproduces_b3_probe"),
                            "scores": {c["id"]: float(v) for c, v in zip(cases, sc)}}
            e = cells[pname]
            print(f"    [{mode}] {pname:8s} all-cells AUC {e['auc_all_cells']:.3f} (strat {e['length']['auc_length_stratified']:.3f}) "
                  f"| incongruent-only AUC {e.get('auc_incongruent_cells', float('nan')):.3f} "
                  f"| flag adv/info {e['flag_rate_advice']:.2f}/{e['flag_rate_information']:.2f}", flush=True)
        out["modes"][mode] = {"probes": {k: {kk: vv for kk, vv in v.items() if kk != "direction"} for k, v in probes.items()},
                              "cells": cells}
    del model
    import torch; torch.cuda.empty_cache()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--models", nargs="*", default=None)
    a = ap.parse_args(argv)
    cfg = load_gates_config(a.config)
    d, cases = load_v2()
    models = a.models or [json.load(open(p))["model"] for p in sorted(glob.glob(str(out_dir(cfg) / "b3" / "*.json")))]
    base = apc_baseline(cases, lambda c: c["proposed_label"] == "advice", "v2/all")
    inc = [c for c in cases if c["cell"] in INCONGRUENT]
    base_inc = apc_baseline(inc, lambda c: c["proposed_label"] == "advice", "v2/incongruent")
    from collections import Counter
    print(f"B3v2: {len(cases)} cases {dict(Counter(c['cell'] for c in cases))}; rule {d['labelling_rule_version']}")
    print(f"B3v2 lexical baseline (the bar a semantic claim must clear): all cells second-person AUC "
          f"{base['second_person_count']:.3f}, length AUC {base['char_length']:.3f}; incongruent cells only "
          f"second-person AUC {base_inc['second_person_count']:.3f}")
    print(f"B3v2 models: {models} (probes recomputed from the same 15 pairs and asserted identical to B3; no fitting to v2)")
    od = out_dir(cfg) / "b3_v2"
    if a.dry_run:
        print("output ->", od / "<model>.json")
        return 0
    for m in models:
        out = run_model(cfg, m, cases)
        out["lexical_baseline"] = {"all_cells": base, "incongruent_cells_only": base_inc}
        out["v2_set"] = {"n": len(cases), "cells": dict(Counter(c["cell"] for c in cases)),
                         "labelling_rule_version": d["labelling_rule_version"]}
        out["provenance"] = provenance(cfg, {"part": "b3_v2", "model": m,
                                             "note": "already-trained probes only; no retraining, no fitting to v2"})
        p = od / f"{safe(m)}.json"; dump_json(out, p)
        print("  wrote", p, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
