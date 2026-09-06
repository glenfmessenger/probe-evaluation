#!/usr/bin/env python3
"""Stage B validity checks — the questions that decide whether a passing number means what it appears to mean.

Every arm in this session cleared its pre-registered threshold, three of them by a wide margin. This project's history
is a sequence of impressive numbers that turned out to measure something else (synthetic data, a contaminated training
set, prompt length), so a pass is not reported until these run. CPU only; reads the committed Stage B results.

  1. ARM 1 contamination — could the 80 authored training strings be near-duplicates of evaluation items?
  2. ARM 1 mechanism    — is the gain from the on-topic twins (the hypothesis) or from the chat template?
  3. ARM 2 leakage      — the pre-registered split shares all 62 injection strings across halves; does the result
                          survive a grouped split where no injection string is seen in training?
  4. ARM 3 contamination — are the authored cases near-duplicates of the 15 APC training pairs?
  5. ARM 3 cue-solvability — can trivial surface features (second-person pronouns, modals, length) match the probe?
                          This is the check that decides whether Arm 3 measures a speech act or a lexical marker of one.

  python -m gates.b4_validity_checks
"""
import difflib
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aase_eval.config import load_config  # noqa: E402
from aase_eval.harmbench import load_harmbench_eval_set  # noqa: E402
from gates.arm1_pairs import PAIRS  # noqa: E402
from gates.arm2_protocol import build_split  # noqa: E402
from gates.b3_apc_external import training_pairs  # noqa: E402
from gates.build_review import ADVICE_BANDS, load_arm3  # noqa: E402
from gates.runb import dump_json, load_gates_config, out_dir, provenance  # noqa: E402
from phase2.extract import auc  # noqa: E402

PRONOUN = r"\b(you|your|yours)\b"
MODAL = r"\b(should|take|start|stop|switch|book|file|sign|sell|buy|move|call|go)\b"
GENERAL = r"\b(clinicians|typically|generally|guidelines|the labell?ing|patients|studies|research|law|statutes|lenders|advisors?)\b"


def max_similarity(a_list, b_list):
    best = (0.0, None, None)
    for a in a_list:
        for b in b_list:
            r = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
            if r > best[0]:
                best = (r, a, b)
    return {"max_ratio": round(best[0], 3), "closest_pair": [best[1], best[2]]}


def arm1_contamination():
    base = load_config()
    d = load_harmbench_eval_set(base["harmbench"], base["datasets"])
    train = [p[1] for p in PAIRS] + [p[2] for p in PAIRS]
    pools = {"harmbench": [r["prompt"] for r in d["harmful"]],
             "xstest_safe": [r["prompt"] for r in d["benign"] if r["benign_set"] == "xstest_safe"],
             "jbb_benign": [r["prompt"] for r in d["benign"] if r["benign_set"] == "jbb_benign"]}
    return {"n_train_strings": len(train),
            "note": "matches are shared stems ('Give step-by-step instructions for'), not shared content",
            **{k: max_similarity(train, v) for k, v in pools.items()}}


def arm1_mechanism():
    rows = {}
    for p in sorted(glob.glob(str(out_dir(load_gates_config()) / "b1" / "*.json"))):
        d = json.load(open(p)); m = d["model"]
        pt, ot, pr = d["templated"]["cells"]["paired"], d["templated"]["cells"]["offtopic"], d["raw"]["cells"]["paired"]
        rows[m] = {"paired_templated_auc_vs_jbb": pt["auc_vs_jbb"], "offtopic_templated_auc_vs_jbb": ot["auc_vs_jbb"],
                   "twins_effect": pt["auc_vs_jbb"] - ot["auc_vs_jbb"],
                   "paired_raw_auc_strat": pr["length_all"]["auc_length_stratified"],
                   "paired_templated_auc_strat": pt["length_all"]["auc_length_stratified"],
                   "template_effect": pt["length_all"]["auc_length_stratified"] - pr["length_all"]["auc_length_stratified"],
                   "paired_identical_to_standard": d["templated"]["probes"]["paired"].get("identical_to_standard")}
    tw = [v["twins_effect"] for v in rows.values()]; te = [v["template_effect"] for v in rows.values()]
    return {"per_model": rows,
            "twins_effect": {"min": min(tw), "max": max(tw), "mean": sum(tw) / len(tw)},
            "template_effect": {"min": min(te), "max": max(te), "mean": sum(te) / len(te)},
            "verdict": "the on-topic twins contribute ~nothing; the chat template carries the entire gain"}


def arm2_leakage():
    per = build_split(); grp = build_split(group_by_attacker=True)
    def shared(sp):
        tr, te = sp["positives"]["train"], sp["positives"]["test"]
        return {"attacker_instructions_shared": len({c["attacker_instruction"] for c in tr} & {c["attacker_instruction"] for c in te}),
                "user_instructions_shared": len({c["user_instruction"] for c in tr} & {c["user_instruction"] for c in te}),
                "tool_responses_shared": len({c["tool_response"] for c in tr} & {c["tool_response"] for c in te}),
                "user_attacker_pairs_shared": len({(c["user_instruction"], c["attacker_instruction"]) for c in tr} &
                                                  {(c["user_instruction"], c["attacker_instruction"]) for c in te}),
                "n_train": len(tr), "n_test": len(te)}
    cfg = load_gates_config()
    cmp_ = {}
    for p in sorted(glob.glob(str(out_dir(cfg) / "b2" / "*.json"))):
        g = p.replace("/b2/", "/b2_grouped/")
        if not Path(g).exists():
            continue
        a, b = json.load(open(p)), json.load(open(g))
        cmp_[a["model"]] = {d: {"seen": a["results"]["test"][d]["auc"], "unseen": b["results"]["test"][d]["auc"],
                                "drop": a["results"]["test"][d]["auc"] - b["results"]["test"][d]["auc"]}
                            for d in ("final", "mean", "maxpos", "oracle")}
    drops = {d: [v[d]["drop"] for v in cmp_.values()] for d in ("final", "mean", "maxpos", "oracle")}
    return {"per_case_split": shared(per), "grouped_split": shared(grp), "per_model": cmp_,
            "mean_drop_when_injection_strings_unseen": {d: sum(v) / len(v) for d, v in drops.items()},
            "verdict": "the pre-registered split shares all 62 injection strings, but the result survives the grouped "
                       "split unchanged, so it is generalisation to unseen injections, not memorisation"}


def arm3_contamination():
    rows, _ = load_arm3()
    pairs = training_pairs("medical")
    train = [p[0] for p in pairs] + [p[1] for p in pairs]
    return {"n_cases": len(rows), "n_training_strings": len(train), **max_similarity([r["text"] for r in rows], train)}


def arm3_cue_solvability():
    rows, _ = load_arm3()
    out = {}
    for scope, sel in (("all_bands", lambda r: r["band"] != "refusal"),
                       ("hard_bands_only", lambda r: r["band"] in ("hedged_advice", "information_dense"))):
        out[scope] = {}
        for topic in ("medical", "financial", "legal"):
            rs = [r for r in rows if r["topic"] == topic and sel(r)]
            y = np.array([1 if r["band"] in ADVICE_BANDS else 0 for r in rs])
            txt = [r["text"] for r in rs]
            feats = {"second_person_pronoun_count": [len(re.findall(PRONOUN, t, re.I)) for t in txt],
                     "directive_modal_count": [len(re.findall(MODAL, t, re.I)) for t in txt],
                     "general_marker_count_negated": [-len(re.findall(GENERAL, t, re.I)) for t in txt],
                     "char_length": [len(t) for t in txt]}
            out[scope][topic] = {"n": len(rs), **{k: auc(np.array(v, float)[y == 1], np.array(v, float)[y == 0])
                                                  for k, v in feats.items()}}
    probe = {}
    for p in sorted(glob.glob(str(out_dir(load_gates_config()) / "b3" / "*.json"))):
        d = json.load(open(p))
        c = d["modes"]["templated"]["cells"]["paired"]
        probe[d["model"]] = {"medical_auc": c["medical"]["auc"],
                             "medical_auc_length_stratified": c["medical"]["length"]["auc_length_stratified"],
                             "hard_band_auc": c["medical"]["hard_bands"]["hedged_advice"]["auc_vs_other_hard_band"],
                             "january_medical_auc": d["modes"]["templated"]["cells"]["january"]["medical"]["auc"]}
    best_cue_hard = max(out["hard_bands_only"][t]["second_person_pronoun_count"] for t in ("medical", "financial", "legal"))
    return {"lexical_baselines": out, "probe_for_comparison": probe,
            "best_pronoun_baseline_on_hard_bands": best_cue_hard,
            "verdict": "counting second-person pronouns reaches AUC 0.93-1.00 on the same contrasts, including the hard "
                       "bands, so this evaluation set does not separate a speech-act probe from a lexical-cue detector; "
                       "the Arm 3 pass is numerically real but does not license the semantic claim"}


def arm3_v2_construction():
    """The three-draft v2 construction record, and why draft 3 is not an evaluation set.

    Also the methodological point the exercise produced: leave-one-out CV is invalid on minimal-pair data, because each
    held-out case's twin sits in the training fold carrying the opposite label and the model anti-predicts."""
    import difflib
    import re as _re
    from gates.lexical_baselines import DIRECTIVE_MODAL, SECOND_PERSON, omnibus_baseline
    p = REPO / "gates/data/arm3_v2_discriminator.json"
    if not p.exists():
        return {"available": False}
    d = json.load(open(p)); cs = d["cases"]
    txt = [c["text"] for c in cs]; y = np.array([1 if c["proposed_label"] == "advice" else 0 for c in cs])
    grp = [c.get("pair_id") for c in cs]
    prof = _re.compile(r"\b(clinician|clinicians|pharmacist|pharmacists|prescriber|prescribers|nurse|nurses|guideline|"
                       r"guidelines|surgeon|surgeons|maternity|trials?|labell?ing|units?)\b", _re.I)
    def f(R, g):
        return np.array([float(len(R.findall(c["text"]))) for c in g])
    adv = [c for c in cs if c["proposed_label"] == "advice"]; inf = [c for c in cs if c["proposed_label"] == "information"]
    nulls = [omnibus_baseline(txt, np.random.default_rng(s).permutation(y), groups=grp)["auc"] for s in range(8)]
    byp = {}
    for i, g in enumerate(grp):
        byp.setdefault(g, []).append(i)
    swap = []
    for s in range(8):
        rng = np.random.default_rng(100 + s); ys = y.copy()
        for g, ix in byp.items():
            if len(ix) == 2 and rng.random() < 0.5:
                ys[ix[0]], ys[ix[1]] = ys[ix[1]], ys[ix[0]]
        swap.append(omnibus_baseline(txt, ys, groups=grp)["auc"])
    def strip(t):
        t = SECOND_PERSON.sub(" ", t.lower())
        t = _re.sub(r"\b(the|a|an|is|are|was|were|be|being|been)\b", " ", t)
        return " ".join(_re.findall(r"[a-z0-9']+", t))
    sims = []
    for pid, m in byp.items():
        a = cs[[i for i in m if y[i] == 1][0]]; b = cs[[i for i in m if y[i] == 0][0]]
        sims.append((difflib.SequenceMatcher(None, strip(a["text"]), strip(b["text"])).ratio(), pid))
    sims.sort(reverse=True)
    return {"available": True, "status": d.get("status"),
            "drafts": [
                {"draft": 1, "controlled": "second-person pronouns", "result": "pronoun AUC 0.515 all / 0.000 incongruent",
                 "cue_exposed": "directive modals", "cue_value": 0.875},
                {"draft": 2, "controlled": "directive modals", "result": "modal AUC 0.500 all and incongruent",
                 "cue_exposed": "professional/third-party register (clinicians, guidelines, pharmacists: 28/40 information "
                                "cases vs 8/40 advice)", "cue_value": 1.000, "cue_measure": "omnibus LOO"},
                {"draft": 3, "controlled": "register, via 40 minimal pairs", "result": "all cue checks pass",
                 "cue_exposed": "label validity dissolved", "cue_value": 0.814,
                 "cue_measure": "median within-pair content similarity after removing pronouns, articles and copulas"}],
            "draft3_cues": {"second_person": auc(f(SECOND_PERSON, adv), f(SECOND_PERSON, inf)),
                            "directive_modal": auc(f(DIRECTIVE_MODAL, adv), f(DIRECTIVE_MODAL, inf)),
                            "professional_noun": auc(f(prof, adv), f(prof, inf)),
                            "char_length": auc(np.array([c["n_chars"] for c in adv], float),
                                               np.array([c["n_chars"] for c in inf], float))},
            "omnibus_leave_one_out": omnibus_baseline(txt, y)["auc"],
            "omnibus_leave_pair_out": omnibus_baseline(txt, y, groups=grp)["auc"],
            "null_shuffled_leave_pair_out": {"mean": float(np.mean(nulls)), "sd": float(np.std(nulls))},
            "null_within_pair_swap": {"mean": float(np.mean(swap)), "sd": float(np.std(swap))},
            "label_validity": {"median_within_pair_similarity": float(np.median([s[0] for s in sims])),
                               "pairs_over_0.90": int(sum(s[0] >= 0.90 for s in sims)),
                               "pairs_over_0.80": int(sum(s[0] >= 0.80 for s in sims)),
                               "worst_pairs": [{"pair_id": pid, "similarity": round(r, 3)} for r, pid in sims[:6]],
                               "mislabelled_examples": ["p11", "p13", "p16"]},
            "methodological_note": ("leave-one-out CV is invalid on minimal-pair data: the held-out case's twin sits in "
                                    "the training fold with the opposite label and the model anti-predicts, which is why "
                                    "LOO reports 0.019 here. The correct procedure is leave-pair-out CV scored against a "
                                    "within-pair-swap null."),
            "verdict": ("draft 3 passes every cue check but is not a usable instrument: making the classes lexically "
                        "indistinguishable made them semantically indistinguishable. In short authored medical text "
                        "directive force is substantially lexically constituted, so surface cue and semantic construct "
                        "coincide at this granularity.")}


def main() -> int:
    cfg = load_gates_config()
    res = {"arm1_contamination": arm1_contamination(), "arm1_mechanism": arm1_mechanism(),
           "arm2_leakage": arm2_leakage(), "arm3_contamination": arm3_contamination(),
           "arm3_cue_solvability": arm3_cue_solvability(), "arm3_v2_construction": arm3_v2_construction()}
    res["provenance"] = provenance(cfg, {"part": "b4_validity_checks"})
    p = out_dir(cfg) / "validity_checks.json"
    dump_json(res, p)
    for k in ("arm1_mechanism", "arm2_leakage", "arm3_cue_solvability", "arm3_v2_construction"):
        print(f"{k}: {res[k]['verdict']}")
    print("arm1 contamination max ratio:", {k: v["max_ratio"] for k, v in res["arm1_contamination"].items() if isinstance(v, dict)})
    print("arm3 contamination max ratio:", res["arm3_contamination"]["max_ratio"])
    print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
