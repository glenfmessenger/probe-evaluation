"""Prompt set for the Llama Guard 3 vs AASE comparison (paper Table `tab:guard_comparison`).

The paper's table was produced on 16 literal prompts embedded in ``aase_vllm/scripts/compare_llama_guard_vllm.py``
(VERIFICATION / SECOND_PASS reports).  For Phase 1 the comparison runs on real, pinned sets:

  harmful  = HarmBench behaviors (same selection as the AF benchmark: eval_config ``harmbench``)
  benign   = the AF benign sets (``harmbench.benign_sets``: XSTest safe 250 primary, JBB benign 100 secondary)

so accuracy / TPR / FPR / F1 for both systems are computed on identical, documented inputs.  The 16-prompt
Table-5 set is preserved in ``legacy_table5_prompts.json`` and is loadable ONLY with ``allow_legacy=True``
(CLI ``--use-legacy-table5-set``); it is never a fallback.
"""
import json
from pathlib import Path
from typing import Any, Dict, List

from .errors import DataLoadError
from .harmbench import load_harmbench_behaviors, load_benign_set

LEGACY_PATH = Path(__file__).resolve().parent / "legacy_table5_prompts.json"


def load_comparison_set(cfg: Dict[str, Any], datasets_cfg: Dict[str, Any], harmbench_dir=None) -> Dict[str, Any]:
    """cfg = eval_config['llama_guard_comparison'] merged with the harmbench section it references."""
    from .config import resolve_path

    hb = cfg["harmbench"]
    d = Path(harmbench_dir) if harmbench_dir else resolve_path(datasets_cfg["harmbench_dir"])
    harmful = load_harmbench_behaviors(d, hb.get("subset", "all"), tuple(hb.get("functional_categories", ("standard", "contextual"))),
                                       bool(hb.get("prepend_context", True)))
    benign: List[Dict[str, Any]] = []
    for name in cfg.get("benign_sets") or hb.get("benign_sets") or [hb.get("benign_set")]:
        if not name:
            raise DataLoadError("llama_guard_comparison needs at least one benign set")
        for r in load_benign_set(name, datasets_cfg):
            benign.append({**r, "benign_set": name})
    cap = cfg.get("max_per_class")
    if cap:
        harmful, benign = harmful[: int(cap)], benign[: int(cap)]
    if not harmful or not benign:
        raise DataLoadError("Llama Guard comparison set is empty")
    return {"harmful": harmful, "benign": benign,
            "summary": {"n_harmful": len(harmful), "n_benign": len(benign), "harmful_source": f"harmbench:{hb.get('subset', 'all')}",
                        "benign_sets": sorted({b['benign_set'] for b in benign}), "n_runs": int(cfg.get("n_runs", 3)),
                        "llama_guard_model": cfg.get("llama_guard_model", "meta-llama/Llama-Guard-3-8B"),
                        "aase_models": cfg.get("aase_models") or ["meta-llama/Llama-3.2-3B-Instruct"]}}


def load_legacy_table5_set(allow_legacy: bool = False, path: Path = LEGACY_PATH) -> Dict[str, Any]:
    if not allow_legacy:
        raise DataLoadError("the 16-prompt Table-5 set is legacy-only; pass --use-legacy-table5-set explicitly. It is never a fallback.")
    if not path.exists():
        raise DataLoadError(f"legacy Table-5 prompt set not found: {path}")
    data = json.loads(path.read_text())
    harmful = [{"id": f"legacy_t5_h{i}", "label": 1, "prompt": p, "source": "legacy_table5"} for i, p in enumerate(data["harmful"])]
    benign = [{"id": f"legacy_t5_b{i}", "label": 0, "prompt": p, "source": "legacy_table5", "benign_set": "legacy_table5"} for i, p in enumerate(data["benign"])]
    return {"harmful": harmful, "benign": benign, "summary": {"n_harmful": len(harmful), "n_benign": len(benign), "harmful_source": "legacy_table5 (16 authored prompts, NOT a benchmark)", "benign_sets": ["legacy_table5"]}}
