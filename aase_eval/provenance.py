"""Run provenance embedded in every results / probe-metadata JSON (D3).

``run_provenance(cfg)`` returns a dict with the git commit (and dirty flag), the sha256 of the eval config file,
the sha256 of every dataset file the run can read, the resolved versions of the packages that matter, and the
host/python/CUDA facts.  It is computed at write time so a results file can always be traced to exact code + data.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional

from .config import REPO_ROOT, resolve_path

PACKAGES = ("numpy", "pyyaml", "torch", "transformers", "vllm", "bitsandbytes", "safetensors", "accelerate", "huggingface_hub", "triton", "xformers", "flashinfer-python")


def sha256_file(path) -> Optional[str]:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state(repo: Path = REPO_ROOT) -> Dict[str, Any]:
    def run(*args):
        try:
            return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None
    commit = run("rev-parse", "HEAD")
    dirty = run("status", "--porcelain")
    return {"commit": commit, "branch": run("rev-parse", "--abbrev-ref", "HEAD"), "dirty": bool(dirty) if dirty is not None else None,
            "dirty_files": (dirty.splitlines()[:20] if dirty else []),
            "submodule_harmbench": run("-C", "third_party/harmbench", "rev-parse", "HEAD")}


def package_versions() -> Dict[str, Optional[str]]:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def dataset_hashes(cfg: Dict[str, Any]) -> Dict[str, Optional[str]]:
    ds = cfg.get("datasets", {})
    files = {
        "eval_config": cfg.get("_config_path"),
        "harmbench_behaviors_text_all": resolve_path(ds["harmbench_dir"]) / "data" / "behavior_datasets" / "harmbench_behaviors_text_all.csv" if "harmbench_dir" in ds else None,
        "jbb_benign_csv": resolve_path(ds["jbb_benign_csv"]) if "jbb_benign_csv" in ds else None,
        "xstest_csv": resolve_path(ds["xstest_csv"]) if "xstest_csv" in ds else None,
        "ailuminate_csv": resolve_path(ds["ailuminate_csv"]) if "ailuminate_csv" in ds else None,
        "aag_benign_eval": resolve_path(cfg.get("injecagent", {}).get("benign_set_path", "datasets/aag_benign_eval.jsonl")),
        "legacy_authored_eval_set": REPO_ROOT / "aase_eval" / "legacy_authored_eval_set.json",
    }
    inj = resolve_path(ds["injecagent_dir"]) if "injecagent_dir" in ds else None
    if inj:
        for name in ("attacker_cases_dh.jsonl", "attacker_cases_ds.jsonl", "user_cases.jsonl", "test_cases_dh_base.json",
                     "test_cases_ds_base.json", "test_cases_dh_enhanced.json", "test_cases_ds_enhanced.json"):
            files[f"injecagent/{name}"] = inj / "data" / name
    return {k: (sha256_file(v) if v else None) for k, v in files.items()}


def cuda_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"available": False}
    try:
        import torch
        info["torch_cuda_version"] = torch.version.cuda
        info["available"] = bool(torch.cuda.is_available())
        if info["available"]:
            info["device_name"] = torch.cuda.get_device_name(0)
            info["device_count"] = torch.cuda.device_count()
    except Exception as e:  # torch missing or broken: record, never fail the run
        info["error"] = str(e)[:200]
    return info


def run_provenance(cfg: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    prov = {
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": git_state(),
        "eval_config_sha256": sha256_file(cfg.get("_config_path")) if cfg.get("_config_path") else None,
        "datasets_sha256": dataset_hashes(cfg),
        "packages": package_versions(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cuda": cuda_info(),
        "argv": sys.argv,
        "env": {k: os.environ.get(k) for k in ("AASE_EVAL_CONFIG", "VLLM_ALLOW_INSECURE_SERIALIZATION", "CUDA_VISIBLE_DEVICES") if k in os.environ},
    }
    if extra:
        prov.update(extra)
    return prov


REQUIRED_PROVENANCE_KEYS = ("written_at", "git", "eval_config_sha256", "datasets_sha256", "packages", "python", "cuda")


def validate_provenance(prov: Dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_PROVENANCE_KEYS if k not in prov]
    if missing:
        raise ValueError(f"results provenance is missing {missing}")
    if not prov["git"].get("commit"):
        raise ValueError("results provenance has no git commit")
    if not prov.get("eval_config_sha256"):
        raise ValueError("results provenance has no eval_config sha256")
    if not any(prov["datasets_sha256"].values()):
        raise ValueError("results provenance has no dataset hashes")
