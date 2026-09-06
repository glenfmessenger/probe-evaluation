"""Single eval configuration (``eval_config.yaml`` at the repo root).

All dataset paths in the config are relative to the repo root unless absolute.
Scripts should call ``load_config()`` once and take every path, count, model
and layer/threshold setting from it instead of editing literals.
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .errors import DataLoadError

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "eval_config.yaml"

_REQUIRED_TOP_LEVEL = ("models", "datasets", "injecagent", "harmbench", "probes", "output")


def resolve_path(p, base: Path = REPO_ROOT) -> Path:
    """Resolve a config path: absolute stays, relative is taken from the repo root.

    ``$VAR`` and ``~`` are expanded so the same config works on the VM.
    """
    p = os.path.expandvars(os.path.expanduser(str(p)))
    path = Path(p)
    return path if path.is_absolute() else (base / path)


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else Path(os.environ.get("AASE_EVAL_CONFIG", DEFAULT_CONFIG_PATH))
    if not cfg_path.exists():
        raise DataLoadError(f"eval config not found: {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    missing = [k for k in _REQUIRED_TOP_LEVEL if k not in cfg]
    if missing:
        raise DataLoadError(f"eval config {cfg_path} is missing required sections: {missing}")
    cfg["_config_path"] = str(cfg_path)
    return cfg
