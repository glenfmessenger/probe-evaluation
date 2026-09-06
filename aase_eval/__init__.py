"""
aase_eval — shared, fail-loud data loaders for the AASE evaluation pipeline.

Every loader in this package reads real data from disk and raises
``DataLoadError`` (naming the file, the expected keys and the keys actually
found) when it cannot.  Nothing in this package ever substitutes synthetic or
authored prompts for missing data.  The only authored prompt set that still
exists (``legacy_authored_eval_set.json``) is loadable exclusively through
``aase_eval.legacy.load_legacy_authored_set`` behind an explicit flag.

Introduced in Phase 0 (branch ``eval-fix``); see PHASE0_REPORT.md.
"""
from .errors import DataLoadError  # noqa: F401
from .config import REPO_ROOT, load_config, resolve_path  # noqa: F401

__all__ = ["DataLoadError", "REPO_ROOT", "load_config", "resolve_path"]
