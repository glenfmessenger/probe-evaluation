"""pytest setup: make the repo root importable and expose the eval config + data paths.

Run from the repo root:  python -m pytest tests/ -q
These tests are CPU-only, need no model, and must pass before every eval session.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aase_eval import load_config, resolve_path  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config(REPO_ROOT / "eval_config.yaml")


@pytest.fixture(scope="session")
def injecagent_dir(cfg):
    d = resolve_path(cfg["datasets"]["injecagent_dir"])
    if not (d / "data").is_dir():
        pytest.skip(f"InjecAgent data not present at {d}")
    return d


@pytest.fixture(scope="session")
def harmbench_dir(cfg):
    d = resolve_path(cfg["datasets"]["harmbench_dir"])
    if not (d / "data" / "behavior_datasets").is_dir():
        pytest.skip(f"HarmBench submodule not checked out at {d} (git submodule update --init)")
    return d
