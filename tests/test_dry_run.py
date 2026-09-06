"""C3 — benchmark_vllm.py --dry-run exercises the full data path with no model.

Also checks that the driver loads data *before* touching any model, that the results-consuming
ROC script refuses empty input, and that the per-case AUC helper is exact.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "aase_vllm/scripts/benchmark_vllm.py"
ROC = REPO / "aase_vllm/scripts/generate_roc_curves_vllm.py"


def _run(*args, check=True):
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=REPO, capture_output=True, text=True, timeout=300, check=False)


def test_dry_run_all(injecagent_dir, harmbench_dir):
    r = _run("--dry-run")
    assert r.returncode == 0, r.stderr[-2000:]
    out = r.stdout
    assert "DRY RUN OK" in out
    assert "injections (label 1): 2108 records, 2108 distinct prompts" in out
    assert "by setting: {'base': 1054, 'enhanced': 1054}" in out
    assert "benign cases (label 0): 120 records, 120 distinct prompts" in out
    assert "subsets: {'injecagent_user': 17, 'agentdojo': 63, 'authored': 40}; hard negatives: 36" in out
    assert out.count("'subset': 'authored'") >= 2 and out.count("'hard_negative_kind': 'natural'") >= 1
    assert "harmful behaviors (label 1): 300 records, 300 distinct prompts" in out
    assert "benign counterpart (label 0): 350 records, 350 distinct prompts" in out
    assert "by benign set: {'xstest_safe': 250, 'jbb_benign': 100}" in out
    # 3 samples per split are printed
    assert out.count("[test_dh_base_") >= 3 and out.count("[injuser_") >= 1 and out.count("[authored_") >= 2
    assert "vllm" not in r.stderr.lower(), "dry run must not import vLLM"


def test_dry_run_injecagent_only(injecagent_dir):
    r = _run("--dry-run", "--benchmark", "injecagent")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "[HarmBench / AF]" not in r.stdout and "[InjecAgent / AAG]" in r.stdout


def test_dry_run_fails_loudly_on_missing_data(tmp_path):
    r = _run("--dry-run", "--benchmark", "injecagent", "--injecagent-dir", str(tmp_path / "missing"))
    assert r.returncode != 0
    assert "DataLoadError" in r.stderr and "missing" in r.stderr
    assert "DRY RUN OK" not in r.stdout
    assert "Now transfer all" not in r.stdout + r.stderr


def test_dry_run_legacy_set_only_with_flag(harmbench_dir):
    r = _run("--dry-run", "--benchmark", "harmbench", "--use-legacy-authored-set")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "legacy_authored (NOT HarmBench)" in r.stdout
    assert "harmful behaviors (label 1): 56 records" in r.stdout and "benign counterpart (label 0): 31 records" in r.stdout


def test_roc_script_refuses_nothing(tmp_path):
    r = subprocess.run([sys.executable, str(ROC), "--results-dir", str(tmp_path)], cwd=REPO, capture_output=True, text=True)
    assert r.returncode != 0 and "DataLoadError" in r.stderr
    r = subprocess.run([sys.executable, str(ROC), "--results-dir", str(tmp_path / "nope")], cwd=REPO, capture_output=True, text=True)
    assert r.returncode != 0 and "DataLoadError" in r.stderr


def test_roc_script_rejects_results_without_per_case(tmp_path):
    (tmp_path / "m.json").write_text(json.dumps({"model": "x", "injecagent": {"detection_rate": 1.0, "details": {"threshold": 0}}}))
    r = subprocess.run([sys.executable, str(ROC), "--results-dir", str(tmp_path)], cwd=REPO, capture_output=True, text=True)
    assert r.returncode != 0 and "per_case" in r.stderr


def test_roc_script_consumes_real_per_case_results(tmp_path):
    rng = np.random.default_rng(0)
    per_case = [{"id": f"p{i}", "label": 1, "score": float(s)} for i, s in enumerate(rng.normal(1, 1, 40))] + \
               [{"id": f"n{i}", "label": 0, "score": float(s)} for i, s in enumerate(rng.normal(0, 1, 60))]
    res = {"model": "org/model-a", "injecagent": {"details": {"threshold": 0.5}, "per_case": per_case, "data_summary": {"n": 100}}}
    (tmp_path / "org_model_a.json").write_text(json.dumps(res))
    out = tmp_path / "roc"
    r = subprocess.run([sys.executable, str(ROC), "--results-dir", str(tmp_path), "--output", str(out)], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    data = json.loads((tmp_path / "roc_data.json").read_text())
    assert len(data["org/model-a"]["aag"]["positive"]) == 40 and len(data["org/model-a"]["aag"]["negative"]) == 60
    assert (out.with_suffix(".png")).exists() or (tmp_path / "roc.png").exists()


def test_auc_helper_is_exact():
    sys.path.insert(0, str(SCRIPT.parent))
    from benchmark_vllm import auc_mann_whitney
    assert auc_mann_whitney(np.array([1, 2, 3]), np.array([0, 0.5])) == 1.0
    assert auc_mann_whitney(np.array([0, 0.5]), np.array([1, 2, 3])) == 0.0
    assert abs(auc_mann_whitney(np.array([1, 2]), np.array([1, 2])) - 0.5) < 1e-12
