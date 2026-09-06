"""C2 — HarmBench + benign-set load tests, and proof that the authored list is gated."""
from pathlib import Path

import pytest

from aase_eval import DataLoadError
from aase_eval.harmbench import (
    load_benign_set, load_harmbench_behaviors, load_harmbench_eval_set, load_jbb_benign, load_xstest_safe,
)
from aase_eval.legacy import LEGACY_PATH, load_legacy_authored_set

EXPECTED_FUNCTIONAL = {"standard": 200, "contextual": 100, "copyright": 100}


def test_harmbench_all_400_rows_and_categories(harmbench_dir):
    recs = load_harmbench_behaviors(harmbench_dir, "all", ("standard", "contextual", "copyright"))
    assert len(recs) == 400
    counts = {}
    for r in recs:
        counts[r["functional_category"]] = counts.get(r["functional_category"], 0) + 1
    assert counts == EXPECTED_FUNCTIONAL
    assert len({r["id"] for r in recs}) == 400


def test_default_filter_is_300_standard_plus_contextual(cfg, harmbench_dir):
    recs = load_harmbench_behaviors(harmbench_dir, cfg["harmbench"]["subset"], tuple(cfg["harmbench"]["functional_categories"]))
    assert len(recs) == 300
    assert not any(r["functional_category"] == "copyright" for r in recs)
    ctx = [r for r in recs if r["functional_category"] == "contextual"]
    assert len(ctx) == 100 and all("\n\n" in r["prompt"] for r in ctx), "contextual behaviors must carry their ContextString"


def test_val_test_split_sizes(harmbench_dir):
    assert len(load_harmbench_behaviors(harmbench_dir, "val", ("standard", "contextual", "copyright"))) == 80
    assert len(load_harmbench_behaviors(harmbench_dir, "test", ("standard", "contextual", "copyright"))) == 320


def test_benign_sets(cfg):
    jbb = load_benign_set("jbb_benign", cfg["datasets"])
    assert len(jbb) == 100 and all(r["label"] == 0 for r in jbb) and len({r["prompt"] for r in jbb}) == 100
    xs = load_benign_set("xstest_safe", cfg["datasets"])
    assert len(xs) == 250 and all(r["label"] == 0 for r in xs)
    with pytest.raises(DataLoadError):
        load_benign_set("authored_31", cfg["datasets"])


def test_full_af_eval_set(cfg, harmbench_dir):
    data = load_harmbench_eval_set(cfg["harmbench"], cfg["datasets"], harmbench_dir)
    assert data["summary"]["n_harmful"] == 300 and data["summary"]["n_benign"] == 350
    assert data["summary"]["benign_sets"] == ["xstest_safe", "jbb_benign"]
    assert data["summary"]["n_benign_by_set"] == {"xstest_safe": 250, "jbb_benign": 100}
    assert all(r["benign_set"] in ("xstest_safe", "jbb_benign") for r in data["benign"])
    assert data["summary"]["harmful_breakdown"]["by_functional_category"] == {"standard": 200, "contextual": 100}
    assert set(data["summary"]["harmful_breakdown"]["by_semantic_category"]) == {
        "chemical_biological", "cybercrime_intrusion", "harassment_bullying", "harmful", "illegal", "misinformation_disinformation"}


def test_missing_harmbench_dir_raises(tmp_path):
    with pytest.raises(DataLoadError, match="harmbench_behaviors_text_all.csv"):
        load_harmbench_behaviors(tmp_path)


def test_wrong_columns_raise(tmp_path):
    p = tmp_path / "data" / "behavior_datasets"
    p.mkdir(parents=True)
    (p / "harmbench_behaviors_text_all.csv").write_text("prompt,category\nfoo,bar\n")
    with pytest.raises(DataLoadError) as ei:
        load_harmbench_behaviors(tmp_path)
    assert "Behavior" in str(ei.value) and "prompt" in str(ei.value)


def test_bad_benign_csv_raises(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(DataLoadError):
        load_jbb_benign(p)
    with pytest.raises(DataLoadError):
        load_xstest_safe(p)


def test_legacy_authored_set_is_gated():
    assert LEGACY_PATH.exists()
    with pytest.raises(DataLoadError, match="legacy-only"):
        load_legacy_authored_set()               # default: refused
    with pytest.raises(DataLoadError):
        load_legacy_authored_set(allow_legacy=False)
    data = load_legacy_authored_set(allow_legacy=True)
    assert len(data["harmful"]) == 56 and len(data["benign"]) == 31
    assert all(r["source"] == "legacy_authored" for r in data["harmful"] + data["benign"])


def test_legacy_set_is_not_reachable_from_harmbench_loader(tmp_path):
    """A missing HarmBench dir must raise, not return the 56-prompt list."""
    with pytest.raises(DataLoadError):
        load_harmbench_eval_set({"subset": "all", "benign_sets": ["jbb_benign"]}, {"harmbench_dir": str(tmp_path),
                                 "jbb_benign_csv": "third_party/benign_sets/jbb_benign_behaviors.csv"}, tmp_path)
