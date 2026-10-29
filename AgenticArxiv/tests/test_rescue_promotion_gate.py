"""Prevent adaptive test-set use in the new research promotion entry point."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from rescue_promotion_gate import evaluate


def summary(ok, task="promotion"):
    return {"details": [{"task_id": task, "trial": 0, "agent_type": "regex",
                         "completed": True, "tool_accurate": ok, "arg_applicable": False,
                         "ref_applicable": False, "parse_fail": 0, "tool_fail": 0,
                         "terminal_semantics_accurate": True}]}


@pytest.fixture
def splits():
    return {"split": {"train": ["train"], "search_dev": ["search"],
                      "promotion_val": ["promotion"], "final_test": ["test"]}}


def test_promotion_requires_positive_gain(splits):
    assert evaluate(summary(False), summary(True), splits)["accepted"]
    assert not evaluate(summary(True), summary(True), splits)["accepted"]
    assert not evaluate(summary(True), summary(False), splits)["accepted"]


@pytest.mark.parametrize("task", ["test", "train", "search"])
def test_other_splits_cannot_be_relabelled_as_promotion(splits, task):
    with pytest.raises(ValueError, match="exactly promotion_val"):
        evaluate(summary(False, task), summary(True, task), splits)


def test_cross_split_overlap_is_rejected(splits):
    splits["split"]["final_test"] = ["promotion"]
    with pytest.raises(ValueError, match="overlapping"):
        evaluate(summary(False), summary(True), splits)


def test_missing_trials_do_not_promote(splits):
    assert not evaluate(summary(False), summary(True), splits, k=3)["accepted"]


def test_error_and_duplicate_rows_are_rejected(splits):
    bad = summary(True)
    bad["error_count"] = 1
    with pytest.raises(ValueError, match="execution errors"):
        evaluate(summary(False), bad, splits)
    bad = summary(True)
    bad["details"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(summary(False), bad, splits)
