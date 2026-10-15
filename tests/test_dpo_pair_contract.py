"""DPO accounting and fail-closed contracts, without producing model traces."""

from __future__ import annotations

import inspect
import json

import pytest

from agentic_arxiv.data_pipeline.dpo_pairs import (
    ACCEPTED_CODES, RESULT_CODES, build_dpo_pairs, summarize_dpo_results,
)
from agentic_arxiv.provenance import (
    dpo_content_hash, dpo_trace_decisions, load_dpo_package, validate_dpo_row,
)
from agentic_arxiv.runtime.react_contract import parse_react_response, validate_react_messages
from agentic_arxiv.tasks.tasks_expanded import EXPANDED_SPECS


def test_requested_trial_slots_remain_in_denominator_without_any_rollouts():
    # Only existing task IDs and requested counters, not fabricated episodes.
    task_ids = [spec.id for spec in EXPANDED_SPECS[:2]]
    rows, report = build_dpo_pairs([], task_ids=task_ids, trials_per_task=5,
                                  min_reward_gap=0.05)
    assert rows == []
    assert report["requested_rollouts"] == report["invalid_rollouts"] == 10
    assert report["requested_combinations"] == 20
    assert report["result_counts"]["sampling_failure"] == 20
    assert sum(report["result_counts"].values()) == 20
    assert report["task_yield_rate"] == report["combination_acceptance_rate"] == 0
    assert len(report["warnings"]) == 1
    for task in report["per_task"].values():
        assert task["requested_rollouts"] == 5
        assert task["requested_combinations"] == 10


def test_unique_primary_codes_distinguish_accepted_duplicate_and_unselected():
    # Pure classification arithmetic; no prompt, response, action, or rollout.
    results = [{"task_id": str(index), "result": code}
               for index, code in enumerate(RESULT_CODES)]
    report = summarize_dpo_results(results, requested_tasks=len(RESULT_CODES),
                                   requested_combinations=len(RESULT_CODES),
                                   requested_rollouts=20, completed_rollouts=17)
    assert report["accepted_combinations"] == len(ACCEPTED_CODES) == 3
    assert report["discarded_combinations"] == len(RESULT_CODES) - 3
    assert report["invalid_rollouts"] == 3
    assert report["duplicate_samples"] == report["not_selected"] == report["final_pairs"] == 1
    assert report["task_yield_rate"] == 1 / len(RESULT_CODES)


def test_counter_mismatch_and_duplicate_selected_task_are_not_silently_accepted():
    with pytest.raises(ValueError, match="请求分母"):
        summarize_dpo_results([], requested_tasks=1, requested_combinations=1,
                              requested_rollouts=2, completed_rollouts=0)
    with pytest.raises(ValueError, match="最多一个"):
        summarize_dpo_results([{"task_id": "same", "result": "selected"}] * 2,
                              requested_tasks=1, requested_combinations=2,
                              requested_rollouts=3, completed_rollouts=3)


@pytest.mark.parametrize("gap", [-1, float("inf"), float("nan"), True])
def test_invalid_pairing_thresholds_fail_before_any_sampling(gap):
    with pytest.raises(ValueError):
        build_dpo_pairs([], task_ids=[EXPANDED_SPECS[0].id], trials_per_task=5,
                        min_reward_gap=gap)


def test_old_string_data_and_incomplete_package_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="格式不兼容"):
        validate_dpo_row({}, {}, {}, decision_parser=parse_react_response,
                         messages_validator=validate_react_messages)
    data = tmp_path / "unpublished.jsonl"
    manifest = tmp_path / "unpublished.jsonl.manifest.json"
    manifest.write_text(json.dumps({"status": "incomplete"}), encoding="utf-8")
    with pytest.raises(ValueError, match="未完成"):
        load_dpo_package(data, decision_parser=parse_react_response,
                         messages_validator=validate_react_messages)


def test_content_fingerprints_preserve_exact_whitespace():
    assert dpo_content_hash(" ") != dpo_content_hash("")
    assert dpo_content_hash({"a": 1, "b": 2}) == dpo_content_hash({"b": 2, "a": 1})


def test_real_runtime_validators_are_mandatory_and_unknown_stops_are_unusable():
    # Source checks are explicitly not dynamic tokenizer or trajectory evidence.
    for function in (load_dpo_package, validate_dpo_row, dpo_trace_decisions):
        for name in ("decision_parser", "messages_validator"):
            assert inspect.signature(function).parameters[name].default is inspect.Parameter.empty
    source = inspect.getsource(dpo_trace_decisions)
    assert 'request.get("finish_reason") == "stop"' in source
    assert 'step.get("framework_generated")' in source
    assert 'step.get("policy_text") != parsed.policy_text' in source
