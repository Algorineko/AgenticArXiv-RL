"""Exact-context, first-action-divergence offline preference selection.

This module consumes actual recorded episodes. It never samples responses,
executes tools, repairs missing history, or manufactures negative examples.
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from itertools import combinations
from typing import Any, Mapping, Sequence

from agentic_arxiv.provenance import (
    DPO_DATA_KIND, DPO_DATA_VERSION, DPO_PAIRING_VERSION, DPO_RESULT_CODES,
    dpo_content_hash, dpo_sample_fingerprint, dpo_trace_decisions,
    require_training_protocol,
)
from agentic_arxiv.runtime.react_contract import (
    parse_react_response, protocol_metadata, validate_react_messages,
)


RESULT_CODES = DPO_RESULT_CODES
ACCEPTED_CODES = {"selected", "not_selected", "duplicate_sample"}
_IDENTITY_FIELDS = ("protocol_metadata", "source_model", "initialization", "snapshot", "sampling")


def _finite(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _compare(left: Mapping[str, Any] | None, right: Mapping[str, Any] | None,
             min_reward_gap: float) -> tuple[str, dict | None]:
    if left is None or right is None:
        return "sampling_failure", None
    if any(trace.get("status") != "completed" for trace in (left, right)):
        failures = {trace.get("failure_code") for trace in (left, right)
                    if trace.get("status") != "completed"}
        return ("nonfinite_score" if "nonfinite_score" in failures else "sampling_failure"), None
    identity = left.get("identity")
    if (not isinstance(identity, Mapping) or identity != right.get("identity")
            or any(identity.get(key) is None for key in _IDENTITY_FIELDS)
            or left.get("task_id") != right.get("task_id")
            or left.get("source_split") != right.get("source_split")):
        return "source_mismatch", None
    try:
        require_training_protocol(identity, protocol_metadata(), source_label="DPO trace",
                                  require_chat_template_hash=True)
    except ValueError:
        return "source_mismatch", None
    for trace in (left, right):
        if not isinstance(trace.get("requests"), list):
            return "missing_record", None
        if not trace["requests"]:
            return "missing_response", None
    try:
        decisions = [dpo_trace_decisions(trace, decision_parser=parse_react_response,
                                        messages_validator=validate_react_messages)
                     for trace in (left, right)]
    except (ValueError, KeyError, TypeError, AttributeError):
        return "missing_record", None
    for index in range(max(map(len, decisions))):
        if any(index >= len(side) for side in decisions):
            return "missing_response", None
        a, b = decisions[0][index], decisions[1][index]
        # Fingerprints are not a substitute for comparing actual full values.
        if a[0]["messages"] != b[0]["messages"]:
            return "context_mismatch", None
        if not a[2] or not b[2]:
            return "unusable_decision", None
        if a[3] == b[3]:
            continue
        if not _finite(left.get("reward")) or not _finite(right.get("reward")):
            return "nonfinite_score", None
        gap = abs(left["reward"] - right["reward"])
        if gap <= min_reward_gap:
            return "insufficient_reward_gap", None
        high, low = (left, right) if left["reward"] > right["reward"] else (right, left)
        high_decision, low_decision = (a, b) if high is left else (b, a)
        row = {
            "kind": DPO_DATA_KIND, "version": DPO_DATA_VERSION,
            "prompt": copy.deepcopy(a[0]["messages"]),
            "chosen": [{"role": "assistant", "content": high_decision[1].policy_text}],
            "rejected": [{"role": "assistant", "content": low_decision[1].policy_text}],
            "task_id": left["task_id"], "source_split": left["source_split"],
            "decision_step": index,
            "chosen_trace_id": high["trace_id"], "rejected_trace_id": low["trace_id"],
            "chosen_reward": high["reward"], "rejected_reward": low["reward"],
            "reward_gap": gap, "context_fingerprint": dpo_content_hash(a[0]["messages"]),
            "protocol_metadata": copy.deepcopy(identity["protocol_metadata"]),
        }
        row["sample_fingerprint"] = dpo_sample_fingerprint(row)
        return "accepted", row
    return "no_action_divergence", None


def summarize_dpo_results(results: Sequence[Mapping[str, Any]], *, requested_tasks: int,
                          requested_combinations: int, requested_rollouts: int,
                          completed_rollouts: int) -> dict:
    """Pure accounting with requested denominators, including failed trials."""
    counts = Counter(item["result"] for item in results)
    if set(counts) - set(RESULT_CODES):
        raise ValueError("DPO 未知组合结果码")
    if len(results) != requested_combinations or sum(counts.values()) != requested_combinations:
        raise ValueError("DPO 组合结果总数与请求分母不一致")
    if (requested_tasks < 0 or requested_rollouts < 0
            or not 0 <= completed_rollouts <= requested_rollouts):
        raise ValueError("DPO 请求/完成数非法")
    counts = {code: counts[code] for code in RESULT_CODES}
    accepted = sum(counts[code] for code in ACCEPTED_CODES)
    selected_tasks = {item["task_id"] for item in results if item["result"] == "selected"}
    if len(selected_tasks) != counts["selected"] or len(selected_tasks) > requested_tasks:
        raise ValueError("DPO 每任务最多一个最终偏好对")
    return {
        "requested_tasks": requested_tasks, "tasks_with_pairs": len(selected_tasks),
        "task_yield_rate": len(selected_tasks) / requested_tasks if requested_tasks else 0.0,
        "requested_rollouts": requested_rollouts, "completed_rollouts": completed_rollouts,
        "invalid_rollouts": requested_rollouts - completed_rollouts,
        "requested_combinations": requested_combinations, "accepted_combinations": accepted,
        "discarded_combinations": requested_combinations - accepted,
        "combination_acceptance_rate": accepted / requested_combinations if requested_combinations else 0.0,
        "duplicate_samples": counts["duplicate_sample"], "not_selected": counts["not_selected"],
        "final_pairs": counts["selected"], "result_counts": counts,
    }


def build_dpo_pairs(traces: Sequence[Mapping[str, Any]], *, task_ids: Sequence[str],
                    trials_per_task: int, min_reward_gap: float,
                    yield_warning_threshold: float = 0.25) -> tuple[list[dict], dict]:
    """Select at most one maximum-gap pair per task, preserving every trial pair."""
    if (not _finite(min_reward_gap) or min_reward_gap < 0
            or not _finite(yield_warning_threshold) or not 0 <= yield_warning_threshold <= 1
            or isinstance(trials_per_task, bool) or not isinstance(trials_per_task, int)
            or trials_per_task < 2):
        raise ValueError("DPO 配对阈值和每任务采样数非法")
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("DPO 需要非空且不重复的请求任务")
    by_trial, trace_ids = {}, set()
    for trace in traces:
        task_id, trial_id = trace.get("task_id"), trace.get("trial_id")
        trace_id = trace.get("trace_id")
        if (task_id not in task_ids or isinstance(trial_id, bool) or not isinstance(trial_id, int)
                or not 0 <= trial_id < trials_per_task or (task_id, trial_id) in by_trial
                or not isinstance(trace_id, str) or not trace_id or trace_id in trace_ids):
            raise ValueError("DPO trace 的任务/trial/ID 不属于唯一请求位置")
        by_trial[task_id, trial_id] = trace
        trace_ids.add(trace_id)
    rows, results, per_task = [], [], {}
    for task_id in task_ids:
        task_results, candidates = [], []
        for left_trial, right_trial in combinations(range(trials_per_task), 2):
            code, row = _compare(by_trial.get((task_id, left_trial)),
                                 by_trial.get((task_id, right_trial)), min_reward_gap)
            result = {"task_id": task_id, "trial_ids": [left_trial, right_trial], "result": code}
            task_results.append(result)
            if row is not None:
                candidates.append((row, result))
        # Stable tie-break is independent of input trace ordering and chosen side.
        candidates.sort(key=lambda item: (-item[0]["reward_gap"], item[0]["decision_step"],
                                          *item[1]["trial_ids"]))
        seen = set()
        selected = False
        for row, result in candidates:
            fingerprint = row["sample_fingerprint"]
            if fingerprint in seen:
                result["result"] = "duplicate_sample"
            else:
                seen.add(fingerprint)
                result["result"] = "not_selected" if selected else "selected"
                if not selected:
                    rows.append(row)
                    selected = True
        completed = sum(trace.get("status") == "completed" for (tid, _), trace in by_trial.items()
                        if tid == task_id)
        per_task[task_id] = summarize_dpo_results(
            task_results, requested_tasks=1, requested_combinations=len(task_results),
            requested_rollouts=trials_per_task, completed_rollouts=completed,
        )
        results.extend(task_results)
    report = summarize_dpo_results(
        results, requested_tasks=len(task_ids), requested_combinations=(
            len(task_ids) * trials_per_task * (trials_per_task - 1) // 2),
        requested_rollouts=len(task_ids) * trials_per_task,
        completed_rollouts=sum(trace.get("status") == "completed" for trace in traces),
    )
    report.update(pairing_version=DPO_PAIRING_VERSION, min_reward_gap=min_reward_gap,
                  yield_warning_threshold=yield_warning_threshold, per_task=per_task,
                  combinations=results, warnings=[])
    if report["task_yield_rate"] < yield_warning_threshold:
        report["warnings"].append("低于任务产出率警戒线；未调整采样、上下文条件或奖励阈值")
    return rows, report
