"""只读加载与汇总已有 GRPO rollout 审计记录。"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from agentic_arxiv.provenance import (
    canonical_hash,
    evaluation_protocol_relation,
    read_protocol_metadata,
)
from agentic_arxiv.scoring.metrics import classify_blocked_terminal_semantics
from agentic_arxiv.tasks.tasks_expanded import get_expanded_tasks


FORMAT_FLAGS = {
    "empty_output",
    "missing_action",
    "invalid_action",
    "missing_thought",
    "unbalanced_think_tags",
    "multiple_think_blocks",
    "multiple_actions_in_turn",
    "hallucinated_observation",
    "unknown_tool",
}


def load_records(path: Path) -> List[Dict[str, Any]]:
    """读取原始记录，不为历史缺失字段补入当前协议。"""
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"JSON 无效: {path}:{line_number}: {exc}") from exc
        if not isinstance(item, dict):
            raise SystemExit(f"记录必须为对象: {path}:{line_number}")
        protocol = read_protocol_metadata(item)
        if protocol["metadata_status"] == "unknown":
            protocol = read_protocol_metadata(item.get("trajectory"))
        trajectory = item.get("trajectory")
        trajectory = trajectory if isinstance(trajectory, Mapping) else {}
        training = read_protocol_metadata(item.get(
            "training_protocol_metadata", trajectory.get("training_protocol_metadata")
        ))
        records.append({
            **item, "protocol_metadata": protocol, "training_protocol_metadata": training,
            "protocol_relation": evaluation_protocol_relation(training, protocol),
        })
    if not records:
        raise SystemExit(f"未找到 rollout 记录: {path}")
    return records


def summarize(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总已保存的异常标记和分组，不执行任务或改写奖励。"""
    rows = list(records)
    anomaly_counts: Counter[str] = Counter()
    task_samples: Counter[str] = Counter()
    task_anomalies: Counter[str] = Counter()
    groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    format_anomalous = 0
    tasks_by_id = {task["id"]: task for task in get_expanded_tasks()}
    terminal_counts: Counter[str] = Counter()
    terminal_rewards: Dict[str, List[float]] = defaultdict(list)
    task_terminal_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    protocol_counts: Counter[str] = Counter()
    protocols: Dict[str, Dict[str, Any]] = {}
    relations: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        protocol = read_protocol_metadata(row)
        if protocol["metadata_status"] == "unknown":
            protocol = read_protocol_metadata(row.get("trajectory"))
        trajectory = row.get("trajectory")
        trajectory = trajectory if isinstance(trajectory, Mapping) else {}
        training = read_protocol_metadata(row.get(
            "training_protocol_metadata", trajectory.get("training_protocol_metadata")
        ))
        relation = evaluation_protocol_relation(training, protocol)
        protocol_key = canonical_hash(relation)
        protocols[protocol_key] = protocol
        relations[protocol_key] = relation
        protocol_counts[protocol_key] += 1
        flags = set(row.get("active_anomalies") or [])
        task_id = str(row.get("task_id"))
        task_samples[task_id] += 1
        anomaly_counts.update(flags)
        if flags:
            task_anomalies[task_id] += 1
        if flags & FORMAT_FLAGS:
            format_anomalous += 1
        groups[(protocol_key, row.get("batch_index"), row.get("group_index"))].append(row)
        task = tasks_by_id.get(task_id)
        if task and task.get("expected_terminal_mode") == "blocked":
            trajectory = row.get("trajectory") or {}
            semantic = classify_blocked_terminal_semantics(
                task, trajectory.get("history") or []
            )
            terminal_counts[semantic] += 1
            terminal_rewards[semantic].append(float(row.get("reward") or 0.0))
            task_terminal_counts[task_id][semantic] += 1

    zero_std_groups = 0
    duplicate_groups = 0
    for group_rows in groups.values():
        if abs(float(group_rows[0].get("group_reward_std") or 0.0)) <= 1e-12:
            zero_std_groups += 1
        completions = {
            json.dumps(row.get("raw_assistant_turns"), ensure_ascii=False, sort_keys=True)
            for row in group_rows
        }
        if len(completions) == 1:
            duplicate_groups += 1

    group_count = len(groups)
    anomalous_samples = sum(bool(row.get("active_anomalies")) for row in rows)
    terminal_sample_count = sum(terminal_counts.values())
    return {
        "protocol_groups": [
            {"protocol_metadata": protocols[key], "protocol_relation": relations[key],
             "sample_count": protocol_counts[key]}
            for key in sorted(protocol_counts)
        ],
        "mixed_protocols": len(protocol_counts) > 1,
        "sample_count": len(rows),
        "group_count": group_count,
        "anomalous_sample_count": anomalous_samples,
        "anomalous_sample_fraction": anomalous_samples / len(rows),
        "format_anomalous_sample_count": format_anomalous,
        "format_anomalous_sample_fraction": format_anomalous / len(rows),
        "zero_reward_std_group_count": zero_std_groups,
        "zero_reward_std_group_fraction": zero_std_groups / group_count if group_count else 0.0,
        "exact_duplicate_group_count": duplicate_groups,
        "exact_duplicate_group_fraction": duplicate_groups / group_count if group_count else 0.0,
        "anomaly_counts": dict(anomaly_counts.most_common()),
        "terminal_semantics": {
            "applicable_sample_count": terminal_sample_count,
            "counts": dict(sorted(terminal_counts.items())),
            "rates": {
                name: count / terminal_sample_count
                for name, count in sorted(terminal_counts.items())
            } if terminal_sample_count else {},
            "mean_reward_by_class": {
                name: sum(values) / len(values)
                for name, values in sorted(terminal_rewards.items())
            },
            "tasks": {
                task_id: dict(sorted(counts.items()))
                for task_id, counts in sorted(task_terminal_counts.items())
            },
        },
        "tasks": {
            task_id: {
                "sample_count": count,
                "anomalous_sample_count": task_anomalies[task_id],
                "anomalous_sample_fraction": task_anomalies[task_id] / count,
            }
            for task_id, count in sorted(task_samples.items())
        },
    }
