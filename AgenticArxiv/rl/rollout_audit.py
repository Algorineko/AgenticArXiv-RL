"""Persist auditable GRPO rollout groups as JSONL.

Reward curves answer whether sampled rewards differ, but they cannot explain
*why*.  This module records the raw assistant turns, normalized trajectory,
reward breakdown, and deterministic anomaly flags for every sampled rollout.
It deliberately stays independent of TRL so the checks are cheap to unit test
and remain usable when the trainer API changes.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _assistant_turns(completion: Any, trajectory: Mapping[str, Any]) -> List[str]:
    """Return model-authored text only, excluding environment observations."""
    raw_turns = trajectory.get("raw_assistant_turns")
    if isinstance(raw_turns, list):
        return [str(item or "") for item in raw_turns]

    if isinstance(completion, str):
        return [completion]
    if isinstance(completion, list):
        return [
            str(message.get("content") or "")
            for message in completion
            if isinstance(message, Mapping) and message.get("role") == "assistant"
        ]
    return [str(completion or "")]


def _parsed_tool_names(history: Sequence[Mapping[str, Any]]) -> List[str]:
    names: List[str] = []
    for step in history:
        action = str(step.get("action") or "")
        if action in {"FINISH", "FORCE_STOP", "ERROR", "PARSE_ERROR"}:
            continue
        try:
            parsed = json.loads(action)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict) and parsed.get("name"):
            names.append(str(parsed["name"]))
    return names


def detect_rollout_anomalies(
    completion: Any,
    trajectory: Mapping[str, Any],
    *,
    allowed_tools: Iterable[str] = (),
) -> Dict[str, bool]:
    """Apply structural checks to one model-generated multi-turn trajectory."""
    turns = _assistant_turns(completion, trajectory)
    history = trajectory.get("history") or []
    actions = [str(step.get("action") or "") for step in history]
    tool_steps = [
        step for step, action in zip(history, actions)
        if action not in {"FINISH", "FORCE_STOP", "ERROR", "PARSE_ERROR"}
    ]
    parse_error = any(
        bool(step.get("parse_failed")) or action == "PARSE_ERROR"
        for step, action in zip(history, actions)
    )
    action_markers = sum(turn.count("Action:") for turn in turns)
    think_counts = [
        (turn.count("<think>"), turn.count("</think>"))
        for turn in turns
    ]
    known = set(allowed_tools)
    used_tools = _parsed_tool_names(history)
    tool_error = any(
        str(step.get("observation") or "").startswith("工具执行失败: ")
        for step in tool_steps
    )
    missing_thought = any(not str(step.get("thought") or "").strip() for step in tool_steps)
    last_action = actions[-1] if actions else ""

    return {
        "empty_output": not any(turn.strip() for turn in turns),
        "missing_action": parse_error and action_markers == 0,
        "invalid_action": parse_error and action_markers > 0,
        "missing_thought": missing_thought,
        # Check each assistant turn independently. Multiple valid ReAct turns
        # may each contain one <think> block and must not be flagged globally.
        "unbalanced_think_tags": any(opened != closed for opened, closed in think_counts),
        "multiple_think_blocks": any(
            opened > 1 or closed > 1 for opened, closed in think_counts
        ),
        "multiple_actions_in_turn": any(turn.count("Action:") > 1 for turn in turns),
        # Only inspect model-authored turns.  Environment-authored Observation
        # suffixes are intentionally excluded by _assistant_turns().
        "hallucinated_observation": any("Observation:" in turn for turn in turns),
        "unknown_tool": bool(known and any(name not in known for name in used_tools)),
        "tool_error": tool_error,
        "unfinished": bool(history) and last_action != "FINISH",
        "clipped": bool(trajectory.get("clipped")),
        "reached_max_turns": bool(trajectory.get("reached_max_turns")),
    }


class RolloutAuditWriter:
    """Append per-sample audit records and retain a compact aggregate summary."""

    def __init__(
        self,
        path: Path,
        *,
        num_generations: int,
        allowed_tools: Iterable[str] = (),
        max_samples: int = 0,
    ) -> None:
        if num_generations < 1:
            raise ValueError("num_generations must be at least 1")
        if max_samples < 0:
            raise ValueError("max_samples cannot be negative")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self.num_generations = int(num_generations)
        self.allowed_tools = set(allowed_tools)
        self.max_samples = int(max_samples)
        self.batch_count = 0
        self.group_count = 0
        self.seen_samples = 0
        self.saved_samples = 0
        self.anomalous_samples = 0
        self.anomaly_counts: Counter[str] = Counter()
        self.task_samples: Counter[str] = Counter()
        self.task_anomalous_samples: Counter[str] = Counter()
        self.task_anomalies: Dict[str, Counter[str]] = defaultdict(Counter)

    def record_batch(
        self,
        *,
        completions: Sequence[Any],
        task_ids: Sequence[Optional[str]],
        trajectories: Sequence[Mapping[str, Any]],
        breakdowns: Sequence[Any],
        rewards: Sequence[float],
        training_step: int,
    ) -> None:
        size = len(completions)
        lengths = {
            "completions": size,
            "task_ids": len(task_ids),
            "trajectories": len(trajectories),
            "breakdowns": len(breakdowns),
            "rewards": len(rewards),
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(f"rollout audit batch length mismatch: {lengths}")
        if size % self.num_generations != 0:
            raise ValueError(
                "rollout audit cannot recover GRPO groups: "
                f"samples={size}, num_generations={self.num_generations}"
            )

        records: List[Dict[str, Any]] = []
        batch_index = self.batch_count
        self.batch_count += 1
        for start in range(0, size, self.num_generations):
            stop = start + self.num_generations
            group_task_ids = [str(item) for item in task_ids[start:stop]]
            if len(set(group_task_ids)) != 1:
                raise ValueError(
                    "rollout audit found mixed task ids inside one GRPO group: "
                    f"group={group_task_ids}"
                )
            group_rewards = [float(value) for value in rewards[start:stop]]
            mean = sum(group_rewards) / len(group_rewards)
            group_std = math.sqrt(
                sum((value - mean) ** 2 for value in group_rewards) / len(group_rewards)
            )
            group_index = self.group_count
            self.group_count += 1

            for offset, sample_index in enumerate(range(start, stop)):
                completion = completions[sample_index]
                trajectory = trajectories[sample_index]
                task_id = group_task_ids[offset]
                anomalies = detect_rollout_anomalies(
                    completion,
                    trajectory,
                    allowed_tools=self.allowed_tools,
                )
                active_anomalies = sorted(name for name, active in anomalies.items() if active)
                self.seen_samples += 1
                self.task_samples[task_id] += 1
                if active_anomalies:
                    self.anomalous_samples += 1
                    self.task_anomalous_samples[task_id] += 1
                    self.anomaly_counts.update(active_anomalies)
                    self.task_anomalies[task_id].update(active_anomalies)

                if self.max_samples and self.saved_samples >= self.max_samples:
                    continue
                records.append({
                    "schema_version": 1,
                    "training_step": int(training_step),
                    "batch_index": batch_index,
                    "group_index": group_index,
                    "generation_index": offset,
                    "task_id": task_id,
                    "raw_completion": _jsonable(completion),
                    "raw_assistant_turns": _assistant_turns(completion, trajectory),
                    "trajectory": _jsonable(trajectory),
                    "reward": group_rewards[offset],
                    "group_reward_mean": mean,
                    "group_reward_std": group_std,
                    "reward_breakdown": _jsonable(breakdowns[sample_index]),
                    "anomalies": anomalies,
                    "active_anomalies": active_anomalies,
                })
                self.saved_samples += 1

        if records:
            with self.path.open("a", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()

    def summary(self) -> Dict[str, Any]:
        task_summary: Dict[str, Any] = {}
        for task_id in sorted(self.task_samples):
            samples = self.task_samples[task_id]
            counts = dict(sorted(self.task_anomalies[task_id].items()))
            task_summary[task_id] = {
                "sample_count": samples,
                "anomalous_sample_count": self.task_anomalous_samples[task_id],
                "anomalous_sample_fraction": (
                    self.task_anomalous_samples[task_id] / samples if samples else 0.0
                ),
                "anomaly_occurrence_count": sum(counts.values()),
                "anomaly_counts": counts,
            }
        return {
            "schema_version": 1,
            "trace_path": str(self.path),
            "num_generations": self.num_generations,
            "batch_count": self.batch_count,
            "group_count": self.group_count,
            "seen_samples": self.seen_samples,
            "saved_samples": self.saved_samples,
            "anomalous_samples": self.anomalous_samples,
            "anomalous_sample_fraction": (
                self.anomalous_samples / self.seen_samples if self.seen_samples else 0.0
            ),
            "anomaly_counts": dict(sorted(self.anomaly_counts.items())),
            "tasks": task_summary,
        }

    def save_summary(self, path: Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.summary(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return output
