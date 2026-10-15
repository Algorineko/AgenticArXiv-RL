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

from agentic_arxiv.provenance import canonical_hash, evaluation_protocol_relation, read_protocol_metadata
from agentic_arxiv.scoring.health import _assistant_turns, detect_rollout_anomalies


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


class RolloutAuditWriter:
    """Append per-sample audit records and retain a compact aggregate summary."""

    def __init__(
        self,
        path: Path,
        *,
        num_generations: int,
        allowed_tools: Iterable[str] = (),
        max_samples: int = 0,
        protocol_metadata: Optional[Mapping[str, Any]] = None,
        training_metadata: Optional[Mapping[str, Any]] = None,
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
        self.protocol_metadata = read_protocol_metadata(protocol_metadata)
        self.training_protocol_metadata = read_protocol_metadata(training_metadata)
        self.protocol_counts: Counter[str] = Counter()
        self.protocols: Dict[str, Dict[str, Any]] = {}
        self.protocol_relations: Dict[str, Dict[str, Any]] = {}
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
                declared_protocol = read_protocol_metadata(trajectory)
                if declared_protocol["metadata_status"] == "unknown":
                    declared_protocol = dict(self.protocol_metadata)
                else:
                    for key, value in self.protocol_metadata.items():
                        if key == "metadata_status" or value is None:
                            continue
                        if declared_protocol.get(key) not in (None, value):
                            raise ValueError(f"rollout audit protocol mismatch: {key}")
                training_protocol = read_protocol_metadata(trajectory.get("training_protocol_metadata"))
                if training_protocol["metadata_status"] == "unknown":
                    training_protocol = dict(self.training_protocol_metadata)
                else:
                    for key, value in self.training_protocol_metadata.items():
                        if key == "metadata_status" or value is None:
                            continue
                        if training_protocol.get(key) not in (None, value):
                            raise ValueError(f"rollout audit model protocol mismatch: {key}")
                relation = evaluation_protocol_relation(training_protocol, declared_protocol)
                protocol_key = canonical_hash(relation)
                self.protocols[protocol_key] = declared_protocol
                self.protocol_relations[protocol_key] = relation
                self.protocol_counts[protocol_key] += 1
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
                    "protocol_metadata": declared_protocol,
                    "training_protocol_metadata": training_protocol,
                    "protocol_relation": relation,
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
            "protocol_metadata": dict(self.protocol_metadata),
            "training_protocol_metadata": dict(self.training_protocol_metadata),
            "protocol_relation": evaluation_protocol_relation(
                self.training_protocol_metadata, self.protocol_metadata
            ),
            "protocol_groups": [
                {"protocol_metadata": self.protocols[key],
                 "protocol_relation": self.protocol_relations[key],
                 "sample_count": self.protocol_counts[key]}
                for key in sorted(self.protocol_counts)
            ],
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
