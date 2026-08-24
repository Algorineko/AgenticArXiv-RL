"""Multi-granular, verifiable rewards for agentic RL.

The reward keeps the public ``RewardCalculator.compute_reward`` API used by
the rollout code, while exposing a component breakdown for logging and tests.
It adapts LLM-TIR's format/correctness/process curriculum to ReAct trajectories.
"""

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from benchmark.metrics import TaskMetrics, argument_match_score, extract_metrics


TERMINAL_ACTIONS = {"FINISH", "FORCE_STOP", "ERROR"}


@dataclass(frozen=True)
class RewardSchedule:
    """Curriculum weights at one training step."""

    format: float
    tool: float
    argument: float
    process: float
    outcome: float


@dataclass(frozen=True)
class RewardBreakdown:
    """Auditable reward components, each bounded to ``[-1, 1]``."""

    total: float
    format: float
    tool: float
    argument: float
    process: float
    outcome: float
    weights: RewardSchedule

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["weights"] = asdict(self.weights)
        return data


class RewardCalculator:
    """Compute hierarchical rewards from a complete ReAct trajectory.

    The default 30-step curriculum mirrors LLM-TIR: structural behavior is
    learned first, then semantic correctness receives its full weight.
    """

    def __init__(
        self,
        curriculum_steps: int = 30,
        early_correctness_scale: float = 1.0 / 3.0,
        weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.curriculum_steps = max(0, curriculum_steps)
        self.early_correctness_scale = early_correctness_scale
        self.weights = {
            "format": 1.0,
            "tool": 3.0,
            "argument": 2.0,
            "process": 1.0,
            "outcome": 3.0,
        }
        if weights:
            self.set_weights(dict(weights))

    def schedule(self, training_step: int = 0) -> RewardSchedule:
        scale = (
            self.early_correctness_scale
            if training_step < self.curriculum_steps
            else 1.0
        )
        return RewardSchedule(
            format=self.weights["format"],
            tool=self.weights["tool"] * scale,
            argument=self.weights["argument"] * scale,
            process=self.weights["process"],
            outcome=self.weights["outcome"] * scale,
        )

    def compute_reward(
        self,
        task_def: Dict[str, Any],
        result: Dict[str, Any],
        agent_type: str = "regex",
        trial: int = 0,
        session_id: str = "rl_train",
        training_step: int = 0,
    ) -> Tuple[float, TaskMetrics]:
        breakdown, metrics = self.compute_reward_breakdown(
            task_def, result, agent_type, trial, session_id, training_step
        )
        return breakdown.total, metrics

    def compute_reward_breakdown(
        self,
        task_def: Dict[str, Any],
        result: Dict[str, Any],
        agent_type: str = "regex",
        trial: int = 0,
        session_id: str = "rl_train",
        training_step: int = 0,
    ) -> Tuple[RewardBreakdown, TaskMetrics]:
        metrics = extract_metrics(task_def, result, agent_type, trial, session_id)
        history = result.get("history", [])
        components = {
            "format": self._format_score(history),
            "tool": self._tool_score(metrics.tool_call_sequence, metrics.expected_tools),
            "argument": self._argument_score(history, task_def.get("expected_tool_args")),
            "process": self._process_score(history, metrics),
            "outcome": self._outcome_score(metrics),
        }
        schedule = self.schedule(training_step)
        active = {
            name: weight
            for name, weight in asdict(schedule).items()
            if not (name == "argument" and components[name] is None)
        }
        denominator = sum(abs(weight) for weight in active.values()) or 1.0
        total = sum(active[name] * components[name] for name in active) / denominator
        breakdown = RewardBreakdown(
            total=round(_clip(total), 6),
            format=round(components["format"], 6),
            tool=round(components["tool"], 6),
            argument=round(components["argument"] or 0.0, 6),
            process=round(components["process"], 6),
            outcome=round(components["outcome"], 6),
            weights=schedule,
        )
        return breakdown, metrics

    def get_weights(self) -> Dict[str, float]:
        return self.weights.copy()

    def set_weights(self, new_weights: Dict[str, float]) -> None:
        unknown = set(new_weights) - set(self.weights)
        if unknown:
            raise ValueError(f"Unknown reward weights: {sorted(unknown)}")
        if any(value < 0 for value in new_weights.values()):
            raise ValueError("Reward weights must be non-negative")
        self.weights.update(new_weights)

    @staticmethod
    def _format_score(history: Sequence[Dict[str, Any]]) -> float:
        if not history:
            return -1.0
        valid = 0
        for step in history:
            action = step.get("action", "")
            if action in TERMINAL_ACTIONS:
                valid += 1
                continue
            parsed = _parse_action(action)
            if parsed and isinstance(parsed.get("name"), str) and isinstance(
                parsed.get("parameters", parsed.get("args", {})), dict
            ):
                valid += 1
        return _scale_ratio(valid, len(history))

    @staticmethod
    def _tool_score(actual: Sequence[str], expected: Sequence[str]) -> float:
        # expected 为空表示正确行为是一次工具都不调（category="infeasible"）。
        # 调了就给 -1.0 而不是 0.0：普通任务调错工具时 f1=0 → 2*0-1 = -1，
        # 「本该什么都不做却动了手」不该比「做错了」罚得更轻。
        if not expected:
            return 1.0 if not actual else -1.0
        lcs = _lcs_length(actual, expected)
        precision = lcs / len(actual) if actual else 0.0
        recall = lcs / len(expected)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return 2 * f1 - 1

    @staticmethod
    def _argument_score(
        history: Sequence[Dict[str, Any]],
        expected_args: Optional[Sequence[Optional[Mapping[str, Any]]]],
    ) -> Optional[float]:
        # 比对逻辑已提取到 benchmark/metrics.py，供 benchmark 报告共用；
        # 这里只做 [0,1] → [-1,1] 的缩放，语义与原实现一致。
        score = argument_match_score(history, expected_args)
        return None if score is None else 2 * score - 1

    @staticmethod
    def _process_score(history: Sequence[Dict[str, Any]], metrics: TaskMetrics) -> float:
        if not history:
            return -1.0
        good_steps = 0.0
        for step in history:
            action = step.get("action", "")
            observation = str(step.get("observation", ""))
            if action in TERMINAL_ACTIONS or _parse_action(action):
                good_steps += 1.0
            if step.get("parse_failed") or "无法解析" in observation:
                good_steps -= 1.0
        failures = metrics.parse_failures + metrics.tool_exec_failures
        extras = max(0, len(metrics.tool_call_sequence) - len(metrics.expected_tools))
        return _clip(good_steps / len(history) - 0.25 * failures - 0.1 * extras)

    @staticmethod
    def _outcome_score(metrics: TaskMetrics) -> float:
        if metrics.termination_type == "ERROR":
            return -1.0
        if metrics.termination_type == "FORCE_STOP":
            return -0.5
        if metrics.task_completed and metrics.tool_call_accurate:
            return 1.0
        if metrics.task_completed:
            return 0.25
        return -0.25


def compute_step_reward(step_dict: Dict[str, Any], metrics: TaskMetrics) -> float:
    """Backward-compatible dense reward for one environment transition."""
    action = step_dict.get("action", "")
    observation = str(step_dict.get("observation", ""))
    reward = 0.1 if action in TERMINAL_ACTIONS or _parse_action(action) else -0.2
    if step_dict.get("parse_failed") or "无法解析" in observation:
        reward -= 0.2
    if any(marker in observation for marker in ("错误:", "工具执行失败:", "Error")):
        reward -= 0.3
    return _clip(reward)


def compute_group_relative_advantages(
    rewards: Sequence[float], group_ids: Sequence[Any], epsilon: float = 1e-6
) -> list:
    """Normalize rewards within each prompt group, as used by GRPO.

    Returning zero for a constant-reward group avoids unstable gradients and
    makes this helper useful in lightweight trainers and unit tests.
    """
    if len(rewards) != len(group_ids):
        raise ValueError("rewards and group_ids must have the same length")
    grouped: Dict[Any, list] = {}
    for index, group_id in enumerate(group_ids):
        grouped.setdefault(group_id, []).append(index)
    advantages = [0.0] * len(rewards)
    for indices in grouped.values():
        values = [float(rewards[index]) for index in indices]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        if std <= epsilon:
            continue
        for index, value in zip(indices, values):
            advantages[index] = (value - mean) / (std + epsilon)
    return advantages


def _parse_action(action: Any) -> Optional[Dict[str, Any]]:
    if isinstance(action, dict):
        return action
    if not isinstance(action, str) or action in TERMINAL_ACTIONS:
        return None
    try:
        parsed = json.loads(action)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    row = [0] * (len(right) + 1)
    for left_item in left:
        previous = 0
        for j, right_item in enumerate(right, 1):
            saved = row[j]
            row[j] = previous + 1 if left_item == right_item else max(row[j], row[j - 1])
            previous = saved
    return row[-1]


def _scale_ratio(numerator: int, denominator: int) -> float:
    return 2 * numerator / denominator - 1 if denominator else -1.0


def _clip(value: float) -> float:
    return max(-1.0, min(1.0, value))
