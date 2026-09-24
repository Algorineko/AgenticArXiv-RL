"""Multi-granular, verifiable rewards for agentic RL.

The reward keeps the public ``RewardCalculator.compute_reward`` API used by
the rollout code, while exposing a component breakdown for logging and tests.
It adapts LLM-TIR's format/correctness/process curriculum to ReAct trajectories.
"""

import ast
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from benchmark.metrics import (
    TaskMetrics,
    argument_match_score,
    classify_blocked_terminal_semantics,
    extract_metrics,
    lcs_length,
)


TERMINAL_ACTIONS = {"FINISH", "FORCE_STOP", "ERROR"}

# Observations reach the scorer as a truncated ``repr``/``json`` string rather
# than a parsed payload, and the two call sites disagree on quoting.  Matching
# both spellings keeps the check working without re-parsing a half-truncated
# document.
_EMPTY_FIGURE_COUNT_RE = re.compile(r"['\"]count['\"]\s*:\s*0(?![\d.])")


def _reports_zero_figures(observation: str) -> bool:
    """True when an ``extract_paper_figures`` payload says it found nothing."""
    return bool(_EMPTY_FIGURE_COUNT_RE.search(observation))


def _has_figure_answer(observation: str) -> bool:
    """仅在图表结果完整且论文标识、答案均非空时返回真。"""
    for parse in (json.loads, ast.literal_eval):
        try:
            payload = parse(observation)
        except (ValueError, SyntaxError, TypeError, RecursionError):
            continue
        return (
            isinstance(payload, dict)
            and isinstance(payload.get("paper_id"), str)
            and bool(payload["paper_id"].strip())
            and isinstance(payload.get("answer"), str)
            and bool(payload["answer"].strip())
        )
    return False


@dataclass(frozen=True)
class RewardSchedule:
    """Curriculum weights at one training step."""

    format: float
    tool: float
    argument: float
    process: float
    outcome: float
    # Kept at zero in the legacy weighted average; these are logged so a
    # dashboard can distinguish policy progress from a safety-gate activation.
    result_quality: float = 0.0
    efficiency: float = 0.0


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
    # Diagnostics/gates introduced for result-grounded reward.  They are kept
    # outside the legacy weighted average so existing curriculum curves remain
    # comparable; severe failures cap the final reward below.
    result_quality: float = 0.0
    efficiency: float = 0.0

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
            result_quality=0.0,
            efficiency=0.0,
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
        forced_finish = bool(result.get("forced_finish")) or any(
            bool(step.get("forced_finish")) for step in history
        )
        components = {
            "format": self._format_score(history),
            "tool": self._tool_score(metrics.tool_call_sequence, metrics.expected_tools),
            "argument": self._argument_score(
                history,
                task_def.get("expected_tool_args"),
                task_def.get("expected_tools"),
                task_def.get("expected_paper_ids"),
            ),
            "process": self._process_score(history, metrics),
            "outcome": self._outcome_score(
                task_def, history, metrics, forced_finish=forced_finish
            ),
            "result_quality": self._result_quality_score(task_def, history, metrics),
            "efficiency": self._efficiency_score(task_def, history, metrics),
        }
        schedule = self.schedule(training_step)
        active = {
            name: weight
            for name, weight in asdict(schedule).items()
            if not (name == "argument" and components[name] is None)
        }
        denominator = sum(abs(weight) for weight in active.values()) or 1.0
        total = sum(active[name] * components[name] for name in active) / denominator
        total = self._apply_safety_gates(
            total,
            result=result,
            history=history,
            metrics=metrics,
            result_quality=components["result_quality"],
            efficiency=components["efficiency"],
        )
        breakdown = RewardBreakdown(
            total=round(_clip(total), 6),
            format=round(components["format"], 6),
            tool=round(components["tool"], 6),
            argument=round(components["argument"] or 0.0, 6),
            process=round(components["process"], 6),
            outcome=round(components["outcome"], 6),
            weights=schedule,
            result_quality=round(components["result_quality"], 6),
            efficiency=round(components["efficiency"], 6),
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
        lcs = lcs_length(actual, expected)
        precision = lcs / len(actual) if actual else 0.0
        recall = lcs / len(expected)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return 2 * f1 - 1

    @staticmethod
    def _argument_score(
        history: Sequence[Dict[str, Any]],
        expected_args: Optional[Sequence[Optional[Mapping[str, Any]]]],
        expected_tools: Optional[Sequence[str]] = None,
        expected_paper_ids: Optional[Sequence[Optional[str]]] = None,
    ) -> Optional[float]:
        # 比对逻辑在 benchmark/metrics.py，供 benchmark 报告共用；
        # 这里只做 [0,1] → [-1,1] 的缩放。
        score = argument_match_score(
            history, expected_args, expected_tools, expected_paper_ids
        )
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
    def _outcome_score(
        task_def: Mapping[str, Any],
        history: Sequence[Dict[str, Any]],
        metrics: TaskMetrics,
        forced_finish: bool = False,
    ) -> float:
        if metrics.termination_type == "ERROR":
            return -1.0
        if metrics.termination_type == "FORCE_STOP":
            return -0.5
        # FINISH only means the policy chose to stop. It cannot turn a failed
        # environment transition into a successful outcome. Without this guard,
        # a trajectory that calls the expected tool with an unresolvable ref can
        # still receive positive outcome credit and outrank a clean trajectory.
        if metrics.tool_exec_failures > 0:
            return -1.0
        # “不可行”任务的正确结果不是空轨迹本身，而是识别能力边界并向用户
        # 解释原因。否则一句泛化的“任务已完成”+ FINISH 会与真正的拒绝同分，
        # GRPO 反而会强化 false completion。该契约由任务声明显式开启，普通
        # completed 任务不受文本启发式影响。
        if (
            task_def.get("expected_terminal_mode") == "blocked"
            and metrics.task_completed
            and metrics.tool_call_accurate
        ):
            terminal_semantics = classify_blocked_terminal_semantics(task_def, history)
            if terminal_semantics == "explained_block":
                return 1.0
            if terminal_semantics == "false_completion":
                return -0.25
            # 没有说完成，也没有给出可识别的阻塞原因：保留少量格式/停止分，
            # 但明显低于正确解释，促使模型学会可审计的终止理由。
            return 0.0
        if metrics.task_completed and metrics.tool_call_accurate:
            # 工具名对了不等于任务完成。`tool_call_accurate` 只比较工具序列，
            # 问 cs.CL 却搜了 cs.AI 在这里完全相等——要的东西根本没拿到，
            # outcome 却给满分。这样 outcome 就成了 tool 档的复制品，权重 3
            # 白送给「照着签名瞎填参数」的策略（issue #17 第 2 条）。
            # 下限保持 0.25，与「FINISH 了但工具序列错」持平：参数全错不该
            # 罚得比工具全错还狠。任务未声明参数标准答案时 arg_score 恒为 1.0，
            # 该分支行为与原实现一致。
            semantic = min(metrics.arg_score, metrics.ref_score)
            outcome = max(0.25, 2 * semantic - 1)
            if forced_finish:
                # `synthesize_trajectory` 在单步 rollout 末尾自动补的 FINISH
                # 不是策略的终止决策：outcome 封顶到「部分完成」档即可。
                # 抑制范围只限 outcome 本身——format/tool/argument 的组内
                # 区分度必须保留，否则单轮任务整组轨迹在安全门处被削平，
                # 参数维度的优势在 GRPO 组内比较里彻底消失。
                outcome = min(outcome, 0.25)
            return outcome
        if metrics.task_completed:
            return 0.25
        return -0.25

    @staticmethod
    def _result_quality_score(
        task_def: Mapping[str, Any],
        history: Sequence[Dict[str, Any]],
        metrics: TaskMetrics,
    ) -> float:
        """Score whether tool observations show a useful, grounded result.

        Tool names and arguments are necessary but not sufficient: a search
        fallback, an empty result, or a failed download must not look like a
        successful episode merely because the policy selected the right tool.
        The evaluator intentionally stays deterministic and uses the tool
        observation/state contract rather than an LLM judge.
        """
        expected = list(task_def.get("expected_tools") or [])
        actual_steps = [
            step for step in history
            if str(step.get("action", "")) not in TERMINAL_ACTIONS
            and _parse_action(step.get("action", "")) is not None
        ]
        if not expected:
            return 1.0 if not actual_steps else -1.0
        if not actual_steps:
            return -1.0

        scores = []
        for index, step in enumerate(actual_steps):
            observation = str(step.get("observation", "") or "")
            action = _parse_action(step.get("action", "")) or {}
            tool_name = str(action.get("name", ""))
            if (
                step.get("parse_failed")
                or "工具执行失败" in observation
                or "无法解析" in observation
                or "offline_fallback" in observation
                or "回退结果" in observation
            ):
                scores.append(-1.0)
                continue
            if not observation.strip():
                scores.append(-0.5)
                continue

            # The expected tool is checked separately by _tool_score.  Here we
            # only grade whether its observation represents useful work.
            if tool_name in {"get_recently_submitted_cs_papers", "search_arxiv_papers"}:
                good = (
                    "成功获取" in observation
                    or "论文" in observation
                    or "paper" in observation.lower()
                    or "'id'" in observation
                    or '"id"' in observation
                )
            elif tool_name in {
                "download_arxiv_pdf",
                "translate_arxiv_pdf",
                "get_paper_cache_status",
            }:
                good = any(
                    marker in observation
                    for marker in ("READY", "成功", "已创建", "status", "pdf_ready")
                )
            elif tool_name == "analyze_figure":
                # 空答案和被截断的结果都不能作为已完成的图表分析。
                scores.append(1.0 if _has_figure_answer(observation) else -1.0)
                continue
            elif tool_name in {"get_paper_content", "summarize_paper"}:
                # The reading tools answer with the resolved paper and the
                # extracted text.  Grounding is the point: a payload that names
                # a paper but carries no text is not useful work, even though
                # the call itself did not raise.
                good = "paper_id" in observation and any(
                    marker in observation
                    for marker in (
                        "'summary'", '"summary"',
                        "'content'", '"content"',
                        "'answer'", '"answer"',
                    )
                )
            elif tool_name == "extract_paper_figures":
                # "The call succeeded" is not the claim being graded here: a
                # paper may genuinely have no embedded raster figures, and the
                # tool reports that as an empty list rather than an error.
                # Producing at least one figure file is the useful outcome.
                good = "paper_id" in observation and not _reports_zero_figures(
                    observation
                )
            else:
                good = True
            scores.append(1.0 if good else 0.0)

        # Extra calls are not allowed to turn a grounded result into a full
        # score.  They are already penalized by process/tool, but the result
        # gate should also see them.
        if len(actual_steps) > len(expected):
            scores.extend([-1.0] * (len(actual_steps) - len(expected)))
        return _clip(sum(scores) / max(1, len(scores)))

    @staticmethod
    def _efficiency_score(
        task_def: Mapping[str, Any],
        history: Sequence[Dict[str, Any]],
        metrics: TaskMetrics,
    ) -> float:
        """Small, task-normalized cost signal for redundant tool calls."""
        expected_count = len(task_def.get("expected_tools") or [])
        actual_count = len(metrics.tool_call_sequence)
        if expected_count == 0:
            return 1.0 if actual_count == 0 else -1.0
        excess = max(0, actual_count - expected_count)
        error_count = metrics.parse_failures + metrics.tool_exec_failures
        penalty = excess / expected_count + 0.25 * error_count
        return _clip(1.0 - 2.0 * penalty)

    @staticmethod
    def _apply_safety_gates(
        total: float,
        *,
        result: Mapping[str, Any],
        history: Sequence[Dict[str, Any]],
        metrics: TaskMetrics,
        result_quality: float,
        efficiency: float,
    ) -> float:
        """Apply non-compensable failure caps after the legacy score."""
        capped = float(total)
        hard_invalid = (
            metrics.termination_type == "ERROR"
            or metrics.parse_failures > 0
            or metrics.tool_exec_failures > 0
            or result_quality <= -0.75
        )
        if hard_invalid:
            # A failed execution must remain distinguishable from a false
            # FINISH and cannot be rescued by format/tool points.
            capped = min(capped, -0.75)
        elif metrics.false_finish:
            capped = min(capped, -0.25)

        # ``synthesize_trajectory`` 用框架自动补的 FINISH 让单步 rollout 成形。
        # 「不拿完整终止奖励」由 `_outcome_score` 的 forced_finish 分支在
        # outcome 内部执行；此处不再对总分截断——否则 format/tool/argument
        # 的组内区分度会在单轮任务上被一起削平（组内奖励全为同一常数，
        # GRPO 无梯度），恰好破坏本文件其余 safety gate 试图维持的区分度。

        if efficiency <= -0.75:
            capped = min(capped, -0.25)
        # ``history`` is intentionally accepted above so a future gate can
        # inspect trajectory-level markers without changing this API.
        return _clip(capped)


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


def _scale_ratio(numerator: int, denominator: int) -> float:
    return 2 * numerator / denominator - 1 if denominator else -1.0


def _clip(value: float) -> float:
    return max(-1.0, min(1.0, value))
