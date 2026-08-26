"""Deterministic policies for checking benchmark-score sensitivity.

These policies never call tools. They construct valid ReAct trajectories and
score them with the same ``RewardCalculator`` used by rollout and training.
The result is a reproducible diagnostic: if a deliberately weak policy scores
near the reference trajectory, the task set or reward needs investigation.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from rl.reward import RewardCalculator


_SEARCH_ACTION = (
    "get_recently_submitted_cs_papers",
    {"aspect": "AI", "days": 7, "max_results": 5},
)
_TOOL_ACTIONS = (
    _SEARCH_ACTION,
    ("download_arxiv_pdf", {"ref": 1}),
    ("translate_arxiv_pdf", {"ref": 1}),
    ("get_paper_cache_status", {"ref": 1}),
)


def _tool_step(name: str, args: Mapping[str, Any]) -> Dict[str, str]:
    action = json.dumps({"name": name, "args": dict(args)}, ensure_ascii=False)
    return {"action": action, "observation": "synthetic baseline action"}


def _finish_step() -> Dict[str, str]:
    return {"action": "FINISH", "observation": "synthetic baseline finish"}


def _result(history: List[Dict[str, str]]) -> Dict[str, Any]:
    return {
        "history": history,
        "timing": {},
        "token_usage": {},
        "iteration_count": len(history),
    }


class BaselinePolicy:
    """A deterministic policy that returns a synthetic ReAct trajectory."""

    name = "baseline"

    def build_result(self, task: Mapping[str, Any], seed: int) -> Dict[str, Any]:
        raise NotImplementedError


class ReferencePolicy(BaselinePolicy):
    """Replay each task declaration's reference tool path as a score ceiling."""

    name = "reference"

    def build_result(self, task: Mapping[str, Any], seed: int) -> Dict[str, Any]:
        expected_tools = task.get("expected_tools", [])
        expected_args = task.get("expected_tool_args") or []
        history = []
        for index, name in enumerate(expected_tools):
            args = expected_args[index] if index < len(expected_args) else {}
            history.append(_tool_step(name, args or {}))
        history.append(_finish_step())
        return _result(history)


class AlwaysFinishPolicy(BaselinePolicy):
    """Terminate immediately, regardless of the task request."""

    name = "always_finish"

    def build_result(self, task: Mapping[str, Any], seed: int) -> Dict[str, Any]:
        return _result([_finish_step()])


class AlwaysSearchPolicy(BaselinePolicy):
    """Make the same valid search call for every task, then terminate."""

    name = "always_search"

    def build_result(self, task: Mapping[str, Any], seed: int) -> Dict[str, Any]:
        name, args = _SEARCH_ACTION
        return _result([_tool_step(name, args), _finish_step()])


class RandomToolPolicy(BaselinePolicy):
    """Choose one valid tool call per task from a seed-stable pseudo-random draw."""

    name = "random_tool"

    def build_result(self, task: Mapping[str, Any], seed: int) -> Dict[str, Any]:
        task_id = str(task.get("id", ""))
        digest = sha256(f"{seed}:{task_id}".encode("utf-8")).digest()
        name, args = _TOOL_ACTIONS[int.from_bytes(digest[:8], "big") % len(_TOOL_ACTIONS)]
        return _result([_tool_step(name, args), _finish_step()])


class WrongArgsPolicy(BaselinePolicy):
    """Replay the reference tool names but with deliberately wrong arguments so that triggers the arg_score check."""

    name = "wrong_args"

    _WRONG_ARGS_BY_TOOL = {
        "get_recently_submitted_cs_papers": {
            "aspect": "nonexistent_topic", "days": 7, "max_results": 5,
        },
        "download_arxiv_pdf": {"ref": -1},
        "translate_arxiv_pdf": {"ref": -1},
        "get_paper_cache_status": {"ref": -1},
    }

    def build_result(self, task: Mapping[str, Any], seed: int) -> Dict[str, Any]:
        expected_tools = task.get("expected_tools", [])
        history = [
            _tool_step(name, self._WRONG_ARGS_BY_TOOL.get(name, {"ref": -1}))
            for name in expected_tools
        ]
        history.append(_finish_step())
        return _result(history)


ALL_POLICIES: Dict[str, BaselinePolicy] = {
    policy.name: policy
    for policy in (
        ReferencePolicy(),
        AlwaysFinishPolicy(),
        AlwaysSearchPolicy(),
        RandomToolPolicy(),
        WrongArgsPolicy(),
    )
}


@dataclass(frozen=True)
class BaselineResult:
    """One policy/task score, kept JSON-friendly for reports and CI artifacts."""

    policy: str
    task_id: str
    category: str
    sample_seed: int
    reward: float
    finished: bool
    exact_tool_path: bool
    exact_reference: bool
    arg_score: Optional[float]
    components: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "task_id": self.task_id,
            "category": self.category,
            "sample_seed": self.sample_seed,
            "reward": self.reward,
            "finished": self.finished,
            "exact_tool_path": self.exact_tool_path,
            "exact_reference": self.exact_reference,
            "arg_score": self.arg_score,
            "components": self.components,
        }


@dataclass(frozen=True)
class PolicySummary:
    """Aggregate score-sensitivity indicators for one baseline policy."""

    policy: str
    task_count: int
    sample_count: int
    mean_reward: float
    reward_std: float
    min_reward: float
    max_reward: float
    finish_rate: float
    exact_tool_path_rate: float
    mean_arg_score: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "task_count": self.task_count,
            "sample_count": self.sample_count,
            "mean_reward": self.mean_reward,
            "reward_std": self.reward_std,
            "min_reward": self.min_reward,
            "max_reward": self.max_reward,
            "finish_rate": self.finish_rate,
            "exact_tool_path_rate": self.exact_tool_path_rate,
            "mean_arg_score": self.mean_arg_score,
        }


@dataclass(frozen=True)
class TaskScoreSummary:
    """Per-task aggregate used to surface suspiciously strong weak policies."""

    policy: str
    task_id: str
    category: str
    sample_count: int
    mean_reward: float
    max_reward: float
    exact_tool_path_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "task_id": self.task_id,
            "category": self.category,
            "sample_count": self.sample_count,
            "mean_reward": self.mean_reward,
            "max_reward": self.max_reward,
            "exact_tool_path_rate": self.exact_tool_path_rate,
        }


def resolve_policies(names: Iterable[str]) -> List[BaselinePolicy]:
    """Resolve policy names and fail early on typos in a CLI invocation."""

    selected = []
    for name in names:
        try:
            selected.append(ALL_POLICIES[name])
        except KeyError as exc:
            choices = ", ".join(ALL_POLICIES)
            raise ValueError(f"Unknown baseline policy {name!r}; choose from: {choices}") from exc
    return selected


def evaluate_baselines(
    tasks: Sequence[Mapping[str, Any]],
    policies: Iterable[BaselinePolicy],
    *,
    seed: int = 42,
    random_samples: int = 20,
    training_step: int = 100,
) -> List[BaselineResult]:
    """Score every policy/task pair without an LLM, network call, or tool execution."""

    if random_samples < 1:
        raise ValueError("random_samples must be at least 1")

    calculator = RewardCalculator()
    results = []
    for policy in policies:
        sample_seeds = (
            range(seed, seed + random_samples)
            if isinstance(policy, RandomToolPolicy)
            else (seed,)
        )
        for sample_seed in sample_seeds:
            for task in tasks:
                breakdown, metrics = calculator.compute_reward_breakdown(
                    dict(task),
                    policy.build_result(task, sample_seed),
                    agent_type=f"baseline:{policy.name}",
                    training_step=training_step,
                )
                expected_args = task.get("expected_tool_args")
                arg_score = metrics.arg_score if expected_args else None
                results.append(BaselineResult(
                    policy=policy.name,
                    task_id=str(task["id"]),
                    category=str(task.get("category", "uncategorized")),
                    sample_seed=sample_seed,
                    reward=breakdown.total,
                    finished=metrics.task_completed,
                    exact_tool_path=metrics.tool_call_accurate,
                    exact_reference=(
                        metrics.tool_call_accurate
                        and (arg_score is None or arg_score == 1.0)
                    ),
                    arg_score=arg_score,
                    components=breakdown.to_dict(),
                ))
    return results


def summarize_baselines(results: Sequence[BaselineResult]) -> List[PolicySummary]:
    """Summarize score separation while keeping FINISH distinct from success."""

    by_policy: Dict[str, List[BaselineResult]] = {}
    for result in results:
        by_policy.setdefault(result.policy, []).append(result)

    summaries = []
    for policy, rows in by_policy.items():
        if not rows:
            continue
        arg_scores = [row.arg_score for row in rows if row.arg_score is not None]
        summaries.append(PolicySummary(
            policy=policy,
            task_count=len({row.task_id for row in rows}),
            sample_count=len(rows),
            mean_reward=round(mean(row.reward for row in rows), 6),
            reward_std=round(pstdev(row.reward for row in rows), 6),
            min_reward=round(min(row.reward for row in rows), 6),
            max_reward=round(max(row.reward for row in rows), 6),
            finish_rate=round(mean(row.finished for row in rows), 6),
            exact_tool_path_rate=round(mean(row.exact_tool_path for row in rows), 6),
            mean_arg_score=round(mean(arg_scores), 6) if arg_scores else None,
        ))
    return summaries


def highest_scoring_tasks(
    results: Sequence[BaselineResult], limit_per_policy: int = 5
) -> List[TaskScoreSummary]:
    """Return high-scoring tasks whose trajectories do not match the reference."""

    if limit_per_policy < 1:
        raise ValueError("limit_per_policy must be at least 1")

    grouped: Dict[tuple, List[BaselineResult]] = {}
    for result in results:
        if result.policy == "reference" or result.exact_reference:
            continue
        key = (result.policy, result.task_id, result.category)
        grouped.setdefault(key, []).append(result)

    by_policy: Dict[str, List[TaskScoreSummary]] = {}
    for (policy, task_id, category), rows in grouped.items():
        by_policy.setdefault(policy, []).append(TaskScoreSummary(
            policy=policy,
            task_id=task_id,
            category=category,
            sample_count=len(rows),
            mean_reward=round(mean(row.reward for row in rows), 6),
            max_reward=round(max(row.reward for row in rows), 6),
            exact_tool_path_rate=round(mean(row.exact_tool_path for row in rows), 6),
        ))

    top = []
    for rows in by_policy.values():
        rows.sort(key=lambda row: (-row.mean_reward, row.task_id))
        top.extend(rows[:limit_per_policy])
    return top


def reference_gap_failures(
    summaries: Sequence[PolicySummary], min_gap: float = 0.3
) -> List[str]:
    """Return health-check failures when weak policies approach the reference."""

    if min_gap < 0:
        raise ValueError("min_gap must be non-negative")
    by_policy = {summary.policy: summary for summary in summaries}
    reference = by_policy.get("reference")
    if reference is None:
        return ["reference policy is required for the reward-gap health check"]

    failures = []
    for policy, summary in by_policy.items():
        if policy == "reference":
            continue
        gap = reference.mean_reward - summary.mean_reward
        if gap < min_gap:
            failures.append(
                f"{policy}: reference gap {gap:.3f} is below required {min_gap:.3f}"
            )
    return failures


def category_gap_failures(
    results: Sequence[BaselineResult], min_gap: float = 0.3
) -> List[str]:
    """Per-category health check: an aggregate mean hides single-category holes.

    ``reference_gap_failures`` compares whole-task-set means, so one leaky
    category is diluted by the rest. Search tasks once left ``always_search``
    only 0.167 below the reference while the aggregate check still passed.

    Rows where a policy reproduced the reference trajectory are excluded: a
    degenerate policy that happens to emit the reference solution has earned
    the reference score, and on ``infeasible`` tasks ``always_finish`` *is*
    the reference. A policy with no other rows in that category is skipped.
    """

    if min_gap < 0:
        raise ValueError("min_gap must be non-negative")

    reference: Dict[str, List[float]] = {}
    others: Dict[tuple, List[float]] = {}
    for result in results:
        if result.policy == "reference":
            reference.setdefault(result.category, []).append(result.reward)
        elif not result.exact_reference:
            others.setdefault((result.category, result.policy), []).append(result.reward)

    if not reference:
        return ["reference policy is required for the reward-gap health check"]

    failures = []
    for (category, policy), rewards in sorted(others.items()):
        if category not in reference:
            continue
        gap = mean(reference[category]) - mean(rewards)
        if gap < min_gap:
            failures.append(
                f"{category}/{policy}: reference gap {gap:.3f} is below "
                f"required {min_gap:.3f}"
            )
    return failures


def render_markdown(
    summaries: Sequence[PolicySummary],
    *,
    task_set: str,
    seed: int,
    training_step: int,
    top_tasks: Sequence[TaskScoreSummary] = (),
    min_reference_gap: Optional[float] = None,
    health_failures: Sequence[str] = (),
    min_category_gap: Optional[float] = None,
    category_failures: Sequence[str] = (),
) -> str:
    """Render a small report suitable for terminal output and saved artifacts."""

    lines = [
        "# Benchmark Degenerate-policy Baselines",
        "",
        f"Task set: `{task_set}` | seed: `{seed}` | training step: `{training_step}`",
        "",
        "These synthetic policies never execute tools. `finish_rate` only means the "
        "trajectory ended in `FINISH`; it is intentionally reported separately from "
        "tool-path accuracy and reward.",
        "",
        "| Policy | Tasks | Samples | Mean reward | Reward std | Reward range | Finish rate | Exact tool path | Mean arg score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        arg_score = (
            f"{summary.mean_arg_score:.3f}"
            if summary.mean_arg_score is not None
            else "N/A"
        )
        lines.append(
            "| {policy} | {tasks} | {samples} | {mean:.3f} | {std:.3f} | "
            "[{low:.3f}, {high:.3f}] | {finish:.1%} | {path:.1%} | {args} |".format(
                policy=summary.policy,
                tasks=summary.task_count,
                samples=summary.sample_count,
                mean=summary.mean_reward,
                std=summary.reward_std,
                low=summary.min_reward,
                high=summary.max_reward,
                finish=summary.finish_rate,
                path=summary.exact_tool_path_rate,
                args=arg_score,
            )
        )

    if min_reference_gap is not None:
        status = "FAIL" if health_failures else "PASS"
        lines.extend([
            "",
            f"Health check: **{status}** (minimum reference gap: `{min_reference_gap:.3f}`)",
        ])
        lines.extend(f"- {failure}" for failure in health_failures)

    if min_category_gap is not None:
        status = "FAIL" if category_failures else "PASS"
        lines.extend([
            "",
            f"Per-category check: **{status}** (minimum gap: `{min_category_gap:.3f}`). "
            "The aggregate check above averages over the whole task set, so a "
            "single leaky category is diluted by the rest.",
        ])
        lines.extend(f"- {failure}" for failure in category_failures)

    if top_tasks:
        lines.extend([
            "",
            "## Highest-scoring non-reference trajectories",
            "",
            "| Policy | Task | Category | Samples | Mean reward | Max reward | Exact tool path |",
            "|---|---|---|---:|---:|---:|---:|",
        ])
        for task in top_tasks:
            lines.append(
                "| {policy} | {task_id} | {category} | {samples} | {mean:.3f} | "
                "{maximum:.3f} | {path:.1%} |".format(
                    policy=task.policy,
                    task_id=task.task_id,
                    category=task.category,
                    samples=task.sample_count,
                    mean=task.mean_reward,
                    maximum=task.max_reward,
                    path=task.exact_tool_path_rate,
                )
            )
    return "\n".join(lines) + "\n"
