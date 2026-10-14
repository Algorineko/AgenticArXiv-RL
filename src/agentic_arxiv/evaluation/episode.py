"""接收现有模型客户端、任务和环境工厂的完整 episode 评测服务。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Sequence

from agentic_arxiv.contracts import ParseKind, project_result
from agentic_arxiv.runtime.agent_engine import ReActAgent
from agentic_arxiv.runtime.react_contract import parse_react_response
from agentic_arxiv.scoring.metrics import is_strict_success
from agentic_arxiv.tasks.initialization import compile_initialization


def execute_episode(
    client: Any,
    task: Mapping[str, Any],
    environment_factory: Callable[[], Any],
    *,
    initialization_plan=None,
    max_iterations: int = 6,
    llm_extra: Mapping[str, Any] | None = None,
    session_id: str = "evaluation",
    model: str | None = None,
) -> Dict[str, Any]:
    """独立执行并清理环境；准备任务和语义参考仍由调用方显式负责。"""
    environment = environment_factory()
    agent = None
    try:
        agent = ReActAgent(
            client, env=environment, max_iterations=max_iterations,
            llm_extra=dict(llm_extra or {}),
        )
        return agent.run(
            task["task"], agent_model=model, session_id=session_id,
            initialization_plan=initialization_plan,
        )
    finally:
        if agent is not None:
            agent.close()
        else:
            environment.close()


@dataclass
class EpisodeEvaluation:
    """完整 episode 的模型无关统计；不含训练停止策略或 writer。"""

    mean_reward: float
    task_completion_rate: float
    per_task_rewards: Dict[str, float]
    num_generations_per_task: int = 4
    parse_rate: float = 0.0
    episode_count: int = 0
    max_iterations: int = 6
    termination_counts: Dict[str, int] = field(default_factory=dict)
    protocol: Dict[str, Any] = field(default_factory=dict)


def evaluate_episodes(
    client: Any,
    tasks: Sequence[Mapping[str, Any]],
    environment_factory: Callable[[], Any],
    reward_calc: Any,
    *,
    num_generations: int = 4,
    max_iterations: int = 6,
    temperature: float = 1.0,
    max_new_tokens: int = 256,
    training_step: int = 0,
    session_prefix: str = "canary",
) -> EpisodeEvaluation:
    """复用既有 Canary 统计口径，不加载模型或准备评分参考答案。"""
    if num_generations < 1 or max_iterations < 1:
        raise ValueError("评测采样数和完整 episode 轮数必须为正数")
    task_lookup = {task["id"]: task for task in tasks}
    all_rewards: List[float] = []
    per_task: Dict[str, List[float]] = {}
    completed = total = parsed = decisions = 0
    termination_counts: Dict[str, int] = {}
    protocol: Dict[str, Any] = {}
    for task in tasks:
        tid = task["id"]
        initialization_plan = compile_initialization(task, task_lookup=task_lookup)
        task_rewards = per_task.setdefault(tid, [])
        for trial in range(num_generations):
            result = execute_episode(
                client, task, environment_factory,
                initialization_plan=initialization_plan,
                max_iterations=max_iterations,
                llm_extra={"temperature": temperature, "max_tokens": max_new_tokens},
                session_id=f"{session_prefix}_{tid}_{trial}",
            )
            breakdown, metrics = reward_calc.compute_reward_breakdown(
                task, project_result(result), training_step=training_step,
            )
            reward = float(breakdown.total)
            task_rewards.append(reward)
            if is_strict_success(metrics):
                completed += 1
            total += 1
            all_rewards.append(reward)
            reason = result.get("termination_reason") or metrics.termination_type
            termination_counts[reason] = termination_counts.get(reason, 0) + 1
            if not protocol:
                protocol = dict(result.get("protocol_relation") or {})
            for history_step in result["history"]:
                # TOKEN_LIMIT 含实际采样文本，其他框架标记不是模型决策。
                if history_step.get("framework_generated") and history_step.get("termination_reason") != "TOKEN_LIMIT":
                    continue
                raw = history_step.get("raw_response")
                if raw is None:
                    continue
                decisions += 1
                if parse_react_response(raw).kind != ParseKind.PARSE_ERROR:
                    parsed += 1
    mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else -1.0
    completion_rate = completed / total if total > 0 else 0.0
    return EpisodeEvaluation(
        mean_reward=round(mean_reward, 4),
        task_completion_rate=round(completion_rate, 4),
        per_task_rewards={k: round(sum(v) / len(v), 4) for k, v in per_task.items()},
        num_generations_per_task=num_generations,
        parse_rate=round(parsed / decisions, 4) if decisions else 0.0,
        episode_count=total,
        max_iterations=max_iterations,
        termination_counts=termination_counts,
        protocol=protocol,
    )
