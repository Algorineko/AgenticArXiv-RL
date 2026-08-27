"""Rollout 循环（收集 trajectory）

核心功能：
1. 加载任务（from benchmark/tasks）
2. 执行 Agent（支持 MockArxivEnv 离线快照环境与本地模型）
3. 计算 reward（使用 RewardCalculator）
4. 保存 trajectory（JSONL 格式）

使用方式：
    python -m AgenticArxiv.rl.rollout search_01 traces/train/
    python -m AgenticArxiv.rl.rollout --all --snapshot data/mock_arxiv_snapshot.json
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import fire

# 添加 AgenticArxiv 到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# RL 路径不依赖数据库：会话状态走内存 store
os.environ.setdefault("STORE_BACKEND", "memory")

from agents.agent_engine import ReActAgent
from agents.side_effects import LocalSideEffectManager
from benchmark.task_spec import build
from benchmark.tasks import get_all_tasks, get_task_by_id
from rl.env import MockArxivEnv
from rl.reward import RewardCalculator
from rl.trajectory import create_trajectory, save_trajectory
from utils.llm_client import TransformersLLMClient, get_env_llm_client


def _create_agent(
    model: Optional[str] = None,
    snapshot: Optional[str] = None,
    temperature: float = 0.1,
    max_iterations: int = 5,
):
    env = None
    if snapshot:
        snapshot_path = Path(snapshot)
        if snapshot_path.exists():
            env = MockArxivEnv(snapshot_path=snapshot_path, mode="replay")
        else:
            env = MockArxivEnv(mode="auto")
    else:
        default_snap = Path(__file__).resolve().parents[2] / "data" / "mock_arxiv_snapshot.json"
        if default_snap.exists():
            env = MockArxivEnv(snapshot_path=default_snap, mode="replay")

    if model and Path(model).exists():
        llm_client = TransformersLLMClient(model)
    else:
        llm_client = get_env_llm_client()

    agent = ReActAgent(
        llm_client,
        side_effect_mgr=LocalSideEffectManager(),
        env=env,
        max_iterations=max_iterations,
        llm_extra={"temperature": temperature},
    )
    return agent, llm_client


def rollout_single_task(
    task_id: str,
    output_dir: str = "traces/train",
    session_id: str = "rl_rollout",
    agent: Optional[ReActAgent] = None,
    llm_client: Optional[Any] = None,
    task_set: str = "basic",
) -> None:
    """对单个任务执行 rollout"""
    task_def = get_task_by_id(task_id)
    if not task_def and task_set == "expanded":
        from benchmark.tasks_expanded import EXPANDED_SPECS
        expanded_tasks = build(EXPANDED_SPECS)
        for t in expanded_tasks:
            if t["id"] == task_id:
                task_def = t
                break

    if not task_def:
        print(f"❌ 任务 {task_id} 不存在")
        return

    print(f"📋 任务: {task_def['id']} - {task_def['task']}")

    if agent is None:
        agent, llm_client = _create_agent()

    print(f"🤖 执行 Agent...")
    result = agent.run(task_def["task"], session_id=session_id)

    # 计算 reward
    reward_calc = RewardCalculator()
    reward_breakdown, metrics = reward_calc.compute_reward_breakdown(
        task_def, result, agent_type="regex", trial=0, session_id=session_id
    )
    reward = reward_breakdown.total

    print(f"✅ 任务完成")
    print(f"   Reward: {reward:.2f}")
    print(
        f"   Metrics: task_completed={metrics.task_completed}, "
        f"tool_call_accurate={metrics.tool_call_accurate}, "
        f"parse_failures={metrics.parse_failures}, "
        f"tool_exec_failures={metrics.tool_exec_failures}"
    )

    # 构造 trajectory
    model_name = getattr(llm_client, "model", "") if llm_client else ""
    traj = create_trajectory(
        task_id=task_def["id"],
        task=task_def["task"],
        session_id=session_id,
        history=result.get("history", []),
        final_reward=reward,
        metrics={
            "task_completed": metrics.task_completed,
            "tool_call_accurate": metrics.tool_call_accurate,
            "parse_failures": metrics.parse_failures,
            "tool_exec_failures": metrics.tool_exec_failures,
            "termination_type": metrics.termination_type,
            "tool_call_sequence": metrics.tool_call_sequence,
            "expected_tools": metrics.expected_tools,
        },
        model=model_name,
        termination_type=metrics.termination_type,
        reward_components=reward_breakdown.to_dict(),
    )

    output_path = Path(output_dir) / f"rollout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    save_trajectory(traj, output_path)
    print(f"💾 Trajectory 保存至: {output_path}")


def rollout_all_tasks(
    output_dir: str = "traces/train",
    session_id_prefix: str = "rl_rollout",
    model: Optional[str] = None,
    snapshot: Optional[str] = None,
    temperature: float = 0.1,
    max_iterations: int = 5,
    task_set: str = "basic",
) -> None:
    """对所有任务执行 rollout"""
    if task_set == "expanded":
        from benchmark.tasks_expanded import EXPANDED_SPECS
        tasks = build(EXPANDED_SPECS)
    else:
        tasks = get_all_tasks()

    print(f"📚 共 {len(tasks)} 个任务 (task_set={task_set})")
    agent, llm_client = _create_agent(
        model=model, snapshot=snapshot, temperature=temperature, max_iterations=max_iterations
    )

    for i, task_def in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(tasks)}] 任务: {task_def['id']}")
        print(f"{'='*60}")

        session_id = f"{session_id_prefix}_{task_def['id']}"
        rollout_single_task(
            task_def["id"],
            output_dir=output_dir,
            session_id=session_id,
            agent=agent,
            llm_client=llm_client,
            task_set=task_set,
        )

    print(f"\n✅ 全部 rollout 完成")


def main(
    task_id: Optional[str] = None,
    output_dir: str = "traces/train",
    all: bool = False,
    model: Optional[str] = None,
    snapshot: Optional[str] = None,
    temperature: float = 0.1,
    max_iterations: int = 5,
    task_set: str = "basic",
):
    """Rollout 主函数"""
    if all:
        rollout_all_tasks(
            output_dir=output_dir,
            model=model,
            snapshot=snapshot,
            temperature=temperature,
            max_iterations=max_iterations,
            task_set=task_set,
        )
    elif task_id:
        agent, llm_client = _create_agent(
            model=model, snapshot=snapshot, temperature=temperature, max_iterations=max_iterations
        )
        rollout_single_task(
            task_id=task_id,
            output_dir=output_dir,
            agent=agent,
            llm_client=llm_client,
            task_set=task_set,
        )
    else:
        print("用法:")
        print("  python -m AgenticArxiv.rl.rollout search_01 traces/train/")
        print("  python -m AgenticArxiv.rl.rollout --all --output_dir traces/train/")


if __name__ == "__main__":
    fire.Fire(main)

