"""Rollout 循环（收集 trajectory）

核心功能：
1. 加载任务（from benchmark/tasks）
2. 执行 Agent（使用 NoOpSideEffectManager）
3. 计算 reward（使用 RewardCalculator）
4. 保存 trajectory（JSONL 格式）

使用方式：
    python -m AgenticArxiv.rl.rollout search_01 traces/train/
"""

import sys
from pathlib import Path
from datetime import datetime
import fire

# 添加 AgenticArxiv 到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.tasks import get_task_by_id, get_all_tasks
from agents.agent_engine import ReActAgent
from agents.side_effects import NoOpSideEffectManager
from utils.llm_client import get_env_llm_client
from rl.reward import RewardCalculator
from rl.trajectory import create_trajectory, save_trajectory


def rollout_single_task(
    task_id: str,
    output_dir: str = "traces/train",
    session_id: str = "rl_rollout",
) -> None:
    """对单个任务执行 rollout

    Args:
        task_id: 任务 ID（如 "search_01"）
        output_dir: 输出目录（JSONL 文件）
        session_id: 会话 ID
    """
    # 1. 加载任务
    task_def = get_task_by_id(task_id)
    if not task_def:
        print(f"❌ 任务 {task_id} 不存在")
        return

    print(f"📋 任务: {task_def['id']} - {task_def['task']}")

    # 2. 创建 Agent（使用 NoOpSideEffectManager）
    llm_client = get_env_llm_client()
    agent = ReActAgent(llm_client, side_effect_mgr=NoOpSideEffectManager())

    # 3. 执行 Agent
    print(f"🤖 执行 Agent...")
    result = agent.run(task_def["task"], session_id=session_id)

    # 4. 计算 reward
    reward_calc = RewardCalculator()
    reward, metrics = reward_calc.compute_reward(
        task_def, result, agent_type="regex", trial=0, session_id=session_id
    )

    print(f"✅ 任务完成")
    print(f"   Reward: {reward:.2f}")
    print(f"   Metrics: task_completed={metrics.task_completed}, "
          f"tool_call_accurate={metrics.tool_call_accurate}, "
          f"parse_failures={metrics.parse_failures}, "
          f"tool_exec_failures={metrics.tool_exec_failures}")

    # 5. 构造 trajectory
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
        model=llm_client.model if hasattr(llm_client, "model") else "",
        termination_type=metrics.termination_type,
    )

    # 6. 保存 trajectory
    output_path = Path(output_dir) / f"rollout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    save_trajectory(traj, output_path)
    print(f"💾 Trajectory 保存至: {output_path}")


def rollout_all_tasks(
    output_dir: str = "traces/train",
    session_id_prefix: str = "rl_rollout",
) -> None:
    """对所有任务执行 rollout

    Args:
        output_dir: 输出目录
        session_id_prefix: 会话 ID 前缀
    """
    tasks = get_all_tasks()
    print(f"📚 共 {len(tasks)} 个任务")

    for i, task_def in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(tasks)}] 任务: {task_def['id']}")
        print(f"{'='*60}")

        session_id = f"{session_id_prefix}_{task_def['id']}"
        rollout_single_task(task_def["id"], output_dir, session_id)

    print(f"\n✅ 全部 rollout 完成")


def main(
    task_id: str = None,
    output_dir: str = "traces/train",
    all: bool = False,
):
    """Rollout 主函数

    Args:
        task_id: 任务 ID（如 "search_01"）
        output_dir: 输出目录
        all: 是否对所有任务执行 rollout

    Examples:
        python -m AgenticArxiv.rl.rollout search_01 traces/train/
        python -m AgenticArxiv.rl.rollout --all --output_dir traces/train/
    """
    if all:
        rollout_all_tasks(output_dir)
    elif task_id:
        rollout_single_task(task_id, output_dir)
    else:
        print("用法:")
        print("  python -m AgenticArxiv.rl.rollout <task_id> <output_dir>")
        print("  python -m AgenticArxiv.rl.rollout --all --output_dir <output_dir>")


if __name__ == "__main__":
    fire.Fire(main)
