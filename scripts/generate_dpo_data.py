"""从 SFT 模型 rollout 生成 DPO 训练数据

DPO 数据格式：
- 每条数据包含 (prompt, chosen, rejected)
- chosen: reward 最高那条轨迹的首个工具调用
- rejected: reward 最低那条轨迹的首个工具调用

策略：
1. 用 SFT 模型对每个 task rollout 多次（如 5 次）
2. 按 reward 排序，取最高与最低两条轨迹
3. 从各自轨迹中取**首个工具调用**作为 chosen / rejected

为什么取首个工具调用，而不是最后一步：
    正常结束的轨迹，最后一步 action 恒为字符串 "FINISH"
    （见 agents/base_agent.py 的执行循环），因此若取 history[-1]，
    chosen 与 rejected 会同时等于 "FINISH"、被判为无效而全部跳过，
    实际产出 0 条样本。

    首个工具调用则是可比的：此时两条轨迹的历史都为空，
    面对的是同一个状态，差异只来自策略选择的动作本身。

运行方式：
    python scripts/generate_dpo_data.py
"""

import os
import sys
import json
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "AgenticArxiv"

# 添加 AgenticArxiv 到 Python 路径
sys.path.insert(0, str(PACKAGE_ROOT))

# RL 路径不依赖数据库：会话状态走内存 store
os.environ.setdefault("STORE_BACKEND", "memory")

from benchmark.tasks import get_all_tasks
from agents.agent_engine import ReActAgent
from agents.side_effects import LocalSideEffectManager
from utils.llm_client import TransformersLLMClient
from rl.reward import RewardCalculator
from rl.env import MockArxivEnv


# 终止标记，不是真实的工具调用
TERMINAL_ACTIONS = ("FINISH", "FORCE_STOP", "ERROR")


def first_tool_action(result: dict) -> str:
    """取轨迹中第一个真实的工具调用 action（JSON 字符串）。

    没有任何工具调用时返回空字符串（例如模型直接 FINISH）。
    """
    for step in (result or {}).get("history", []) or []:
        action = step.get("action", "")
        if action and action not in TERMINAL_ACTIONS:
            return action
    return ""


def build_preference_pair(rollouts: list, task_def: dict, min_reward_gap: float = 0.0):
    """从同一任务的多条 rollout 构造一条偏好样本，无法构造时返回 None。"""
    if len(rollouts) < 2:
        return None

    # Extremes can have the same first action even when a useful, different
    # action exists in the middle. Search every valid pair and keep the largest
    # reward gap instead of silently throwing that preference signal away.
    candidates = []
    for chosen_rollout in rollouts:
        chosen = first_tool_action(chosen_rollout["result"])
        if not chosen:
            continue
        for rejected_rollout in rollouts:
            rejected = first_tool_action(rejected_rollout["result"])
            gap = chosen_rollout["reward"] - rejected_rollout["reward"]
            if rejected and chosen != rejected and gap > min_reward_gap:
                candidates.append((gap, chosen, rejected))
    if not candidates:
        return None
    _, chosen, rejected = max(candidates, key=lambda item: item[0])

    return {
        "prompt": task_def["task"],
        "chosen": chosen,
        "rejected": rejected,
    }


def generate_dpo_dataset(
    num_rollouts_per_task: int = 5,
    model: str = None,
    output: str = None,
    snapshot: str = None,
    device: str = "auto",
    dtype: str = "auto",
    temperature: float = 0.8,
    seed: int = 42,
    min_reward_gap: float = 0.05,
):
    """
    从 SFT 模型 rollout 生成 DPO 数据集

    Args:
        num_rollouts_per_task: 每个任务 rollout 次数
    """

    sft_model_path = Path(model) if model else REPO_ROOT / "outputs" / "sft" / "final"
    if not sft_model_path.exists():
        print(f"❌ SFT 模型不存在: {sft_model_path}")
        print(f"请先运行: python -m AgenticArxiv.rl.train_sft")
        return

    print(f"📦 加载本地 SFT 模型: {sft_model_path}")
    llm_client = TransformersLLMClient(
        str(sft_model_path), device=device, dtype=dtype, seed=seed
    )
    snapshot_path = Path(snapshot) if snapshot else REPO_ROOT / "data" / "mock_arxiv_snapshot.json"
    env = None
    if snapshot_path.exists():
        env = MockArxivEnv(snapshot_path=snapshot_path, mode="replay")
        print(f"🗂 使用离线快照: {snapshot_path}")
    else:
        print(
            f"⚠️ 未找到离线快照 {snapshot_path}，工具将使用实时网络。"
            "如需可复现生成，请先运行 python -m AgenticArxiv.rl.build_snapshot"
        )
    agent = ReActAgent(
        llm_client,
        side_effect_mgr=LocalSideEffectManager(),
        env=env,
        llm_extra={"temperature": temperature},
    )

    reward_calc = RewardCalculator()
    dpo_data = []
    tasks = get_all_tasks()

    print(f"📚 共 {len(tasks)} 个任务，每个 rollout {num_rollouts_per_task} 次")

    for i, task_def in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] 任务: {task_def['id']} - {task_def['task']}")

        rollouts = []

        for j in range(num_rollouts_per_task):
            try:
                result = agent.run(
                    task_def["task"],
                    session_id=f"dpo_gen_{task_def['id']}_r{j}"
                )
                reward, metrics = reward_calc.compute_reward(task_def, result)
                rollouts.append({
                    "result": result,
                    "reward": reward,
                    "metrics": metrics,
                })
                print(f"   rollout {j+1}: reward={reward:.2f}")

            except Exception as e:
                print(f"   rollout {j+1}: ❌ {e}")

        if len(rollouts) < 2:
            print(f"   ⚠️  rollout 数量不足，跳过")
            continue

        pair = build_preference_pair(rollouts, task_def, min_reward_gap=min_reward_gap)
        if pair is None:
            print(f"   ⚠️  chosen/rejected 无效（无工具调用、动作相同或奖励无差异），跳过")
            continue

        dpo_data.append(pair)
        rewards = [r["reward"] for r in rollouts]

        print(f"   ✅ chosen_reward={max(rewards):.2f}, rejected_reward={min(rewards):.2f}")

    # 保存到 JSONL
    output_path = Path(output) if output else REPO_ROOT / "data" / "dpo" / "dpo_train.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ DPO 数据生成完成：{len(dpo_data)} 条样本 → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从本地 SFT 模型生成 DPO 偏好数据")
    parser.add_argument("--model", default=None, help="本地 SFT 模型路径")
    parser.add_argument("--output", default=None, help="输出 JSONL 路径")
    parser.add_argument("--snapshot", default=None, help="离线 arXiv 快照路径")
    parser.add_argument("--num_rollouts_per_task", type=int, default=5)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_reward_gap", type=float, default=0.05)
    generate_dpo_dataset(**vars(parser.parse_args()))
