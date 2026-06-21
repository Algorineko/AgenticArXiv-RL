"""从 SFT 模型 rollout 生成 DPO 训练数据

DPO 数据格式：
- 每条数据包含 (prompt, chosen, rejected)
- chosen: reward 最高的 action
- rejected: reward 最低的 action

策略：
1. 用 SFT 模型对每个 task rollout 多次（如 5 次）
2. 按 reward 排序，reward 最高的作为 chosen，最低的作为 rejected
3. 构造 DPO 格式

运行方式：
    python scripts/generate_dpo_data.py
"""

import sys
import json
from pathlib import Path

# 添加 AgenticArxiv 到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "AgenticArxiv"))

from benchmark.tasks import get_all_tasks
from agents.agent_engine import ReActAgent
from agents.side_effects import NoOpSideEffectManager
from utils.llm_client import get_env_llm_client
from rl.reward import RewardCalculator


def generate_dpo_dataset(num_rollouts_per_task: int = 5):
    """
    从 SFT 模型 rollout 生成 DPO 数据集

    Args:
        num_rollouts_per_task: 每个任务 rollout 次数
    """

    sft_model_path = Path("outputs/sft/final")
    if not sft_model_path.exists():
        print(f"❌ SFT 模型不存在: {sft_model_path}")
        print(f"请先运行: python -m rl.train_sft")
        return

    # TODO: 加载 SFT 模型（需要修改 llm_client 支持本地模型）
    # 目前占位：使用原始 LLM
    llm_client = get_env_llm_client()
    agent = ReActAgent(llm_client, side_effect_mgr=NoOpSideEffectManager())

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

        # 排序
        rollouts.sort(key=lambda x: x["reward"], reverse=True)

        # 取最优和最差
        best = rollouts[0]
        worst = rollouts[-1]

        # 提取最后一步的 action（或第一步，取决于策略）
        best_action = best["result"]["history"][-1].get("action", "") if best["result"]["history"] else ""
        worst_action = worst["result"]["history"][-1].get("action", "") if worst["result"]["history"] else ""

        if not best_action or not worst_action or best_action == worst_action:
            print(f"   ⚠️  chosen/rejected 无效，跳过")
            continue

        dpo_data.append({
            "prompt": task_def["task"],
            "chosen": best_action,
            "rejected": worst_action,
        })

        print(f"   ✅ chosen_reward={best['reward']:.2f}, rejected_reward={worst['reward']:.2f}")

    # 保存到 JSONL
    output_path = Path("data/dpo/dpo_train.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ DPO 数据生成完成：{len(dpo_data)} 条样本 → {output_path}")


if __name__ == "__main__":
    generate_dpo_dataset()
