"""从 benchmark tasks 生成 SFT 训练数据

SFT 数据格式：
- 每一步 (thought, action) 对作为一条训练样本
- 使用 messages 格式（system/user/assistant）
- 只学习 action（工具调用 JSON）

运行方式：
    python scripts/generate_sft_data.py
"""

import os
import sys
import json
from pathlib import Path

# 添加 AgenticArxiv 到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "AgenticArxiv"))

# RL 路径不依赖数据库：会话状态走内存 store
os.environ.setdefault("STORE_BACKEND", "memory")

from benchmark.tasks import get_all_tasks
from agents.agent_engine import ReActAgent
from agents.side_effects import LocalSideEffectManager
from utils.llm_client import get_env_llm_client


def generate_sft_dataset():
    """执行所有 benchmark tasks，收集成功的 trajectories 作为 expert demos"""

    llm_client = get_env_llm_client()
    agent = ReActAgent(llm_client, side_effect_mgr=LocalSideEffectManager())

    sft_data = []
    tasks = get_all_tasks()

    print(f"📚 共 {len(tasks)} 个任务")

    for i, task_def in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] 任务: {task_def['id']} - {task_def['task']}")

        try:
            result = agent.run(task_def["task"], session_id=f"sft_gen_{task_def['id']}")

            # 只保留成功的 trajectories
            if result.get("iteration_count", 0) > 0 and result.get("history"):
                print(f"   ✅ 成功，共 {len(result['history'])} 步")

                # 转为 SFT 格式（每一步作为一条训练样本）
                for step in result["history"]:
                    action = step.get("action", "")
                    if not action or action == "FINISH":
                        continue

                    sft_data.append({
                        "messages": [
                            {
                                "role": "system",
                                "content": "你是一个 arXiv 论文检索 Agent，可以调用工具完成任务。"
                            },
                            {
                                "role": "user",
                                "content": task_def["task"]
                            },
                            {
                                "role": "assistant",
                                "content": action  # 只学习 action
                            }
                        ]
                    })
            else:
                print(f"   ⚠️  执行失败或无有效步骤")

        except Exception as e:
            print(f"   ❌ 执行出错: {e}")

    # 保存到 JSONL
    output_path = Path("data/sft/sft_train.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ SFT 数据生成完成：{len(sft_data)} 条样本 → {output_path}")


if __name__ == "__main__":
    generate_sft_dataset()
