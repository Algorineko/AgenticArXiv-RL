"""从 SFT 模型 rollout 生成 DPO 偏好数据

DPO 数据格式：
- 每条数据包含 (prompt, chosen, rejected)
- 支持多轮交互下的共同前缀发散步（divergence turn）抽取：在多步复合任务中，
  若前置步骤相同而在后续步骤产生分歧，能精准提取该决策点的状态与偏好对
- 提示词使用与推理对齐的 ReAct Prompt 模板，包含当前决策步的前置历史

策略：
1. 用 SFT 模型对每个 task rollout 多次（如 5 次）
2. 在所有有效轨迹对中比对，找到最早的分歧动作决策点（前缀历史相同，当前动作不同）
3. 选取奖励差异最大的决策点构造偏好样本

运行方式：
    python scripts/generate_dpo_data.py
    python scripts/generate_dpo_data.py --model outputs/sft/final --num_rollouts_per_task 8
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "AgenticArxiv"

# 添加 AgenticArxiv 到 Python 路径
sys.path.insert(0, str(PACKAGE_ROOT))

# RL 路径不依赖数据库：会话状态走内存 store
os.environ.setdefault("STORE_BACKEND", "memory")

import tools.arxiv_tool  # noqa: F401
import tools.cache_status_tool  # noqa: F401
import tools.pdf_download_tool  # noqa: F401
import tools.pdf_translate_tool  # noqa: F401
from tools.bootstrap import register_all_tools
register_all_tools()

from agents.agent_engine import ReActAgent
from agents.prompt_templates import format_tool_description, get_react_prompt
from agents.side_effects import LocalSideEffectManager
from benchmark.task_spec import build
from benchmark.tasks import BENCHMARK_SPECS, get_all_tasks
from rl.env import MockArxivEnv
from rl.reward import RewardCalculator
from tools.tool_registry import registry
from utils.llm_client import TransformersLLMClient

# 终止标记
TERMINAL_ACTIONS = ("FINISH", "FORCE_STOP", "ERROR", "PARSE_ERROR")


def _format_history_text(steps: List[Dict[str, Any]]) -> str:
    """将前置步骤格式化为 ReAct 历史文本"""
    parts = []
    for s in steps:
        thought = s.get("thought", "")
        action = s.get("action", "")
        if isinstance(action, dict):
            action = json.dumps(action, ensure_ascii=False)
        obs = s.get("observation", "")
        parts.append(f"Thought: {thought}\nAction: {action}\nObservation: {obs}")
    return "\n\n".join(parts)


def _canonical_action_str(action: Any) -> str:
    """归一化动作字符串用于比对"""
    if isinstance(action, dict):
        return json.dumps(action, sort_keys=True, ensure_ascii=False)
    if isinstance(action, str):
        try:
            parsed = json.loads(action)
            if isinstance(parsed, dict):
                return json.dumps(parsed, sort_keys=True, ensure_ascii=False)
        except Exception:
            pass
        return action.strip()
    return str(action).strip()


def find_divergence_step(
    chosen_history: List[Dict[str, Any]],
    rejected_history: List[Dict[str, Any]],
) -> Optional[Tuple[int, Dict[str, Any], Dict[str, Any]]]:
    """寻找两条轨迹在相同前缀历史下的第一个动作分歧点。

    Returns:
        (divergence_step_index, chosen_step, rejected_step) 或 None
    """
    min_len = min(len(chosen_history), len(rejected_history))

    for k in range(min_len):
        c_step = chosen_history[k]
        r_step = rejected_history[k]

        c_act = _canonical_action_str(c_step.get("action", ""))
        r_act = _canonical_action_str(r_step.get("action", ""))

        if c_act != r_act:
            # 两个动作产生分歧，且 0..k-1 步完全一致
            if not c_act or not r_act:
                return None
            return k, c_step, r_step

    # 若共同长度内动作完全一致，但长度不同（一个提前 FINISH 或超时）
    if len(chosen_history) != len(rejected_history):
        k = min_len
        c_step = chosen_history[k] if k < len(chosen_history) else {"thought": "任务完成", "action": "FINISH"}
        r_step = rejected_history[k] if k < len(rejected_history) else {"thought": "任务完成", "action": "FINISH"}
        c_act = _canonical_action_str(c_step.get("action", ""))
        r_act = _canonical_action_str(r_step.get("action", ""))
        if c_act != r_act:
            return k, c_step, r_step

    return None


def build_preference_pair(
    rollouts: List[Dict[str, Any]],
    task_def: Dict[str, Any],
    tools_description: str,
    min_reward_gap: float = 0.05,
) -> Optional[Dict[str, Any]]:
    """从同一任务的多条 rollout 构造最优偏好样本。

    支持多轮前缀发散抽取与 ReAct Prompt 完整对齐。
    """
    if len(rollouts) < 2:
        return None

    candidates = []

    for chosen_r in rollouts:
        c_hist = (chosen_r.get("result") or {}).get("history") or []
        if not c_hist:
            continue

        for rejected_r in rollouts:
            r_hist = (rejected_r.get("result") or {}).get("history") or []
            if not r_hist:
                continue

            gap = chosen_r["reward"] - rejected_r["reward"]
            if gap <= min_reward_gap:
                continue

            div = find_divergence_step(c_hist, r_hist)
            if div is None:
                continue

            step_idx, c_step, r_step = div
            c_action = c_step.get("action", "")
            r_action = r_step.get("action", "")

            # 构造共同前缀提示词
            prefix_history = c_hist[:step_idx]
            history_text = _format_history_text(prefix_history)
            prompt = get_react_prompt(
                task=task_def["task"],
                tools_description=tools_description,
                history=history_text,
            )

            # 格式化 chosen 与 rejected 内容
            c_thought = c_step.get("thought", "")
            r_thought = r_step.get("thought", "")

            c_chosen = (
                f"Thought: {c_thought}\nAction: {c_action}"
                if c_thought
                else f"Action: {c_action}"
            )
            r_rejected = (
                f"Thought: {r_thought}\nAction: {r_action}"
                if r_thought
                else f"Action: {r_action}"
            )

            candidates.append((gap, prompt, c_chosen, r_rejected, step_idx))

    if not candidates:
        return None

    # 优先选取奖励差距最大的分歧点
    gap, best_prompt, best_chosen, best_rejected, step_idx = max(
        candidates, key=lambda item: item[0]
    )

    return {
        "prompt": best_prompt,
        "chosen": best_chosen,
        "rejected": best_rejected,
        "task_id": task_def.get("id", ""),
        "divergence_step": step_idx,
        "reward_gap": round(gap, 4),
    }


def generate_dpo_dataset(
    num_rollouts_per_task: int = 5,
    model: Optional[str] = None,
    output: Optional[str] = None,
    snapshot: Optional[str] = None,
    device: str = "auto",
    dtype: str = "auto",
    temperature: float = 0.8,
    seed: int = 42,
    min_reward_gap: float = 0.05,
    task_set: str = "basic",
) -> None:
    """从 SFT 模型 rollout 生成 DPO 数据集"""
    sft_model_path = Path(model) if model else REPO_ROOT / "outputs" / "sft" / "final"
    if not sft_model_path.exists():
        print(f"❌ SFT 模型不存在: {sft_model_path}")
        print(f"请先运行: python -m AgenticArxiv.rl.train_sft")
        return

    print(f"📦 加载本地 SFT 模型: {sft_model_path}")
    llm_client = TransformersLLMClient(
        str(sft_model_path), device=device, dtype=dtype, seed=seed
    )
    snapshot_path = (
        Path(snapshot) if snapshot else REPO_ROOT / "data" / "mock_arxiv_snapshot.json"
    )
    env = None
    if snapshot_path.exists():
        env = MockArxivEnv(snapshot_path=snapshot_path, mode="replay")
        print(f"🗂 使用离线快照: {snapshot_path}")
    else:
        print(f"⚠️ 未找到离线快照 {snapshot_path}，使用自动回放环境")
        env = MockArxivEnv(mode="auto")

    agent = ReActAgent(
        llm_client,
        side_effect_mgr=LocalSideEffectManager(),
        env=env,
        llm_extra={"temperature": temperature},
    )

    reward_calc = RewardCalculator()
    tools_desc = format_tool_description(registry.list_tools())
    dpo_data: List[Dict[str, Any]] = []

    if task_set == "expanded":
        try:
            from benchmark.tasks_expanded import EXPANDED_SPECS
            tasks = build(EXPANDED_SPECS)
        except ImportError:
            tasks = get_all_tasks()
    else:
        tasks = get_all_tasks()

    print(f"📚 共 {len(tasks)} 个任务，每个任务采样 {num_rollouts_per_task} 次")

    for i, task_def in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] 任务: {task_def['id']} - {task_def['task']}")
        rollouts = []

        for j in range(num_rollouts_per_task):
            try:
                result = agent.run(
                    task_def["task"],
                    session_id=f"dpo_gen_{task_def['id']}_r{j}",
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
            print(f"   ⚠️ rollout 数量不足，跳过")
            continue

        pair = build_preference_pair(
            rollouts, task_def, tools_desc, min_reward_gap=min_reward_gap
        )
        if pair is None:
            print(f"   ⚠️ 未发现有效分歧决策点（动作相同或奖励无显著差异），跳过")
            continue

        dpo_data.append(pair)
        rewards = [r["reward"] for r in rollouts]
        print(
            f"   ✅ 提取偏好对: gap={pair['reward_gap']:.2f}, step={pair['divergence_step']}"
        )

    # 保存到 JSONL
    output_path = (
        Path(output) if output else REPO_ROOT / "data" / "dpo" / "dpo_train.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ DPO 数据生成完成：共 {len(dpo_data)} 条偏好样本 → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从本地 SFT 模型生成 DPO 偏好数据")
    parser.add_argument("--model", default=None, help="本地 SFT 模型路径")
    parser.add_argument("--output", default=None, help="输出 JSONL 路径")
    parser.add_argument("--snapshot", default=None, help="离线 arXiv 快照路径")
    parser.add_argument("--num_rollouts_per_task", type=int, default=5)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_reward_gap", type=float, default=0.05)
    parser.add_argument(
        "--task_set", choices=["basic", "expanded"], default="basic"
    )
    generate_dpo_dataset(**vars(parser.parse_args()))

