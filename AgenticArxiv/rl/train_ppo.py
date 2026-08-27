"""PPO 训练脚本（使用 TRL PPOTrainer + Verifiable Reward）

PPO（Proximal Policy Optimization）：
- 目标：使用 Actor-Critic 架构与可验证规则奖励（RLVR）在线优化策略
- 优势：标准的策略梯度方法，配合 Value Head 估计状态价值，具备强大的策略收敛能力
- 数据：在线算法，直接由 benchmark/tasks.py 派生 Prompt 数据集

奖励与评估：
- 沿用项目标准的五分量可验证奖励（RewardCalculator）
- 支持离线快照环境回放（MockArxivEnv）
- 支持训练过程中的 Canary 监控与阶段验证

使用方式：
    python -m AgenticArxiv.rl.train_ppo
    python -m AgenticArxiv.rl.train_ppo --model outputs/grpo/final --batch_size 4 --lr 1e-6
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent

# 添加 AgenticArxiv 到 Python 路径
sys.path.insert(0, str(PACKAGE_ROOT))

# 奖励计算走内存 store
os.environ.setdefault("STORE_BACKEND", "memory")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

import tools.arxiv_tool  # noqa: F401  触发工具注册
import tools.cache_status_tool  # noqa: F401
import tools.pdf_download_tool  # noqa: F401
import tools.pdf_translate_tool  # noqa: F401

from benchmark.tasks import get_all_tasks
from rl.canary import CanaryCallback, CanaryEvaluator
from rl.grpo_reward import build_prompt_dataset, load_mock_env, synthesize_trajectory
from rl.observability import describe_logging, resolve_report_to
from rl.precision import precision_flags
from rl.reward import RewardCalculator
from rl.stage_verifier import StageVerifier

DEFAULT_SNAPSHOT = REPO_ROOT / "data" / "mock_arxiv_snapshot.json"


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


def main(
    model: str = "outputs/grpo/final",
    output_dir: str = "outputs/ppo",
    epochs: int = 1,
    batch_size: int = 4,
    mini_batch_size: int = 2,
    gradient_accumulation_steps: int = 1,
    lr: float = 1e-6,
    init_kl_coef: float = 0.05,
    max_completion_length: int = 256,
    temperature: float = 0.7,
    snapshot: Optional[str] = None,
    canary_steps: int = 50,
    min_canary_reward: float = -1.0,
    canary_patience: int = 3,
    verify: bool = True,
    report_to: str = "none",
    run_name: Optional[str] = None,
):
    backends = resolve_report_to(report_to)
    logging_dir = str(REPO_ROOT / output_dir / "logs")

    model_path = REPO_ROOT / model
    if model_path.exists():
        resolved = str(model_path)
    elif "/" in model and not model.startswith(("outputs/", "./", "/")):
        resolved = model
    else:
        # 回退到 DPO 或 SFT 模型
        fallback_paths = [
            REPO_ROOT / "outputs" / "dpo" / "final",
            REPO_ROOT / "outputs" / "sft" / "final",
        ]
        resolved = None
        for fb in fallback_paths:
            if fb.exists():
                resolved = str(fb)
                print(f"ℹ️ 未找到指定模型 {model_path}，自动回退至上一阶段: {resolved}")
                break
        if not resolved:
            raise SystemExit(
                f"❌ 未找到本地模型 {model_path}\n"
                f"请先运行 train_sft / train_dpo / train_grpo，或用 --model 指定检查点"
            )

    print(f"📦 加载 PPO Policy & Value Head 模型: {resolved}")
    tokenizer = AutoTokenizer.from_pretrained(resolved)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLMWithValueHead.from_pretrained(resolved)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(resolved)

    # 数据集
    tasks = get_all_tasks()
    tasks_by_id = {t["id"]: t for t in tasks}
    rows = build_prompt_dataset(tasks)
    train_dataset = Dataset.from_list(rows)
    print(f"📚 任务数: {len(tasks)}（PPO 在线采样）")

    # 快照环境
    snapshot_path = Path(snapshot) if snapshot else DEFAULT_SNAPSHOT
    env = load_mock_env(snapshot_path)
    print(f"🗂 离线快照环境: {snapshot_path}")

    reward_calc = RewardCalculator()

    config = PPOConfig(
        model_name=resolved,
        learning_rate=lr,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        init_kl_coef=init_kl_coef,
        log_with=backends[0] if backends else None,
        project_kwargs={"logging_dir": logging_dir} if backends else None,
    )

    print(describe_logging(backends, logging_dir if backends else None))
    print(f"🚀 开始 PPO 训练 (batch_size={batch_size}, mini_batch_size={mini_batch_size})...")

    trainer = PPOTrainer(
        config=config,
        model=policy,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=collator,
    )

    generation_kwargs = {
        "max_new_tokens": max_completion_length,
        "temperature": temperature,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
    }

    device = trainer.accelerator.device

    for epoch in range(epochs):
        print(f"\n🔄 Epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(trainer.dataloader):
            query_tensors = []
            prompt_texts = []
            task_ids = batch.get("task_id", [])

            for prompt in batch["prompt"]:
                if isinstance(prompt, list):
                    text = tokenizer.apply_chat_template(
                        prompt, tokenize=False, add_generation_prompt=True
                    )
                else:
                    text = str(prompt)
                prompt_texts.append(text)
                ids = tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)
                query_tensors.append(ids.to(device))

            # 采样生成响应
            response_tensors = []
            for query in query_tensors:
                response = trainer.generate(query, **generation_kwargs)
                response_tensors.append(response.squeeze(0)[len(query):])

            # 解码并计算可验证奖励
            rewards = []
            for resp_tensor, tid in zip(response_tensors, task_ids):
                resp_text = tokenizer.decode(resp_tensor, skip_special_tokens=True)
                task_def = tasks_by_id.get(tid)
                if task_def is None:
                    rewards.append(torch.tensor(0.0, device=device))
                    continue

                result = synthesize_trajectory(resp_text, env=env)
                breakdown, _ = reward_calc.compute_reward_breakdown(task_def, result)
                reward_val = float(breakdown.total)
                rewards.append(torch.tensor(reward_val, device=device))

            # PPO 更新步骤
            stats = trainer.step(query_tensors, response_tensors, rewards)
            mean_r = torch.stack(rewards).mean().item()
            print(f"   Step {step + 1}: mean_reward={mean_r:.4f}, kl={stats.get('objective/kl', 0):.4f}")

    final_dir = REPO_ROOT / output_dir / "final"
    trainer.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\n✅ PPO 训练完成，模型已保存: {final_dir}")

    if verify:
        print(f"\n🔍 运行 PPO 阶段验证...")
        verifier = StageVerifier(grpo_min_reward=-0.2)
        report = verifier.verify_grpo(
            model_path=str(final_dir),
            tokenizer=tokenizer,
            env=env,
        )
        verifier.save_report(report, final_dir)
        print(report.summary())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="PPO 强化学习训练")
    p.add_argument("--model", default="outputs/grpo/final", help="基座检查点路径")
    p.add_argument("--output_dir", default="outputs/ppo", help="模型输出目录")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--mini_batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--init_kl_coef", type=float, default=0.05)
    p.add_argument("--max_completion_length", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--snapshot", default=None)
    p.add_argument("--canary_steps", type=int, default=50)
    p.add_argument("--min_canary_reward", type=float, default=-1.0)
    p.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--report_to", default="none")
    p.add_argument("--run_name", default=None)
    args = p.parse_args()
    main(**vars(args))
