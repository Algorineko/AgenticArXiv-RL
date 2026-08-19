"""GRPO 训练脚本（使用 TRL GRPOTrainer）

GRPO（Group Relative Policy Optimization）：
- 目标：用 verifiable reward 在线训练
- 优势：无需 reward model、无需 value model（比 PPO 更轻量）
- 数据：**不需要预先生成**。GRPO 是在线算法，只要 prompt + 可验证奖励，
        数据集直接由 benchmark/tasks.py 派生

奖励定义见 rl/grpo_reward.py：把模型生成的一步补成最小完整轨迹，
再交给 rl/reward.py 已有的 RewardCalculator 打分，不引入第二套标准。

使用方式：
    python -m AgenticArxiv.rl.train_grpo
    python -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --num_generations 8
"""

import argparse
import dataclasses
import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent

# 添加 AgenticArxiv 到 Python 路径
sys.path.insert(0, str(PACKAGE_ROOT))

# 奖励计算要执行工具，会话状态走内存，不依赖数据库
os.environ.setdefault("STORE_BACKEND", "memory")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

import tools.arxiv_tool  # noqa: F401  触发工具注册
import tools.cache_status_tool  # noqa: F401
import tools.pdf_download_tool  # noqa: F401
import tools.pdf_translate_tool  # noqa: F401

from benchmark.tasks import get_all_tasks
from rl.grpo_reward import (
    build_prompt_dataset,
    load_mock_env,
    make_grpo_reward_fn,
    parse_react_action,
)

DEFAULT_SNAPSHOT = REPO_ROOT / "data" / "mock_arxiv_snapshot.json"


def _precision_flags():
    """只有 CUDA 才开 bf16/fp16；CPU / Apple MPS 上开混合精度会训练失败。"""
    import torch
    if torch.cuda.is_available():
        return {"bf16": torch.cuda.is_bf16_supported(), "fp16": not torch.cuda.is_bf16_supported()}
    return {}


def _gold_completion_tokens(tokenizer, tasks) -> int:
    """标准动作渲染成 ReAct 文本后的最大 token 数。

    用于校验 max_completion_length：设小了模型永远吐不出完整动作，
    奖励恒为下限、组内方差为 0、梯度为 0，而日志上只表现为
    completions/clipped_ratio=1，很容易被误读成「模型太差」。
    """
    import json as _json
    lengths = [0]
    for task in tasks:
        for name in task.get("expected_tools", []) or []:
            text = f'Thought: xxx\nAction: {_json.dumps({"name": name, "args": {}}, ensure_ascii=False)}'
            lengths.append(len(tokenizer(text)["input_ids"]))
    return max(lengths)


def main(
    model: str = "outputs/dpo/final",
    output_dir: str = "outputs/grpo",
    epochs: int = 1,
    batch_size: int = 4,
    grad_accum: int = 1,
    lr: float = 1e-5,
    beta: float = 0.04,
    num_generations: int = 4,
    max_completion_length: int = 256,
    temperature: float = 1.0,
    snapshot: str = None,
):
    model_path = REPO_ROOT / model
    if model_path.exists():
        resolved = str(model_path)
    elif "/" in model and not model.startswith(("outputs/", "./", "/")):
        resolved = model                    # 形如 org/name 的 HF 仓库
    else:
        raise SystemExit(
            f"❌ 未找到本地模型 {model_path}\n"
            f"请先运行 train_sft / train_dpo，或用 --model 指定其它检查点"
        )

    print(f"📦 加载模型: {resolved}")
    tokenizer = AutoTokenizer.from_pretrained(resolved)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(resolved)

    # --- 数据集：直接由任务集派生，无需预生成 ---
    tasks = get_all_tasks()
    rows = build_prompt_dataset(tasks)
    train_dataset = Dataset.from_list(rows)
    print(f"📚 任务数: {len(tasks)}（GRPO 为在线算法，无需预生成轨迹数据）")

    # --- 生成长度守卫 ---
    need = _gold_completion_tokens(tokenizer, tasks)
    if max_completion_length < need:
        raise SystemExit(
            f"❌ max_completion_length={max_completion_length} 小于标准动作所需的 {need} tokens，"
            f"模型不可能生成完整动作，奖励会恒为下限、梯度为 0。\n"
            f"请改用 --max_completion_length {need + 64}"
        )

    # --- 奖励函数 ---
    snapshot_path = Path(snapshot) if snapshot else DEFAULT_SNAPSHOT
    env = load_mock_env(snapshot_path)
    if env is None:
        print(f"⚠️  未找到快照 {snapshot_path}，奖励将不执行工具"
              f"（仅由格式/工具名/参数决定）。生成快照: python -m AgenticArxiv.rl.build_snapshot")
    else:
        print(f"🗂  使用离线快照执行工具: {snapshot_path}")
    reward_fn = make_grpo_reward_fn({t["id"]: t for t in tasks}, env=env)

    # GRPO 要求生成批量能被 num_generations 整除
    if batch_size % num_generations != 0:
        batch_size = num_generations
        print(f"  调整 per_device_train_batch_size = {batch_size}（需被 num_generations 整除）")

    cfg_kwargs = {
        "output_dir": str(REPO_ROOT / output_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "beta": beta,
        "num_generations": num_generations,
        "max_completion_length": max_completion_length,
        "temperature": temperature,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": [],
        **_precision_flags(),
    }
    # GRPOConfig 的字段在 TRL 各版本间有增删，按实际安装版本过滤，
    # 避免因为一个参数名不存在就整个训练起不来
    valid = {f.name for f in dataclasses.fields(GRPOConfig)}
    dropped = sorted(k for k in cfg_kwargs if k not in valid)
    if dropped:
        print(f"  提示：当前 TRL 不支持这些 GRPOConfig 参数，已忽略 -> {dropped}")
    config = GRPOConfig(**{k: v for k, v in cfg_kwargs.items() if k in valid})

    print(f"🚀 开始 GRPO 训练（每个 prompt 采样 {num_generations} 条，规则奖励）")
    trainer = GRPOTrainer(
        model=policy,
        reward_funcs=reward_fn,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    final_dir = REPO_ROOT / output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"✅ GRPO 训练完成，模型已保存: {final_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GRPO 训练")
    p.add_argument("--model", default="outputs/dpo/final")
    p.add_argument("--output_dir", default="outputs/grpo")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--max_completion_length", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--snapshot", default=None)
    a = p.parse_args()
    main(a.model, a.output_dir, a.epochs, a.batch_size, a.grad_accum, a.lr, a.beta,
         a.num_generations, a.max_completion_length, a.temperature, a.snapshot)
