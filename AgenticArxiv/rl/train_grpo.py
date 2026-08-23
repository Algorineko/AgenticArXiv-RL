"""GRPO 训练脚本（使用 TRL GRPOTrainer）

GRPO（Group Relative Policy Optimization）：
- 目标：用 verifiable reward 在线训练
- 优势：无需 reward model、无需 value model（比 PPO 更轻量）
- 数据：**不需要预先生成**。GRPO 是在线算法，只要 prompt + 可验证奖励，
        数据集直接由 benchmark/tasks.py 派生

奖励定义见 rl/grpo_reward.py：TRL 原生执行多轮 tool-calling，把每轮环境
observation 插回上下文，并将完整轨迹交给 RewardCalculator 打分。

训练中每 N 步自动运行 canary 评估（在固定小任务集上验证模型未退化），
训练结束后可选运行阶段验证（检查模型是否达到最低质量阈值）。

使用方式：
    python -m AgenticArxiv.rl.train_grpo
    python -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --num_generations 8
    python -m AgenticArxiv.rl.train_grpo --canary_steps 20 --min_canary_reward -0.3
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
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

import tools.arxiv_tool  # noqa: F401  触发工具注册
import tools.cache_status_tool  # noqa: F401
import tools.pdf_download_tool  # noqa: F401
import tools.pdf_translate_tool  # noqa: F401

from benchmark.tasks import get_all_tasks
from rl.canary import CanaryEvaluator, CanaryCallback
from rl.grpo_reward import (
    build_prompt_dataset,
    load_mock_env,
    make_grpo_reward_fn,
    make_multiturn_rollout_func,
    parse_react_action,
)
from rl.multiturn_env import make_environment_factory
from rl.reward import RewardCalculator
from rl.stage_verifier import StageVerifier

DEFAULT_SNAPSHOT = REPO_ROOT / "data" / "mock_arxiv_snapshot.json"


def _precision_flags():
    """见 rl/precision.py：CUDA 上优先 bf16，退回 fp16；CPU / MPS 不开混合精度。"""
    from rl.precision import precision_flags
    return precision_flags()


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


class RewardVarianceGuard(TrainerCallback):
    """组内奖励方差为 0 时中止训练。

    GRPO 的梯度来自同一 prompt 采样出的若干条轨迹之间的奖励差异。
    若一组内所有样本拿到同一个奖励，优势归零，这一步不产生任何梯度。
    最常见的成因是基座模型还吐不出可解析的动作 —— 每条采样都落到解析
    失败的地板分，方差恒为 0。

    这种失败是静默的：训练照常跑完、loss 看着像在收敛（其实只是 KL 项）、
    checkpoint 照常保存，唯一的迹象是日志里 frac_reward_zero_std 恒为 1，
    而这一栏很容易被忽略。这里把它变成一次响亮的失败。
    """

    def __init__(self, patience: int = 5):
        self.patience = patience
        self.streak = 0
        self.tripped = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        frac = logs.get("frac_reward_zero_std")
        std = logs.get("reward_std")
        if frac is None and std is None:
            return
        dead = (frac is not None and float(frac) >= 1.0) or (
            frac is None and std is not None and float(std) == 0.0
        )
        if not dead:
            self.streak = 0
            return

        self.streak += 1
        if self.streak < self.patience:
            return

        self.tripped = True
        control.should_training_stop = True
        mean = logs.get("reward")
        print(
            f"\n❌ 连续 {self.streak} 步组内奖励方差为 0"
            + (f"（奖励恒为 {float(mean):.4f}）" if mean is not None else "")
            + "，优势全零，训练不产生任何梯度。\n"
            "   最常见原因：基座模型还产不出可解析的 ReAct 动作，每条采样都落到解析失败的地板分。\n"
            "   处理办法：\n"
            "     1) 先跑 SFT 让模型学会输出格式，再用 --model outputs/sft/final 起 GRPO\n"
            "     2) 换更强的基座模型\n"
            "     3) 若确认是任务全在难度谱两端（全对或全错），换一批成功率居中的任务\n"
            "   确实想继续跑，用 --allow_zero_variance 跳过本检查。"
        )


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
    max_turns: int = 4,
    temperature: float = 1.0,
    snapshot: str = None,
    allow_zero_variance: bool = False,
    canary_steps: int = 50,
    min_canary_reward: float = -1.0,
    canary_patience: int = 3,
    verify: bool = True,
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
    if not snapshot_path.exists():
        raise SystemExit(
            f"❌ 多轮 GRPO 需要离线快照 {snapshot_path}\n"
            "   请先运行: python -m AgenticArxiv.rl.build_snapshot"
        )
    env = load_mock_env(snapshot_path)  # canary / verifier 使用
    environment_factory = make_environment_factory(snapshot_path)
    print(f"🗂  使用独立离线环境执行多轮工具调用: {snapshot_path}")
    # structured completions 已携带真实 tool observations，不在 reward 端重复执行。
    reward_fn = make_grpo_reward_fn({t["id"]: t for t in tasks}, env=None)
    rollout_func = make_multiturn_rollout_func(environment_factory, max_turns=max_turns)

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
        rollout_func=rollout_func,
    )

    # --- Canary 回调：每 N 步在固定任务上评估，检测性能退化 ---
    canary_cb = None
    if canary_steps > 0:
        canary_evaluator = CanaryEvaluator(
            model=policy,
            tokenizer=tokenizer,
            reward_calc=RewardCalculator(),
            env=env,
            num_generations=num_generations,
            max_new_tokens=max_completion_length,
            temperature=temperature,
        )
        canary_cb = CanaryCallback(
            evaluator=canary_evaluator,
            steps=canary_steps,
            min_reward=min_canary_reward,
            patience=canary_patience,
        )
        trainer.add_callback(canary_cb)
        print(f"🐤 Canary: 每 {canary_steps} 步评估，阈值={min_canary_reward}，patience={canary_patience}")

    guard = None
    if not allow_zero_variance:
        guard = RewardVarianceGuard()
        trainer.add_callback(guard)
    trainer.train()

    if guard is not None and guard.tripped:
        raise SystemExit(1)
    if canary_cb is not None and canary_cb.tripped:
        raise SystemExit(1)

    final_dir = REPO_ROOT / output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"✅ GRPO 训练完成，模型已保存: {final_dir}")

    # --- 阶段验证：训练后检查模型是否达到最低质量阈值 ---
    if verify:
        print(f"\n🔍 运行 GRPO 阶段验证...")
        verifier = StageVerifier()
        report = verifier.verify_grpo(
            model_path=str(final_dir),
            tokenizer=tokenizer,
            env=env,
        )
        verifier.save_report(report, final_dir)
        print(report.summary())
        if not report.passed:
            raise SystemExit(
                f"\n❌ GRPO 阶段验证失败，模型未达到最低质量阈值。\n"
                f"   检查 {final_dir / 'verification_report.json'} 了解详情。\n"
                f"   跳过验证用 --no-verify。"
            )


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
    p.add_argument(
        "--allow_zero_variance", action="store_true",
        help="跳过「组内奖励方差为 0」检查。默认开启该检查：方差为 0 时"
             "GRPO 不产生任何梯度，训练会静默空转",
    )
    p.add_argument("--max_completion_length", type=int, default=256)
    p.add_argument(
        "--max_turns", type=int, default=4,
        help="每条 rollout 最多执行多少轮工具调用；环境 observation 不计入策略 loss",
    )
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--snapshot", default=None)
    p.add_argument(
        "--canary_steps", type=int, default=50,
        help="每隔多少步运行 canary 评估（0 表示禁用）",
    )
    p.add_argument(
        "--min_canary_reward", type=float, default=-1.0,
        help="Canary 奖励下限。低于此值连续 patience 次则停止训练。"
             "默认 -1.0（理论最低分）表示禁用检查",
    )
    p.add_argument(
        "--canary_patience", type=int, default=3,
        help="Canary 奖励连续低于阈值多少次后停止训练",
    )
    p.add_argument(
        "--verify", action=argparse.BooleanOptionalAction, default=True,
        help="训练结束后运行阶段验证（默认开启）",
    )
    main(**vars(p.parse_args()))
