"""OPD（On-Policy Distillation）训练脚本（使用 TRL GKDTrainer）

OPD 是一个**完整的训练范式**而非单一 trick：学生模型自己采样动作（on-policy），
教师模型对每个 token 给出 logprob，损失取二者的 reverse-KL
D_KL(π_student ‖ π_teacher)（mode-seeking：学生分布收敛到教师在学生自己的
高概率动作上）。它不需要奖励、不需要 value model，是 SFT（off-policy 蒸馏）
的 on-policy 升级，也是 GRPO 的互替路径——有强教师时用蒸馏省掉 RL 的
探索成本；没有教师时仍应走 GRPO（verifiable reward 路线）。

本实现支持两种模式：
- **单轮（默认）**：保留 TRL GKDTrainer 的原始行为，便于复现实验。
- **多轮（--max_turns > 1）**：学生执行 Action → Observation → Action，
  Observation 仅作为上下文且 loss mask 为 0，reverse-KL 只计算学生生成 token。
  每条轨迹使用独立 MockArxivEnv，避免工具和会话状态串扰。
- **教师需本地 logprob**：外部 API 教师拿不到逐 token logprob，因此
  --teacher 必须是本地 HF 模型或 HF 仓库名。上限是教师：蒸馏不会超过
  教师在本任务上的水平。
- **beta=1.0 即 reverse-KL**：TRL `generalized_jsd_loss` 里 beta=1 走
  `kl_div(teacher_log_probs, student_log_probs)` = D_KL(π_s‖π_t)
  （beta=0 是反方向的 D_KL(π_t‖π_s)，mass-covering，不是 OPD 要的）。
  tests/test_opd.py 有数值单测锁死这个方向。
- 已在 trl 1.5.1（`trl.experimental.gkd`）上验证；更老的 TRL 顶层也有
  GKDTrainer，导入做了兼容，但数据列格式以 1.x 的 DataCollatorForChatML
  为准，过老的版本会在启动时报错。

使用方式：
    python -m AgenticArxiv.rl.train_opd --model outputs/sft/final
    python -m AgenticArxiv.rl.train_opd --model outputs/sft/final --teacher Qwen/Qwen2.5-7B-Instruct
    python -m AgenticArxiv.rl.train_opd --max_turns 4 --snapshot data/mock_arxiv_snapshot.json
    python -m AgenticArxiv.rl.train_opd --report_to tensorboard
"""

import argparse
import dataclasses
import inspect
import os
import sys
from pathlib import Path
from typing import Optional

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent

# 添加 AgenticArxiv 到 Python 路径
sys.path.insert(0, str(PACKAGE_ROOT))

# Canary / 阶段验证要执行工具打分，会话状态走内存，不依赖数据库
os.environ.setdefault("STORE_BACKEND", "memory")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import tools.arxiv_tool  # noqa: F401  触发工具注册（canary / 阶段验证打分要用）
import tools.cache_status_tool  # noqa: F401
import tools.pdf_download_tool  # noqa: F401
import tools.pdf_translate_tool  # noqa: F401

from benchmark.tasks import get_all_tasks
from benchmark.tasks_expanded import get_expanded_tasks
from rl.canary import CanaryCallback, CanaryEvaluator
from rl.grpo_reward import build_prompt_dataset
from rl.observability import describe_logging, resolve_report_to
from rl.opd_multiturn import (
    MultiTurnOPDCollator,
    make_multiturn_gkd_trainer,
    validate_tokenizer_compatibility,
)
from rl.reward import RewardCalculator
from rl.stage_verifier import StageVerifier

DEFAULT_SNAPSHOT = REPO_ROOT / "data" / "mock_arxiv_snapshot.json"
DEFAULT_TEACHER = "Qwen/Qwen2.5-7B-Instruct"

# TRL generalized_jsd_loss 的方向开关：beta=1 ⇒ D_KL(π_student‖π_teacher)。
# 这是 OPD 的 reverse-KL（mode-seeking）；beta=0 是反方向，别用。
REVERSE_KL_BETA = 1.0


def _precision_flags():
    """见 rl/precision.py：CUDA 上优先 bf16，退回 fp16；CPU / MPS 不开混合精度。"""
    from rl.precision import precision_flags
    return precision_flags()


def _resolve_model_path(model: str) -> str:
    """本地路径优先；形如 org/name 的当作 HF 仓库（与 train_grpo 同一套约定）。"""
    model_path = REPO_ROOT / model
    if model_path.exists():
        return str(model_path)
    if "/" in model and not model.startswith(("outputs/", "./", "/")):
        return model
    raise SystemExit(
        f"❌ 未找到本地模型 {model_path}\n"
        f"请先运行 train_sft，或用 --model / --teacher 指定其它检查点 / HF 仓库名"
    )


def _import_gkd():
    """GKD 在 TRL 里搬过家：1.x 在 trl.experimental.gkd，更早的版本在顶层。

    导入失败要在这里响亮地报错，而不是等训练跑完才发现根本没有蒸馏。
    """
    try:
        from trl.experimental.gkd import GKDConfig, GKDTrainer
        return GKDConfig, GKDTrainer
    except ImportError:
        pass
    try:
        from trl import GKDConfig, GKDTrainer  # type: ignore
        return GKDConfig, GKDTrainer
    except ImportError:
        pass
    try:
        from importlib.metadata import version
        installed = version("trl")
    except Exception:  # noqa: BLE001
        installed = "未知"
    raise SystemExit(
        f"❌ 当前 TRL（{installed}）找不到 GKDTrainer / GKDConfig。\n"
        "   OPD 依赖 TRL 自带的 on-policy 蒸馏实现：\n"
        "     - trl >= 1.0: trl.experimental.gkd\n"
        "     - 更早版本: trl 顶层\n"
        "   升级: pip install -U 'trl>=1.0'"
    )


def _filter_cfg_kwargs(cfg_kwargs: dict, valid_names: set) -> tuple:
    """按当前 TRL 实际存在的字段过滤配置。

    GKDConfig 各版本字段有增删（例如 1.x 移除了 sft_alpha），按字段过滤
    避免因为一个参数名不存在就整个训练起不来。返回 (过滤后, 被丢弃的键)。
    """
    dropped = sorted(k for k in cfg_kwargs if k not in valid_names)
    return {k: v for k, v in cfg_kwargs.items() if k in valid_names}, dropped


def _load_tasks(task_set: str):
    """选择 OPD 训练任务集（无奖励信号，也就没有 GRPO 那套组内方差 / 切分问题）。

    默认仍是 benchmark/tasks.py 的冒烟任务；完整任务集显式 --task_set expanded。
    """
    pool = get_expanded_tasks() if task_set == "expanded" else get_all_tasks()
    if task_set != "expanded":
        print(
            f"⚠️  正在用 benchmark/tasks.py 的 {len(pool)} 条冒烟任务训练。"
            "完整任务集用 --task_set expanded。"
        )
    return pool


def build_opd_dataset(tasks, tokenizer) -> tuple:
    """把 ReAct prompt 渲染成 GKD 需要的列。返回 (Dataset, 最大 prompt token 数)。

    GKDTrainer(1.x) 默认挂 DataCollatorForChatML：带 input_ids 的行走
    「已 tokenize」分支，`prompts` 张量按 prompt 列重新 tokenize 得到。
    这里自己调 apply_chat_template(add_generation_prompt=True)，与 GRPO
    多轮 rollout_func 的渲染完全一致，保证 OPD / GRPO / 推理三方输入分布相同。

    没有 completion 列是刻意的：lmbda=1.0 时 GKD 每步都用学生采样替换
    input_ids/labels，completion 列根本用不上。
    """
    rows = []
    max_prompt_tokens = 0
    for row in build_prompt_dataset(tasks):
        messages = row["prompt"]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"❌ tokenizer 渲染 chat template 失败：{exc}\n"
                "   OPD 的学生通常是 SFT 产物，tokenizer 应带 chat template；"
                "请检查 --model 指向的检查点。"
            )
        ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
        max_prompt_tokens = max(max_prompt_tokens, len(ids))
        rows.append({"prompt": text, "input_ids": ids, "task_id": row["task_id"]})
    return Dataset.from_list(rows), max_prompt_tokens


def _teacher_load_kwargs() -> dict:
    """教师模型的加载精度。

    教师是冻结的 eval 模块（全程 no_grad），CUDA 上跟随混合精度策略用
    bf16 权重省显存；fp16 / CPU / MPS 场景退回 fp32 —— fp16 权重推理
    容易溢出，教师省这点显存不值得冒这个险。
    """
    if not _precision_flags().get("bf16"):
        return {}
    loader = AutoModelForCausalLM.from_pretrained
    name = "dtype" if "dtype" in inspect.signature(loader).parameters else "torch_dtype"
    return {name: torch.bfloat16}


def _gold_action_tokens(tokenizer, tasks) -> int:
    """标准动作渲染成 ReAct 文本后的最大 token 数（与 train_grpo 同一套体检）。

    max_new_tokens 设小了模型永远吐不出完整动作，蒸馏信号从第一步起就是
    截断的残句 —— 这类失败在 loss 曲线上毫无异常，必须在启动时拦下。
    """
    import json as _json
    lengths = [0]
    for task in tasks:
        for name in task.get("expected_tools", []) or []:
            text = f'Thought: xxx\nAction: {_json.dumps({"name": name, "args": {}}, ensure_ascii=False)}'
            lengths.append(len(tokenizer(text)["input_ids"]))
    return max(lengths)


def _model_context_limit(model) -> Optional[int]:
    """Return a usable decoder context limit, ignoring tokenizer sentinel values."""
    config = getattr(model, "config", None)
    candidates = []
    for name in ("max_position_embeddings", "n_positions", "max_sequence_length"):
        value = getattr(config, name, None)
        if isinstance(value, int) and 0 < value < 10_000_000:
            candidates.append(value)
    return min(candidates) if candidates else None


def main(
    model: str = "outputs/sft/final",
    teacher: str = DEFAULT_TEACHER,
    output_dir: str = "outputs/opd",
    epochs: int = 1,
    max_steps: int = -1,
    batch_size: int = 2,
    grad_accum: int = 1,
    lr: float = 1e-5,
    temperature: float = 0.9,
    lmbda: float = 1.0,
    beta: float = REVERSE_KL_BETA,
    max_new_tokens: int = 256,
    max_turns: int = 1,
    max_observation_tokens: int = 256,
    snapshot: str = None,
    task_set: str = "default",
    canary_steps: int = 50,
    min_canary_reward: float = -1.0,
    canary_patience: int = 3,
    verify: bool = True,
    report_to: str = "none",
    run_name: str = None,
):
    # 先校验日志后端再加载模型：参数写错时应立刻失败，而不是等模型加载完
    backends = resolve_report_to(report_to)
    logging_dir = str(REPO_ROOT / output_dir / "logs")

    student_path = _resolve_model_path(model)
    teacher_path = _resolve_model_path(teacher)
    GKDConfig, GKDTrainer = _import_gkd()

    if lmbda != 1.0:
        # lmbda<1 会偶发走离线分支，而本数据集没有 completion 列，
        # 标签全为 -100，loss 会除零变 nan —— 与其静默 nan，不如启动即拦。
        raise SystemExit(
            f"❌ --lmbda 必须为 1.0（当前 {lmbda}）。\n"
            "   本脚本的 OPD 数据集只含 prompt（蒸馏目标完全由教师逐 token 给出），\n"
            "   不存在可用的离线 completion；<1.0 时 GKD 会偶发走离线分支拿到\n"
            "   全 -100 的标签，loss 直接 nan。要混合离线数据请自行扩展数据集。"
        )

    print(f"📦 加载学生模型: {student_path}")
    tokenizer = AutoTokenizer.from_pretrained(student_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    student = AutoModelForCausalLM.from_pretrained(student_path)

    print(f"🎓 加载教师模型: {teacher_path}")
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_path)
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    if max_turns > 1:
        try:
            validate_tokenizer_compatibility(tokenizer, teacher_tokenizer)
        except ValueError as exc:
            raise SystemExit(f"❌ {exc}") from exc
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_path, **_teacher_load_kwargs())
    teacher_model.requires_grad_(False)  # 教师全程 eval + no_grad，双保险防意外更新
    print(f"   显存提示：学生与教师同驻显存；吃紧时换更小的教师（如 Qwen2.5-3B/1.5B-Instruct）")

    # --- 数据集：由任务集派生，无需预生成 ---
    tasks = _load_tasks(task_set)
    train_dataset, max_prompt_tokens = build_opd_dataset(tasks, tokenizer)
    print(f"📚 任务数: {len(tasks)}（OPD 为在线算法，prompt 即数据，无需预生成轨迹）")

    # --- 生成长度体检 ---
    need = _gold_action_tokens(tokenizer, tasks)
    if max_new_tokens < need:
        raise SystemExit(
            f"❌ max_new_tokens={max_new_tokens} 小于标准动作所需的 {need} tokens，"
            f"采样永远截断，蒸馏信号从第一步起就是残句。\n"
            f"请改用 --max_new_tokens {need + 64}"
        )

    multiturn = max_turns > 1
    if max_turns < 1:
        raise SystemExit("❌ --max_turns 必须至少为 1")
    if max_observation_tokens < 0:
        raise SystemExit("❌ --max_observation_tokens 不能为负数")

    # Multi-turn mode needs room for every assistant turn and every inserted
    # observation.  A small fixed marker allowance covers ``Observation:`` and
    # the following ``Thought:`` prompt without tying this guard to one tokenizer.
    if multiturn:
        max_sequence_length = (
            max_prompt_tokens
            + max_turns * max_new_tokens
            + (max_turns - 1) * (max_observation_tokens + 32)
        )
        context_limits = [
            value
            for value in (
                _model_context_limit(student),
                _model_context_limit(teacher_model),
            )
            if value is not None
        ]
        context_limit = min(context_limits) if context_limits else None
        if context_limit is not None and max_sequence_length > context_limit:
            raise SystemExit(
                f"❌ 多轮 OPD 最坏需要 {max_sequence_length} tokens，但模型上下文仅 "
                f"{context_limit}。请降低 --max_turns / --max_new_tokens / "
                "--max_observation_tokens。"
            )
    else:
        max_sequence_length = max_prompt_tokens + max_new_tokens

    cfg_kwargs = {
        "output_dir": str(REPO_ROOT / output_dir),
        "num_train_epochs": epochs,
        "max_steps": max_steps,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "temperature": temperature,
        "lmbda": lmbda,
        "beta": beta,
        "max_new_tokens": max_new_tokens,
        # SFTConfig.max_length：ChatML collator 的截断预算，prompt + 生成余量
        "max_length": max_sequence_length,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": backends,
        "run_name": run_name or Path(output_dir).name,
        **_precision_flags(),
    }
    valid = {f.name for f in dataclasses.fields(GKDConfig)}
    if "beta" not in valid:
        raise SystemExit(
            "❌ 当前 TRL 的 GKDConfig 没有 beta 字段，无法选择 reverse-KL 方向，"
            "蒸馏方向不可控，拒绝启动。请升级 TRL。"
        )
    cfg, dropped = _filter_cfg_kwargs(cfg_kwargs, valid)
    if dropped:
        print(f"  提示：当前 TRL 不支持这些 GKDConfig 参数，已忽略 -> {dropped}")
    config = GKDConfig(**cfg)

    print(describe_logging(backends, logging_dir if backends else None))
    mode_description = f"多轮，max_turns={max_turns}" if multiturn else "单轮兼容模式"
    print(
        f"🚀 开始 OPD 训练（{mode_description}，lmbda={lmbda} 全 on-policy 采样，"
        f"beta={beta} → {'reverse-KL D_KL(学生‖教师)' if beta == 1.0 else 'JSD 插值'}）"
    )

    # Multi-turn rollout itself requires the deterministic environment even if
    # Canary and final verification are disabled.
    needs_env = multiturn or canary_steps > 0 or verify
    env = None
    environment_factory = None
    if needs_env:
        snapshot_path = Path(snapshot) if snapshot else DEFAULT_SNAPSHOT
        if not snapshot_path.exists():
            reason = "多轮 OPD / Canary / 阶段验证" if multiturn else "Canary / 阶段验证"
            raise SystemExit(
                f"❌ {reason}需要离线快照 {snapshot_path}\n"
                "   请先运行: python -m AgenticArxiv.rl.build_snapshot"
                "（单轮模式可用 --canary_steps 0 --no-verify 跳过）"
            )
        from rl.grpo_reward import load_mock_env
        env = load_mock_env(snapshot_path)
        if multiturn:
            from rl.multiturn_env import make_environment_factory
            environment_factory = make_environment_factory(snapshot_path)

    trainer_class = make_multiturn_gkd_trainer(GKDTrainer) if multiturn else GKDTrainer
    trainer_kwargs = {}
    if multiturn:
        trainer_kwargs.update({
            "data_collator": MultiTurnOPDCollator(tokenizer.pad_token_id),
            "environment_factory": environment_factory,
            "tasks_by_id": {task["id"]: task for task in tasks},
            "max_turns": max_turns,
            "max_observation_tokens": max_observation_tokens,
            "max_sequence_length": max_sequence_length,
        })
    trainer = trainer_class(
        model=student,
        teacher_model=teacher_model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        **trainer_kwargs,
    )

    # --- Canary 回调：每 N 步在固定任务上评估，检测学生退化 ---
    # OPD 本身无奖励信号，但产出模型仍要在环境里过关；快照只在 canary / 验证时需要
    canary_cb = None
    if canary_steps > 0:
        canary_evaluator = CanaryEvaluator(
            model=student,
            tokenizer=tokenizer,
            reward_calc=RewardCalculator(),
            env=env,
            num_generations=2,  # OPD 无组内对比，2 条采样足够给 canary 降噪
            max_new_tokens=max_new_tokens,
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

    trainer.train()

    if canary_cb is not None and canary_cb.tripped:
        raise SystemExit(1)

    final_dir = REPO_ROOT / output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"✅ OPD 训练完成，学生模型已保存: {final_dir}")

    if verify:
        print(f"\n🔍 运行 OPD 产出模型阶段验证（沿用 GRPO 阈值）...")
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
                f"\n❌ OPD 阶段验证失败，学生未达到最低质量阈值"
                f"（上限是教师，先确认教师自己能过这关）。\n"
                f"   检查 {final_dir / 'verification_report.json'} 了解详情。\n"
                f"   跳过验证用 --no-verify。"
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="OPD（on-policy 蒸馏）训练")
    p.add_argument("--model", default="outputs/sft/final", help="学生模型（通常为 SFT 产物）")
    p.add_argument(
        "--teacher", default=DEFAULT_TEACHER,
        help="教师模型：本地路径或 HF 仓库名（需能取逐 token logprob，不能用外部 API）",
    )
    p.add_argument("--output_dir", default="outputs/opd")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument(
        "--max_steps", type=int, default=-1,
        help="限制优化步数；-1 按 epochs 完整训练，可用 1 做 GPU 端到端烟雾验证",
    )
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=0.9, help="学生 on-policy 采样温度")
    p.add_argument(
        "--lmbda", type=float, default=1.0,
        help="on-policy 采样比例；本实现只支持 1.0（数据集无离线 completion）",
    )
    p.add_argument(
        "--beta", type=float, default=REVERSE_KL_BETA,
        help="GKD 损失插值：1.0=reverse-KL D_KL(学生‖教师)（OPD 标准，默认）；"
             "0.0=反方向；其余为 JSD 插值",
    )
    p.add_argument("--max_new_tokens", type=int, default=256, help="学生单步动作生成预算")
    p.add_argument(
        "--max_turns", type=int, default=1,
        help="每条 on-policy 轨迹的最大 ReAct 轮数；1 保留原单轮 OPD，>1 启用多轮 OPD",
    )
    p.add_argument(
        "--max_observation_tokens", type=int, default=256,
        help="每轮写回上下文的 Observation token 上限；这些 token 的 loss mask 恒为 0",
    )
    p.add_argument("--snapshot", default=None)
    p.add_argument(
        "--task_set", choices=["default", "expanded"], default="default",
        help="default=benchmark/tasks.py 的 8 条冒烟任务（保持现状）；expanded=完整基准集",
    )
    p.add_argument(
        "--canary_steps", type=int, default=50,
        help="每隔多少步运行 canary 评估（0 表示禁用）",
    )
    p.add_argument(
        "--min_canary_reward", type=float, default=-1.0,
        help="Canary 奖励下限。低于此值连续 patience 次则停止训练",
    )
    p.add_argument(
        "--canary_patience", type=int, default=3,
        help="Canary 奖励连续低于阈值多少次后停止训练",
    )
    p.add_argument(
        "--verify", action=argparse.BooleanOptionalAction, default=True,
        help="训练结束后运行阶段验证（默认开启）",
    )
    p.add_argument(
        "--report_to", default="none",
        help="训练曲线记到哪：none / auto / tensorboard / wandb（可逗号分隔）",
    )
    p.add_argument(
        "--run_name", default=None,
        help="本次运行在 TensorBoard / wandb 里的名字，默认取 output_dir 末段",
    )
    main(**vars(p.parse_args()))
