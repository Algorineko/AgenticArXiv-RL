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
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent

# 添加 AgenticArxiv 到 Python 路径
sys.path.insert(0, str(PACKAGE_ROOT))

# 奖励计算要执行工具，会话状态走内存，不依赖数据库
os.environ.setdefault("STORE_BACKEND", "memory")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from rl import trl_compat  # noqa: F401  (torch<2.6 时兜底 FSDPModule，须在 trl 之前)
from trl import GRPOConfig, GRPOTrainer

import tools.arxiv_tool  # noqa: F401  触发工具注册
import tools.cache_status_tool  # noqa: F401
import tools.pdf_download_tool  # noqa: F401
import tools.pdf_translate_tool  # noqa: F401

from benchmark.tasks import get_all_tasks
from benchmark.splits import DEFAULT_SPLIT_PATH, load_split
from benchmark.semantic_oracle import attach_expected_paper_ids
from benchmark.tasks_expanded import get_expanded_tasks
from rl.canary import CanaryEvaluator, CanaryCallback
from rl.grpo_reward import (
    SUPPORTED_TOOL_NAMES,
    build_prompt_dataset,
    load_mock_env,
    make_grpo_reward_fn,
    make_multiturn_rollout_func,
    parse_react_action,
)
from rl.multiturn_env import make_environment_factory
from rl.rollout_audit import RolloutAuditWriter
from rl.observability import (
    RewardComponentTracker,
    describe_logging,
    resolve_report_to,
)
from rl.reward import RewardCalculator
from rl.stage_verifier import StageVerifier
from rl.train_sft import (
    _assert_lora_only_trainable,
    _build_qlora_configs,
    _qlora_runtime_guard,
    _trainable_parameter_stats,
)

DEFAULT_SNAPSHOT = REPO_ROOT / "data" / "mock_arxiv_snapshot.json"


def _precision_flags():
    """见 rl/precision.py：CUDA 上优先 bf16，退回 fp16；CPU / MPS 不开混合精度。"""
    from rl.precision import precision_flags
    return precision_flags()


def _build_reward_calculator(curriculum_steps: int) -> RewardCalculator:
    """构造本次运行唯一的奖励配置，并拒绝含糊的负步数。"""
    if curriculum_steps < 0:
        raise SystemExit("❌ reward_curriculum_steps 不能为负数")
    return RewardCalculator(curriculum_steps=curriculum_steps)


def _load_tasks(task_set: str, split: Optional[str]):
    """选择训练任务集。

    默认仍是 benchmark/tasks.py 的 8 条冒烟任务 —— 换默认值会让此前所有
    训练曲线不可比，所以要用完整任务集必须显式 --task_set expanded。

    `--split rl_train` 取的是 train 中成功率处于中间带的那部分：GRPO 的
    优势是组内相对的，成功率贴近 0 或 1 的任务，同一 prompt 采样出的轨迹
    奖励一致，组内方差为零、优势为零、不产生任何梯度。把它们放进训练集
    是纯粹的空转，还会把 frac_reward_zero_std 顶到 1 触发方差守卫。
    """
    pool = get_expanded_tasks() if task_set == "expanded" else get_all_tasks()
    if not split:
        if task_set != "expanded":
            print(
                f"⚠️  正在用 benchmark/tasks.py 的 {len(pool)} 条冒烟任务训练。"
                "完整任务集用 --task_set expanded；\n"
                "    只训练有梯度的那部分用 --task_set expanded "
                f"--split rl_train（{DEFAULT_SPLIT_PATH.name}）"
            )
        return pool

    wanted = set(load_split(split))
    by_id = {t["id"]: t for t in pool}
    chosen = [by_id[tid] for tid in sorted(wanted) if tid in by_id]
    missing = wanted - set(by_id)
    if missing:
        # 切分按完整任务集划定，缺任务说明任务池选错了，
        # 此时训练集会静默变小，训出来的东西与切分不对应。
        raise SystemExit(
            f"❌ 切分 '{split}' 里有 {len(missing)} 条任务不在当前任务集中，"
            f"例如 {sorted(missing)[:3]}\n"
            "   切分按 --task_set expanded 的完整任务集划定"
        )
    print(f"📑 使用切分 {split}（{len(chosen)} 条）")
    return chosen


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


def _global_step(state) -> int:
    """当前步数，取不到就按 0（视作还在开局宽限期内）。

    宽限期只用来区分「开局就没梯度」和「练到没梯度」，判错的代价是
    多报一次故障；为它让训练崩掉才是更坏的结果。
    """
    try:
        return int(getattr(state, "global_step", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _adapter_base_model(adapter_path: Path, override: Optional[str] = None) -> str:
    """Resolve the immutable base model behind an SFT LoRA adapter."""
    if override:
        base = Path(override)
        return str(base.resolve()) if base.exists() else override
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        return ""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    value = str(config.get("base_model_name_or_path") or "")
    if not value:
        raise SystemExit(f"❌ adapter_config 未声明 base_model_name_or_path: {config_path}")
    base = Path(value)
    if base.exists():
        return str(base.resolve())
    if "/" in value and not value.startswith(("outputs/", "./", "/")):
        return value
    raise SystemExit(
        f"❌ SFT adapter 引用的基座模型不存在: {value}\n"
        "   adapter 换机器后绝对路径可能失效，请显式传 --base_model"
    )


def _load_policy(
    model_path: str,
    *,
    qlora: bool,
    base_model: Optional[str],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
):
    """Load either an existing SFT adapter or a fresh policy for GRPO."""
    import torch

    local = Path(model_path)
    is_adapter = local.is_dir() and (local / "adapter_config.json").exists()
    if not qlora:
        print("⚠️  --no-qlora：加载完整精度策略，不属于单张4090冻结方案")
        policy = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=local.exists()
        )
        policy.config.use_cache = False
        return policy

    _qlora_runtime_guard()
    quantization, fresh_lora = _build_qlora_configs(
        lora_r, lora_alpha, lora_dropout
    )
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

    source = _adapter_base_model(local, base_model) if is_adapter else (
        str(Path(base_model).resolve()) if base_model and Path(base_model).exists()
        else base_model or model_path
    )
    source_path = Path(source)
    print(
        "🪶 GRPO QLoRA: 4-bit NF4 + double quant + BF16 compute\n"
        f"   base={source}\n"
        f"   adapter={model_path if is_adapter else '(new)'}"
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = AutoModelForCausalLM.from_pretrained(
        source,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=source_path.exists(),
    )
    base = prepare_model_for_kbit_training(
        base, use_gradient_checkpointing=False
    )
    policy = (
        PeftModel.from_pretrained(base, model_path, is_trainable=True)
        if is_adapter
        else get_peft_model(base, fresh_lora)
    )
    policy.config.use_cache = False
    trainable, total, ratio = _assert_lora_only_trainable(policy)
    print(
        f"🔧 参数统计: trainable={trainable:,}, model_parameters={total:,}, "
        f"trainable_ratio={ratio:.4f}%"
    )
    return policy


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

    def __init__(self, patience: int = 5, grace_steps: int = 20):
        self.patience = patience
        # 只有开局 grace_steps 步内的零方差才算故障。此后再出现，通常是策略
        # 已经在当前任务集上做满了 —— 尤其 --split rl_train 只有十来条中间带
        # 任务，模型学会之后每组都是满分，方差自然归零。那是「该换任务了」，
        # 不是「训练坏了」，把它当故障会白扔一个已经练好的 checkpoint。
        self.grace_steps = grace_steps
        self.streak = 0
        self.tripped = False
        self.converged = False
        self.stop_step = None

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

        control.should_training_stop = True
        mean = logs.get("reward")
        step = _global_step(state)
        self.stop_step = step
        if step > self.grace_steps:
            # 训练后段的连续零方差更像局部采样窗口已饱和，而非开局就没有
            # 可学习信号。正常停止并保留 checkpoint，但不要据此宣称整个任务集
            # 都已收敛：随机采样可能连续命中已经学会的容易任务。
            self.converged = True
            print(
                f"\n⏹  截至第 {step} 步已连续 {self.streak} 次记录到组内奖励方差为 0"
                + (f"（奖励恒为 {float(mean):.4f}）" if mean is not None else "")
                + "，当前采样窗口已饱和，提前停止并保存。\n"
                "   请先做固定评测确认是否真正收敛；若仍需继续训练，应重新划中间带"
                "（成功率会随训练漂移，旧的 rl_train 会逐渐变成 ceiling）或扩充任务集。"
            )
            return

        self.tripped = True
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
    max_steps: int = -1,
    batch_size: int = 4,
    grad_accum: int = 1,
    lr: float = 1e-5,
    beta: float = 0.04,
    reward_curriculum_steps: int = 30,
    num_generations: int = 4,
    max_completion_length: int = 256,
    max_turns: int = 4,
    temperature: float = 1.0,
    seed: int = 42,
    snapshot: str = None,
    task_set: str = "default",
    split: str = None,
    allow_zero_variance: bool = False,
    canary_steps: int = 50,
    min_canary_reward: float = -1.0,
    canary_patience: int = 3,
    verify: bool = True,
    report_to: str = "none",
    run_name: str = None,
    qlora: bool = True,
    base_model: str = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    gradient_checkpointing: bool = True,
    optim: str = "paged_adamw_8bit",
    save_steps: int = 0,
    save_total_limit: int = 2,
    resume_from_checkpoint: str = None,
    save_rollout_traces: bool = False,
    rollout_trace_path: str = None,
    rollout_trace_max_samples: int = 0,
):
    # 先校验日志后端再加载模型：参数写错时应立刻失败，而不是等模型加载完
    backends = resolve_report_to(report_to)
    if rollout_trace_max_samples < 0:
        raise SystemExit("❌ rollout_trace_max_samples 不能为负数")
    if save_steps < 0:
        raise SystemExit("❌ save_steps 不能为负数")
    if save_steps > 0 and save_total_limit < 1:
        raise SystemExit("❌ 启用周期 checkpoint 时 save_total_limit 必须至少为 1")
    resolved_resume_checkpoint = None
    if resume_from_checkpoint:
        resume_path = Path(resume_from_checkpoint)
        if not resume_path.is_absolute():
            resume_path = REPO_ROOT / resume_path
        if not resume_path.exists():
            raise SystemExit(f"❌ resume_from_checkpoint 不存在: {resume_path}")
        resolved_resume_checkpoint = str(resume_path)
    logging_dir = str(REPO_ROOT / output_dir / "logs")

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

    # 先验证任务与快照，再加载模型。语义标准答案损坏时应在占用显存前失败。
    tasks = _load_tasks(task_set, split)
    snapshot_path = Path(snapshot) if snapshot else DEFAULT_SNAPSHOT
    if not snapshot_path.exists():
        raise SystemExit(
            f"❌ 多轮 GRPO 需要离线快照 {snapshot_path}\n"
            "   请先运行: python -m AgenticArxiv.rl.build_snapshot"
        )
    tasks = attach_expected_paper_ids(tasks, snapshot_path)
    required_turns = max((len(t.get("expected_tools") or []) + 1 for t in tasks), default=1)
    if max_turns < required_turns:
        raise SystemExit(
            f"❌ max_turns={max_turns} 小于当前任务标准链所需的 {required_turns} "
            "（工具步骤 + FINISH），最长任务永远无法完成"
        )

    # --- 数据集：先验证身份元数据，再加载 tokenizer / 模型 ---
    # 数据或 datasets 版本不兼容时应在占用 GPU 前失败。
    rows = build_prompt_dataset(tasks)
    train_dataset = Dataset.from_list(rows)
    expected_hidden_ids = [row["task_id"] for row in rows]
    actual_hidden_ids = [
        row["prompt"][0].get("_task_id")
        for row in train_dataset
    ]
    if actual_hidden_ids != expected_hidden_ids:
        raise SystemExit(
            "❌ datasets 转换后隐藏 task_id 元数据丢失，"
            "不能安全区分相同文本、不同 setup 的 GRPO 任务"
        )
    # ``task_id`` also lives in each structured prompt as rollout-only metadata.
    # Keep this text lookup solely as a compatibility fallback for unique
    # prompts. Duplicate visible prompts are valid when their hidden setup is
    # different, so they must not be forced into a lossy text -> id mapping.
    prompt_task_ids = {}
    ambiguous_prompt_texts = set()
    for row in rows:
        prompt_text = row["prompt"][0]["content"]
        hidden_task_id = row["prompt"][0].get("_task_id")
        if hidden_task_id != row["task_id"]:
            raise SystemExit(
                "❌ GRPO prompt 缺少可信的隐藏 task_id 元数据: "
                f"row={row['task_id']}, hidden={hidden_task_id}"
            )
        if prompt_text in ambiguous_prompt_texts:
            continue
        previous = prompt_task_ids.get(prompt_text)
        if previous is not None and previous != row["task_id"]:
            prompt_task_ids.pop(prompt_text)
            ambiguous_prompt_texts.add(prompt_text)
        else:
            prompt_task_ids[prompt_text] = row["task_id"]
    if ambiguous_prompt_texts:
        print(
            f"🪪 检测到 {len(ambiguous_prompt_texts)} 个重复可见 prompt；"
            "使用不会进入模型 token 的隐藏 task_id 选择各自 setup"
        )
    print(f"📚 任务数: {len(tasks)}（GRPO 为在线算法，无需预生成轨迹数据）")

    print(f"📦 加载 tokenizer: {resolved}")
    tokenizer = AutoTokenizer.from_pretrained(
        resolved, local_files_only=Path(resolved).exists()
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy = _load_policy(
        resolved,
        qlora=qlora,
        base_model=base_model,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    # --- 生成长度守卫 ---
    need = _gold_completion_tokens(tokenizer, tasks)
    if max_completion_length < need:
        raise SystemExit(
            f"❌ max_completion_length={max_completion_length} 小于标准动作所需的 {need} tokens，"
            f"模型不可能生成完整动作，奖励会恒为下限、梯度为 0。\n"
            f"请改用 --max_completion_length {need + 64}"
        )
    env = load_mock_env(snapshot_path)  # canary / verifier 使用
    environment_factory = make_environment_factory(snapshot_path)
    print(f"🗂  使用独立离线环境执行多轮工具调用: {snapshot_path}")
    # structured completions 已携带真实 tool observations，不在 reward 端重复执行。
    tracker = RewardComponentTracker()
    reward_calc = _build_reward_calculator(reward_curriculum_steps)
    initial_weights = reward_calc.schedule(training_step=0)
    print(
        "⚖️  奖励课程: "
        f"curriculum_steps={reward_curriculum_steps}, initial_weights={initial_weights}"
    )
    auditor = None
    resolved_trace_path = None
    if save_rollout_traces or rollout_trace_path:
        resolved_trace_path = (
            Path(rollout_trace_path)
            if rollout_trace_path
            else REPO_ROOT / output_dir / "rollout_traces.jsonl"
        )
        if not resolved_trace_path.is_absolute():
            resolved_trace_path = REPO_ROOT / resolved_trace_path
        auditor = RolloutAuditWriter(
            resolved_trace_path,
            num_generations=num_generations,
            allowed_tools=SUPPORTED_TOOL_NAMES,
            max_samples=rollout_trace_max_samples,
        )
        limit_text = (
            str(rollout_trace_max_samples)
            if rollout_trace_max_samples
            else "不限"
        )
        print(
            f"🔎 原始 rollout 审计: {resolved_trace_path} "
            f"(保存上限={limit_text})"
        )
    tasks_by_id = {t["id"]: t for t in tasks}
    reward_fn = make_grpo_reward_fn(
        tasks_by_id,
        env=None,
        reward_calc=reward_calc,
        tracker=tracker,
        auditor=auditor,
    )
    rollout_func = make_multiturn_rollout_func(
        environment_factory,
        max_turns=max_turns,
        tasks_by_id=tasks_by_id,
        prompt_task_ids=prompt_task_ids,
    )

    # GRPO 要求生成批量能被 num_generations 整除
    if batch_size % num_generations != 0:
        batch_size = num_generations
        print(f"  调整 per_device_train_batch_size = {batch_size}（需被 num_generations 整除）")

    cfg_kwargs = {
        "output_dir": str(REPO_ROOT / output_dir),
        "num_train_epochs": epochs,
        "max_steps": max_steps,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "beta": beta,
        "num_generations": num_generations,
        "max_completion_length": max_completion_length,
        "temperature": temperature,
        "seed": seed,
        "data_seed": seed,
        "logging_steps": 1,
        "logging_first_step": True,
        "save_strategy": "steps" if save_steps > 0 else "no",
        "save_steps": save_steps if save_steps > 0 else 500,
        "save_total_limit": save_total_limit,
        "optim": optim if qlora else "adamw_torch_fused",
        "gradient_checkpointing": gradient_checkpointing,
        "gradient_checkpointing_kwargs": (
            {"use_reentrant": False} if gradient_checkpointing else None
        ),
        "report_to": backends,
        "run_name": run_name or Path(output_dir).name,
        **_precision_flags(),
    }
    # GRPOConfig 的字段在 TRL 各版本间有增删，按实际安装版本过滤，
    # 避免因为一个参数名不存在就整个训练起不来
    valid = {f.name for f in dataclasses.fields(GRPOConfig)}
    dropped = sorted(k for k in cfg_kwargs if k not in valid)
    if dropped:
        print(f"  提示：当前 TRL 不支持这些 GRPOConfig 参数，已忽略 -> {dropped}")
    config = GRPOConfig(**{k: v for k, v in cfg_kwargs.items() if k in valid})

    print(describe_logging(backends, logging_dir if backends else None))
    print(f"🚀 开始 GRPO 训练（每个 prompt 采样 {num_generations} 条，规则奖励）")
    trainer = GRPOTrainer(
        model=policy,
        reward_funcs=reward_fn,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        rollout_func=rollout_func,
    )
    # 必须在 trainer 建好之后接线。即使 report_to=none，TRL 仍会把 _metrics
    # 汇总到控制台；report_to 只控制是否转发到 TensorBoard/wandb，不应决定
    # 是否采集指标。此前的条件绑定还会让 _pending 在长训练中持续增长。
    tracker.bind(trainer)

    # --- Canary 回调：每 N 步在固定任务上评估，检测性能退化 ---
    canary_cb = None
    if canary_steps > 0:
        canary_evaluator = CanaryEvaluator(
            model=policy,
            tokenizer=tokenizer,
            # Canary 与训练必须使用同一奖励课程，否则监控值和训练目标不可比。
            reward_calc=reward_calc,
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
    try:
        train_result = trainer.train(
            resume_from_checkpoint=resolved_resume_checkpoint
        )
    except BaseException:
        # JSONL is flushed after every reward batch. Also persist its compact
        # summary on OOM, Ctrl-C, or trainer failure so the last healthy batch
        # remains diagnosable instead of disappearing with the failed run.
        if auditor is not None:
            failed_summary_path = auditor.save_summary(
                resolved_trace_path.with_suffix(".summary.json")
            )
            print(f"🔎 训练中断，已保留 rollout 审计: {failed_summary_path}")
        raise

    # 放在方差/canary 退出检查之前：即使保护器提前停止，也保留诊断数据。
    reward_summary_path = tracker.save_task_summary(
        REPO_ROOT / output_dir / "reward_probe_summary.json"
    )
    print(f"📊 逐任务奖励统计: {reward_summary_path}")
    rollout_audit_summary_path = None
    rollout_audit_summary = None
    if auditor is not None:
        rollout_audit_summary_path = auditor.save_summary(
            resolved_trace_path.with_suffix(".summary.json")
        )
        rollout_audit_summary = auditor.summary()
        print(
            f"🔎 rollout 审计摘要: {rollout_audit_summary_path} | "
            f"已保存={rollout_audit_summary['saved_samples']}, "
            f"异常样本={rollout_audit_summary['anomalous_samples']}"
        )

    if guard is not None and guard.tripped:
        raise SystemExit(1)
    if canary_cb is not None and canary_cb.tripped:
        raise SystemExit(1)

    final_dir = REPO_ROOT / output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    if qlora:
        required_adapter_files = (
            final_dir / "adapter_config.json",
            final_dir / "adapter_model.safetensors",
        )
        missing = [str(path) for path in required_adapter_files if not path.exists()]
        if missing:
            raise SystemExit(f"❌ GRPO完成但adapter文件缺失: {missing}")
    import torch
    actual_steps = int(getattr(trainer.state, "global_step", 0) or 0)
    stopped_early = max_steps > 0 and actual_steps < max_steps
    if guard is not None and guard.converged:
        stop_reason = "reward_variance_window_saturated"
    elif canary_cb is not None and canary_cb.tripped:
        stop_reason = "canary_threshold"
    elif stopped_early:
        stop_reason = "trainer_stopped_early"
    else:
        stop_reason = "max_steps_or_epochs_completed"
    manifest = {
        "stage": "grpo_qlora" if qlora else "grpo_full",
        "source_model": resolved,
        "base_model": base_model,
        "task_set": task_set,
        "split": split,
        "task_count": len(tasks),
        "train_loss": float(train_result.training_loss),
        "learning_rate": lr,
        "beta": beta,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "temperature": temperature,
        "frozen_policy_probe": lr == 0.0,
        "max_steps": max_steps,
        "requested_max_steps": max_steps,
        "actual_steps": actual_steps,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "reward_variance_guard": (
            {
                "enabled": True,
                "converged": guard.converged,
                "tripped": guard.tripped,
                "patience": guard.patience,
                "grace_steps": guard.grace_steps,
                "stop_step": guard.stop_step,
            }
            if guard is not None
            else {"enabled": False}
        ),
        "reward_curriculum_steps": reward_curriculum_steps,
        "initial_reward_weights": dataclasses.asdict(initial_weights),
        "rollout_audit": (
            {
                "trace_path": str(resolved_trace_path),
                "summary_path": str(rollout_audit_summary_path),
                "seen_samples": rollout_audit_summary["seen_samples"],
                "saved_samples": rollout_audit_summary["saved_samples"],
                "anomalous_samples": rollout_audit_summary["anomalous_samples"],
            }
            if rollout_audit_summary is not None
            else None
        ),
        "num_generations": num_generations,
        "max_turns": max_turns,
        "max_completion_length": max_completion_length,
        "seed": seed,
        "save_steps": save_steps,
        "save_total_limit": save_total_limit if save_steps > 0 else None,
        "resumed_from_checkpoint": resolved_resume_checkpoint,
        "peak_vram_gib": (
            torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        ),
    }
    trainable, model_parameters, trainable_ratio = _trainable_parameter_stats(trainer.model)
    manifest.update({
        "trainable_parameters": trainable,
        "model_parameters": model_parameters,
        "trainable_ratio_percent": trainable_ratio,
    })
    if not math.isfinite(manifest["train_loss"]):
        raise SystemExit(f"❌ GRPO training_loss不是有限数: {manifest['train_loss']}")
    (final_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
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
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument(
        "--reward_curriculum_steps", type=int, default=30,
        help="奖励课程持续步数。默认30保持历史实验兼容；SFT后直接优化工具、参数和结果时"
             "显式传0，从第一步启用完整语义权重。该值会写入training_manifest.json",
    )
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
    p.add_argument(
        "--seed", type=int, default=42,
        help="训练与 rollout 采样种子；对比两个策略的原始轨迹时必须保持一致",
    )
    p.add_argument(
        "--qlora", action=argparse.BooleanOptionalAction, default=True,
        help="使用4-bit QLoRA继续训练SFT adapter（单4090默认方案）",
    )
    p.add_argument(
        "--base_model", default=None,
        help="SFT adapter对应的基座模型；默认读取adapter_config.json",
    )
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True,
    )
    p.add_argument("--optim", default="paged_adamw_8bit")
    p.add_argument(
        "--save_steps", type=int, default=0,
        help="每隔多少个 optimizer step 保存可恢复 checkpoint；0 表示只保存最终模型",
    )
    p.add_argument(
        "--save_total_limit", type=int, default=2,
        help="最多保留多少个周期 checkpoint（最终模型不计入）",
    )
    p.add_argument(
        "--resume_from_checkpoint", default=None,
        help="从指定 checkpoint 继续模型、优化器和调度器状态；需保持原 output_dir 与训练参数",
    )
    p.add_argument("--snapshot", default=None)
    p.add_argument(
        "--task_set", choices=["default", "expanded"], default="default",
        help="default=benchmark/tasks.py 的 8 条冒烟任务（保持现状）；"
             "expanded=完整基准集。改默认值会让既有训练曲线不可比，故需显式指定",
    )
    p.add_argument(
        "--split", default=None, metavar="[FILE:]NAME",
        help=f"只用某一份切分训练，如 rl_train（用 {DEFAULT_SPLIT_PATH.name}）或 "
             "data/splits/v2.json:train。rl_train 只含成功率中间带的任务——"
             "GRPO 的梯度来自组内奖励方差，两端的任务不产生梯度。需配合 --task_set expanded",
    )
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
    p.add_argument(
        "--report_to", default="none",
        help="训练曲线记到哪：none / auto / tensorboard / wandb（可逗号分隔）。"
             "记录 reward、kl、grad_norm、frac_reward_zero_std，"
             "以及 format/tool/argument/process/outcome 五个奖励分量与当前课程权重",
    )
    p.add_argument(
        "--run_name", default=None,
        help="本次运行在 TensorBoard / wandb 里的名字，默认取 output_dir 末段",
    )
    p.add_argument(
        "--save_rollout_traces", action="store_true",
        help="保存每条 GRPO 原始生成、工具轨迹、奖励分解和格式异常。"
             "默认写到 output_dir/rollout_traces.jsonl",
    )
    p.add_argument(
        "--rollout_trace_path", default=None,
        help="自定义 rollout 审计 JSONL 路径；传入该参数会自动启用审计",
    )
    p.add_argument(
        "--rollout_trace_max_samples", type=int, default=0,
        help="最多写入多少条原始 rollout（0=全部）；摘要统计始终覆盖全部样本",
    )
    main(**vars(p.parse_args()))
