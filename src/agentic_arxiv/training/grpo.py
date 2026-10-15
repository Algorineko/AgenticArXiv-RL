"""GRPO 训练脚本（使用 TRL GRPOTrainer）

GRPO（Group Relative Policy Optimization）：
- 目标：用 verifiable reward 在线训练
- 优势：无需 reward model、无需 value model
- 数据：**不需要预先生成**。GRPO 是在线算法，只要 prompt + 可验证奖励，
        数据集直接由 benchmark/tasks.py 派生

奖励定义见 rl/grpo_reward.py：TRL 原生执行多轮 tool-calling，把每轮环境
observation 插回上下文，并将完整轨迹交给 RewardCalculator 打分。

训练中每 N 步自动运行 canary 评估（在固定小任务集上验证模型未退化），
训练结束后可选运行阶段验证（检查模型是否达到最低质量阈值）。

使用方式：
    agentic-arxiv train grpo --task-set expanded --split data/splits/v7_86.json:train
    agentic-arxiv train grpo --smoke
"""

import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Optional

#: 当前源码主线路径的显式默认，不据此推断历史运行有效值。
DEFAULT_BETA = 0.04

from agentic_arxiv.training.callbacks import TrainerCallback
from agentic_arxiv.training.distributed import TrainingContext, grpo_generation_batch_size
from agentic_arxiv.training.precision import quantized_model_device_map

# 工具集从 tools/bootstrap.py 的注册表来，不在这里手写模块清单：
# 手写的那份漏掉过 paper_content / paper_summary / paper_figures，后果是
# prompt 里的工具列表少了三个，而策略的动作空间缩小是**静默**的。
from agentic_arxiv.tools.bootstrap import require_all_tools

from agentic_arxiv.tasks.task_io import load_tasks
from agentic_arxiv.tasks.splits import DEFAULT_SPLIT_PATH
from agentic_arxiv.tasks.reference import attach_expected_paper_ids
from agentic_arxiv.training.canary import CanaryEvaluator, CanaryCallback
from agentic_arxiv.training.mainline_config import (
    config_file_path,
    load_mainline_config,
    require_supported_config,
    resolve_model_source,
    stage_config,
    validate_grpo_options,
)
from agentic_arxiv.provenance import (
    build_run_manifest, file_fingerprint, model_input_fingerprint,
    require_training_protocol, source_fingerprint,
)
from agentic_arxiv.runtime.react_contract import protocol_metadata
from agentic_arxiv.training.grpo_reward import (
    SUPPORTED_TOOL_NAMES,
    build_prompt_dataset,
    load_mock_env,
    make_grpo_reward_fn,
    make_multiturn_rollout_func,
)
from agentic_arxiv.runtime.multiturn_env import make_environment_factory
from agentic_arxiv.training.rollout_audit import RolloutAuditWriter
from agentic_arxiv.training.observability import (
    RewardComponentTracker,
    describe_logging,
    resolve_report_to,
)
from agentic_arxiv.scoring.reward import RewardCalculator
from agentic_arxiv.training.stage_verifier import StageVerifier
from agentic_arxiv.training.model_setup import (
    _assert_lora_only_trainable,
    _build_qlora_configs,
    _qlora_runtime_guard,
    _trainable_parameter_stats,
)
from agentic_arxiv.config import get_workspace, workspace_path

DEFAULT_SNAPSHOT = Path("data/mock_arxiv_snapshot.json")
GRPO_DEFAULTS = stage_config("grpo")["defaults"]


def _precision_flags():
    """见 rl/precision.py：CUDA 上优先 bf16，退回 fp16；CPU / MPS 不开混合精度。"""
    from agentic_arxiv.training.precision import precision_flags
    return precision_flags()


def _build_reward_calculator(curriculum_steps: int) -> RewardCalculator:
    """构造本次运行唯一的奖励配置，并拒绝含糊的负步数。"""
    if curriculum_steps < 0:
        raise SystemExit("❌ reward_curriculum_steps 不能为负数")
    return RewardCalculator(curriculum_steps=curriculum_steps)


def _load_tasks(task_set: Optional[str], split: Optional[str], *, smoke: bool = False):
    """正式训练显式选训练切分；8 条冒烟任务只在显式 smoke 模式下使用。"""
    if smoke:
        if task_set not in (None, "default"):
            raise SystemExit("冒烟模式只允许 benchmark/tasks.py 的 8 条任务")
        if split and split.rsplit(":", 1)[-1] not in ("train", "rl_train"):
            raise SystemExit("冒烟模式不能使用 dev/iid_test/ood_test 等非训练切分")
        return load_tasks("default", split)
    if task_set != "expanded" or not split:
        raise SystemExit(
            "正式 GRPO 必须显式传 --task-set expanded 与 --split [FILE:]train/rl_train；"
            "8 条冒烟任务需显式 --smoke，不能隐式训练完整任务池"
        )
    if split.rsplit(":", 1)[-1] not in ("train", "rl_train"):
        raise SystemExit("正式 GRPO 仅允许 train/rl_train 训练切分，不能使用留出集")
    split_path, separator, split_name = split.rpartition(":")
    if separator and not Path(split_path).is_absolute():
        split = f"{workspace_path(split_path)}:{split_name}"
    return load_tasks(task_set, split)


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
        return resolve_model_source(override)
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        raise SystemExit(f"适配器元数据缺失: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SystemExit(f"adapter_config 必须为 JSON 对象: {config_path}")
    value = str(config.get("base_model_name_or_path") or "")
    if not value:
        raise SystemExit(f"❌ adapter_config 未声明 base_model_name_or_path: {config_path}")
    try:
        return resolve_model_source(value)
    except ValueError as exc:
        raise SystemExit(
            f"SFT adapter 引用的基座模型不存在: {value}；"
            "换机器后可显式传 --base-model 指定同一基座的有效位置"
        ) from exc


def _artifact_metadata(model_path: str, *, base_model: Optional[str] = None) -> dict:
    """GRPO 只消费明确的本地产物，不能用基座或远程 API 替代丢失的 SFT。"""
    local = Path(resolve_model_source(model_path, local_only=True))
    manifest_path = local / "training_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"训练模型缺少必要 training_manifest: {manifest_path}")
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise SystemExit(f"training_manifest 必须是 JSON 对象: {manifest_path}")
    stage = metadata.get("stage")
    if stage not in ("sft_qlora", "sft_full", "grpo_qlora", "grpo_full"):
        raise SystemExit(f"不支持的训练产物 stage={stage!r}: {manifest_path}")
    is_adapter = (local / "adapter_config.json").is_file()
    if is_adapter != stage.endswith("_qlora"):
        raise SystemExit("训练 manifest 的全参/适配器类型与产物文件不一致")
    if is_adapter:
        if not metadata.get("base_model"):
            raise SystemExit("适配器 training_manifest 缺少 base_model")
        if not any((local / name).is_file() for name in (
            "adapter_model.safetensors", "adapter_model.bin"
        )):
            raise SystemExit(f"适配器权重缺失: {local}")
        adapter_config = json.loads((local / "adapter_config.json").read_text(encoding="utf-8"))
        declared_base = adapter_config.get("base_model_name_or_path")
        if not declared_base or declared_base != metadata["base_model"]:
            raise SystemExit("适配器基座声明与 training_manifest 的 base_model 不一致")
        _adapter_base_model(local, base_model)
    else:
        if not (local / "config.json").is_file():
            raise SystemExit(f"全参模型 config.json 缺失: {local}")
        if not any((local / name).is_file() for name in (
            "model.safetensors", "model.safetensors.index.json",
            "pytorch_model.bin", "pytorch_model.bin.index.json"
        )):
            raise SystemExit(f"全参模型权重或分片索引缺失: {local}")
    return metadata


def _load_policy(
    model_path: str,
    *,
    qlora: bool,
    base_model: Optional[str],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
):
    """按全参/适配器产物合同加载，QLoRA 只是资源策略。"""
    import torch
    from transformers import AutoModelForCausalLM

    _artifact_metadata(model_path, base_model=base_model)
    local = Path(resolve_model_source(model_path, local_only=True))
    model_path = str(local)
    is_adapter = local.is_dir() and (local / "adapter_config.json").exists()
    if not is_adapter and base_model:
        raise SystemExit("全参 SFT 产物不能用 --base-model 替换训练起点")
    TrainingContext.from_environment().prepare_device(qlora=qlora)
    if not qlora:
        print("⚠️  --no-qlora：加载完整精度策略，不属于单张4090冻结方案")
        source = _adapter_base_model(local, base_model) if is_adapter else model_path
        # 全参保留路径原本不传 dtype；适配器合并合同也不借机改写此省略语义。
        policy = AutoModelForCausalLM.from_pretrained(
            source, local_files_only=Path(source).exists(),
        )
        if is_adapter:
            from peft import PeftModel

            policy = PeftModel.from_pretrained(policy, model_path).merge_and_unload()
        for parameter in policy.parameters():
            parameter.requires_grad = True
        policy.config.use_cache = False
        return policy

    _qlora_runtime_guard()
    quantization, fresh_lora = _build_qlora_configs(
        lora_r, lora_alpha, lora_dropout
    )
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

    source = _adapter_base_model(local, base_model) if is_adapter else model_path
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
        device_map=quantized_model_device_map(),
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
            "   确实想继续跑，用 --allow-zero-variance 跳过本检查。"
        )


def main(
    model: str = GRPO_DEFAULTS["model"],
    output_dir: str = GRPO_DEFAULTS["output_dir"],
    epochs: int = GRPO_DEFAULTS["epochs"],
    max_steps: int = GRPO_DEFAULTS["max_steps"],
    batch_size: int = GRPO_DEFAULTS["batch_size"],
    grad_accum: int = GRPO_DEFAULTS["grad_accum"],
    lr: float = GRPO_DEFAULTS["lr"],
    beta: float = GRPO_DEFAULTS["beta"],
    reward_curriculum_steps: int = GRPO_DEFAULTS["reward_curriculum_steps"],
    num_generations: int = GRPO_DEFAULTS["num_generations"],
    max_completion_length: int = GRPO_DEFAULTS["max_completion_length"],
    max_turns: int = GRPO_DEFAULTS["max_turns"],
    temperature: float = GRPO_DEFAULTS["temperature"],
    loss_type: str = GRPO_DEFAULTS["loss_type"],
    epsilon_high: float = GRPO_DEFAULTS["epsilon_high"],
    mask_truncated_completions: bool = GRPO_DEFAULTS["mask_truncated_completions"],
    importance_sampling_level: str = GRPO_DEFAULTS["importance_sampling_level"],
    scale_rewards: str = GRPO_DEFAULTS["scale_rewards"],
    seed: int = GRPO_DEFAULTS["seed"],
    snapshot: str = GRPO_DEFAULTS["snapshot"],
    task_set: str = GRPO_DEFAULTS["task_set"],
    split: str = GRPO_DEFAULTS["split"],
    smoke: bool = GRPO_DEFAULTS["smoke"],
    allow_zero_variance: bool = GRPO_DEFAULTS["allow_zero_variance"],
    canary_steps: int = GRPO_DEFAULTS["canary_steps"],
    min_canary_reward: float = GRPO_DEFAULTS["min_canary_reward"],
    canary_patience: int = GRPO_DEFAULTS["canary_patience"],
    verify: bool = GRPO_DEFAULTS["verify"],
    report_to: str = GRPO_DEFAULTS["report_to"],
    run_name: str = GRPO_DEFAULTS["run_name"],
    qlora: bool = GRPO_DEFAULTS["qlora"],
    base_model: str = GRPO_DEFAULTS["base_model"],
    lora_r: int = GRPO_DEFAULTS["lora_r"],
    lora_alpha: int = GRPO_DEFAULTS["lora_alpha"],
    lora_dropout: float = GRPO_DEFAULTS["lora_dropout"],
    gradient_checkpointing: bool = GRPO_DEFAULTS["gradient_checkpointing"],
    optim: str = GRPO_DEFAULTS["optim"],
    save_steps: int = GRPO_DEFAULTS["save_steps"],
    save_total_limit: int = GRPO_DEFAULTS["save_total_limit"],
    resume_from_checkpoint: str = GRPO_DEFAULTS["resume_from_checkpoint"],
    save_rollout_traces: bool = GRPO_DEFAULTS["save_rollout_traces"],
    rollout_trace_path: str = GRPO_DEFAULTS["rollout_trace_path"],
    rollout_trace_max_samples: int = GRPO_DEFAULTS["rollout_trace_max_samples"],
    config_path: str = None,
):
    requested_config = dict(locals())
    # 操作启动后才设置训练进程选项，包导入不改变用户环境。
    os.environ.setdefault("STORE_BACKEND", "memory")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from datasets import Dataset
    from transformers import AutoTokenizer
    # torch<2.6 的兼容保护必须先于 TRL 实际导入。
    from agentic_arxiv.training import trl_compat  # noqa: F401
    from trl import GRPOConfig, GRPOTrainer
    configured = stage_config("grpo", config_path)
    execution = TrainingContext.from_environment()
    generation_batch = grpo_generation_batch_size(batch_size, grad_accum, num_generations, execution)
    try:
        validate_grpo_options(
            loss_type=loss_type,
            importance_sampling_level=importance_sampling_level,
            scale_rewards=scale_rewards,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    # 工具没注册齐就直接失败：模型不会因为工具列表不全而报错，
    # 它会编工具名，训练照跑、奖励照记，只是那些工具对应的任务永远做不成。
    require_all_tools("GRPO 训练")

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
        resume_path = workspace_path(resume_from_checkpoint)
        if not resume_path.exists():
            raise SystemExit(f"❌ resume_from_checkpoint 不存在: {resume_path}")
        resolved_resume_checkpoint = str(resume_path)
    logging_dir = str(workspace_path(output_dir) / "logs")

    # 先验证任务与快照，再加载模型。语义标准答案损坏时应在占用显存前失败。
    tasks = _load_tasks(task_set, split, smoke=smoke)
    try:
        resolved = resolve_model_source(model, local_only=True)
        source_metadata = _artifact_metadata(resolved, base_model=base_model)
        require_training_protocol(
            source_metadata, protocol_metadata(), source_label="GRPO 输入模型",
            require_chat_template_hash=True,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    actual_base_model = (
        _adapter_base_model(Path(resolved), base_model)
        if (Path(resolved) / "adapter_config.json").is_file()
        else resolved
    )
    snapshot_path = workspace_path(snapshot or DEFAULT_SNAPSHOT)
    if not snapshot_path.exists():
        raise SystemExit(
            f"❌ 多轮 GRPO 需要离线快照 {snapshot_path}\n"
            "   请准备已有离线快照；快照入口为 agentic-arxiv data snapshot"
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
    current_protocol = protocol_metadata(tokenizer)
    try:
        require_training_protocol(
            current_protocol, current_protocol, source_label="当前 GRPO tokenizer",
            require_chat_template_hash=True,
        )
        require_training_protocol(
            source_metadata, current_protocol, source_label="GRPO 输入模型",
            require_chat_template_hash=True,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
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
            f"请改用 --max-completion-length {need + 64}"
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
            workspace_path(rollout_trace_path)
            if rollout_trace_path
            else workspace_path(output_dir) / "rollout_traces.jsonl"
        )
        if execution.is_main:
            auditor = RolloutAuditWriter(
                resolved_trace_path,
                num_generations=num_generations,
                allowed_tools=SUPPORTED_TOOL_NAMES,
                max_samples=rollout_trace_max_samples,
                protocol_metadata=current_protocol,
                training_metadata=source_metadata,
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
        reward_calc=reward_calc,
        tracker=tracker,
        auditor=auditor,
        distributed=execution,
        num_generations=num_generations,
    )
    rollout_func = make_multiturn_rollout_func(
        environment_factory,
        max_turns=max_turns,
        tasks_by_id=tasks_by_id,
        prompt_task_ids=prompt_task_ids,
        training_metadata=source_metadata,
    )
    print(f"🧮 全局生成批量={generation_batch}，world_size={execution.world_size}，每组={num_generations}")

    cfg_kwargs = {
        **configured["trainer_parameters"],
        "output_dir": str(workspace_path(output_dir)),
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

    # 未设置字段仍省略，不把某个 TRL 版本的默认值猜成当前请求值。
    if loss_type is not None:
        cfg_kwargs["loss_type"] = loss_type
    if epsilon_high is not None:
        cfg_kwargs["epsilon_high"] = epsilon_high
    if mask_truncated_completions is not None:
        cfg_kwargs["mask_truncated_completions"] = mask_truncated_completions
    if importance_sampling_level is not None:
        cfg_kwargs["importance_sampling_level"] = importance_sampling_level
    if scale_rewards is not None:
        cfg_kwargs["scale_rewards"] = scale_rewards
    cfg_kwargs.update(execution.trainer_kwargs(stage="grpo"))
    require_supported_config(GRPOConfig, cfg_kwargs)
    config = GRPOConfig(**cfg_kwargs)
    if config.generation_batch_size != generation_batch:
        raise SystemExit("TRL 实际 generation_batch_size 与全局 batch/累积配置不一致")

    # 单进程时钉住单卡（避免 Trainer 走 DataParallel 的段错误）；
    # accelerate launch 下 accelerator 自己管设备，这里自动跳过。
    from agentic_arxiv.training.precision import pin_single_gpu

    pin_single_gpu(config)

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
    execution.validate_trainer(trainer, stage="grpo")
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
            distributed=execution,
        )
        trainer.add_callback(canary_cb)
        print(f"🐤 Canary: 每 {canary_steps} 步评估，阈值={min_canary_reward}，patience={canary_patience}")

    guard = None
    if not allow_zero_variance:
        guard = RewardVarianceGuard()
        trainer.add_callback(guard)

    memory = None

    def save_run_diagnostic(reason: str) -> dict | None:
        """保护器或异常退出仍保存可得事实，不把失败运行标为成功 final。"""
        if not execution.is_main:
            return None
        step = getattr(trainer.state, "global_step", None)
        planned = getattr(trainer.state, "max_steps", None)
        split_path = (split.rpartition(":")[0] or DEFAULT_SPLIT_PATH) if split else None
        if split_path and not Path(split_path).is_absolute():
            split_path = workspace_path(split_path)
        record = build_run_manifest(
            requested_config=requested_config,
            resolved_config=config.to_dict(),
            derived_values={
                "per_device_train_batch_size": batch_size,
                "effective_batch_size": execution.effective_batch_size(batch_size, grad_accum),
                "generation_batch_size": generation_batch,
                "distributed_training": execution.metadata(),
                "memory": memory,
                "task_count": len(tasks),
                "actual_task_ids": [task["id"] for task in tasks],
                "actual_base_model": actual_base_model,
                "trainer_planned_steps": planned,
            },
            source_baseline={
                "configuration_baseline": load_mainline_config(config_path)["source_baseline"],
                "execution_source": source_fingerprint(
                    [Path(__file__), Path(__file__).with_name("mainline_config.py"),
                     Path(__file__).with_name("grpo_reward.py"),
                     Path(__file__).with_name("canary.py"),
                     Path(__file__).with_name("observability.py"),
                     Path(__file__).with_name("model_setup.py"),
                     Path(__file__).with_name("distributed.py"),
                     Path(__file__).with_name("precision.py")], repo_root=get_workspace(),
                ),
            },
            inputs={
                "model_manifest": file_fingerprint(Path(resolved) / "training_manifest.json"),
                "source_model": {**model_input_fingerprint(resolved),
                                 "stage": source_metadata["stage"]},
                "base_model": model_input_fingerprint(actual_base_model),
                "snapshot": file_fingerprint(snapshot_path),
                "split": file_fingerprint(split_path),
                "mainline_configuration": file_fingerprint(config_file_path(config_path)),
            },
            resource_overrides={**execution.resource_overrides(), **{
                key: {"baseline": GRPO_DEFAULTS[key], "requested": requested_config[key]}
                for key in ("qlora", "batch_size", "grad_accum", "gradient_checkpointing",
                            "optim", "lora_r", "lora_alpha", "lora_dropout")
                if requested_config[key] != GRPO_DEFAULTS[key]
            }},
            requested_steps=max_steps,
            protocol_metadata=current_protocol,
            actual_steps=step,
            stop_reason=reason,
            protections={
                "reward_variance_guard": {
                    "enabled": guard is not None,
                    "tripped": guard.tripped if guard is not None else None,
                    "converged": guard.converged if guard is not None else None,
                    "patience": guard.patience if guard is not None else None,
                    "grace_steps": guard.grace_steps if guard is not None else None,
                    "stop_step": guard.stop_step if guard is not None else None,
                },
                "canary_steps": canary_steps,
                "canary_min_reward": min_canary_reward,
                "canary_patience": canary_patience,
                "canary_tripped": canary_cb.tripped if canary_cb is not None else None,
                "verification_enabled": verify,
            },
        )
        record_path = workspace_path(output_dir) / "run_manifest.json"
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        return record

    try:
        train_result = trainer.train(
            resume_from_checkpoint=resolved_resume_checkpoint
        )
    except BaseException as exc:
        try:
            save_run_diagnostic(f"training_exception:{type(exc).__name__}")
        except Exception as record_error:
            print(f"训练异常时未能保存运行诊断: {type(record_error).__name__}")
        # JSONL is flushed after every reward batch. Also persist its compact
        # summary on OOM, Ctrl-C, or trainer failure so the last healthy batch
        # remains diagnosable instead of disappearing with the failed run.
        if auditor is not None:
            failed_summary_path = auditor.save_summary(
                resolved_trace_path.with_suffix(".summary.json")
            )
            print(f"🔎 训练中断，已保留 rollout 审计: {failed_summary_path}")
        raise

    train_loss = float(train_result.training_loss)
    if not math.isfinite(train_loss):
        try:
            save_run_diagnostic("invalid_training_loss")
        except Exception as record_error:
            print(f"无效训练 loss 时未能保存运行诊断: {type(record_error).__name__}")
        raise SystemExit(f"❌ GRPO training_loss不是有限数: {train_loss}")

    actual_steps = int(getattr(trainer.state, "global_step", 0) or 0)
    planned_steps = max_steps if max_steps > 0 else getattr(trainer.state, "max_steps", None)
    stopped_early = bool(planned_steps and actual_steps < planned_steps)
    if guard is not None and guard.tripped:
        stop_reason = "reward_variance_initial_signal_missing"
    elif guard is not None and guard.converged:
        stop_reason = "reward_variance_window_saturated"
    elif canary_cb is not None and canary_cb.tripped:
        stop_reason = "canary_threshold"
    elif stopped_early:
        stop_reason = "trainer_stopped_early"
    else:
        stop_reason = "max_steps_or_epochs_completed"
    memory = execution.memory_summary()
    run_record = execution.run_on_main(lambda: save_run_diagnostic(stop_reason))

    # 保护器提前停止也保留全球奖励与完整 generation 组的审计统计。
    reward_summary_path = execution.run_on_main(lambda: tracker.save_task_summary(
        workspace_path(output_dir) / "reward_probe_summary.json",
        protocol_metadata=current_protocol,
        training_metadata=source_metadata,
    ))
    print(f"📊 逐任务奖励统计: {reward_summary_path}")

    def save_rollout_summary():
        if auditor is None:
            return None, None
        path = auditor.save_summary(resolved_trace_path.with_suffix(".summary.json"))
        summary = auditor.summary()
        print(f"🔎 rollout 审计摘要: {path} | 已保存={summary['saved_samples']}, "
              f"异常样本={summary['anomalous_samples']}")
        return path, summary

    rollout_audit_summary_path, rollout_audit_summary = execution.run_on_main(save_rollout_summary)
    if guard is not None and guard.tripped:
        raise SystemExit(1)
    if canary_cb is not None and canary_cb.tripped:
        raise SystemExit(1)

    final_dir = workspace_path(output_dir) / "final"
    trainer.save_model(str(final_dir))

    def save_final_artifacts():
        tokenizer.save_pretrained(str(final_dir))
        if qlora:
            required_adapter_files = (
                final_dir / "adapter_config.json",
                final_dir / "adapter_model.safetensors",
            )
            missing = [str(path) for path in required_adapter_files if not path.exists()]
            if missing:
                raise SystemExit(f"❌ GRPO完成但adapter文件缺失: {missing}")
        manifest = {
            "stage": "grpo_qlora" if qlora else "grpo_full",
            "source_model": resolved,
            "base_model": actual_base_model,
            "task_set": "default" if smoke else task_set,
            "split": split,
            "smoke": smoke,
            "task_count": len(tasks),
            "train_loss": train_loss,
            "learning_rate": lr,
            "beta": beta,
            "loss_type": loss_type,
            "epsilon_high": epsilon_high,
            "mask_truncated_completions": mask_truncated_completions,
            "importance_sampling_level": importance_sampling_level,
            "scale_rewards": scale_rewards,
            "batch_size": batch_size,
            "effective_batch_size": execution.effective_batch_size(batch_size, grad_accum),
            "generation_batch_size": generation_batch,
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
            "peak_vram_gib": memory["peak_allocated_gib"] or 0.0,
        }
        trainable, model_parameters, trainable_ratio = _trainable_parameter_stats(trainer.model)
        manifest.update({
            "trainable_parameters": trainable,
            "model_parameters": model_parameters,
            "trainable_ratio_percent": trainable_ratio,
        })
        manifest.update(run_record)
        (final_dir / "training_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"✅ GRPO 训练完成，模型已保存: {final_dir}")

    execution.run_on_main(save_final_artifacts)

    if verify:
        def verify_final():
            print("\n🔍 运行 GRPO 阶段验证...")
            verifier = StageVerifier()
            report = verifier.verify_grpo(model_path=str(final_dir), tokenizer=tokenizer, env=env)
            verifier.save_report(report, final_dir)
            print(report.summary())
            if not report.passed:
                raise SystemExit(
                    "\n❌ GRPO 阶段验证失败，模型未达到最低质量阈值。\n"
                    f"   检查 {final_dir / 'verification_report.json'} 了解详情。\n"
                    "   跳过验证用 --no-verify。"
                )

        execution.run_on_main(verify_final)
