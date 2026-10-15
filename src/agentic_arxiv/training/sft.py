"""SFT/QLoRA 训练脚本（使用 TRL SFTTrainer）

SFT（Supervised Fine-Tuning）：
- 目标：让模型学会基本的工具调用格式
- 数据：expert demonstrations（从 benchmark tasks 生成）
- 输出：SFT 全参或适配器产物（直接作为 GRPO 的起点）

使用方式：
    agentic-arxiv train sft --inspect-only --max-length 4096
    agentic-arxiv train sft --max-steps 30 --max-length 4096 --no-verify
"""

import dataclasses
import hashlib
import json
import math
import time
from importlib.metadata import version
from pathlib import Path

from agentic_arxiv.training.observability import describe_logging, resolve_report_to
from agentic_arxiv.training.distributed import TrainingContext
from agentic_arxiv.training.dataloader_resume import preserve_managed_dataloader_epoch
from agentic_arxiv.training.managed import (
    prepare_managed_training, attach_managed_training, validate_managed_config,
    record_training_result,
    managed_cancellation_check,
)
from agentic_arxiv.training.stage_verifier import StageVerifier
from agentic_arxiv.training.mainline_config import (
    config_file_path,
    load_mainline_config,
    require_supported_config,
    resolve_model_source,
    stage_config,
)
from agentic_arxiv.provenance import (
    build_run_manifest, file_fingerprint, model_input_fingerprint,
    read_protocol_metadata, require_training_protocol, source_fingerprint,
)
from agentic_arxiv.runtime.react_contract import protocol_metadata, validate_react_messages
from agentic_arxiv.config import get_workspace, workspace_path
from agentic_arxiv.training.model_setup import (
    QLORA_TARGET_MODULES,
    _assert_lora_only_trainable,
    _build_qlora_configs,
    _qlora_runtime_guard,
    _trainable_parameter_stats,
)

SFT_DEFAULTS = stage_config("sft")["defaults"]


def _verification_environment(snapshot, managed):
    """Bind managed verification to the frozen input, never the container cwd."""
    if managed is not None:
        inputs = managed.identity["inputs"].get("inputs", {})
        reference = inputs.get("snapshot", {})
        declared = reference.get("path") if isinstance(reference, dict) else None
        if not isinstance(declared, str) or not declared:
            raise ValueError("受管 SFT 阶段验证需要明确 snapshot 输入")
        if snapshot and workspace_path(snapshot).resolve() != Path(declared).resolve():
            raise ValueError("SFT 阶段验证快照与受管输入身份不一致")
        snapshot = declared
    if not snapshot:
        # Preserve the historical workspace fallback for unmanaged callers.
        return None
    from agentic_arxiv.runtime.env import MockArxivEnv

    return MockArxivEnv(snapshot_path=workspace_path(snapshot), mode="replay")


def _filter_dataclass_kwargs(config_cls, kwargs: dict) -> tuple[dict, list[str]]:
    """按当前安装版本的 dataclass 字段过滤跨版本配置项。

    TRL 的配置类会随版本增删参数。训练脚本仍显式构造完整配置字典，
    这里仅在实例化前移除当前版本不存在的键，并把它们返回给调用方告警。
    """
    supported = {field.name for field in dataclasses.fields(config_cls)}
    dropped = sorted(set(kwargs) - supported)
    return {key: value for key, value in kwargs.items() if key in supported}, dropped


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_data_manifest(data_path: Path, manifest_path: Path | None = None) -> dict:
    """冻结训练输入：行数、类型与 SHA256 任一漂移都拒绝训练。"""
    manifest_path = manifest_path or data_path.with_suffix(data_path.suffix + ".manifest.json")
    if not manifest_path.exists():
        raise SystemExit(
            f"❌ 缺少训练数据 manifest: {manifest_path}\n"
            "   正式 QLoRA 默认只接受 build_sft_train_mix.py 生成的可审计数据；"
            "临时实验可显式传 --skip-data-manifest-check"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "qlora_sft_train_mix":
        raise SystemExit(f"❌ 训练数据 manifest kind 错误: {manifest.get('kind')!r}")
    actual_hash = _sha256_file(data_path)
    if manifest.get("output_sha256") != actual_hash:
        raise SystemExit(
            "❌ 训练数据 SHA256 与 manifest 不一致，文件可能在冻结后被修改\n"
            f"   manifest={manifest.get('output_sha256')}\n   actual={actual_hash}"
        )
    actual_rows = sum(1 for line in data_path.open("r", encoding="utf-8") if line.strip())
    if manifest.get("output_rows") != actual_rows:
        raise SystemExit(
            f"❌ 训练数据行数不一致: manifest={manifest.get('output_rows')}, "
            f"actual={actual_rows}"
        )
    if manifest.get("unique_sample_fingerprints") != actual_rows:
        raise SystemExit("❌ manifest 显示训练集中存在重复样本")
    print(
        f"🔒 数据审计通过: rows={actual_rows}, "
        f"semantic_tasks={manifest.get('semantic_task_instances')}, sha256={actual_hash[:12]}…"
    )
    return manifest


def _precision_flags():
    """见 rl/precision.py：CUDA 上优先 bf16，退回 fp16；CPU / MPS 不开混合精度。"""
    from agentic_arxiv.training.precision import precision_flags
    return precision_flags()


def _load_sft_rows(path: Path, *, expected_protocol: dict | None = None) -> list:
    """Read only the columns training needs.

    The JSONL also carries provenance columns (``generation_parameters``,
    ``sample_sha256`` and friends) that belong in the artifact but not in the
    tensors.  ``datasets`` infers a single Arrow schema for the whole file, and
    those structs do not share a shape between rows — the parametric rows add
    ``section``/``style``/``max_words``, search rows have none of them — so one
    broader row makes the cast fail for the entire dataset.  Selecting the
    training columns here keeps the audit trail on disk without letting it
    break the loader.
    """
    rows = []

    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if expected_protocol is not None:
                require_training_protocol(
                    row, expected_protocol, source_label=f"SFT 数据第 {line_number} 行",
                )
                _validate_training_messages(row)
            if "messages" in row:
                rows.append({"messages": row["messages"]})
            else:
                rows.append(
                    {
                        "prompt": row.get("prompt") or [],
                        "completion": row.get("completion") or [],
                    }
                )

    if not rows:
        raise SystemExit(f"训练数据为空: {path}")

    return rows


def _messages_of(row):
    """兼容两种数据格式：{"messages": [...]} 与 {"prompt": [...], "completion": [...]}。"""
    if "messages" in row:
        return list(row["messages"])
    return list(row.get("prompt") or []) + list(row.get("completion") or [])


def _to_prompt_completion(row):
    """将共同多轮历史前缀与唯一当前 assistant completion 分开。

    此前 assistant/setup/Observation 全部留在 prompt。显式拆分后，
    ``completion_only_loss=True`` 可以在不依赖模型 chat template 是否提供
    assistant mask 的情况下，仅监督当前目标响应。
    """
    messages = list(row["messages"])
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("SFT 样本必须以 assistant message 结尾")
    return {
        "prompt": messages[:-1],
        "completion": messages[-1:],
    }


def _validate_training_messages(row: dict) -> None:
    """训练标签只能是一条当前 assistant，不能把历史动作当作 completion。"""
    if "messages" in row:
        converted = _to_prompt_completion(row)
        if "prompt" in row or "completion" in row:
            if converted != {"prompt": row.get("prompt"), "completion": row.get("completion")}:
                raise ValueError("SFT messages 与 prompt/completion 不一致")
    else:
        converted = {"prompt": row.get("prompt"), "completion": row.get("completion")}
    prompt, completion = converted["prompt"], converted["completion"]
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("SFT prompt 必须包含共同初始 user 消息")
    if not isinstance(completion, list) or len(completion) != 1:
        raise ValueError("SFT completion 必须仅包含当前一条 assistant 消息")
    validate_react_messages(prompt)
    validate_react_messages(prompt + completion, require_completion=True)


def _training_model_metadata(source: str, expected_protocol: dict) -> dict | None:
    """保留显式 SFT 基座入口；已有训练产物不能以未知协议冒充兼容。"""
    root = Path(source)
    manifest = root / "training_manifest.json"
    if manifest.is_file():
        return require_training_protocol(
            manifest, expected_protocol, source_label="SFT 输入模型",
            require_chat_template_hash=True,
        )
    if root.is_dir() and (
        (root / "adapter_config.json").is_file()
        or root.resolve().is_relative_to(workspace_path("outputs"))
    ):
        raise ValueError(f"SFT 已有训练产物缺少必要 training_manifest: {manifest}")
    # 这里只有调用者指定的 SFT 起点，不由缺 manifest 推断其真实训练历史。
    return None


def _token_length(tokenizer, messages) -> int:
    # 先渲染成字符串再计数：apply_chat_template(tokenize=True) 在 transformers 5.x
    # 返回 BatchEncoding，len() 数到的是字段数而不是 token 数。
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return len(tokenizer(text)["input_ids"])


def _check_lengths(tokenizer, dataset, max_length: int):
    """样本超过 max_length 就会被从右截断，而右边正是唯一带监督信号的 assistant 部分。

    截断是静默的：训练照常跑完、loss 照常下降、checkpoint 照常保存，
    只是模型没学到任何东西。这里在训练前把它变成一次响亮的失败。
    """
    lengths = sorted(_token_length(tokenizer, _messages_of(row)) for row in dataset)
    total = len(lengths)
    if not total:
        raise SystemExit("❌ 数据集为空")

    over = sum(1 for n in lengths if n > max_length)
    longest = lengths[-1]
    print(
        f"   token 长度: 中位 {lengths[total // 2]} / p90 {lengths[int(total * 0.9)]} / max {longest}"
    )

    if over > total * 0.01:
        raise SystemExit(
            f"❌ {over}/{total} ({over / total:.0%}) 的样本超过 max_length={max_length}，"
            f"会被截断掉 assistant 部分，训练将无监督信号。\n"
            f"   请改用 --max-length {longest + 64}（或缩短 prompt 中的工具描述）"
        )
    if over:
        print(f"   ⚠️  {over}/{total} 个样本超过 max_length={max_length}，这部分会被截断")


def main(
    model: str = SFT_DEFAULTS["model"],
    data: str = SFT_DEFAULTS["data"],
    data_manifest: str = SFT_DEFAULTS["data_manifest"],
    output_dir: str = SFT_DEFAULTS["output_dir"],
    epochs: int = SFT_DEFAULTS["epochs"],
    batch_size: int = SFT_DEFAULTS["batch_size"],
    grad_accum: int = SFT_DEFAULTS["grad_accum"],
    lr: float = SFT_DEFAULTS["lr"],
    max_length: int = SFT_DEFAULTS["max_length"],
    max_steps: int = SFT_DEFAULTS["max_steps"],
    qlora: bool = SFT_DEFAULTS["qlora"],
    lora_r: int = SFT_DEFAULTS["lora_r"],
    lora_alpha: int = SFT_DEFAULTS["lora_alpha"],
    lora_dropout: float = SFT_DEFAULTS["lora_dropout"],
    gradient_checkpointing: bool = SFT_DEFAULTS["gradient_checkpointing"],
    optim: str = SFT_DEFAULTS["optim"],
    inspect_only: bool = SFT_DEFAULTS["inspect_only"],
    skip_data_manifest_check: bool = SFT_DEFAULTS["skip_data_manifest_check"],
    seed: int = SFT_DEFAULTS["seed"],
    verify: bool = SFT_DEFAULTS["verify"],
    min_parse_rate: float = SFT_DEFAULTS["min_parse_rate"],
    report_to: str = SFT_DEFAULTS["report_to"],
    run_name: str = SFT_DEFAULTS["run_name"],
    config_path: str = None,
    managed_context: str = None,
    resume_from_checkpoint: str = None,
    snapshot: str = None,
):
    managed = prepare_managed_training(
        managed_context, stage="sft", output_dir=output_dir,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    import torch
    requested_config = dict(locals())
    requested_config.pop("torch")
    requested_config.pop("managed")
    verification_env = _verification_environment(snapshot, managed) if verify else None
    # 纯数据、标签和参数保护可以独立检查；执行训练时才加载训练栈。
    from agentic_arxiv.training import trl_compat  # noqa: F401
    from trl import SFTConfig, SFTTrainer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import Dataset

    configured = stage_config("sft", config_path)
    validate_managed_config(managed, requested_config, configured)
    execution = TrainingContext.from_environment()
    if configured["trainer_parameters"].get("completion_only_loss") is not True:
        raise SystemExit("SFT 主线必须仅监督当前 assistant completion")

    train_data_path = workspace_path(
        data or "data/sft/sft_react_messages_v3_train_mix.jsonl"
    )
    out_path = workspace_path(output_dir)
    # 先校验日志后端再加载模型：参数写错时应立刻失败
    backends = resolve_report_to(report_to)
    logging_dir = str(managed.log_dir if managed is not None else out_path / "logs")

    print(f"📚 加载 SFT 数据集: {train_data_path}")
    if not train_data_path.exists():
        raise SystemExit(
            f"❌ 数据集不存在: {train_data_path}\n"
            "   请准备已有训练输入；数据混合入口为 agentic-arxiv data sft mix"
        )
    manifest_path = (workspace_path(data_manifest) if data_manifest else
                     train_data_path.with_suffix(train_data_path.suffix + ".manifest.json"))
    current_protocol = protocol_metadata()
    try:
        require_training_protocol(manifest_path, current_protocol, source_label="SFT 数据 manifest")
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    manifest = None
    if not skip_data_manifest_check:
        manifest = _verify_data_manifest(train_data_path, manifest_path)
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print("⚠️  已跳过数据manifest校验；该运行不能作为正式可复现实验")

    try:
        resolved_model = resolve_model_source(model)
        source_protocol = _training_model_metadata(resolved_model, current_protocol)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"📦 加载 tokenizer: {resolved_model}")
    tokenizer = AutoTokenizer.from_pretrained(resolved_model, local_files_only=Path(resolved_model).exists())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    current_protocol = protocol_metadata(tokenizer)
    try:
        require_training_protocol(
            current_protocol, current_protocol, source_label="当前 SFT tokenizer",
            require_chat_template_hash=True,
        )
        require_training_protocol(manifest, current_protocol, source_label="SFT 数据 manifest")
        source_protocol = _training_model_metadata(resolved_model, current_protocol)
        training_rows = _load_sft_rows(train_data_path, expected_protocol=current_protocol)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    train_dataset = Dataset.from_list(training_rows)
    if "messages" in train_dataset.column_names:
        train_dataset = train_dataset.map(
            _to_prompt_completion,
            remove_columns=["messages"],
            desc="Converting SFT data to prompt-completion format",
        )
    print(f"   样本数: {len(train_dataset)}")

    # --- 长度守卫 ---
    _check_lengths(tokenizer, train_dataset, max_length)

    effective_batch = execution.effective_batch_size(batch_size, grad_accum)
    print(
        f"🧮 训练计划: rows={len(train_dataset)}, micro_batch={batch_size}, "
        f"grad_accum={grad_accum}, world_size={execution.world_size}, effective_batch={effective_batch}, "
        f"max_length={max_length}, max_steps={max_steps}, epochs={epochs}"
    )
    if inspect_only:
        print("✅ inspect_only 完成：数据、manifest、token长度均通过；尚未加载模型或占用训练显存")
        return

    quantization_config = None
    peft_config = None
    policy = resolved_model
    from agentic_arxiv.training.precision import quantized_model_device_map

    execution.prepare_device(qlora=qlora, allow_sharded=not qlora)
    if qlora:
        _qlora_runtime_guard()
        quantization_config, peft_config = _build_qlora_configs(
            lora_r, lora_alpha, lora_dropout
        )
        print(
            "🪶 QLoRA: 4-bit NF4 + double quant + BF16 compute | "
            f"r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}\n"
            f"   target_modules={list(QLORA_TARGET_MODULES)}"
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        from peft import get_peft_model, prepare_model_for_kbit_training

        policy = AutoModelForCausalLM.from_pretrained(
            resolved_model,
            quantization_config=quantization_config,
            device_map=quantized_model_device_map(),
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=Path(resolved_model).exists(),
        )
        # TRL 0.29.1 会按 SFTConfig 启用 gradient checkpointing 和 input grads；
        # 此处只做一次 k-bit 冻结/LayerNorm准备，不重复打开 checkpointing。
        policy = prepare_model_for_kbit_training(
            policy, use_gradient_checkpointing=False
        )
        policy = get_peft_model(policy, peft_config)
    else:
        print("⚠️  --no-qlora：将加载完整模型；该运行不属于本项目冻结的4090方案")
        # 权重精度跟随 _precision_flags：bf16 训练器配 bf16 权重，fp16 回退
        # （T4 / V100 等 pre-Ampere 卡）和 CPU 配 fp32 权重。不显式指定时
        # Transformers 5 按 checkpoint 原始精度加载，Qwen2.5 是 bf16，在不支持
        # bf16 的卡上第一步 matmul 就崩（见 rl/precision.py）。
        from agentic_arxiv.training.precision import trainable_model_load_kwargs

        policy = AutoModelForCausalLM.from_pretrained(
            resolved_model,
            local_files_only=Path(resolved_model).exists(),
            **trainable_model_load_kwargs(AutoModelForCausalLM.from_pretrained),
        )
        if optim == "paged_adamw_8bit":
            optim = "adamw_torch_fused"

    # KV cache 用于自回归生成，会保留每层历史 K/V；训练时既占显存，又与
    # gradient checkpointing 的重计算策略冲突，因此在模型侧统一关闭。
    policy.config.use_cache = False

    # use_cache 属于模型 forward/generation 配置，不是 TRL 0.29.1 的 SFTConfig
    # 字段。QLoRA 分支已在 policy.config.use_cache=False 关闭 KV cache，以兼容
    # gradient checkpointing；不要再把它传给 TrainingArguments。
    config_kwargs = {
        **configured["trainer_parameters"],
        "output_dir": str(out_path),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "max_length": max_length,  # TRL>=0.20 用 max_length
        "max_steps": max_steps,
        "optim": optim,
        "gradient_checkpointing": gradient_checkpointing,
        "gradient_checkpointing_kwargs": (
            {"use_reentrant": False} if gradient_checkpointing else None
        ),
        "seed": seed,
        "data_seed": seed,
        "report_to": backends,
        "run_name": run_name or out_path.name,
        **_precision_flags(),  # 只有 CUDA 才开混合精度
    }
    config_kwargs.update(execution.trainer_kwargs(stage="sft"))
    if managed is not None:
        config_kwargs["logging_dir"] = logging_dir
    require_supported_config(SFTConfig, config_kwargs)
    config = SFTConfig(**config_kwargs)

    # 单进程时钉住单卡（避免 Trainer 走 DataParallel 的段错误）；
    # accelerate launch 下 accelerator 自己管设备，这里自动跳过。
    from agentic_arxiv.training.precision import pin_single_gpu

    pin_single_gpu(config)

    print(describe_logging(backends, logging_dir if backends else None))
    print(f"🚀 开始 SFT 训练...")
    trainer_kwargs = {
        "model": policy,
        "args": config,
        "train_dataset": train_dataset,
        "processing_class": tokenizer,
    }
    trainer = SFTTrainer(**trainer_kwargs)
    execution.validate_trainer(trainer, stage="sft", allow_sharded=not qlora)
    stop_control, _checkpoint = attach_managed_training(
        trainer, managed, execution, resume_from_checkpoint=resume_from_checkpoint,
    )

    trainable, total, ratio = _trainable_parameter_stats(trainer.model)
    if qlora:
        trainable, total, ratio = _assert_lora_only_trainable(trainer.model)
    print(
        f"🔧 参数统计: trainable={trainable:,}, model_parameters={total:,}, "
        f"trainable_ratio={ratio:.4f}%"
    )
    if torch.cuda.is_available():
        print(f"   Trainer初始化显存: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    memory = None

    def save_run_diagnostic(reason: str) -> dict | None:
        """保存实际可得的运行事实；异常记录不产生成功 final。"""
        if not execution.is_main:
            return None
        record = build_run_manifest(
            requested_config=requested_config,
            resolved_config=config.to_dict(),
            derived_values={
                "effective_batch_size": effective_batch,
                "distributed_training": execution.metadata(),
                "memory": memory,
                "trainable_parameters": trainable,
                "model_parameters": total,
                "trainer_planned_steps": getattr(trainer.state, "max_steps", None),
                "data_protocol": read_protocol_metadata(manifest),
                "source_model_protocol": source_protocol,
                "source_model_kind": (
                    "project_training_artifact" if source_protocol is not None else
                    "local_unclassified_input" if Path(resolved_model).is_dir() else "requested_sft_base"
                ),
                **({"training_loss_semantics": "Trainer 原始返回值，不作为跨尝试累计均值或恢复等价证据",
                    "attempt_loss_sum": None, "attempt_mean_loss": None} if managed is not None else {}),
            },
            source_baseline={
                "configuration_baseline": load_mainline_config(config_path)["source_baseline"],
                "execution_source": source_fingerprint(
                    [Path(__file__), Path(__file__).with_name("mainline_config.py"),
                     Path(__file__).with_name("model_setup.py"),
                     Path(__file__).with_name("distributed.py"),
                     Path(__file__).with_name("precision.py")],
                    repo_root=get_workspace(),
                ),
            },
            inputs={
                "model": model_input_fingerprint(resolved_model),
                "data": file_fingerprint(train_data_path),
                "data_manifest": file_fingerprint(
                    workspace_path(data_manifest) if data_manifest
                    else train_data_path.with_suffix(train_data_path.suffix + ".manifest.json")
                ),
                "mainline_configuration": file_fingerprint(config_file_path(config_path)),
            },
            resource_overrides={**execution.resource_overrides(), **{
                key: {"baseline": SFT_DEFAULTS[key], "requested": requested_config[key]}
                for key in ("qlora", "batch_size", "grad_accum", "gradient_checkpointing",
                            "optim", "lora_r", "lora_alpha", "lora_dropout")
                if requested_config[key] != SFT_DEFAULTS[key]
            }},
            requested_steps=max_steps,
            protocol_metadata=current_protocol,
            actual_steps=getattr(trainer.state, "global_step", None),
            stop_reason=reason,
            protections={
                "data_manifest_check": not skip_data_manifest_check,
                "completion_only_loss": True,
                "length_guard": True,
                "verification_enabled": verify,
            },
        )
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "run_manifest.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return record

    training_started = time.monotonic()
    try:
        with preserve_managed_dataloader_epoch(
                trainer, enabled=managed is not None and bool(resume_from_checkpoint)):
            train_result = (trainer.train(resume_from_checkpoint=str(workspace_path(resume_from_checkpoint)))
                            if resume_from_checkpoint else trainer.train())
    except BaseException as exc:
        if stop_control is not None:
            stop_control.mark_failed(exc, getattr(trainer.state, "global_step", None))
        record_training_result(managed, execution, training="failed", validation="not_run",
                               global_step=getattr(trainer.state, "global_step", None),
                               planned_steps=getattr(trainer.state, "max_steps", None),
                               reason=f"training_exception:{type(exc).__name__}",
                               best_effort=True,
                               training_elapsed_seconds=time.monotonic() - training_started)
        try:
            save_run_diagnostic(f"training_exception:{type(exc).__name__}")
        except Exception as record_error:
            print(f"训练异常时未能保存运行诊断: {type(record_error).__name__}")
        raise
    training_elapsed = time.monotonic() - training_started
    train_loss = float(train_result.training_loss)
    if not math.isfinite(train_loss):
        record_training_result(managed, execution, training="failed", validation="not_run",
                               global_step=getattr(trainer.state, "global_step", None),
                               planned_steps=getattr(trainer.state, "max_steps", None),
                               reason="invalid_training_loss", training_elapsed_seconds=training_elapsed,
                               best_effort=True)
        try:
            save_run_diagnostic("invalid_training_loss")
        except Exception as record_error:
            print(f"无效训练 loss 时未能保存运行诊断: {type(record_error).__name__}")
        raise SystemExit(f"❌ training_loss不是有限数: {train_loss}")
    actual_steps = int(getattr(trainer.state, "global_step", 0) or 0)
    planned_steps = getattr(trainer.state, "max_steps", None)
    if managed is not None and (type(planned_steps) is not int or planned_steps < 1):
        record_training_result(managed, execution, training="failed", validation="not_run",
                               global_step=actual_steps, reason="missing_trainer_budget",
                               training_elapsed_seconds=training_elapsed)
        raise ValueError("受管 SFT 缺少 Trainer 实际核定的总优化步数，不能宣称预算完成")
    if managed is None and max_steps > 0:
        planned_steps = max_steps
    stop_reason = (
        "trainer_stopped_early"
        if planned_steps and actual_steps < planned_steps
        else "max_steps_or_epochs_completed"
    )
    if stop_reason == "trainer_stopped_early" and stop_control is not None and stop_control.user_stopped:
        stop_reason = "user_safe_stop"
    memory = execution.memory_summary()
    run_record = execution.run_on_main(lambda: save_run_diagnostic(stop_reason))
    if stop_reason != "max_steps_or_epochs_completed":
        record_training_result(
            managed, execution,
            training="user_safe_stop" if stop_reason == "user_safe_stop" else "protection_stopped",
            validation="not_run", global_step=actual_steps, reason=stop_reason,
            planned_steps=planned_steps,
            checkpoint=stop_control.checkpoint if stop_control is not None else None,
            training_elapsed_seconds=training_elapsed,
        )
        print(f"SFT 训练已停止: {stop_reason}；未导出 final 或运行正常完成验证")
        return

    final_output_dir = out_path / "final"
    trainer.save_model(str(final_output_dir))

    def save_final_artifacts():
        # Trainer.save_model 由所有 rank 参加，其余文件只由主进程写入。
        tokenizer.save_pretrained(str(final_output_dir))
        peak_vram = memory["peak_allocated_gib"]
        if qlora:
            required_adapter_files = (
                final_output_dir / "adapter_config.json",
                final_output_dir / "adapter_model.safetensors",
            )
            missing = [str(path) for path in required_adapter_files if not path.exists()]
            if missing:
                raise SystemExit(f"❌ QLoRA训练结束但适配器文件缺失: {missing}")

        training_manifest = {
            "stage": "sft_qlora" if qlora else "sft_full",
            "base_model": resolved_model,
            "data": str(train_data_path),
            "data_sha256": _sha256_file(train_data_path),
            "data_manifest_kind": manifest.get("kind") if manifest else None,
            "output": str(final_output_dir),
            "max_steps": max_steps,
            "epochs": epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "effective_batch_size": effective_batch,
            "learning_rate": lr,
            "max_length": max_length,
            "optimizer": optim,
            "seed": seed,
            "train_loss": train_loss,
            "trainable_parameters": trainable,
            "model_parameters_seen_by_trainer": total,
            "trainable_ratio_percent": ratio,
            "peak_vram_gib": peak_vram,
            "qlora": {
                "load_in_4bit": qlora,
                "quant_type": "nf4" if qlora else None,
                "double_quant": qlora,
                "compute_dtype": "bfloat16" if qlora else None,
                "r": lora_r if qlora else None,
                "alpha": lora_alpha if qlora else None,
                "dropout": lora_dropout if qlora else None,
                "target_modules": list(QLORA_TARGET_MODULES) if qlora else [],
            },
            "versions": {
                package: version(package)
                for package in ("torch", "transformers", "trl", "peft", "bitsandbytes")
                if qlora or package not in ("peft", "bitsandbytes")
            },
        }
        training_manifest.update(run_record)
        final_output_dir.mkdir(parents=True, exist_ok=True)
        (final_output_dir / "training_manifest.json").write_text(
            json.dumps(training_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"✅ SFT训练完成: loss={train_loss:.6f}, "
            + (f"peak_vram={peak_vram:.2f} GiB, " if peak_vram is not None else "")
            + f"模型已保存: {final_output_dir}"
        )

    execution.run_on_main(save_final_artifacts)
    record_training_result(managed, execution, training="budget_completed",
                           validation="in_progress" if verify else "skipped",
                           global_step=actual_steps, reason=stop_reason,
                           planned_steps=planned_steps,
                           training_elapsed_seconds=training_elapsed)

    # 其余 rank 等待结果；主进程验证失败也必须通知它们，不能挂在 barrier。
    if verify:
        def verify_final():
            print("\n🔍 运行 SFT 阶段验证...")
            verifier = StageVerifier(sft_min_parse_rate=min_parse_rate,
                                     cancel_requested=managed_cancellation_check(managed))
            report = verifier.verify_sft(model_path=str(final_output_dir), env=verification_env)
            verifier.save_report(report, final_output_dir)
            print(report.summary())
            if not report.passed:
                print(
                    "\n⚠️  SFT 阶段验证未通过，但模型已保存。"
                    f"请在继续 GRPO 前检查 {final_output_dir / 'verification_report.json'}。"
                )
            return ("interrupted" if getattr(report, "coverage", {}).get("execution") == "interrupted"
                    else "passed" if report.passed else "failed")

        verification_started = time.monotonic()
        try:
            verification = execution.run_on_main(verify_final)
        except BaseException as exc:
            record_training_result(managed, execution, training="budget_completed",
                                   validation="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                   global_step=actual_steps, reason=f"verification_exception:{type(exc).__name__}",
                                   planned_steps=planned_steps,
                                   best_effort=True,
                                   training_elapsed_seconds=training_elapsed,
                                   stage_validation_elapsed_seconds=time.monotonic() - verification_started)
            raise
        record_training_result(managed, execution, training="budget_completed",
                               validation=verification,
                               global_step=actual_steps, reason=stop_reason,
                               planned_steps=planned_steps,
                               training_elapsed_seconds=training_elapsed,
                               stage_validation_elapsed_seconds=time.monotonic() - verification_started)
