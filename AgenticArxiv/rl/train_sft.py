"""SFT/QLoRA 训练脚本（使用 TRL SFTTrainer）

SFT（Supervised Fine-Tuning）：
- 目标：让模型学会基本的工具调用格式
- 数据：expert demonstrations（从 benchmark tasks 生成）
- 输出：SFT 模型（作为 DPO/GRPO 的起点）

使用方式：
    python -m AgenticArxiv.rl.train_sft --inspect_only --max_length 4096
    python -m AgenticArxiv.rl.train_sft --max_steps 30 --max_length 4096 --no-verify
    python -m AgenticArxiv.rl.train_sft --verify --min_parse_rate 0.5
"""

import argparse
import dataclasses
import hashlib
import json
import math
import sys
from importlib.metadata import version
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent

# 添加 AgenticArxiv 到 Python 路径
sys.path.insert(0, str(PACKAGE_ROOT))

# 先做旧 torch 的 trl 兼容（torch<2.6 时兜底 FSDPModule），再 import trl
from rl import trl_compat  # noqa: F401
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from rl.observability import describe_logging, resolve_report_to
from rl.stage_verifier import StageVerifier


QLORA_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


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
            "临时实验可显式传 --skip_data_manifest_check"
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


def _qlora_runtime_guard() -> None:
    """在加载3GB模型前检查GPU和关键依赖，避免晚失败。"""
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("❌ QLoRA 需要 CUDA；当前 torch.cuda.is_available()=False")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("❌ 当前GPU不支持BF16；本项目冻结的4090配置要求BF16计算")
    required = {"transformers": "4.57.6", "trl": "0.29.1", "peft": "0.17.1", "bitsandbytes": "0.45.5"}
    installed = {}
    for package in required:
        try:
            installed[package] = version(package)
        except Exception as exc:
            raise SystemExit(f"❌ QLoRA依赖缺失: {package}: {exc}") from exc
    print("🧩 QLoRA依赖: " + ", ".join(f"{k}={v}" for k, v in installed.items()))
    free, total = torch.cuda.mem_get_info()
    print(
        f"🎮 GPU: {torch.cuda.get_device_name(0)} | "
        f"空闲 {free / 2**30:.2f} / {total / 2**30:.2f} GiB"
    )
    if total < 20 * 2**30:
        raise SystemExit("❌ 本配置要求至少20GiB显存")


def _build_qlora_configs(r: int, alpha: int, dropout: float):
    """返回量化与LoRA配置；k-bit准备交给SFTTrainer，只做一次。"""
    import torch
    from peft import LoraConfig
    from transformers import BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    peft = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(QLORA_TARGET_MODULES),
    )
    return quantization, peft


def _trainable_parameter_stats(model) -> tuple[int, int, float]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ratio = 100.0 * trainable / total if total else 0.0
    return trainable, total, ratio


def _assert_lora_only_trainable(model) -> tuple[int, int, float]:
    unexpected = [
        name for name, param in model.named_parameters()
        if param.requires_grad and "lora_" not in name
    ]
    if unexpected:
        raise SystemExit(
            "❌ QLoRA中出现非LoRA可训练参数，例如: " + ", ".join(unexpected[:5])
        )
    stats = _trainable_parameter_stats(model)
    if stats[0] == 0:
        raise SystemExit("❌ 没有任何可训练LoRA参数")
    if not getattr(model, "is_loaded_in_4bit", False):
        raise SystemExit("❌ 模型没有以4-bit加载，拒绝把全参数训练误称为QLoRA")
    return stats


def _precision_flags():
    """见 rl/precision.py：CUDA 上优先 bf16，退回 fp16；CPU / MPS 不开混合精度。"""
    from rl.precision import precision_flags
    return precision_flags()


def _messages_of(row):
    """兼容两种数据格式：{"messages": [...]} 与 {"prompt": [...], "completion": [...]}。"""
    if "messages" in row:
        return list(row["messages"])
    return list(row.get("prompt") or []) + list(row.get("completion") or [])


def _to_prompt_completion(row):
    """把单轮 messages 样本转换成 TRL 的 prompt-completion 格式。

    项目的 assistant message 只包含 Action。显式拆分 prompt/completion 后，
    ``completion_only_loss=True`` 可以在不依赖模型 chat template 是否提供
    assistant mask 的情况下，仅监督 Action。
    """
    messages = list(row["messages"])
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("SFT 样本必须以 assistant message 结尾")
    return {
        "prompt": messages[:-1],
        "completion": messages[-1:],
    }


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
            f"   请改用 --max_length {longest + 64}（或缩短 prompt 中的工具描述）"
        )
    if over:
        print(f"   ⚠️  {over}/{total} 个样本超过 max_length={max_length}，这部分会被截断")


def main(
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    data: str = None,
    data_manifest: str = None,
    output_dir: str = "outputs/sft_qlora",
    epochs: int = 3,
    batch_size: int = 1,
    grad_accum: int = 8,
    lr: float = 1e-4,
    max_length: int = 4096,
    max_steps: int = -1,
    qlora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    gradient_checkpointing: bool = True,
    optim: str = "paged_adamw_8bit",
    inspect_only: bool = False,
    skip_data_manifest_check: bool = False,
    seed: int = 42,
    verify: bool = False,
    min_parse_rate: float = 0.3,
    report_to: str = "none",
    run_name: str = None,
):
    import torch

    train_data_path = (
        Path(data) if data
        else REPO_ROOT / "data" / "sft" / "sft_v3_train_mix.jsonl"
    )
    out_path = REPO_ROOT / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)
    # 先校验日志后端再加载模型：参数写错时应立刻失败
    backends = resolve_report_to(report_to)
    logging_dir = str(out_path / "logs")

    print(f"📚 加载 SFT 数据集: {train_data_path}")
    if not train_data_path.exists():
        raise SystemExit(
            f"❌ 数据集不存在: {train_data_path}\n"
            f"   请先运行: python scripts/build_sft_train_mix.py"
        )
    manifest = None
    if not skip_data_manifest_check:
        manifest_path = Path(data_manifest) if data_manifest else None
        manifest = _verify_data_manifest(train_data_path, manifest_path)
    else:
        print("⚠️  已跳过数据manifest校验；该运行不能作为正式可复现实验")

    resolved_model = str(Path(model).resolve()) if Path(model).exists() else model
    print(f"📦 加载 tokenizer: {resolved_model}")
    tokenizer = AutoTokenizer.from_pretrained(resolved_model, local_files_only=Path(model).exists())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = load_dataset("json", data_files=str(train_data_path), split="train")
    if "messages" in train_dataset.column_names:
        train_dataset = train_dataset.map(
            _to_prompt_completion,
            remove_columns=["messages"],
            desc="Converting SFT data to prompt-completion format",
        )
    print(f"   样本数: {len(train_dataset)}")

    # --- 长度守卫 ---
    _check_lengths(tokenizer, train_dataset, max_length)

    effective_batch = batch_size * grad_accum
    print(
        f"🧮 训练计划: rows={len(train_dataset)}, micro_batch={batch_size}, "
        f"grad_accum={grad_accum}, effective_batch={effective_batch}, "
        f"max_length={max_length}, max_steps={max_steps}, epochs={epochs}"
    )
    if inspect_only:
        print("✅ inspect_only 完成：数据、manifest、token长度均通过；尚未加载模型或占用训练显存")
        return

    quantization_config = None
    peft_config = None
    policy = resolved_model
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
            device_map={"": 0},
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=Path(model).exists(),
        )
        # TRL 0.29.1 会按 SFTConfig 启用 gradient checkpointing 和 input grads；
        # 此处只做一次 k-bit 冻结/LayerNorm准备，不重复打开 checkpointing。
        policy = prepare_model_for_kbit_training(
            policy, use_gradient_checkpointing=False
        )
        policy = get_peft_model(policy, peft_config)
    else:
        print("⚠️  --no-qlora：将加载完整模型；该运行不属于本项目冻结的4090方案")
        policy = AutoModelForCausalLM.from_pretrained(resolved_model)
        if optim == "paged_adamw_8bit":
            optim = "adamw_torch_fused"

    # KV cache 用于自回归生成，会保留每层历史 K/V；训练时既占显存，又与
    # gradient checkpointing 的重计算策略冲突，因此在模型侧统一关闭。
    policy.config.use_cache = False

    # use_cache 属于模型 forward/generation 配置，不是 TRL 0.29.1 的 SFTConfig
    # 字段。QLoRA 分支已在 policy.config.use_cache=False 关闭 KV cache，以兼容
    # gradient checkpointing；不要再把它传给 TrainingArguments。
    config_kwargs = {
        "output_dir": str(out_path),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "max_length": max_length,  # TRL>=0.20 用 max_length
        "completion_only_loss": True,  # prompt 不计 loss，只监督 assistant Action
        "max_steps": max_steps,
        "logging_steps": 1,
        "logging_first_step": True,
        "save_steps": 50,
        "save_total_limit": 3,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "optim": optim,
        "gradient_checkpointing": gradient_checkpointing,
        "gradient_checkpointing_kwargs": (
            {"use_reentrant": False} if gradient_checkpointing else None
        ),
        "seed": seed,
        "data_seed": seed,
        "dataloader_num_workers": 0,
        "dataset_num_proc": 1,
        "packing": False,
        "report_to": backends,
        "run_name": run_name or out_path.name,
        **_precision_flags(),  # 只有 CUDA 才开混合精度
    }
    config_kwargs, dropped_config_keys = _filter_dataclass_kwargs(
        SFTConfig, config_kwargs
    )
    if dropped_config_keys:
        print(
            "⚠️  当前TRL版本不支持以下SFTConfig参数，已忽略: "
            + ", ".join(dropped_config_keys)
        )
    config = SFTConfig(**config_kwargs)

    print(describe_logging(backends, logging_dir if backends else None))
    print(f"🚀 开始 SFT 训练...")
    trainer_kwargs = {
        "model": policy,
        "args": config,
        "train_dataset": train_dataset,
        "processing_class": tokenizer,
    }
    trainer = SFTTrainer(**trainer_kwargs)

    trainable, total, ratio = _trainable_parameter_stats(trainer.model)
    if qlora:
        trainable, total, ratio = _assert_lora_only_trainable(trainer.model)
    print(
        f"🔧 参数统计: trainable={trainable:,}, model_parameters={total:,}, "
        f"trainable_ratio={ratio:.4f}%"
    )
    if torch.cuda.is_available():
        print(f"   Trainer初始化显存: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    train_result = trainer.train()
    train_loss = float(train_result.training_loss)
    if not math.isfinite(train_loss):
        raise SystemExit(f"❌ training_loss不是有限数: {train_loss}")

    final_output_dir = out_path / "final"
    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))
    if qlora:
        required_adapter_files = (
            final_output_dir / "adapter_config.json",
            final_output_dir / "adapter_model.safetensors",
        )
        missing = [str(path) for path in required_adapter_files if not path.exists()]
        if missing:
            raise SystemExit(f"❌ QLoRA训练结束但适配器文件缺失: {missing}")

    peak_vram = (
        torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    )
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

    # --- 阶段验证：检查模型是否能产出可解析的输出 ---
    if verify:
        print(f"\n🔍 运行 SFT 阶段验证...")
        verifier = StageVerifier(sft_min_parse_rate=min_parse_rate)
        report = verifier.verify_sft(model_path=str(final_output_dir))
        verifier.save_report(report, final_output_dir)
        print(report.summary())
        if not report.passed:
            print(
                f"\n⚠️  SFT 阶段验证未通过，但模型已保存。"
                f"请在继续 DPO/GRPO 前检查 {final_output_dir / 'verification_report.json'}。"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT / 单卡4-bit QLoRA训练")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data", default=None)
    parser.add_argument("--data_manifest", default=None)
    parser.add_argument("--output_dir", default="outputs/sft_qlora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--max_length", type=int, default=4096,
        help="超过此长度的样本会被从右截断，而右边正是 assistant 目标；"
             "训练前会做长度体检，不匹配直接报错",
    )
    parser.add_argument(
        "--max_steps", type=int, default=-1,
        help="限制优化步数；先用30做显存冒烟，-1表示按epochs完整训练",
    )
    parser.add_argument(
        "--qlora", action=argparse.BooleanOptionalAction, default=True,
        help="默认启用4-bit QLoRA；--no-qlora是全参数路径，不属于冻结的4090方案",
    )
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument("--optim", default="paged_adamw_8bit")
    parser.add_argument(
        "--inspect_only", action="store_true",
        help="只审计manifest和token长度，不加载模型、不开始训练",
    )
    parser.add_argument(
        "--skip_data_manifest_check", action="store_true",
        help="允许临时数据运行；结果不得作为正式可复现实验",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--verify", action=argparse.BooleanOptionalAction, default=False,
        help="训练结束后运行阶段验证（默认关闭；冒烟可显式写 --no-verify）",
    )
    parser.add_argument(
        "--min_parse_rate", type=float, default=0.3,
        help="SFT 验证的最低可解析率阈值（默认 0.3）",
    )
    parser.add_argument(
        "--report_to", default="none",
        help="训练曲线记到哪：none / auto / tensorboard / wandb（可逗号分隔）",
    )
    parser.add_argument(
        "--run_name", default=None,
        help="本次运行在 TensorBoard / wandb 里的名字，默认取 output_dir 末段",
    )
    main(**vars(parser.parse_args()))
