"""训练共享的产物加载、QLoRA 配置、硬件保护与参数统计。"""

from pathlib import Path
from typing import Optional

from agentic_arxiv.config import resolve_model_source
from agentic_arxiv.llm.artifacts import adapter_base_model, artifact_metadata
from agentic_arxiv.training.distributed import TrainingContext
from agentic_arxiv.training.precision import quantized_model_device_map

from importlib.metadata import version

QLORA_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def _qlora_runtime_guard() -> None:
    """在加载3GB模型前检查GPU和关键依赖，避免晚失败。"""
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("❌ QLoRA 需要 CUDA；当前 torch.cuda.is_available()=False")
    from agentic_arxiv.training.precision import cuda_supports_bf16

    # torch>=2.3 的 is_bf16_supported() 默认连“模拟”也算支持，T4 也返回 True，
    # 然后在第一次 matmul 才崩；这里要的是原生 bf16 计算（见 rl/precision.py）。
    if not cuda_supports_bf16():
        raise SystemExit("❌ 当前GPU不支持BF16；本项目冻结的4090配置要求BF16计算")
    required = {"transformers": "4.57.6", "trl": "0.29.1", "peft": "0.17.1", "bitsandbytes": "0.45.5"}
    installed = {}
    for package in required:
        try:
            installed[package] = version(package)
        except Exception as exc:
            raise SystemExit(f"❌ QLoRA依赖缺失: {package}: {exc}") from exc
    print("🧩 QLoRA依赖: " + ", ".join(f"{k}={v}" for k, v in installed.items()))
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    print(
        f"🎮 GPU {device}: {torch.cuda.get_device_name(device)} | "
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


def load_training_policy(
    model_path: str,
    *,
    qlora: bool,
    base_model: Optional[str],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    allowed_stages=None,
    expected_protocol=None,
    local_base_only: bool = False,
    model_dtype=None,
):
    """按全参/适配器产物合同加载，QLoRA 只是资源策略。"""
    import torch
    from transformers import AutoModelForCausalLM

    artifact_metadata(
        model_path, base_model=base_model, allowed_stages=allowed_stages,
        expected_protocol=expected_protocol, local_base_only=local_base_only,
    )
    local = Path(resolve_model_source(model_path, local_only=True))
    model_path = str(local)
    is_adapter = local.is_dir() and (local / "adapter_config.json").exists()
    if not is_adapter and base_model:
        raise SystemExit("全参 SFT 产物不能用 --base-model 替换训练起点")
    TrainingContext.from_environment().prepare_device(qlora=qlora)
    if not qlora:
        print("⚠️  --no-qlora：加载完整精度策略，不属于单张4090冻结方案")
        source = adapter_base_model(local, base_model, local_only=local_base_only) if is_adapter else model_path
        # GRPO 默认仍不传 dtype；DPO 可显式遵循共享精度策略。
        dtype_kwargs = {"dtype": model_dtype} if model_dtype is not None else {}
        policy = AutoModelForCausalLM.from_pretrained(
            source, local_files_only=local_base_only or Path(source).exists(),
            **dtype_kwargs,
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

    source = adapter_base_model(local, base_model, local_only=local_base_only) if is_adapter else model_path
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
        local_files_only=local_base_only or source_path.exists(),
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


def load_reference_policy(
    model_path: str,
    *,
    qlora: bool,
    base_model: Optional[str] = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    allowed_stages=None,
    expected_protocol=None,
    local_base_only: bool = True,
    model_dtype=None,
):
    """独立加载完整 SFT 参考；不新建 adapter，也不关闭已有 SFT adapter。"""
    import torch
    from transformers import AutoModelForCausalLM

    artifact_metadata(
        model_path, base_model=base_model, allowed_stages=allowed_stages,
        expected_protocol=expected_protocol, local_base_only=local_base_only,
    )
    local = Path(resolve_model_source(model_path, local_only=True))
    is_adapter = (local / "adapter_config.json").is_file()
    if not is_adapter and base_model:
        raise SystemExit("全参 SFT 产物不能用 --base-model 替换训练起点")
    TrainingContext.from_environment().prepare_device(qlora=qlora)
    source = (adapter_base_model(local, base_model, local_only=local_base_only)
              if is_adapter else str(local))
    kwargs = {"local_files_only": local_base_only or Path(source).exists()}
    if not qlora and model_dtype is not None:
        kwargs["dtype"] = model_dtype
    if qlora:
        _qlora_runtime_guard()
        quantization, _ = _build_qlora_configs(lora_r, lora_alpha, lora_dropout)
        kwargs.update(
            quantization_config=quantization, device_map=quantized_model_device_map(),
            dtype=torch.bfloat16, low_cpu_mem_usage=True,
        )
    reference = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    if is_adapter:
        from peft import PeftModel

        reference = PeftModel.from_pretrained(
            reference, str(local), is_trainable=False, local_files_only=True,
        )
        if not qlora:
            reference = reference.merge_and_unload()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    reference.config.use_cache = False
    reference.eval()
    return reference
