"""SFT/GRPO 共享的 QLoRA 配置、硬件保护与参数统计。"""

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
