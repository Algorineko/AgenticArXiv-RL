"""SFT 训练脚本（使用 TRL SFTTrainer）

SFT（Supervised Fine-Tuning）：
- 目标：让模型学会基本的工具调用格式
- 数据：expert demonstrations（从 benchmark tasks 生成）
- 输出：SFT 模型（作为 DPO/GRPO 的起点）

使用方式：
    python -m AgenticArxiv.rl.train_sft
    python -m AgenticArxiv.rl.train_sft --model HuggingFaceTB/SmolLM2-135M-Instruct --max_length 3072
"""

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent

# 添加 AgenticArxiv 到 Python 路径
sys.path.insert(0, str(PACKAGE_ROOT))

from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def _precision_flags():
    """见 rl/precision.py：CUDA 上优先 bf16，退回 fp16；CPU / MPS 不开混合精度。"""
    from rl.precision import precision_flags
    return precision_flags()


def _messages_of(row):
    """兼容两种数据格式：{"messages": [...]} 与 {"prompt": [...], "completion": [...]}。"""
    if "messages" in row:
        return list(row["messages"])
    return list(row.get("prompt") or []) + list(row.get("completion") or [])


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
    output_dir: str = "outputs/sft",
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 2e-5,
    max_length: int = 3072,
):
    train_data_path = Path(data) if data else REPO_ROOT / "data" / "sft" / "sft_train.jsonl"
    out_path = REPO_ROOT / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)

    print(f"📦 加载模型: {model}")
    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(model)

    print(f"📚 加载 SFT 数据集: {train_data_path}")
    if not train_data_path.exists():
        raise SystemExit(
            f"❌ 数据集不存在: {train_data_path}\n"
            f"   请先运行: python scripts/generate_sft_data.py"
        )

    train_dataset = load_dataset("json", data_files=str(train_data_path), split="train")
    print(f"   样本数: {len(train_dataset)}")

    # --- 长度守卫 ---
    _check_lengths(tokenizer, train_dataset, max_length)

    config = SFTConfig(
        output_dir=str(out_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        max_length=max_length,    # TRL>=0.20 用 max_length（旧名 max_seq_length 已移除）
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        **_precision_flags(),     # 只有 CUDA 才开 fp16
    )

    print(f"🚀 开始 SFT 训练...")
    trainer = SFTTrainer(
        model=policy,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,   # TRL>=0.13 用 processing_class（旧名 tokenizer 已移除）
    )
    trainer.train()

    final_output_dir = out_path / "final"
    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))
    print(f"✅ SFT 训练完成，模型已保存: {final_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT 训练")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data", default=None)
    parser.add_argument("--output_dir", default="outputs/sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--max_length", type=int, default=3072,
        help="超过此长度的样本会被从右截断，而右边正是 assistant 目标；"
             "训练前会做长度体检，不匹配直接报错",
    )
    main(**vars(parser.parse_args()))
