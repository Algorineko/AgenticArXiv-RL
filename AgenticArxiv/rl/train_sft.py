"""SFT 训练脚本（使用 TRL SFTTrainer）

SFT（Supervised Fine-Tuning）：
- 目标：让模型学会基本的工具调用格式
- 数据：expert demonstrations（从 benchmark tasks 生成）
- 输出：SFT 模型（作为 DPO/GRPO 的起点）

使用方式：
    python -m rl.train_sft
"""

import sys
from pathlib import Path

# 添加 AgenticArxiv 到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def _precision_flags():
    """只有 CUDA 才开 fp16；CPU / Apple MPS 上开 fp16 会训练失败。"""
    import torch
    return {"fp16": True} if torch.cuda.is_available() else {}


def main():
    """SFT 训练主函数"""

    # 1. 配置
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # 基座模型
    train_data_path = "data/sft/sft_train.jsonl"
    output_dir = "./outputs/sft"

    print(f"📦 加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 2. 加载 SFT 数据集
    print(f"📚 加载 SFT 数据集: {train_data_path}")
    if not Path(train_data_path).exists():
        print(f"❌ 数据集不存在: {train_data_path}")
        print(f"请先运行: python scripts/generate_sft_data.py")
        return

    train_dataset = load_dataset("json", data_files=train_data_path, split="train")
    print(f"   样本数: {len(train_dataset)}")

    # 3. 配置 SFT
    config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        max_length=1024,          # TRL>=0.20 用 max_length（旧名 max_seq_length 已移除）
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        **_precision_flags(),     # 只有 CUDA 才开 fp16
    )

    # 4. 训练
    print(f"🚀 开始 SFT 训练...")
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,   # TRL>=0.13 用 processing_class（旧名 tokenizer 已移除）
    )
    trainer.train()

    # 5. 保存
    final_output_dir = f"{output_dir}/final"
    trainer.save_model(final_output_dir)
    print(f"✅ SFT 训练完成，模型已保存: {final_output_dir}")


if __name__ == "__main__":
    main()
