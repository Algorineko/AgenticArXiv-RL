"""DPO 训练脚本（使用 TRL DPOTrainer）

DPO（Direct Preference Optimization）：
- 目标：让模型偏好正确的工具选择，拒绝错误路由
- 数据：chosen/rejected 对（从 SFT 模型 rollout 生成）
- 输出：DPO 模型（作为 GRPO 的起点）

使用方式：
    python -m rl.train_dpo
"""

import sys
from pathlib import Path

# 添加 AgenticArxiv 到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def _precision_flags():
    """只有 CUDA 才开 fp16；CPU / Apple MPS 上开 fp16 会训练失败。"""
    import torch
    return {"fp16": True} if torch.cuda.is_available() else {}


def main():
    """DPO 训练主函数"""

    # 1. 配置
    model_name = "./outputs/sft/final"  # 从 SFT 模型继续
    train_data_path = "data/dpo/dpo_train.jsonl"
    output_dir = "./outputs/dpo"

    print(f"📦 加载 SFT 模型: {model_name}")
    if not Path(model_name).exists():
        print(f"❌ SFT 模型不存在: {model_name}")
        print(f"请先运行: python -m rl.train_sft")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)  # reference model

    # 2. 加载 DPO 数据集
    print(f"📚 加载 DPO 数据集: {train_data_path}")
    if not Path(train_data_path).exists():
        print(f"❌ 数据集不存在: {train_data_path}")
        print(f"请先运行: python scripts/generate_dpo_data.py")
        return

    train_dataset = load_dataset("json", data_files=train_data_path, split="train")
    print(f"   样本数: {len(train_dataset)}")

    # 3. 配置 DPO
    config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        beta=0.1,  # DPO 温度系数
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        **_precision_flags(),     # 只有 CUDA 才开 fp16
    )

    # 4. 训练
    print(f"🚀 开始 DPO 训练...")
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,   # TRL>=0.13 用 processing_class（旧名 tokenizer 已移除）
    )
    trainer.train()

    # 5. 保存
    final_output_dir = f"{output_dir}/final"
    trainer.save_model(final_output_dir)
    print(f"✅ DPO 训练完成，模型已保存: {final_output_dir}")


if __name__ == "__main__":
    main()
