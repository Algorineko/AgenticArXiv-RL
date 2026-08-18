"""DPO 训练脚本（使用 TRL DPOTrainer）

DPO（Direct Preference Optimization）：
- 目标：让模型偏好正确的工具选择，拒绝错误路由
- 数据：chosen/rejected 对（从 SFT 模型 rollout 生成）
- 输出：DPO 模型（作为 GRPO 的起点）

使用方式：
    python -m AgenticArxiv.rl.train_dpo
"""

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent

# 添加 AgenticArxiv 到 Python 路径
sys.path.insert(0, str(PACKAGE_ROOT))

from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def main():
    """DPO 训练主函数"""

    # 1. 配置
    model_path = REPO_ROOT / "outputs" / "sft" / "final"  # 从 SFT 模型继续
    model_name = str(model_path)
    train_data_path = REPO_ROOT / "data" / "dpo" / "dpo_train.jsonl"
    output_dir = REPO_ROOT / "outputs" / "dpo"

    print(f"📦 加载 SFT 模型: {model_name}")
    if not model_path.exists():
        print(f"❌ SFT 模型不存在: {model_path}")
        print(f"请先运行: python -m AgenticArxiv.rl.train_sft")
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

    train_dataset = load_dataset("json", data_files=str(train_data_path), split="train")
    print(f"   样本数: {len(train_dataset)}")

    # 3. 配置 DPO
    config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        beta=0.1,  # DPO 温度系数
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        fp16=True,
    )

    # 4. 训练
    print(f"🚀 开始 DPO 训练...")
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()

    # 5. 保存
    final_output_dir = output_dir / "final"
    trainer.save_model(str(final_output_dir))
    print(f"✅ DPO 训练完成，模型已保存: {final_output_dir}")


if __name__ == "__main__":
    main()
