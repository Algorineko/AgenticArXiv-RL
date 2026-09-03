"""DPO 训练脚本（使用 TRL DPOTrainer）

DPO（Direct Preference Optimization）：
- 目标：让模型偏好正确的工具选择，拒绝错误路由
- 数据：chosen/rejected 对（从 SFT 模型 rollout 生成）
- 输出：DPO 模型（作为 GRPO 的起点）

使用方式：
    python -m AgenticArxiv.rl.train_dpo
    python -m AgenticArxiv.rl.train_dpo --verify --min_reward -0.2
"""

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent

# 添加 AgenticArxiv 到 Python 路径
sys.path.insert(0, str(PACKAGE_ROOT))

# 先做旧 torch 的 trl 兼容（torch<2.6 时兜底 FSDPModule），再 import trl
from rl import trl_compat  # noqa: F401
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from rl.observability import describe_logging, resolve_report_to
from rl.stage_verifier import StageVerifier


def _precision_flags():
    """见 rl/precision.py：CUDA 上优先 bf16，退回 fp16；CPU / MPS 不开混合精度。"""
    from rl.precision import precision_flags
    return precision_flags()


def main(
    model: str = None,
    data: str = None,
    output_dir: str = None,
    verify: bool = False,
    min_reward: float = -0.3,
    report_to: str = "none",
    run_name: str = None,
):
    """DPO 训练主函数"""

    # 先校验日志后端再加载模型：参数写错时应立刻失败
    backends = resolve_report_to(report_to)
    # 1. 配置
    model_path = Path(model) if model else REPO_ROOT / "outputs" / "sft" / "final"
    model_name = str(model_path)
    train_data_path = Path(data) if data else REPO_ROOT / "data" / "dpo" / "dpo_train.jsonl"
    out_dir = Path(output_dir) if output_dir else REPO_ROOT / "outputs" / "dpo"

    print(f"📦 加载 SFT 模型: {model_name}")
    if not model_path.exists():
        print(f"❌ SFT 模型不存在: {model_path}")
        print(f"请先运行: python -m AgenticArxiv.rl.train_sft")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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
    logging_dir = str(Path(out_dir) / "logs")
    print(describe_logging(backends, logging_dir if backends else None))

    config = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        beta=0.1,  # DPO 温度系数
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        report_to=backends,
        run_name=run_name or Path(out_dir).name,
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
    final_output_dir = out_dir / "final"
    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))
    print(f"✅ DPO 训练完成，模型已保存: {final_output_dir}")

    # --- 阶段验证：检查模型是否达到最低奖励阈值 ---
    if verify:
        print(f"\n🔍 运行 DPO 阶段验证...")
        verifier = StageVerifier(dpo_min_reward=min_reward)
        report = verifier.verify_dpo(model_path=str(final_output_dir))
        verifier.save_report(report, final_output_dir)
        print(report.summary())
        if not report.passed:
            print(
                f"\n⚠️  DPO 阶段验证未通过，但模型已保存。"
                f"请在继续 GRPO 前检查 {final_output_dir / 'verification_report.json'}。"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO 训练")
    parser.add_argument("--model", default=None, help="SFT 模型路径")
    parser.add_argument("--data", default=None, help="DPO 数据集路径")
    parser.add_argument("--output_dir", default=None, help="输出目录")
    parser.add_argument(
        "--verify", action="store_true", default=False,
        help="训练结束后运行阶段验证（检查模型奖励是否达标）",
    )
    parser.add_argument(
        "--min_reward", type=float, default=-0.3,
        help="DPO 验证的最低平均奖励阈值（默认 -0.3）",
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
