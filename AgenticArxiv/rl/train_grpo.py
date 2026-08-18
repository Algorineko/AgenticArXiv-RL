"""GRPO 训练脚本（使用 TRL GRPOTrainer）

GRPO（Group Relative Policy Optimization）：
- 目标：用 verifiable reward 在线训练
- 优势：无需 value model（比 PPO 更轻量）
- 数据：在线 rollout + reward 计算

使用方式：
    python -m AgenticArxiv.rl.train_grpo
"""

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent

# 添加 AgenticArxiv 到 Python 路径
sys.path.insert(0, str(PACKAGE_ROOT))

from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from rl.reward import RewardCalculator


def main():
    """GRPO 训练主函数"""

    # 1. 配置
    model_path = REPO_ROOT / "outputs" / "dpo" / "final"  # 从 DPO 模型继续
    model_name = str(model_path)
    output_dir = REPO_ROOT / "outputs" / "grpo"

    print(f"📦 加载 DPO 模型: {model_name}")
    if not model_path.exists():
        print(f"❌ DPO 模型不存在: {model_path}")
        print(f"请先运行: python -m AgenticArxiv.rl.train_dpo")
        print(f"或者直接从 SFT 模型开始: 修改 model_path = REPO_ROOT / 'outputs' / 'sft' / 'final'")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)

    # 2. 定义 reward_model
    reward_calc = RewardCalculator()

    def reward_fn(responses, prompts):
        """
        GRPO reward 函数

        Args:
            responses: 模型生成的响应列表
            prompts: 对应的 prompt 列表

        Returns:
            List[float]: 每个响应的奖励
        """
        # TODO: 实现 reward 计算逻辑
        # 1. 解析 responses 为 action
        # 2. 执行工具得到 observation
        # 3. 调用 reward_calc.compute_reward()
        # 4. 返回 reward 列表

        # 占位实现
        return [0.0 for _ in responses]

    # 3. 配置 GRPO
    config = GRPOConfig(
        model_name=model_name,
        learning_rate=1e-5,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        num_sample_generations=4,  # 每个 prompt 采样 4 条 trajectory
    )

    # 4. 训练
    print(f"🚀 开始 GRPO 训练...")
    print(f"⚠️  GRPO 训练脚本骨架已就绪，需完成 reward_fn TODO")

    # trainer = GRPOTrainer(
    #     model=model,
    #     ref_model=ref_model,
    #     args=config,
    #     reward_model=reward_fn,
    #     tokenizer=tokenizer,
    # )
    # trainer.train()

    # 5. 保存
    # final_output_dir = output_dir / "final"
    # trainer.save_model(str(final_output_dir))
    # print(f"✅ GRPO 训练完成，模型已保存: {final_output_dir}")


if __name__ == "__main__":
    main()
