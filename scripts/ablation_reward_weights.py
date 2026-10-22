#!/usr/bin/env python3
"""多粒度奖励权重消融实验脚本

基于当前最佳 8B GRPO 模型（strict 72.8%），系统性验证五分量奖励权重配置
对训练效果的影响。实验设计遵循单变量控制原则，每次仅改变一个维度。

实验组设计：
1. Baseline：当前默认权重 (format=1, tool=3, argument=2, process=1, outcome=3)
2. Uniform：均匀权重 (1, 1, 1, 1, 1) - 验证差异化权重的必要性
3. OutcomeOnly：仅保留最终目标 (0, 0, 0, 0, 3) - 对比稀疏奖励
4. NoOutcome：移除最终目标 (1, 3, 2, 1, 0) - 验证 outcome 的重要性
5. StrongFormat：增强格式正确性 (3, 3, 2, 1, 3) - 强化结构化行为
6. ToolFocus：工具序列为主 (1, 5, 1, 1, 3) - 强化工具选择准确性
7. ArgumentFocus：参数匹配为主 (1, 3, 4, 1, 3) - 强化参数传递准确性
8. ProcessRich：密集过程反馈 (1, 3, 2, 3, 3) - 强化中间步骤质量
9. NoCurriculum：无课程学习 (1, 3, 2, 1, 3, curriculum_steps=0)

用法：
    python scripts/ablation_reward_weights.py --base_model outputs/sft/final
    python scripts/ablation_reward_weights.py --quick_test  # 快速测试（10 steps）
    python scripts/ablation_reward_weights.py --dry_run    # 仅生成配置，不训练
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "AgenticArxiv"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


@dataclass
class RewardWeightConfig:
    """奖励权重配置"""
    name: str
    description: str
    format: float
    tool: float
    argument: float
    process: float
    outcome: float
    curriculum_steps: int = 30

    def to_config_dict(self) -> Dict:
        """生成完整的配置字典，包含 weights 子字段"""
        return {
            **asdict(self),
            "weights": {
                "format": self.format,
                "tool": self.tool,
                "argument": self.argument,
                "process": self.process,
                "outcome": self.outcome,
            }
        }


# 定义消融实验组
ABLATION_CONFIGS = [
    RewardWeightConfig(
        name="baseline",
        description="当前默认配置（基线）",
        format=1.0, tool=3.0, argument=2.0, process=1.0, outcome=3.0,
        curriculum_steps=30
    ),
    RewardWeightConfig(
        name="uniform",
        description="均匀权重 - 验证差异化权重的必要性",
        format=1.0, tool=1.0, argument=1.0, process=1.0, outcome=1.0,
        curriculum_steps=30
    ),
    RewardWeightConfig(
        name="outcome_only",
        description="仅保留最终目标 - 对比稀疏奖励",
        format=0.0, tool=0.0, argument=0.0, process=0.0, outcome=3.0,
        curriculum_steps=30
    ),
    RewardWeightConfig(
        name="no_outcome",
        description="移除最终目标 - 验证 outcome 的重要性",
        format=1.0, tool=3.0, argument=2.0, process=1.0, outcome=0.0,
        curriculum_steps=30
    ),
    RewardWeightConfig(
        name="strong_format",
        description="增强格式正确性 - 强化结构化行为",
        format=3.0, tool=3.0, argument=2.0, process=1.0, outcome=3.0,
        curriculum_steps=30
    ),
    RewardWeightConfig(
        name="tool_focus",
        description="工具序列为主 - 强化工具选择准确性",
        format=1.0, tool=5.0, argument=1.0, process=1.0, outcome=3.0,
        curriculum_steps=30
    ),
    RewardWeightConfig(
        name="argument_focus",
        description="参数匹配为主 - 强化参数传递准确性",
        format=1.0, tool=3.0, argument=4.0, process=1.0, outcome=3.0,
        curriculum_steps=30
    ),
    RewardWeightConfig(
        name="process_rich",
        description="密集过程反馈 - 强化中间步骤质量",
        format=1.0, tool=3.0, argument=2.0, process=3.0, outcome=3.0,
        curriculum_steps=30
    ),
    RewardWeightConfig(
        name="no_curriculum",
        description="无课程学习 - 验证课程学习的必要性",
        format=1.0, tool=3.0, argument=2.0, process=1.0, outcome=3.0,
        curriculum_steps=0
    ),
]


def run_single_experiment(
    config: RewardWeightConfig,
    base_model: str,
    output_dir: Path,
    num_steps: int = 100,
    num_generations: int = 4,
    task_set: str = "expanded",
    split: str = "rl_train",
    dry_run: bool = False,
) -> Dict:
    """运行单个消融实验"""
    exp_dir = output_dir / config.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置（包含 weights 子字段供 train_grpo.py 读取）
    config_path = exp_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config.to_config_dict(), f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"实验组: {config.name}")
    print(f"描述: {config.description}")
    print(f"权重: format={config.format}, tool={config.tool}, argument={config.argument}, "
          f"process={config.process}, outcome={config.outcome}")
    print(f"课程学习步数: {config.curriculum_steps}")
    print(f"输出目录: {exp_dir}")
    print(f"配置文件: {config_path}")
    print(f"{'='*60}")

    if dry_run:
        print("🔍 Dry run 模式，跳过实际训练")
        print(f"✅ 配置已保存至: {config_path}")
        return {"status": "dry_run", "config": asdict(config), "config_path": str(config_path)}

    # 构建训练命令（使用 --reward_config 参数）
    cmd = [
        sys.executable, "-m", "AgenticArxiv.rl.train_grpo",
        "--model", base_model,
        "--output_dir", str(exp_dir),
        "--max_steps", str(num_steps),
        "--num_generations", str(num_generations),
        "--reward_curriculum_steps", str(config.curriculum_steps),
        "--reward_config", str(config_path),  # 传入配置文件路径
        "--task_set", task_set,
        "--split", split,
        "--save_steps", str(num_steps),
        "--logging_steps", "10",
    ]

    print(f"\n🚀 训练命令: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=3600
        )

        # 保存日志
        log_path = exp_dir / "training.log"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"=== STDOUT ===\n{result.stdout}\n\n")
            f.write(f"=== STDERR ===\n{result.stderr}\n\n")
            f.write(f"=== RETURN CODE: {result.returncode} ===\n")

        if result.returncode != 0:
            print(f"❌ 训练失败，返回码: {result.returncode}")
            print(f"详细日志: {log_path}")
            return {
                "status": "failed",
                "config": asdict(config),
                "returncode": result.returncode,
                "log_path": str(log_path)
            }

        print(f"✅ 训练完成")

        # 运行评测
        print(f"📊 开始评测...")
        eval_cmd = [
            sys.executable, "-m", "AgenticArxiv.benchmark.run_benchmark",
            str(exp_dir / "final"),
            "--output_dir", str(exp_dir / "eval"),
            "--task_set", task_set,
        ]

        eval_result = subprocess.run(
            eval_cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=1800
        )

        eval_log_path = exp_dir / "eval.log"
        with open(eval_log_path, "w", encoding="utf-8") as f:
            f.write(f"=== STDOUT ===\n{eval_result.stdout}\n\n")
            f.write(f"=== STDERR ===\n{eval_result.stderr}\n\n")
            f.write(f"=== RETURN CODE: {eval_result.returncode} ===\n")

        if eval_result.returncode != 0:
            print(f"⚠️  评测失败，返回码: {eval_result.returncode}")
        else:
            print(f"✅ 评测完成")

        return {
            "status": "success",
            "config": asdict(config),
            "output_dir": str(exp_dir),
            "log_path": str(log_path),
            "eval_log_path": str(eval_log_path)
        }

    except subprocess.TimeoutExpired:
        print(f"⏰ 训练超时")
        return {"status": "timeout", "config": asdict(config)}
    except Exception as e:
        print(f"❌ 异常: {e}")
        return {"status": "error", "config": asdict(config), "error": str(e)}


def summarize_results(output_dir: Path) -> None:
    """汇总所有实验结果"""
    print(f"\n{'='*60}")
    print("📊 汇总实验结果")
    print(f"{'='*60}\n")

    summary = []

    for config in ABLATION_CONFIGS:
        exp_dir = output_dir / config.name
        eval_summary = exp_dir / "eval" / "summary.json"

        if not eval_summary.exists():
            print(f"⚠️  {config.name}: 无评测结果")
            continue

        with open(eval_summary, encoding="utf-8") as f:
            metrics = json.load(f)

        summary.append({
            "name": config.name,
            "description": config.description,
            "weights": {
                "format": config.format,
                "tool": config.tool,
                "argument": config.argument,
                "process": config.process,
                "outcome": config.outcome,
            },
            "curriculum_steps": config.curriculum_steps,
            "metrics": metrics
        })

        print(f"✅ {config.name}:")
        print(f"   strict: {metrics.get('strict_success_rate', 0)*100:.1f}%")
        print(f"   完成率: {metrics.get('completion_rate', 0)*100:.1f}%")
        print(f"   tool: {metrics.get('tool_accuracy', 0)*100:.1f}%")
        print(f"   argument: {metrics.get('argument_accuracy', 0)*100:.1f}%")

    # 保存汇总
    summary_path = output_dir / "ablation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n📝 汇总已保存至: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="多粒度奖励权重消融实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="outputs/sft/final",
        help="基础模型路径（默认: outputs/sft/final）"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("artifacts/ablation_reward_weights"),
        help="实验输出目录（默认: artifacts/ablation_reward_weights）"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="每个实验的训练步数（默认: 100）"
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=4,
        help="每步生成数（默认: 4）"
    )
    parser.add_argument(
        "--task_set",
        type=str,
        default="expanded",
        help="任务集（默认: expanded）"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="rl_train",
        help="数据集划分（默认: rl_train）"
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        help="指定要运行的实验（默认: 全部）"
    )
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="快速测试模式（10 steps）"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run 模式，仅生成配置不实际训练"
    )

    args = parser.parse_args()

    if args.quick_test:
        args.num_steps = 10
        print("⚡ 快速测试模式：10 steps")

    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 筛选实验组
    experiments = ABLATION_CONFIGS
    if args.experiments:
        experiments = [c for c in ABLATION_CONFIGS if c.name in args.experiments]
        if not experiments:
            print(f"❌ 未找到指定的实验: {args.experiments}")
            print(f"可用实验: {[c.name for c in ABLATION_CONFIGS]}")
            return 1

    print(f"🔬 将运行 {len(experiments)} 个实验组")
    print(f"📁 输出目录: {args.output_dir}")

    # 运行所有实验
    results = []
    for config in experiments:
        result = run_single_experiment(
            config=config,
            base_model=args.base_model,
            output_dir=args.output_dir,
            num_steps=args.num_steps,
            num_generations=args.num_generations,
            task_set=args.task_set,
            split=args.split,
            dry_run=args.dry_run,
        )
        results.append(result)

    # 保存所有实验的元信息
    meta_path = args.output_dir / "experiments_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "total": len(experiments),
            "experiments": [c.to_config_dict() for c in experiments],
            "results": results,
            "args": vars(args)
        }, f, indent=2, ensure_ascii=False)

    print(f"\n📝 实验元信息已保存至: {meta_path}")

    # 汇总结果
    if not args.dry_run:
        summarize_results(args.output_dir)

    print(f"\n{'='*60}")
    print("✅ 所有实验完成")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
