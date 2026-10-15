"""轻量 SFT/GRPO 参数声明；帮助不导入训练器或读取用户配置。"""

from __future__ import annotations

import argparse
import json

from agentic_arxiv.config import read_default_mainline_config


def _add_shared_arguments(parser, defaults):
    parser.add_argument("--model", default=defaults["model"], help="基座模型或本地模型目录")
    parser.add_argument("--output-dir", default=defaults["output_dir"], help="训练产物目录")
    parser.add_argument("--epochs", type=int, default=defaults["epochs"], help="训练轮数")
    parser.add_argument("--max-steps", type=int, default=defaults["max_steps"],
                        help="优化步数上限；-1 表示按 epochs 训练")
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"],
                        help="每设备训练批量")
    parser.add_argument("--grad-accum", type=int, default=defaults["grad_accum"],
                        help="梯度累积步数")
    parser.add_argument("--lr", type=float, default=defaults["lr"], help="学习率")
    parser.add_argument("--qlora", action=argparse.BooleanOptionalAction,
                        default=defaults["qlora"],
                        help="启用 4-bit QLoRA；--no-qlora 使用全参数路径")
    parser.add_argument("--lora-r", type=int, default=defaults["lora_r"], help="LoRA 秩")
    parser.add_argument("--lora-alpha", type=int, default=defaults["lora_alpha"],
                        help="LoRA 缩放参数")
    parser.add_argument("--lora-dropout", type=float, default=defaults["lora_dropout"],
                        help="LoRA dropout 比例")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                        default=defaults["gradient_checkpointing"], help="启用梯度检查点")
    parser.add_argument("--optim", default=defaults["optim"], help="优化器名称")
    parser.add_argument("--seed", type=int, default=defaults["seed"],
                        help="训练与采样随机种子")
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction,
                        default=defaults["verify"], help="训练结束后运行阶段验证")
    parser.add_argument("--report-to", default=defaults["report_to"],
                        help="曲线记录目标：none / auto / tensorboard / wandb，可逗号分隔")
    parser.add_argument("--run-name", default=defaults["run_name"],
                        help="TensorBoard / wandb 运行名称，默认取输出目录末段")
    parser.add_argument("--config", dest="config_path", default=None,
                        help="主线配置 JSON；显式 CLI 参数优先于配置")


def _add_sft_arguments(parser, defaults):
    _add_shared_arguments(parser, defaults)
    parser.add_argument("--data", default=defaults["data"], help="已有 SFT 数据文件")
    parser.add_argument("--data-manifest", default=defaults["data_manifest"],
                        help="已有 SFT 数据 manifest")
    parser.add_argument("--max-length", type=int, default=defaults["max_length"],
                        help="训练序列长度上限；训练前检查长度，避免截断 assistant 目标")
    parser.add_argument("--inspect-only", action="store_true", default=defaults["inspect_only"],
                        help="只审计 manifest 和 token 长度，不加载模型或开始训练")
    parser.add_argument("--skip-data-manifest-check", action="store_true",
                        default=defaults["skip_data_manifest_check"],
                        help="仅跳过数据 kind/hash/行数校验，不能绕过协议兼容检查")
    parser.add_argument("--min-parse-rate", type=float, default=defaults["min_parse_rate"],
                        help="SFT 阶段验证的最低可解析率")


def _add_grpo_arguments(parser, defaults):
    _add_shared_arguments(parser, defaults)
    parser.add_argument("--beta", type=float, default=defaults["beta"], help="KL 惩罚权重")
    parser.add_argument("--reward-curriculum-steps", type=int,
                        default=defaults["reward_curriculum_steps"],
                        help="奖励课程持续步数；显式传 0 从首步启用完整语义权重")
    parser.add_argument("--num-generations", type=int, default=defaults["num_generations"],
                        help="每个提示的生成组大小")
    parser.add_argument("--allow-zero-variance", action="store_true",
                        default=defaults["allow_zero_variance"],
                        help="跳过组内奖励方差为 0 的检查；默认保留空转保护")
    parser.add_argument("--max-completion-length", type=int,
                        default=defaults["max_completion_length"], help="每轮策略生成 token 上限")
    parser.add_argument("--max-turns", type=int, default=defaults["max_turns"],
                        help="每条 rollout 最多执行轮数，含 FINISH；环境观测不计入策略 loss")
    parser.add_argument("--temperature", type=float, default=defaults["temperature"],
                        help="策略采样温度")
    parser.add_argument("--loss-type", choices=["grpo"], default=defaults["loss_type"],
                        help="主线损失字段；未设置时不传给 TRL，不猜填有效值")
    parser.add_argument("--epsilon-high", type=float, default=defaults["epsilon_high"],
                        help="重要性比上界；未设置时不传给 TRL")
    parser.add_argument("--mask-truncated-completions", action=argparse.BooleanOptionalAction,
                        default=defaults["mask_truncated_completions"],
                        help="剔除触及生成长度上限的轨迹；未设置时不传给 TRL")
    parser.add_argument("--importance-sampling-level", choices=["token"],
                        default=defaults["importance_sampling_level"],
                        help="主线重要性采样粒度；未设置时不传给 TRL")
    parser.add_argument("--scale-rewards", choices=["group", "none", "batch"],
                        default=defaults["scale_rewards"],
                        help="优势归一化方式；未设置时沿用 TRL 现值")
    parser.add_argument("--base-model", default=defaults["base_model"],
                        help="SFT adapter 对应基座模型；默认读取 adapter_config.json")
    parser.add_argument("--save-steps", type=int, default=defaults["save_steps"],
                        help="周期 checkpoint 间隔，单位为优化步；0 表示只保存最终模型")
    parser.add_argument("--save-total-limit", type=int, default=defaults["save_total_limit"],
                        help="最多保留的周期 checkpoint 数，最终模型不计入")
    parser.add_argument("--resume-from-checkpoint", default=defaults["resume_from_checkpoint"],
                        help="恢复模型、优化器与调度器状态；须保持原输出目录和训练参数")
    parser.add_argument("--snapshot", default=defaults["snapshot"], help="已有离线快照目录")
    parser.add_argument("--task-set", choices=["default", "expanded"],
                        default=defaults["task_set"],
                        help="正式训练选择 expanded 与已有训练切分；default 仅允许显式冒烟")
    parser.add_argument("--split", default=defaults["split"], metavar="[FILE:]NAME",
                        help="已有 train/rl_train 切分；禁止使用 dev/iid_test/ood_test 训练")
    parser.add_argument("--smoke", action="store_true", default=defaults["smoke"],
                        help="显式选择 8 条冒烟任务，不是正式训练切分")
    parser.add_argument("--canary-steps", type=int, default=defaults["canary_steps"],
                        help="Canary 评估间隔，单位为步；0 表示禁用")
    parser.add_argument("--min-canary-reward", type=float, default=defaults["min_canary_reward"],
                        help="Canary 奖励下限；默认 -1.0 表示禁用阈值检查")
    parser.add_argument("--canary-patience", type=int, default=defaults["canary_patience"],
                        help="Canary 奖励连续低于阈值多少次后停止训练")
    parser.add_argument("--save-rollout-traces", action="store_true",
                        default=defaults["save_rollout_traces"],
                        help="保存原始生成、工具轨迹及奖励分解，默认写入输出目录")
    parser.add_argument("--rollout-trace-path", default=defaults["rollout_trace_path"],
                        help="Rollout 审计 JSONL 路径；显式传入会自动启用审计")
    parser.add_argument("--rollout-trace-max-samples", type=int,
                        default=defaults["rollout_trace_max_samples"],
                        help="最多保存的原始 rollout 数；0 表示全部，摘要始终覆盖全部样本")


def register_commands(train_parser, add_command):
    """将独立参数声明注册到统一 CLI，执行组装由调用方负责。"""
    config = json.loads(read_default_mainline_config())
    commands = train_parser.add_subparsers(dest="train_command", required=True)
    sft = add_command(commands, "sft", help="SFT / 单卡或 DDP 4-bit QLoRA；需要 train 安装层")
    sft.description = "SFT / 单卡或 DDP 4-bit QLoRA；执行需要 agentic-arxiv[train]"
    _add_sft_arguments(sft, config["sft"]["defaults"])
    sft.set_defaults(_operation="train.sft")
    grpo = add_command(commands, "grpo", help="GRPO 单卡或 DDP 多轮训练；需要 train 安装层")
    grpo.description = "GRPO 单卡或 DDP 多轮训练；执行需要 agentic-arxiv[train]"
    _add_grpo_arguments(grpo, config["grpo"]["defaults"])
    grpo.set_defaults(_operation="train.grpo")
