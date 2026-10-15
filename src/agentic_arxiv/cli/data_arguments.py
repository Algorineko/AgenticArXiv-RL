"""数据准备命令的轻量参数声明，不加载业务或模型实现。"""

import argparse
import json
from pathlib import Path
from typing import Callable

from agentic_arxiv.config import read_default_mainline_config

DPO_GENERATION_DEFAULTS = json.loads(read_default_mainline_config())["dpo"]["generation_defaults"]


def register_commands(
    data_parser: argparse.ArgumentParser,
    add_command: Callable[..., argparse.ArgumentParser],
) -> None:
    """在已有的数据命令解析器下注册全部数据准备操作。"""
    commands = data_parser.add_subparsers(dest="data_command", required=True)

    snapshot = add_command(commands, "snapshot", help="构建工具回放快照")
    snapshot.set_defaults(_operation="data.snapshot")
    snapshot.add_argument(
        "--snapshot", default="data/mock_arxiv_snapshot.json", help="快照输出路径",
    )
    snapshot.add_argument("--aspects", nargs="+", default=None, help="论文研究方向列表")
    snapshot.add_argument(
        "--keyword-queries", nargs="+", default=None, help="论文关键词查询列表",
    )
    snapshot.add_argument("--max-results", type=int, default=50, help="每次查询的结果上限")
    snapshot.add_argument("--days", type=int, default=30, help="查询最近多少天的论文")
    snapshot.add_argument(
        "--no-pin-references", action="store_true",
        help="不向 AI/CV 池固定任务所需论文，仅用于原始数据研究",
    )
    snapshot.add_argument(
        "--allow-partial", action="store_true",
        help="允许部分查询失败后仍保存快照，不建议用于训练或正式评测",
    )
    snapshot.add_argument(
        "--content-workers", type=int, default=8, help="并行预取 PDF 的线程数",
    )
    snapshot.add_argument(
        "--skip-prefetch", action="store_true",
        help="跳过并行预取，直接从已有 PDF 开始内容抽取",
    )
    snapshot.add_argument(
        "--prefetch-budget", type=int, default=900, help="预取阶段的整体墙钟预算，单位秒",
    )
    snapshot.add_argument(
        "--content-max-ref", type=int, default=0,
        help="每个论文池预取正文与图表的前 N 篇，0 表示全部",
    )
    snapshot.add_argument(
        "--translate-max-ref", type=int, default=1,
        help="每池前 N 篇录制译文页，0 关闭；实际翻译需要 translate 安装层及在线服务",
    )

    backfill = add_command(commands, "backfill-figures", help="补录快照中的图表分析")
    backfill.set_defaults(_operation="data.backfill-figures")
    backfill.add_argument("--snapshot", type=Path, required=True, help="已有快照路径")
    backfill.add_argument(
        "--paper-id", action="append", default=None,
        help="仅处理指定论文，可重复；本地 VLM 后端需要 local 安装层",
    )
    backfill.add_argument(
        "--force", action="store_true", help="覆盖已有图表分析条目",
    )

    sft = add_command(commands, "sft", help="SFT 数据生成、增强与混合")
    sft_commands = sft.add_subparsers(dest="sft_data_command", required=True)

    generate = add_command(sft_commands, "generate", help="生成 SFT 专家数据")
    generate.set_defaults(_operation="data.sft.generate")
    generate.add_argument("--output", default=None, help="输出 JSONL 路径")
    generate.add_argument("--snapshot", default=None, help="离线快照路径")
    generate.add_argument(
        "--mode", choices=("replay", "live", "record"), default="replay",
        help="工具执行模式；默认离线且缺快照失败，在线须显式选择 live",
    )
    generate.add_argument(
        "--use-llm", action="store_true", help="使用外部 LLM API，默认使用确定性专家",
    )
    generate.add_argument(
        "--task-set", choices=("basic", "expanded"), default="basic",
        help="基础或扩展任务集",
    )
    generate.add_argument(
        "--split", default=None, metavar="PATH:NAME",
        help="扩展正式数据须显式指定版本化 train，例如 data/splits/v2_62.json:train",
    )

    parametric = add_command(
        sft_commands, "generate-parametric", help="生成仅训练切分的参数化 SFT 数据",
    )
    parametric.set_defaults(_operation="data.sft.generate-parametric")
    parametric.add_argument(
        "--split-file", default="data/splits/v3_81.json",
        help="版本化切分；全部派生任务的父任务必须属于 train",
    )
    parametric.add_argument(
        "--snapshot", default="data/mock_arxiv_snapshot.json", help="离线快照路径",
    )
    parametric.add_argument(
        "--output", default=None, help="输出 JSONL 路径，默认按是否加入 T5 选择",
    )
    parametric.add_argument(
        "--include-t5", action="store_true", help="加入 T5 图表派生任务，需要 v3_81 训练切分",
    )
    parametric.add_argument(
        "--only-parents", default=None, metavar="ID,ID,...",
        help="仅生成指定训练父任务的派生数据，未知 ID 报错",
    )

    augment = add_command(sft_commands, "augment", help="确定性语言扩增 SFT 种子数据")
    augment.set_defaults(_operation="data.sft.augment")
    augment.add_argument("--input", required=True, help="新协议种子 JSONL")
    augment.add_argument("--output", required=True, help="扩增后的 JSONL")
    augment.add_argument("--manifest", default=None, help="默认读取种子旁的 manifest")
    augment.add_argument(
        "--split-file", default="data/splits/v2_62.json", help="泄漏审计使用的版本化切分",
    )

    augment_parametric = add_command(
        sft_commands, "augment-parametric", help="语言扩增参数化 SFT 种子数据",
    )
    augment_parametric.set_defaults(_operation="data.sft.augment-parametric")
    augment_parametric.add_argument(
        "--input", default="data/sft/sft_react_messages_v3_parametric_seed.jsonl",
        help="参数化种子 JSONL",
    )
    augment_parametric.add_argument(
        "--manifest", default=None, help="默认读取种子旁的 manifest",
    )
    augment_parametric.add_argument(
        "--split-file", default="data/splits/v3_81.json", help="版本化切分路径",
    )
    augment_parametric.add_argument(
        "--output", default="data/sft/sft_react_messages_v3_parametric_linguistic.jsonl",
        help="扩增后的 JSONL 路径",
    )
    augment_parametric.add_argument(
        "--wrapper-set", choices=("base", "targeted"), default="base",
        help="措辞框架：base 为默认数据集；targeted 为与 base 不相交的定向语言扩增",
    )

    mix = add_command(sft_commands, "mix", help="构建最终 SFT 训练混合数据")
    mix.set_defaults(_operation="data.sft.mix")
    mix.add_argument(
        "--original", default="data/sft/sft_react_messages_v3_linguistic.jsonl",
        help="原始任务语言扩增数据路径",
    )
    mix.add_argument(
        "--parametric", default="data/sft/sft_react_messages_v3_parametric_linguistic.jsonl",
        help="参数化任务语言扩增数据路径",
    )
    mix.add_argument(
        "--output", default="data/sft/sft_react_messages_v3_train_mix.jsonl",
        help="训练混合数据输出路径",
    )
    mix.add_argument("--seed", type=int, default=42, help="确定性打乱的随机种子")

    dpo = add_command(commands, "dpo", help="本地 SFT 的真实决策偏好数据；需要 local 安装层")
    dpo_commands = dpo.add_subparsers(dest="dpo_data_command", required=True)
    dpo_generate = add_command(dpo_commands, "generate", help="严格同上下文离线偏好采样；需要 local 安装层")
    dpo_generate.description = "从已有本地 SFT 和显式 expanded :train 采样；需要 agentic-arxiv[local]"
    dpo_generate.set_defaults(_operation="data.dpo.generate")
    dpo_generate.add_argument("--model", default=DPO_GENERATION_DEFAULTS["model"], help="当前协议本地 SFT 产物")
    dpo_generate.add_argument("--base-model", default=DPO_GENERATION_DEFAULTS["base_model"], help="适配器声明的精确本地基座")
    dpo_generate.add_argument("--output", default=DPO_GENERATION_DEFAULTS["output"], help="新批次 JSONL，拒绝覆盖")
    dpo_generate.add_argument("--snapshot", default=DPO_GENERATION_DEFAULTS["snapshot"], help="必需的已有离线快照，不联网补齐")
    dpo_generate.add_argument("--task-set", choices=("expanded",), default=DPO_GENERATION_DEFAULTS["task_set"], help="仅支持 expanded 训练任务")
    dpo_generate.add_argument("--split", default=DPO_GENERATION_DEFAULTS["split"], metavar="PATH:train", help="必需的显式版本化 train 切分")
    for name, value_type, help_text in (
        ("num_trials", int, "每任务独立采样轨迹数"),
        ("temperature", float, "采样温度"),
        ("seed", int, "按任务和 trial 派生独立随机流的种子"),
        ("min_reward_gap", float, "奖励差必须严格大于此值"),
        ("min_task_yield", float, "任务产出率低于此值只报警，不放宽筛选"),
        ("max_turns", int, "轨迹轮数预算，含 FINISH"),
        ("max_new_tokens", int, "每轮新 token 预算"),
        ("reward_curriculum_steps", int, "固定离线评分的课程步数，评分 training_step 为 0"),
    ):
        dpo_generate.add_argument("--" + name.replace("_", "-"), type=value_type,
                                  default=DPO_GENERATION_DEFAULTS[name], help=help_text)

    grpo = add_command(commands, "grpo", help="GRPO 数据选择")
    grpo_commands = grpo.add_subparsers(dest="grpo_data_command", required=True)
    signal = add_command(grpo_commands, "select-signal", help="从既有探测汇总选择有信号的训练任务")
    signal.set_defaults(_operation="data.grpo.select-signal")
    signal.add_argument("--probe", required=True, help="reward_probe_summary.json 路径")
    signal.add_argument(
        "--source-split", required=True,
        help="候选任务来源，仅允许 FILE:train，例如 data/splits/v7_86.json:train",
    )
    signal.add_argument("--model", required=True, help="被探测的策略模型路径")
    signal.add_argument("--output", required=True, help="新切分输出路径，拒绝覆盖已有文件")
    signal.add_argument("--num-generations", type=int, default=4, help="每组采样数量")
    signal.add_argument("--seed", type=int, default=43, help="切分随机种子")
    signal.add_argument(
        "--min-informative-groups", type=int, default=2, help="每个任务至少需要的有信息组数量",
    )
