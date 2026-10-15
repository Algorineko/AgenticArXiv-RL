"""评测与诊断的轻量参数声明，不导入模型或执行模块。"""

from __future__ import annotations

import argparse
from pathlib import Path


def register_commands(evaluate, audit, add_command):
    commands = evaluate.add_subparsers(dest="evaluate_command", required=True)
    run = add_command(commands, "run", help="执行文本 ReAct benchmark")
    run.set_defaults(_operation="evaluate.run")
    run.add_argument("--repeat", type=int, default=3, help="每个任务重复次数")
    run.add_argument("--tasks", choices=("search", "download", "translate", "cache", "composite",
                                        "keyword_search", "ref_form", "optional", "state",
                                        "long_chain", "constraint", "infeasible"))
    run.add_argument("--task-ids", nargs="+")
    run.add_argument("--output", default="data", help="报告输出目录")
    run.add_argument("--model", help="API 模型名或本地模型目录")
    run.add_argument("--backend", choices=("api", "transformers"), default="api")
    run.add_argument("--local-device", default="auto")
    run.add_argument("--local-dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--prefix")
    run.add_argument("--no-thinking", action="store_true", help="关闭模型思维链")
    run.add_argument("--offline", action="store_true", help="只回放已有快照，不联网")
    run.add_argument("--snapshot", help="默认 data/mock_arxiv_snapshot.json")
    run.add_argument("--save-traces", nargs="?", const="", default=None, metavar="PATH")
    run.add_argument("--split", metavar="[FILE:]NAME")
    run.add_argument("--task-set", choices=("default", "expanded"), default="default")

    rollout = add_command(commands, "rollout", help="采集任务轨迹；默认离线")
    rollout.set_defaults(_operation="evaluate.rollout")
    rollout.add_argument("--task-id")
    rollout.add_argument("--output-dir", default="traces/train")
    rollout.add_argument("--all", action="store_true", help="执行全部选定任务")
    rollout.add_argument("--model", help="本地模型目录；需要安装 agentic-arxiv[local]，未指定时使用 API")
    rollout.add_argument("--snapshot")
    rollout.add_argument("--temperature", type=float, default=0.1)
    rollout.add_argument("--max-iterations", type=int, default=6)
    rollout.add_argument("--task-set", choices=("basic", "expanded"), default="basic")
    rollout.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True,
                         help="默认回放快照；--no-offline 显式允许在线执行")

    baseline = add_command(commands, "baselines", help="原有合成策略奖励诊断")
    baseline.set_defaults(_operation="evaluate.baselines")
    baseline.add_argument("--task-set", choices=("default", "expanded"), default="expanded")
    baseline.add_argument("--policies", nargs="+",
                          choices=("reference", "always_finish", "always_search", "random_tool", "wrong_args"),
                          default=["reference", "always_finish", "always_search", "random_tool", "wrong_args"])
    baseline.add_argument("--seed", type=int, default=42)
    baseline.add_argument("--random-samples", type=int, default=20)
    baseline.add_argument("--min-reference-gap", type=float, default=0.3)
    baseline.add_argument("--min-category-gap", type=float, default=0.3)
    baseline.add_argument("--top", type=int, default=5)
    baseline.add_argument("--training-step", type=int, default=100)
    baseline.add_argument("--output", type=Path, help="可选报告目录")

    rescore = add_command(commands, "rescore", help="历史轨迹只读重评分，不执行工具或模型")
    rescore.set_defaults(_operation="evaluate.rescore")
    for name in ("traces", "summary", "output"):
        rescore.add_argument(f"--{name}", required=True)
    rescore.add_argument("--snapshot", help="可选的原快照路径校验；不执行，也不能代替冻结参考")
    rescore.add_argument("--references", help="已有冻结参考 JSON：task_id 映射到逐步 expected_paper_ids")
    rescore.add_argument("--task-set", choices=("default", "expanded"), default="expanded")
    rescore.add_argument("--split", metavar="[FILE:]NAME")

    badcases = add_command(commands, "badcases", help="坏例回放、捕获与显式执行")
    _badcase_common(badcases)
    bad = badcases.add_subparsers(dest="badcases_command")
    badcases.set_defaults(_operation="evaluate.badcases.replay")
    for name, description in (("replay", "历史轨迹只读重评分"),
                              ("capture", "从已有轨迹追加坏例"),
                              ("run", "执行当前本地模型完整 episode；需要 [local]")):
        parser = add_command(bad, name, help=description)
        _badcase_common(parser, suppress_defaults=True)
        parser.set_defaults(_operation=f"evaluate.badcases.{name}")
        if name == "capture":
            parser.add_argument("--traces", type=Path, required=True)
            parser.add_argument("--source", default="")
            parser.add_argument("--dry-run", action="store_true")
        elif name == "run":
            parser.add_argument("--model", required=True, help="本地模型目录；缺失不替换模型或 API")
            parser.add_argument("--offline", action="store_true")
            parser.add_argument("--snapshot", type=Path)
            parser.add_argument("--max-iterations", type=int, default=6)

    diagnostics = audit.add_subparsers(dest="audit_command", required=True)
    rollouts = add_command(diagnostics, "rollouts", help="只读诊断 GRPO 原始轨迹")
    rollouts.set_defaults(_operation="audit.rollouts")
    rollouts.add_argument("trace", type=Path)
    rollouts.add_argument("--show", type=int, default=10)
    rollouts.add_argument("--summary-out", type=Path)
    curriculum = add_command(diagnostics, "curriculum", help="分析已有 TensorBoard 事件；需要 [logging]")
    curriculum.set_defaults(_operation="audit.curriculum")
    curriculum.add_argument("log_dir", type=Path)
    curriculum.add_argument("--curriculum-steps", type=int, default=30)
    assets = add_command(diagnostics, "assets", help="默认只读核对长期研究资产")
    assets.set_defaults(_operation="audit.assets")
    assets.add_argument("--baseline", default="artifacts/mainline_baseline/assets.json")
    modes = assets.add_mutually_exclusive_group()
    modes.add_argument("--freeze", action="store_true", help="显式新建基线，拒绝覆盖")
    modes.add_argument("--dump", action="store_true", help="只打印当前规范化资产 JSON")


def _badcase_common(parser, *, suppress_defaults=False):
    def default(value):
        return argparse.SUPPRESS if suppress_defaults else value

    parser.add_argument("--cases", type=Path, default=default(Path("eval/eval_cases.jsonl")))
    parser.add_argument("--task-set", choices=("default", "expanded"), default=default("expanded"))
    parser.add_argument("--training-step", type=int, default=default(100))
    parser.add_argument("-v", "--verbose", action="store_true", default=default(False))
