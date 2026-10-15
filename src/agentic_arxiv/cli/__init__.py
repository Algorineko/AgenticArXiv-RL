"""唯一受支持的轻量命令入口；仅选定操作后导入业务实现。"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

from .data_arguments import register_commands as register_data_commands
from .evaluation_arguments import register_commands as register_evaluation_commands
from .training_arguments import register_commands as register_training_commands


HELP_COMMANDS = (
    (), ("run",), ("evaluate",), ("evaluate", "run"), ("evaluate", "rollout"),
    ("evaluate", "baselines"), ("evaluate", "rescore"), ("evaluate", "badcases"),
    ("evaluate", "badcases", "replay"), ("evaluate", "badcases", "capture"),
    ("evaluate", "badcases", "run"), ("train",), ("train", "sft"), ("train", "dpo"), ("train", "grpo"),
    ("data",), ("data", "snapshot"), ("data", "backfill-figures"), ("data", "sft"),
    ("data", "sft", "generate"), ("data", "sft", "generate-parametric"),
    ("data", "sft", "augment"), ("data", "sft", "augment-parametric"),
    ("data", "sft", "mix"), ("data", "dpo"), ("data", "dpo", "generate"),
    ("data", "grpo"), ("data", "grpo", "select-signal"),
    ("audit",), ("audit", "rollouts"), ("audit", "curriculum"), ("audit", "assets"),
)


def _add_command(subparsers, name, **kwargs):
    parser = subparsers.add_parser(name, **kwargs)
    # SUPPRESS 避免后层未指定的默认值覆盖前层显式工作区。
    parser.add_argument("--workspace", type=Path, default=argparse.SUPPRESS,
                        help="输入与输出工作区（默认当前目录，可放在命令前后）")
    return parser


def build_parser():
    parser = argparse.ArgumentParser(
        prog="agentic-arxiv", description="AgenticArxiv：文本 ReAct、训练、评测与数据准备",
        epilog="基础 API：agentic-arxiv；本地推理：[local]；训练：[train]；翻译：[translate]；日志：[logging]。",
    )
    parser.add_argument("--workspace", type=Path, default=None,
                        help="输入与输出工作区（默认当前目录，可放在命令前后）")
    commands = parser.add_subparsers(dest="command", required=True)
    run = _add_command(commands, "run", help="普通文本 ReAct；默认在线 API")
    run.set_defaults(_operation="run")
    run.add_argument("task", nargs="?", default="获取最近7天内机器学习领域（ML）的最新论文,最多获取15篇")
    run.add_argument("--offline", action="store_true", help="只回放已有快照，不联网")
    run.add_argument("--snapshot", help="离线快照路径，默认 data/mock_arxiv_snapshot.json")
    run.add_argument("--llm-backend", choices=("api", "transformers"), default="api",
                     help="transformers 后端需要安装 agentic-arxiv[local]")
    run.add_argument("--model", help="API 模型名或已有本地模型目录")
    run.add_argument("--max-iterations", type=int, default=6, help="轮数预算，含 FINISH")
    run.add_argument("--output-dir", type=Path, default=Path("output/runs"), help="结果根目录，每次创建独立子目录")
    evaluate = _add_command(commands, "evaluate", help="评测、rollout 与历史轨迹")
    audit = _add_command(commands, "audit", help="只读诊断与资产核对")
    register_evaluation_commands(evaluate, audit, _add_command)
    train = _add_command(commands, "train", help="SFT / 可选 DPO / GRPO；需要安装 agentic-arxiv[train]")
    register_training_commands(train, _add_command)
    data = _add_command(commands, "data", help="显式数据准备操作")
    register_data_commands(data, _add_command)
    return parser


def parse_args(argv=None, *, check_capabilities=False):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    args.workspace = Path(args.workspace or Path.cwd()).expanduser().resolve()
    if check_capabilities:
        try:
            _check_capabilities(args)
        except ValueError as exc:
            parser.error(str(exc))
    if args._operation.startswith("train.") and args.config_path:
        from agentic_arxiv.config import workspace_path
        from agentic_arxiv.training.mainline_config import stage_config

        path = workspace_path(args.config_path, workspace=args.workspace)
        try:
            stage = args._operation.split(".")[1]
            defaults = stage_config(stage, path)["defaults"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(str(exc))
        # 原 CLI 语义：配置给默认值，显式 CLI 仍覆盖；业务完整校验配置字段。
        leaf = _leaf_parser(parser, ("train", stage))
        leaf.set_defaults(**defaults)
        args = parser.parse_args(argv)
        args.workspace = Path(args.workspace or Path.cwd()).expanduser().resolve()
    _resolve_paths(args)
    return parser, args


def _leaf_parser(parser, commands):
    for name in commands:
        action = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
        parser = action.choices[name]
    return parser


def _resolve_paths(args):
    from agentic_arxiv.config import workspace_path

    operation = args._operation
    names = {
        "run": ("snapshot", "output_dir"),
        "evaluate.run": ("output", "snapshot", "save_traces"),
        "evaluate.rollout": ("output_dir", "snapshot"),
        "evaluate.baselines": ("output",),
        "evaluate.rescore": ("traces", "summary", "output", "snapshot", "references"),
        "audit.rollouts": ("trace", "summary_out"),
        "audit.curriculum": ("log_dir",),
        "audit.assets": ("baseline",),
        "train.sft": ("data", "data_manifest", "output_dir", "config_path"),
        "train.dpo": ("data", "data_manifest", "output_dir", "snapshot", "config_path"),
        "train.grpo": ("output_dir", "snapshot", "resume_from_checkpoint", "rollout_trace_path", "config_path"),
        "data.snapshot": ("snapshot",),
        "data.backfill-figures": ("snapshot",),
        "data.sft.generate": ("output", "snapshot"),
        "data.sft.generate-parametric": ("split_file", "snapshot", "output"),
        "data.sft.augment": ("input", "output", "manifest", "split_file"),
        "data.sft.augment-parametric": ("input", "output", "manifest", "split_file"),
        "data.sft.mix": ("original", "parametric", "output"),
        "data.dpo.generate": ("output", "snapshot"),
        "data.grpo.select-signal": ("probe", "model", "output"),
    }.get(operation, ())
    if operation.startswith("evaluate.badcases."):
        names = ("cases", "traces", "snapshot")
    for name in names:
        value = getattr(args, name, None)
        if value not in (None, ""):
            path = workspace_path(value, workspace=args.workspace)
            setattr(args, name, path if isinstance(value, Path) else str(path))
    for name in ("split", "source_split"):
        value = getattr(args, name, None)
        if value and ":" in value:
            filename, split_name = value.rsplit(":", 1)
            setattr(args, name, f"{workspace_path(filename, workspace=args.workspace)}:{split_name}")
    local_model = ((operation == "run" and args.llm_backend == "transformers")
                   or (operation == "evaluate.run" and args.backend == "transformers")
                   or (operation == "evaluate.rollout" and args.model is not None)
                   or operation in ("evaluate.badcases.run", "train.grpo", "train.dpo", "data.dpo.generate"))
    if local_model and args.model:
        args.model = str(workspace_path(args.model, workspace=args.workspace))
    elif operation == "train.sft":
        args.model = _model_source(args.model, args.workspace)
    if operation in ("train.grpo", "train.dpo", "data.dpo.generate") and args.base_model:
        args.base_model = _model_source(args.base_model, args.workspace)


def _model_source(value, workspace):
    from agentic_arxiv.config import workspace_path

    path = workspace_path(value, workspace=workspace)
    hub_id = len(value.split("/")) == 2 and not value.startswith(("outputs/", "models/", "./", "../", "/", "~"))
    return value if hub_id and not path.is_dir() else str(path)


def _require_capability(extra):
    requirements = {"local": ("torch", "transformers", "accelerate", "peft"),
                    "train": ("torch", "transformers", "accelerate", "trl", "datasets", "peft", "bitsandbytes")}
    missing = [name for name in requirements[extra] if importlib.util.find_spec(name) is None]
    if missing:
        raise ValueError(f"缺少 {extra} 依赖：{', '.join(missing)}；请安装 pip install 'agentic-arxiv[{extra}]'")


def _check_capabilities(args):
    local = ((args._operation == "run" and args.llm_backend == "transformers")
             or (args._operation == "evaluate.run" and args.backend == "transformers")
             or (args._operation == "evaluate.rollout" and args.model is not None)
             or args._operation in ("evaluate.badcases.run", "data.dpo.generate"))
    if args._operation.startswith("train."):
        _require_capability("train")
    elif local:
        _require_capability("local")


def _validate_operation(parser, args):
    from agentic_arxiv.config import workspace_path

    if args._operation == "run":
        if args.max_iterations < 1:
            parser.error("--max-iterations 必须至少为 1，FINISH 也占用一轮")
        if args.snapshot and not args.offline:
            parser.error("--snapshot 必须与 --offline 一起使用")
        if args.offline:
            args.snapshot = Path(args.snapshot) if args.snapshot else workspace_path("data/mock_arxiv_snapshot.json")
            if not args.snapshot.exists():
                parser.error(f"离线模式需要已有快照: {args.snapshot}")
        if args.llm_backend == "transformers" and (not args.model or not Path(args.model).is_dir()):
            parser.error(f"transformers 后端需要存在的 --model 本地模型目录: {args.model or args.workspace}")
    if args._operation == "evaluate.run" and args.backend == "transformers" and not args.model:
        parser.error("--backend transformers 必须同时提供 --model 本地模型目录")


def _operation_kwargs(args):
    excluded = {"workspace", "command", "evaluate_command", "badcases_command", "audit_command",
                "train_command", "data_command", "sft_data_command", "dpo_data_command", "grpo_data_command", "_operation"}
    return {key: value for key, value in vars(args).items() if key not in excluded}


def main(argv=None):
    parser, args = parse_args(argv, check_capabilities=True)
    try:
        from agentic_arxiv.config import initialize_operation

        readonly = args._operation.startswith("audit.") or args._operation in (
            "evaluate.rescore", "evaluate.badcases.replay", "evaluate.badcases.capture")
        initialize_operation(args.workspace, load_env=not readonly)
        _validate_operation(parser, args)
        result = _dispatch(args)
        return result if isinstance(result, int) else 0
    except (OSError, ValueError, ImportError) as exc:
        parser.error(str(exc))


def _dispatch(args):
    operation = args._operation
    kwargs = _operation_kwargs(args)
    if operation == "run":
        from .run import execute

        return execute(args)
    if operation == "train.sft":
        from agentic_arxiv.training.sft import main as train_sft

        return train_sft(**kwargs)
    if operation == "train.grpo":
        from agentic_arxiv.training.grpo import main as train_grpo

        return train_grpo(**kwargs)
    if operation == "train.dpo":
        from agentic_arxiv.training.dpo import main as train_dpo

        return train_dpo(**kwargs)
    if operation == "evaluate.run":
        from agentic_arxiv.evaluation.run_benchmark import execute

        return execute(args)
    if operation == "evaluate.baselines":
        from agentic_arxiv.evaluation.run_baselines import execute

        return execute(args)
    if operation == "evaluate.rollout":
        from agentic_arxiv.evaluation.rollout import run_rollout

        return run_rollout(**kwargs)
    if operation == "evaluate.rescore":
        from agentic_arxiv.evaluation.rescore_traces import execute

        return execute(args)
    if operation.startswith("evaluate.badcases."):
        from agentic_arxiv.evaluation.badcase_replay import cmd_capture, cmd_replay, cmd_run

        return {"replay": cmd_replay, "capture": cmd_capture, "run": cmd_run}[args.badcases_command or "replay"](args)
    if operation == "audit.rollouts":
        from agentic_arxiv.evaluation.audit_rollouts import run

        return run(args)
    if operation == "audit.curriculum":
        from agentic_arxiv.evaluation.curriculum import run

        return run(args)
    if operation == "audit.assets":
        from agentic_arxiv.evaluation.assets import audit_assets, canonical_bytes

        result = audit_assets(args.workspace, baseline=args.baseline, freeze=args.freeze, dump=args.dump)
        if args.dump:
            print(canonical_bytes(result).decode("utf-8"))
        elif args.freeze:
            print(f"已新建资产基线：{result['baseline']}")
        else:
            print("主线业务资产与冻结基线一致")
        return 0
    kwargs["workspace"] = args.workspace
    if operation == "data.snapshot":
        from agentic_arxiv.data_pipeline.build_snapshot import build

        kwargs["snapshot_path"] = kwargs.pop("snapshot")
        kwargs["pin_references"] = not kwargs.pop("no_pin_references")
        return build(**kwargs)
    if operation == "data.backfill-figures":
        from agentic_arxiv.data_pipeline.backfill_figure_analysis import main as backfill

        kwargs["paper_ids"] = kwargs.pop("paper_id")
        return backfill(**kwargs)
    if operation == "data.sft.generate":
        from agentic_arxiv.data_pipeline.generate_sft_data import generate_sft_dataset

        return generate_sft_dataset(**kwargs)
    if operation == "data.sft.generate-parametric":
        from agentic_arxiv.data_pipeline.generate_parametric_sft_data import main as generate

        return generate(**kwargs)
    if operation == "data.sft.augment":
        from agentic_arxiv.data_pipeline.augment_sft_data import main as augment

        return augment(**kwargs)
    if operation == "data.sft.augment-parametric":
        from agentic_arxiv.data_pipeline.augment_parametric_sft_data import main as augment

        return augment(**kwargs)
    if operation == "data.sft.mix":
        from agentic_arxiv.data_pipeline.build_sft_train_mix import main as mix

        return mix(**kwargs)
    if operation == "data.grpo.select-signal":
        from agentic_arxiv.data_pipeline.build_grpo_signal_split import main as select_signal

        return select_signal(**kwargs)
    if operation == "data.dpo.generate":
        from agentic_arxiv.data_pipeline.generate_dpo_data import generate_dpo_dataset

        return generate_dpo_dataset(**kwargs)
    raise ValueError(f"未知操作: {operation}")
