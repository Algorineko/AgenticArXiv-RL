"""在真实隔离安装环境检查入口、资源和轻量依赖；不执行模型或数据生产。"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata, resources
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


LOCAL_DISTRIBUTIONS = {"torch", "transformers", "accelerate", "peft"}
TRAIN_DISTRIBUTIONS = {"trl", "datasets", "bitsandbytes"}
HEAVY_MODULES = LOCAL_DISTRIBUTIONS | TRAIN_DISTRIBUTIONS
DEFAULT_CONFIG_SHA256 = "3479911b31a7740321283a30c70f929f1fd8835be9b9974573d2c377a894b34c"
HELP_COMMANDS = (
    (), ("run",),
    ("evaluate",), ("evaluate", "rollout"), ("evaluate", "run"),
    ("evaluate", "baselines"), ("evaluate", "rescore"),
    ("evaluate", "badcases"), ("evaluate", "badcases", "replay"),
    ("evaluate", "badcases", "capture"), ("evaluate", "badcases", "run"),
    ("train",), ("train", "sft"), ("train", "grpo"),
    ("data",), ("data", "snapshot"), ("data", "backfill-figures"),
    ("data", "sft"), ("data", "sft", "generate"),
    ("data", "sft", "generate-parametric"), ("data", "sft", "augment"),
    ("data", "sft", "augment-parametric"), ("data", "sft", "mix"),
    ("data", "grpo"), ("data", "grpo", "select-signal"),
    ("audit",), ("audit", "rollouts"), ("audit", "curriculum"), ("audit", "assets"),
)


def _installed(name: str) -> bool:
    try:
        metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return False
    return True


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, f"入口失败: {command}\n{result.stdout}\n{result.stderr}"
    return result


def check_installation(layer: str, require_wheel: bool, source_root: Path, history: Path | None) -> dict:
    assert not os.environ.get("PYTHONPATH"), "安装验收禁止设置 PYTHONPATH"
    forbidden = set() if layer == "train" else TRAIN_DISTRIBUTIONS
    if layer == "base":
        forbidden = forbidden | LOCAL_DISTRIBUTIONS
    unexpected = sorted(name for name in forbidden if _installed(name))
    assert not unexpected, f"{layer} 环境被额外依赖掩盖: {unexpected}"
    if layer in {"local", "train"}:
        required = LOCAL_DISTRIBUTIONS | (TRAIN_DISTRIBUTIONS if layer == "train" else set())
        missing = sorted(name for name in required if not _installed(name))
        assert not missing, f"{layer} 安装缺少声明依赖: {missing}"

    import agentic_arxiv

    origin = Path(agentic_arxiv.__file__).resolve()
    if require_wheel:
        assert not origin.is_relative_to(source_root), f"正常安装误从源码加载: {origin}"
        distribution_files = tuple(metadata.distribution("agentic-arxiv").files or ())
        assert distribution_files, "正常wheel没有安装文件清单"
        for item in distribution_files:
            filename = str(item)
            assert not set(Path(filename).parts).intersection({"data", "artifacts", "models", "docs", "tests", "AgenticArxiv"}), (
                f"正常wheel携带了研究资产、归档或旧源码: {filename}"
            )
            assert not filename.endswith((".jsonl", ".py.txt")), f"正常wheel携带了研究输入或非活动基线: {filename}"
    assert not (HEAVY_MODULES & set(sys.modules)), "根包导入加载了本地或训练栈"
    config_bytes = resources.files("agentic_arxiv").joinpath("resources").joinpath("mainline.json").read_bytes()
    assert hashlib.sha256(config_bytes).hexdigest() == DEFAULT_CONFIG_SHA256, "安装资源与迁移前配置不一致"
    json.loads(config_bytes)

    executable = Path(sys.executable).parent / ("agentic-arxiv.exe" if os.name == "nt" else "agentic-arxiv")
    assert executable.is_file(), "当前Python安装环境缺少 agentic-arxiv console script"
    help_checks = []
    guard_checks = []
    with tempfile.TemporaryDirectory(prefix="agentic-arxiv-installed-help-") as temporary:
        cwd = Path(temporary).resolve()
        assert not cwd.is_relative_to(source_root), "帮助检查必须在仓库外目录执行"
        for arguments in HELP_COMMANDS:
            console = _run([str(executable), *arguments, "--help"], cwd)
            module = _run([sys.executable, "-m", "agentic_arxiv", *arguments, "--help"], cwd)
            assert "usage:" in console.stdout.lower(), f"缺少帮助: {arguments}"
            assert console.stdout.replace("agentic-arxiv", "agentic_arxiv") == module.stdout.replace(
                "agentic-arxiv", "agentic_arxiv"
            ), f"模块入口与 console 参数不一致: {arguments}"
            help_checks.append(" ".join(arguments) or "根命令")
        assert not list(cwd.iterdir()), "帮助写入了工作区产物"
        lightweight = _run([
            sys.executable, "-c",
            "import contextlib, io, sys; from agentic_arxiv.cli import main, HELP_COMMANDS; "
            "heavy={'torch','transformers','accelerate','trl','datasets','peft','bitsandbytes'}; "
            "\nfor command in HELP_COMMANDS:\n"
            " with contextlib.redirect_stdout(io.StringIO()):\n"
            "  try: main([*command, '--help'])\n"
            "  except SystemExit as exc: assert exc.code == 0\n"
            " assert not heavy.intersection(sys.modules), command\n"
            "print(len(HELP_COMMANDS))",
        ], cwd)
        assert int(lightweight.stdout.strip()) == len(HELP_COMMANDS), "帮助清单不完整"

        from agentic_arxiv.config import workspace_path

        expected = cwd / "missing-snapshot.json"
        assert workspace_path("missing-snapshot.json", workspace=cwd) == expected
        absolute = source_root / "README.md"
        assert workspace_path(absolute, workspace=cwd) == absolute, "绝对输入路径被改写"

        workspace = cwd / "workspace"
        workspace.mkdir()
        missing_snapshot = str(workspace / "data" / "mock_arxiv_snapshot.json")
        for location in (cwd, workspace):
            missing = subprocess.run([
                sys.executable, "-m", "agentic_arxiv", "--workspace", str(workspace), "run", "--offline",
            ], cwd=location, text=True, capture_output=True, timeout=90)
            assert missing.returncode == 2 and missing_snapshot in missing.stderr, (
                f"缺失输入未报告显式workspace实际路径: {missing.stdout}\n{missing.stderr}"
            )
        assert not list(workspace.iterdir()), "缺失输入检查意外生成了产物"

        guards = [
            ("本地能力或模型缺失", ["run", "--llm-backend", "transformers", "--model", str(workspace / "missing-model")],
             "agentic-arxiv[local]" if layer == "base" else str(workspace / "missing-model")),
            ("训练能力或输入缺失", ["train", "sft", "--model", str(workspace / "missing-model"),
             "--data", str(workspace / "missing-data.jsonl"), "--report-to", "none", "--no-verify", "--no-qlora"],
             str(workspace / "missing-data.jsonl") if layer == "train" else "agentic-arxiv[train]"),
        ]
        offline_env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        for label, arguments, expected_error in guards:
            guarded = subprocess.run([
                sys.executable, "-m", "agentic_arxiv", "--workspace", str(workspace), *arguments,
            ], cwd=cwd, text=True, capture_output=True, timeout=90, env=offline_env)
            assert guarded.returncode != 0 and expected_error in guarded.stdout + guarded.stderr, (
                f"缺失保护不符合安装层: {label}\n{guarded.stdout}\n{guarded.stderr}"
            )
            guard_checks.append(label)
        assert not list(workspace.iterdir()), "缺失能力/输入保护写入了工作区产物"

        if layer in {"local", "train"}:
            _run([
                sys.executable, "-c",
                "import sys; import agentic_arxiv.llm.local; "
                "assert not {'trl','datasets','peft','bitsandbytes'}.intersection(sys.modules)",
            ], cwd)
        if layer == "local":
            _run([
                sys.executable, "-c",
                "import sys; import torch, transformers, accelerate, peft; "
                "assert callable(peft.PeftModel.from_pretrained); "
                "assert not {'trl','datasets','bitsandbytes'}.intersection(sys.modules); "
                "assert not any(name.startswith('agentic_arxiv.training') for name in sys.modules)",
            ], cwd)

    from agentic_arxiv.tools.bootstrap import require_all_tools, registered_tool_count

    require_all_tools("基础安装声明检查")
    assert registered_tool_count() == 10, "基础安装没有完整注册10个工具"
    assert not (HEAVY_MODULES & set(sys.modules)), "工具声明检查加载了模型或训练栈"

    historical_records = None
    if history is not None:
        before = hashlib.sha256(history.read_bytes()).hexdigest()
        from agentic_arxiv.provenance import read_jsonl

        records = read_jsonl(history)
        historical_records = len(records)
        assert historical_records, f"现有历史输入为空: {history}"
        assert hashlib.sha256(history.read_bytes()).hexdigest() == before, "历史输入被修改"
        assert not (HEAVY_MODULES & set(sys.modules)), "历史读取加载了模型或训练栈"

    return {
        "安装层": layer,
        "正常wheel安装": require_wheel,
        "根包位置": str(origin),
        "Python": sys.version,
        "资源SHA256": hashlib.sha256(config_bytes).hexdigest(),
        "帮助命令": help_checks,
        "缺失保护": guard_checks,
        "工具声明数": registered_tool_count(),
        "历史输入": str(history) if history is not None else None,
        "历史记录数": historical_records,
        "未执行": ["真实API", "tokenizer", "模型加载", "训练", "数据生产", "完整任务集"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", choices=("base", "local", "train"), default="base")
    parser.add_argument("--require-wheel", action="store_true")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--history", type=Path, help="已有历史JSONL，只读检查，不生成替代输入")
    args = parser.parse_args()
    result = check_installation(args.layer, args.require_wheel, args.source_root.resolve(), args.history)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
