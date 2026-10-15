"""只读核对主线业务资产；显式冻结仅可新建基线，不调用工具或联网。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys


BASELINE = Path("artifacts/mainline_baseline/assets.json")
FRAMEWORK_PARAMETERS = {"session_id"}


def canonical_bytes(value: object) -> bytes:
    """键排序、无额外空白；列表顺序和原冻结规范完全相同。"""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def collect_assets(workspace: str | Path) -> dict:
    """收集完整声明；调用方应使用 fresh_assets 隔离工具注册顺序。"""
    from agentic_arxiv.tasks.tasks import BENCHMARK_SPECS, BENCHMARK_TASKS
    from agentic_arxiv.tasks.tasks_expanded import EXPANDED_SPECS, EXPANDED_TASKS
    from agentic_arxiv.tools.bootstrap import TOOL_MODULES, require_all_tools
    from agentic_arxiv.tools.tool_registry import registry

    workspace = Path(workspace).resolve()
    require_all_tools("资产核对")
    tools = []
    for tool in registry.list_tools():
        schema = deepcopy(tool["parameters"])
        for name in FRAMEWORK_PARAMETERS:
            schema.get("properties", {}).pop(name, None)
        if "required" in schema:
            schema["required"] = [
                name for name in schema["required"] if name not in FRAMEWORK_PARAMETERS
            ]
        tools.append({"name": tool["name"], "parameters": schema})
    expected_names = [name for names in TOOL_MODULES.values() for name in names]
    if [tool["name"] for tool in tools] != expected_names:
        raise ValueError("实际注册顺序不等于 bootstrap 声明顺序，不能冻结")
    split_files = {}
    for path in sorted((workspace / "data/splits").glob("*.json")):
        split_files[path.relative_to(workspace).as_posix()] = {
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "definition": json.loads(path.read_text(encoding="utf-8")),
        }
    assets = {
        "tools": tools,
        "expanded_task_specs": [asdict(spec) for spec in EXPANDED_SPECS],
        "expanded_tasks": EXPANDED_TASKS,
        "smoke_task_specs": [asdict(spec) for spec in BENCHMARK_SPECS],
        "smoke_tasks": BENCHMARK_TASKS,
        "split_files": split_files,
    }
    # tuple 的规范化表示仍为 JSON 数组，直接比较与落盘比较一致。
    return json.loads(canonical_bytes(assets))


def fresh_assets(workspace: str | Path) -> dict:
    """从已安装包启动独立进程，不注入 PYTHONPATH 或源码搜索路径。"""
    workspace = Path(workspace).resolve()
    result = subprocess.run(
        [sys.executable, "-I", "-c", (
            "import sys; from agentic_arxiv.evaluation.assets import "
            "collect_assets, canonical_bytes; "
            "print(canonical_bytes(collect_assets(sys.argv[1])).decode('utf-8'))"
        ), str(workspace)],
        cwd=workspace, check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def audit_assets(
    workspace: str | Path,
    *,
    baseline: str | Path = BASELINE,
    freeze: bool = False,
    dump: bool = False,
) -> dict:
    """默认只读全内容核对；dump 返回清单，freeze 保留拒绝覆盖保护。"""
    if freeze and dump:
        raise ValueError("--freeze 和 --dump 不能同时使用")
    workspace = Path(workspace).resolve()
    baseline = Path(baseline)
    if not baseline.is_absolute():
        baseline = workspace / baseline
    baseline = baseline.resolve()
    if freeze and baseline.exists():
        raise FileExistsError(f"基线已存在，拒绝覆盖：{baseline}")
    if not freeze and not dump and not baseline.is_file():
        raise FileNotFoundError(f"资产基线不存在：{baseline}")
    assets = fresh_assets(workspace)
    if dump:
        return assets
    if freeze:
        document = {
            "format_version": "mainline-assets-s0-v1",
            "normalization": "UTF-8 JSON；键排序；无额外空白；保留数组顺序；禁止 NaN",
            "scope": "冻结业务 schema（仅排除框架 session_id）、注册顺序、完整 TaskSpec/展开任务与所有历史切分原文内容和文件字节哈希；不证明训练或模型性能",
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=workspace, text=True,
            ).strip(),
            "sources": [
                "src/agentic_arxiv/tools/bootstrap.py",
                "src/agentic_arxiv/tasks/task_spec.py",
                "src/agentic_arxiv/tasks/tasks.py",
                "src/agentic_arxiv/tasks/tasks_expanded.py",
                *assets["split_files"],
            ],
            "section_sha256": {key: sha256(value) for key, value in assets.items()},
            "assets_sha256": sha256(assets),
            "assets": assets,
        }
        baseline.parent.mkdir(parents=True, exist_ok=True)
        with baseline.open("x", encoding="utf-8") as file:
            file.write(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
        return {"status": "baseline_created", "baseline": str(baseline)}
    document = json.loads(baseline.read_text(encoding="utf-8"))
    if assets != document["assets"] or sha256(assets) != document["assets_sha256"]:
        raise ValueError("资产与冻结基线不一致；请检查差异，不要覆盖历史基线")
    return {
        "status": "assets_match", "baseline": str(baseline),
        "assets_sha256": sha256(assets), "sections": list(assets),
    }
