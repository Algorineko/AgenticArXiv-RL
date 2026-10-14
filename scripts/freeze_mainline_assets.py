"""冻结/核对主线业务资产；不调用工具、不联网、不改原任务与切分。"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/mainline_baseline/assets.json"
FRAMEWORK_PARAMETERS = {"session_id"}


def canonical_bytes(value: object) -> bytes:
    """键排序、无额外空白、UTF-8；列表顺序（尤其注册/任务顺序）不变。"""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def collect_assets() -> dict:
    """必须在新进程中收集，避免其他测试先导入工具导致注册顺序污染。"""
    sys.path.insert(0, str(ROOT / "AgenticArxiv"))
    from benchmark.tasks import BENCHMARK_SPECS, BENCHMARK_TASKS
    from benchmark.tasks_expanded import EXPANDED_SPECS, EXPANDED_TASKS
    from tools.bootstrap import TOOL_MODULES, require_all_tools
    from tools.tool_registry import registry

    require_all_tools("资产冻结")
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
    for path in sorted((ROOT / "data/splits").glob("*.json")):
        split_files[path.relative_to(ROOT).as_posix()] = {
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
    # TaskSpec 的 tuple 在 JSON 中表示为数组；直接核对与落盘读取使用同一表示。
    return json.loads(canonical_bytes(assets))


def fresh_assets() -> dict:
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--dump"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze", action="store_true", help="新建基线，拒绝覆盖已存在文件")
    mode.add_argument("--dump", action="store_true", help="仅打印当前规范化清单")
    args = parser.parse_args()
    assets = collect_assets()
    if args.dump:
        print(canonical_bytes(assets).decode("utf-8"))
        return
    if args.freeze:
        document = {
            "format_version": "mainline-assets-s0-v1",
            "normalization": "UTF-8 JSON；键排序；无额外空白；保留数组顺序；禁止 NaN",
            "scope": "冻结业务 schema（仅排除框架 session_id）、注册顺序、完整 TaskSpec/展开任务与所有历史切分原文内容和文件字节哈希；不证明训练或模型性能",
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            ).strip(),
            "sources": [
                "AgenticArxiv/tools/bootstrap.py",
                "AgenticArxiv/benchmark/task_spec.py",
                "AgenticArxiv/benchmark/tasks.py",
                "AgenticArxiv/benchmark/tasks_expanded.py",
                *assets["split_files"],
            ],
            "section_sha256": {key: sha256(value) for key, value in assets.items()},
            "assets_sha256": sha256(assets),
            "assets": assets,
        }
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        with BASELINE.open("x", encoding="utf-8") as file:
            file.write(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
        print(f"已新建资产基线：{BASELINE}")
        return
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    if assets != baseline["assets"] or sha256(assets) != baseline["assets_sha256"]:
        raise SystemExit("资产与冻结基线不一致；请检查差异，不要覆盖历史基线")
    print("主线业务资产与冻结基线一致")


if __name__ == "__main__":
    main()
