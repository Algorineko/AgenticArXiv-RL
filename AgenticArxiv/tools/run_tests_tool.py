"""代码生成域工具: run_tests —— 确定性执行预生成单元测试（v0.8 轨道 A）。

域契约：模型在 Action 里提交函数代码并调用 run_tests(code=..., task_id=...)；
本工具把代码嵌入测试 harness，在隔离子进程中限时执行（-I 模式、无网环境、
CPU/AS 资源限额），返回逐测试通过/失败与错误摘要。测试本体由
benchmark/tasks_code.py 按task_id静态携带 —— 无状态、天然确定性、无需快照回放。

RL 路径的真实实现走 CodeMultiTurnEnv.run_tests（env 方法即工具面）；
本注册表条目服务于 prompt 工具描述与 benchmark runner 的独立调用，二者共享
同一执行核（本文件的 execute_harness），保证训练/评测口径一致。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # 注册表缺失时执行核仍可用（独立测试/非 agent 上下文）
    from tools.tool_registry import registry  # noqa: F401
except ImportError:  # pragma: no cover
    registry = None

SCHEMA = {
    "type": "object",
    "properties": {
        "code": {
            "type": "string",
            "description": "完整 Python 函数定义（def ...），可含 import；将被嵌入测试 harness 执行",
        },
        "task_id": {"type": "string", "description": "任务 id，定位预生成的单元测试集"},
    },
    "required": ["code", "task_id"],
}

TIMEOUT_S = 8
MEM_LIMIT_BYTES = 512 * 1024 * 1024

# harness 模板: 模型代码在前, 检查在后(lambda 内引用函数, NameError 被 check 捕获)
_HARNESS_TEMPLATE = '''\
import json, sys

RESULTS = []

def check(name, fn):
    try:
        fn()
        RESULTS.append([name, True, ""])
    except Exception as e:  # noqa: BLE001
        RESULTS.append([name, False, repr(e)[:200]])

# ---- model code ----
{model_code}
# ---- checks ----
{checks}

print("@@RESULTS@@" + json.dumps(RESULTS))
'''


def _build_harness(model_code: str, checks_src: str) -> str:
    if "def " not in model_code:
        model_code = model_code + "\n# (no def found)"
    return _HARNESS_TEMPLATE.format(model_code=model_code, checks=checks_src)


def _rlimits():
    import resource

    def _apply():
        resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT_BYTES, MEM_LIMIT_BYTES))
        resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT_S, TIMEOUT_S))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))

    return _apply


def execute_harness(model_code: str, checks_src: str) -> Dict[str, Any]:
    """执行核: env 方法与注册表工具共用。返回 JSON 安全的观测 dict。"""
    src = _build_harness(model_code, checks_src)
    with tempfile.TemporaryDirectory(prefix="run_tests_") as td:
        path = os.path.join(td, "harness.py")
        Path(path).write_text(src, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", path],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_S + 4,
                preexec_fn=_rlimits(),
                cwd=td,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "timeout", "results": [], "stdout_tail": ""}
    out = proc.stdout or ""
    marker = "@@RESULTS@@"
    if marker in out:
        try:
            results: List[List[Any]] = json.loads(out.split(marker, 1)[1].strip().splitlines()[0])
        except (ValueError, IndexError):
            results = []
        passed = sum(1 for r in results if r and r[1])
        return {
            "ok": True,
            "results": results,
            "passed": passed,
            "total": len(results),
            "stdout_tail": out[-400:],
            "stderr_tail": (proc.stderr or "")[-400:],
            "returncode": proc.returncode,
        }
    return {
        "ok": False,
        "error": "no-result-marker",
        "results": [],
        "stdout_tail": out[-400:],
        "stderr_tail": (proc.stderr or "")[-400:],
        "returncode": proc.returncode,
    }


def run_tests(code: str = "", task_id: str = "", **_: Any) -> Dict[str, Any]:
    """注册表入口: 从任务定义取 checks 源, 走执行核。"""
    from benchmark.tasks_code import get_code_task

    spec: Optional[Dict[str, Any]] = get_code_task(task_id)
    if spec is None:
        return {"ok": False, "error": f"unknown task_id: {task_id}", "results": []}
    obs = execute_harness(code, spec["checks_src"])
    obs["task_id"] = task_id
    return obs


if registry is not None:
    registry.register_tool(
        name="run_tests",
    description=(
        "提交函数代码并运行该任务的预生成单元测试。返回逐测试通过/失败与错误摘要。"
        "code 须是完整可执行的函数定义(可含 import)；task_id 为当前任务 id。"
    ),
    parameter_schema=SCHEMA,
    func=run_tests,
)
