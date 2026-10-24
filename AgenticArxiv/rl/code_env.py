"""代码生成域环境 (v0.8 轨道 A): CodeMultiTurnEnv — run_tests 工具的确定性执行面。

契约同 rl/multiturn_env.py: reset(task_id) + 每工具一方法。无需快照回放
(run_tests 天然确定性)。注册表条目由本模块加载时注册(工具面=方法面)。
"""
from __future__ import annotations

from typing import Any, Dict

from AgenticArxiv.benchmark.tasks_code import get_code_task


class CodeMultiTurnEnv:
    """单工具环境: agent 提交函数代码 -> run_tests 执行预生成单测。"""

    def __init__(self) -> None:
        self._task: Dict[str, Any] | None = None

    # ---- 工具面 ----
    def run_tests(self, code: str = "", task_id: str = "", **_: Any) -> Dict[str, Any]:
        if self._task is None or (task_id and task_id != self._task["id"]):
            spec = get_code_task(task_id)
            if spec is None:
                return {"ok": False, "error": f"unknown task_id: {task_id}", "results": []}
            self._task = spec
        from AgenticArxiv.tools.run_tests_tool import execute_harness

        obs = execute_harness(code, self._task["checks_src"])
        obs["task_id"] = self._task["id"]
        return obs

    # ---- 环境契约 ----
    def reset(self, task_id: str) -> None:
        spec = get_code_task(task_id)
        if spec is None:
            raise KeyError(f"unknown code task: {task_id}")
        self._task = spec


def make_code_environment_factory():
    def factory():
        return CodeMultiTurnEnv()
    return factory
