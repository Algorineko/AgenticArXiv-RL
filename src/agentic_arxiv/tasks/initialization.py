"""将任务依赖与 setup 纯编译为运行时需要的动作序列。"""

from __future__ import annotations

from typing import Any, Mapping

from agentic_arxiv.tasks.tasks import get_task_by_id


def compile_initialization(
    task_def: Mapping[str, Any], *,
    task_lookup: Any = None,
) -> list[dict[str, Any]]:
    """保留原依赖查找范围、动作顺序和重复，不执行工具或携带评分字段。

    显式目录优先，缺少的依赖仅从原冒烟任务目录查找。前置任务先初始化，
    再添加其标准步骤，最后添加当前任务的 setup；当前任务标准步骤不入计划。
    """
    lookup = task_lookup or {}
    if not isinstance(lookup, Mapping):
        lookup = {task["id"]: task for task in lookup}
    visited: set[str] = set()
    plan: list[dict[str, Any]] = []

    def append_action(name: str, args: Any) -> None:
        plan.append({"name": name, "args": dict(args or {})})

    def initialize(task: Mapping[str, Any]) -> None:
        dependency = task.get("depends_on")
        if dependency:
            if dependency in visited:
                raise ValueError("任务初始化存在循环依赖")
            visited.add(dependency)
            prior = lookup.get(dependency) or get_task_by_id(dependency)
            if prior is None:
                raise ValueError(f"任务初始化依赖不存在: {dependency}")
            initialize(prior)
            tools, args = prior.get("expected_tools") or [], prior.get("expected_tool_args")
            if args is None or len(tools) != len(args):
                raise ValueError("依赖初始化缺少可执行的标准步骤参数")
            for name, arguments in zip(tools, args):
                append_action(name, arguments)
        for action in task.get("setup") or []:
            append_action(action["name"], action.get("args"))

    initialize(task_def)
    return plan
