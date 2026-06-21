# AgenticArxiv/benchmark/metrics.py
"""从 Agent run() 结果中提取性能和准确性指标。"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class TaskMetrics:
    """单次任务执行的完整指标"""
    task_id: str
    agent_type: str
    trial: int
    session_id: str = ""

    # --- 性能 ---
    total_time_ms: int = 0
    iteration_count: int = 0
    total_llm_ms: int = 0
    total_tool_ms: int = 0
    framework_overhead_ms: int = 0
    avg_llm_ms: float = 0.0
    avg_tool_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # --- 准确性 ---
    task_completed: bool = False
    termination_type: str = "UNKNOWN"
    tool_call_sequence: List[str] = field(default_factory=list)
    expected_tools: List[str] = field(default_factory=list)
    tool_call_accurate: bool = False
    parse_failures: int = 0
    tool_exec_failures: int = 0

    # --- 原始数据 ---
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tool_call_sequence"] = ",".join(d["tool_call_sequence"])
        d["expected_tools"] = ",".join(d["expected_tools"])
        return d


def extract_metrics(
    task_def: Dict[str, Any],
    result: Dict[str, Any],
    agent_type: str,
    trial: int,
    session_id: str = "",
) -> TaskMetrics:
    """从 agent.run() 返回值提取 TaskMetrics"""
    history = result.get("history", [])
    timing = result.get("timing", {})
    token_usage = result.get("token_usage", {})

    # --- 性能指标 ---
    total_time_ms = result.get("total_time_ms", 0)
    total_llm_ms = timing.get("total_llm_ms", 0)
    total_tool_ms = timing.get("total_tool_ms", 0)
    framework_overhead_ms = timing.get("framework_overhead_ms", total_time_ms - total_llm_ms - total_tool_ms)
    iteration_count = result.get("iteration_count", len(history))

    effective_steps = max(1, iteration_count)
    avg_llm_ms = round(total_llm_ms / effective_steps, 1)
    avg_tool_ms = round(total_tool_ms / effective_steps, 1)

    # --- 准确性指标 ---
    termination_type = _get_termination_type(history)
    task_completed = termination_type == "FINISH"

    tool_sequence = _extract_tool_sequence(history)
    expected_tools = task_def.get("expected_tools", [])
    tool_call_accurate = _check_tool_sequence(tool_sequence, expected_tools)

    parse_failures = _count_parse_failures(history)
    tool_exec_failures = _count_tool_failures(history)

    error = None
    if termination_type == "ERROR" and history:
        error = history[-1].get("observation", "")

    return TaskMetrics(
        task_id=task_def["id"],
        agent_type=agent_type,
        trial=trial,
        session_id=session_id,
        total_time_ms=total_time_ms,
        iteration_count=iteration_count,
        total_llm_ms=total_llm_ms,
        total_tool_ms=total_tool_ms,
        framework_overhead_ms=framework_overhead_ms,
        avg_llm_ms=avg_llm_ms,
        avg_tool_ms=avg_tool_ms,
        prompt_tokens=token_usage.get("prompt_tokens", 0),
        completion_tokens=token_usage.get("completion_tokens", 0),
        total_tokens=token_usage.get("total_tokens", 0),
        task_completed=task_completed,
        termination_type=termination_type,
        tool_call_sequence=tool_sequence,
        expected_tools=expected_tools,
        tool_call_accurate=tool_call_accurate,
        parse_failures=parse_failures,
        tool_exec_failures=tool_exec_failures,
        error=error,
    )


def _get_termination_type(history: List[Dict]) -> str:
    if not history:
        return "NO_HISTORY"
    last_action = history[-1].get("action", "")
    if last_action == "FINISH":
        return "FINISH"
    elif last_action == "FORCE_STOP":
        return "FORCE_STOP"
    elif last_action == "ERROR":
        return "ERROR"
    # action 是 JSON 字符串（工具调用后没有正常终止）
    return "INCOMPLETE"


def _extract_tool_sequence(history: List[Dict]) -> List[str]:
    """从 history 中提取实际调用的工具名序列"""
    tools = []
    for step in history:
        action = step.get("action", "")
        if action in ("FINISH", "FORCE_STOP", "ERROR"):
            continue
        try:
            action_dict = json.loads(action)
            name = action_dict.get("name", "")
            if name:
                tools.append(name)
        except (json.JSONDecodeError, TypeError):
            pass
    return tools


def _check_tool_sequence(actual: List[str], expected: List[str]) -> bool:
    """检查实际工具调用是否包含所有预期工具（顺序匹配）"""
    if not expected:
        return True
    ei = 0
    for tool in actual:
        if ei < len(expected) and tool == expected[ei]:
            ei += 1
    return ei == len(expected)


def _count_parse_failures(history: List[Dict]) -> int:
    """统计解析失败次数（thought 存在但 action 为终止且非正常 FINISH）"""
    failures = 0
    for i, step in enumerate(history):
        action = step.get("action", "")
        # FINISH 在非最后一步出现（说明解析失败导致提前终止）
        if action == "FINISH" and i < len(history) - 1:
            failures += 1
        # observation 包含"无法解析"
        if "无法解析" in step.get("observation", ""):
            failures += 1
    return failures


def _count_tool_failures(history: List[Dict]) -> int:
    """统计工具执行失败次数"""
    failures = 0
    error_markers = ["错误:", "工具执行失败:", "命令失败", "命令执行超时", "命令执行异常"]
    for step in history:
        action = step.get("action", "")
        if action in ("FINISH", "FORCE_STOP", "ERROR"):
            continue
        obs = step.get("observation", "")
        if any(marker in obs for marker in error_markers):
            failures += 1
    return failures
