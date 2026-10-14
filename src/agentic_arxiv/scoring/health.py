"""从已提供的轨迹计算纯健康指标和结构异常，不执行环境或文件操作。"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from agentic_arxiv.contracts import TOOL_ERROR_PREFIX


def _assistant_turns(completion: Any, trajectory: Mapping[str, Any]) -> List[str]:
    """Return model-authored text only, excluding environment observations."""
    raw_turns = trajectory.get("raw_assistant_turns")
    if isinstance(raw_turns, list):
        return [str(item or "") for item in raw_turns]

    if isinstance(completion, str):
        return [completion]
    if isinstance(completion, list):
        return [
            str(message.get("content") or "")
            for message in completion
            if isinstance(message, Mapping) and message.get("role") == "assistant"
        ]
    return [str(completion or "")]


def _parsed_tool_names(history: Sequence[Mapping[str, Any]]) -> List[str]:
    names: List[str] = []
    for step in history:
        action = str(step.get("action") or "")
        if action in {"FINISH", "FORCE_STOP", "ERROR", "PARSE_ERROR"}:
            continue
        try:
            parsed = json.loads(action)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict) and parsed.get("name"):
            names.append(str(parsed["name"]))
    return names


def detect_rollout_anomalies(
    completion: Any,
    trajectory: Mapping[str, Any],
    *,
    allowed_tools: Iterable[str] = (),
) -> Dict[str, bool]:
    """Apply structural checks to one model-generated multi-turn trajectory."""
    turns = _assistant_turns(completion, trajectory)
    history = trajectory.get("history") or []
    actions = [str(step.get("action") or "") for step in history]
    tool_steps = [
        step for step, action in zip(history, actions)
        if action not in {"FINISH", "FORCE_STOP", "ERROR", "PARSE_ERROR"}
    ]
    parse_error = any(
        bool(step.get("parse_failed")) or action == "PARSE_ERROR"
        for step, action in zip(history, actions)
    )
    action_markers = sum(turn.count("Action:") for turn in turns)
    think_counts = [
        (turn.count("<think>"), turn.count("</think>"))
        for turn in turns
    ]
    known = set(allowed_tools)
    used_tools = _parsed_tool_names(history)
    tool_error = any(
        str(step.get("observation") or "").startswith("工具执行失败: ")
        for step in tool_steps
    )
    missing_thought = any(not str(step.get("thought") or "").strip() for step in tool_steps)
    last_action = actions[-1] if actions else ""

    return {
        "empty_output": not any(turn.strip() for turn in turns),
        "missing_action": parse_error and action_markers == 0,
        "invalid_action": parse_error and action_markers > 0,
        "missing_thought": missing_thought,
        # Check each assistant turn independently. Multiple valid ReAct turns
        # may each contain one <think> block and must not be flagged globally.
        "unbalanced_think_tags": any(opened != closed for opened, closed in think_counts),
        "multiple_think_blocks": any(
            opened > 1 or closed > 1 for opened, closed in think_counts
        ),
        "multiple_actions_in_turn": any(turn.count("Action:") > 1 for turn in turns),
        # Only inspect model-authored turns. Environment-authored Observation
        # suffixes are intentionally excluded by _assistant_turns().
        "hallucinated_observation": any("Observation:" in turn for turn in turns),
        "unknown_tool": bool(known and any(name not in known for name in used_tools)),
        "tool_error": tool_error,
        "unfinished": bool(history) and last_action != "FINISH",
        "clipped": bool(trajectory.get("clipped")),
        "reached_max_turns": bool(trajectory.get("reached_max_turns")),
    }


def trajectory_health(history: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    """从一条 ReAct 轨迹里抽出多轮 rollout 的健康度指标。

    多轮采样落地后，``reward`` 掉下去可能是策略退化，也可能是环境侧出了问题
    （比如工具一直报错、或者模型压根没学会收尾而是每次都跑满 max_turns）。
    这几个量把这两类原因分开。

    Returns:
        - ``turns``：这条轨迹用了几步
        - ``finished``：是否以 FINISH 正常收尾（0/1）。持续为 0 说明模型在
          「跑满轮数」而不是「决定结束」，此时 reward 低与工具能力无关。
        - ``parse_error_rate``：解析失败步数占比
        - ``tool_error_rate``：工具调用步中执行失败的占比
    """
    turns = len(history)
    actions = [str(step.get("action", "")) for step in history]
    parse_errors = sum(
        1 for step, action in zip(history, actions)
        if step.get("parse_failed") or action == "PARSE_ERROR"
    )
    tool_steps = [
        step for step, action in zip(history, actions)
        if action not in ("FINISH", "PARSE_ERROR")
    ]
    tool_errors = sum(
        1 for step in tool_steps
        if str(step.get("observation", "")).startswith(TOOL_ERROR_PREFIX)
    )
    return {
        "turns": float(turns),
        "finished": 1.0 if actions and actions[-1] == "FINISH" else 0.0,
        "parse_error_rate": parse_errors / turns,
        "tool_error_rate": (tool_errors / len(tool_steps)) if tool_steps else 0.0,
    }
