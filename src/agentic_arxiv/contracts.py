"""轻量动作、观测与轨迹契约；不读取当前协议或执行环境。"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, fields
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence


DEFAULT_OBSERVATION_BODY_LIMIT = 4000
TOOL_ERROR_PREFIX = "工具执行失败: "
_FRAMEWORK_ARG_KEYS = {"session_id", "_resolved_paper_id"}
_BODY_FIELDS = {
    "content", "summary", "abstract", "text", "full_text",
    "translated_content", "answer",
}
_FIELD_ORDER = (
    "ref", "id", "paper_id", "title", "status", "error", "offline",
    "source", "page", "total_pages", "count", "references", "papers",
)


class ParseKind(str, Enum):
    CALL = "CALL"
    FINISH = "FINISH"
    PARSE_ERROR = "PARSE_ERROR"


class TerminationType(str, Enum):
    FINISH = "FINISH"
    PARSE_ERROR = "PARSE_ERROR"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    MAX_TURNS = "MAX_TURNS"
    TOKEN_LIMIT = "TOKEN_LIMIT"
    MODEL_END = "MODEL_END"


@dataclass(frozen=True)
class ParsedAction:
    kind: ParseKind
    thought: str = ""
    action: Optional[Dict[str, Any]] = None
    raw_response: str = ""
    policy_text: str = ""
    error: Optional[str] = None


@dataclass(frozen=True)
class RenderedObservation:
    text: str
    truncation: Dict[str, Any] = field(default_factory=dict)


def is_finish_action(value: Any) -> bool:
    """模型只有完整 FINISH 能请求结束；框架标记不属于模型动作。"""
    return isinstance(value, str) and value.strip().upper() == "FINISH"


def business_action(action: Mapping[str, Any]) -> Dict[str, Any]:
    """框架字段不进入动作评分或标签；原模型输出另外完整保留。"""
    return {
        "name": action["name"],
        "args": {
            key: copy.deepcopy(value)
            for key, value in action["args"].items()
            if key not in _FRAMEWORK_ARG_KEYS
        },
    }


def render_observation(
    result: Any, body_limit: Optional[int] = DEFAULT_OBSERVATION_BODY_LIMIT,
    include_reference_map: bool = True,
) -> RenderedObservation:
    """稳定展示完整标识/状态，仅逐字段裁剪明确的正文。

    不对整个字符串结果粗暴切片：不透明字符串中的标识或错误位置未知。
    ``body_limit=None`` 用于已有评分 API 的完整结果投影。
    """
    if body_limit is not None and body_limit < 0:
        raise ValueError("观测正文预算必须非负")
    clipped = []
    remaining = body_limit
    body_chars = 0
    visible_body_chars = 0

    def key_order(key: Any):
        text = str(key)
        rank = _FIELD_ORDER.index(text) if text in _FIELD_ORDER else len(_FIELD_ORDER)
        return rank, text

    def visible(value: Any, path: str, field_name: str = "") -> Any:
        nonlocal remaining, body_chars, visible_body_chars
        if isinstance(value, Mapping):
            return {
                key: visible(value[key], f"{path}.{key}", str(key))
                for key in sorted(value, key=key_order)
            }
        if isinstance(value, (list, tuple)):
            return [visible(item, f"{path}[{i}]") for i, item in enumerate(value)]
        if isinstance(value, str) and field_name in _BODY_FIELDS:
            body_chars += len(value)
            kept = len(value) if remaining is None else min(len(value), remaining)
            visible_body_chars += kept
            if remaining is not None:
                remaining -= kept
            if kept < len(value):
                clipped.append({
                    "path": path, "original_chars": len(value), "visible_chars": kept,
                })
                return value[:kept] + "\n[正文已裁剪]"
        return value

    payload = result
    if include_reference_map and isinstance(result, list) and all(
        isinstance(item, Mapping) and "id" in item for item in result
    ):
        payload = {
            "count": len(result),
            "references": [
                {"ref": i, "paper_id": item["id"]}
                for i, item in enumerate(result, 1)
            ],
            "papers": result,
        }
    shown = visible(payload, "$")
    text = shown if isinstance(shown, str) else json.dumps(
        shown, ensure_ascii=False, separators=(",", ":"), default=str,
    )
    if isinstance(result, Mapping):
        error = result.get("error")
        if error or result.get("status") in {"FAILED", "ERROR"}:
            reason = str(error or result.get("status"))
            text = f"{TOOL_ERROR_PREFIX}{reason}\n{text}"
    return RenderedObservation(text, {
        "body_limit": body_limit, "body_chars": body_chars,
        "visible_body_chars": visible_body_chars,
        "truncated": bool(clipped), "fields": clipped,
    })


def copy_tool_result(value: Any) -> Any:
    """保存完整结构；资产时间转为 ISO 字符串，不修改工具返回对象。"""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {key: copy_tool_result(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [copy_tool_result(item) for item in value]
    return copy.deepcopy(value)


def make_history_step(
    parsed: ParsedAction, result: Any = None, observation: Any = None,
    error: Optional[str] = None, llm_latency_ms: int = 0,
    tool_latency_ms: int = 0, termination_reason: Optional[str] = None,
    token_usage: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """保留审计结果，同时暴露既有评分所需的动作和观测字段。"""
    actual_action = business_action(parsed.action) if parsed.action is not None else None
    action = json.dumps(actual_action, ensure_ascii=False) if actual_action else str(
        getattr(parsed.kind, "value", parsed.kind)
    )
    if parsed.kind == ParseKind.PARSE_ERROR:
        observation = f"无法解析 Action: {parsed.error or '格式错误'}"
        error = parsed.error
        termination_reason = TerminationType.PARSE_ERROR.value
    elif parsed.kind == ParseKind.FINISH:
        observation = "模型请求结束" if observation is None else observation
        termination_reason = TerminationType.FINISH.value
    elif error:
        if result is None:
            result = {"status": "FAILED", "error": str(error)}
        elif not isinstance(result, Mapping) or not result.get("error"):
            observation = f"{TOOL_ERROR_PREFIX}{error}"
    rendered = observation if isinstance(observation, RenderedObservation) else (
        render_observation(result) if observation is None else RenderedObservation(str(observation))
    )
    # 引用映射只装饰模型观测，不能凭新增 "papers"/"paper_id" 字段改变评分。
    scoring = render_observation(
        result, body_limit=None, include_reference_map=False,
    ).text if result is not None else rendered.text
    if error and not scoring.startswith((TOOL_ERROR_PREFIX, "无法解析")):
        scoring = f"{TOOL_ERROR_PREFIX}{error}\n{scoring}"
    return {
        "thought": parsed.thought, "action": action,
        "observation": rendered.text, "visible_observation": rendered.text,
        "scoring_observation": scoring,
        "parse_failed": parsed.kind == ParseKind.PARSE_ERROR,
        "raw_response": parsed.raw_response, "policy_text": parsed.policy_text,
        "parsed_action": actual_action, "tool_result": copy_tool_result(result),
        "observation_metadata": dict(rendered.truncation), "error": error,
        "termination_reason": termination_reason,
        "llm_latency_ms": llm_latency_ms, "tool_latency_ms": tool_latency_ms,
        "token_usage": dict(token_usage or {}),
    }


def make_terminal_step(
    reason: str, thought: str = "", error: str = "", raw_response: str = "",
    policy_text: str = "",
) -> Dict[str, Any]:
    """记录实际框架错误/预算，不伪造模型 FINISH 或模型响应。"""
    reason = str(getattr(reason, "value", reason))
    actions = {
        TerminationType.RUNTIME_ERROR.value: "ERROR",
        TerminationType.MAX_TURNS.value: "FORCE_STOP",
        TerminationType.TOKEN_LIMIT.value: "FORCE_STOP",
        TerminationType.MODEL_END.value: "FORCE_STOP",
    }
    if reason not in actions:
        raise ValueError(f"未知框架结束原因: {reason}")
    return {
        "thought": thought, "action": actions[reason], "observation": error,
        "visible_observation": error, "scoring_observation": error,
        "raw_response": raw_response, "policy_text": policy_text,
        "parsed_action": None, "tool_result": None,
        "parse_failed": False, "error": error or None,
        "termination_reason": reason, "framework_generated": True,
    }


def project_history(history: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    """新轨迹用未裁结果评分，旧记录保持原观测文本和语义。"""
    return [
        {**step, "observation": step["scoring_observation"]}
        if step.get("scoring_observation") is not None else dict(step)
        for step in history
    ]


def project_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {**result, "history": project_history(result.get("history") or [])}


@dataclass
class TrajectoryStep:
    """单个执行步骤。"""

    step: int
    thought: str
    action: str
    observation: str
    llm_latency_ms: int = 0
    tool_latency_ms: int = 0
    parse_failed: bool = False
    raw_response: Optional[str] = None
    policy_text: Optional[str] = None
    parsed_action: Optional[Dict[str, Any]] = None
    tool_result: Any = None
    visible_observation: Optional[str] = None
    scoring_observation: Optional[str] = None
    observation_metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    termination_reason: Optional[str] = None
    token_usage: Dict[str, Any] = field(default_factory=dict)
    framework_generated: bool = False
    extra_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """完整轨迹；缺失的历史协议字段保持未知。"""

    task_id: str
    task: str
    session_id: str
    steps: List[TrajectoryStep]
    final_reward: float
    metrics: Dict[str, Any]
    timestamp: str
    model: str = ""
    termination_type: str = ""
    reward_components: Dict[str, Any] = field(default_factory=dict)
    termination_reason: Optional[str] = None
    runtime_version: Optional[str] = None
    context_version: Optional[str] = None
    training_protocol: Optional[str] = None
    prompt_template_hash: Optional[str] = None
    chat_template_hash: Optional[str] = None
    observation_body_limit: Optional[int] = None
    training_protocol_metadata: Optional[Dict[str, Any]] = None
    protocol_relation: Optional[Dict[str, Any]] = None
    token_usage: Optional[Dict[str, Any]] = None
    extra_fields: Dict[str, Any] = field(default_factory=dict)


def _known_fields(cls, data: Mapping[str, Any]) -> Dict[str, Any]:
    """读取未知扩展字段时保留原值，不补造历史协议或原始响应。"""
    names = {item.name for item in fields(cls)}
    known = {key: value for key, value in data.items() if key in names}
    extras = dict(known.get("extra_fields") or {})
    extras.update({key: value for key, value in data.items() if key not in names})
    known["extra_fields"] = extras
    return known


def _flatten_extensions(data: Dict[str, Any]) -> Dict[str, Any]:
    extras = data.pop("extra_fields", {})
    return {**extras, **data}


def trajectory_to_dict(traj: Trajectory) -> Dict[str, Any]:
    """转换嵌套数据契约，同时保留历史扩展字段。"""
    data = _flatten_extensions(asdict(traj))
    data["steps"] = [_flatten_extensions(step) for step in data["steps"]]
    return data


def trajectory_from_dict(data: Mapping[str, Any]) -> Trajectory:
    """只消费已提供记录；不查询当前协议或修改输入。"""
    document = dict(data)
    steps_data = document.pop("steps", [])
    steps = [TrajectoryStep(**_known_fields(TrajectoryStep, step)) for step in steps_data]
    return Trajectory(steps=steps, **_known_fields(Trajectory, document))


def create_trajectory(
    task_id: str,
    task: str,
    session_id: str,
    history: List[Dict[str, Any]],
    final_reward: float,
    metrics: Dict[str, Any],
    model: str = "",
    termination_type: str = "FINISH",
    reward_components: Optional[Dict[str, Any]] = None,
    termination_reason: Optional[str] = None,
    protocol_fields: Optional[Mapping[str, Any]] = None,
    training_protocol_metadata: Optional[Mapping[str, Any]] = None,
    protocol_relation: Optional[Mapping[str, Any]] = None,
    token_usage: Optional[Mapping[str, Any]] = None,
) -> Trajectory:
    """从执行记录构造轨迹；当前协议事实必须由调用方显式提供。"""
    steps = []
    for i, step_dict in enumerate(history):
        step_data = {"thought": "", "action": "", "observation": "", **step_dict, "step": i + 1}
        steps.append(TrajectoryStep(**_known_fields(TrajectoryStep, step_data)))

    # 执行上下文已知不代表 checkpoint 的训练协议已知。
    metadata = dict(protocol_fields) if protocol_fields is not None else {}
    protocol_names = {
        "runtime_version", "context_version", "training_protocol",
        "prompt_template_hash", "chat_template_hash", "observation_body_limit",
    }
    return Trajectory(
        task_id=task_id,
        task=task,
        session_id=session_id,
        steps=steps,
        final_reward=final_reward,
        metrics=metrics,
        timestamp=datetime.now().isoformat(),
        model=model,
        termination_type=termination_type,
        reward_components=reward_components or {},
        termination_reason=termination_reason,
        training_protocol_metadata=(
            dict(training_protocol_metadata) if training_protocol_metadata is not None else None
        ),
        protocol_relation=dict(protocol_relation) if protocol_relation is not None else None,
        token_usage=dict(token_usage) if token_usage is not None else None,
        **{key: value for key, value in metadata.items() if key in protocol_names},
    )
