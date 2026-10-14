"""坏例回放：保留历史只读重评分与显式完整 episode 执行的边界。

历史 replay 只使用已保存的轨迹和提供的任务声明，不创建环境、不执行工具、
不补充语义参考答案。实际 run_cases 使用共同运行时和独立的参考准备服务。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from agentic_arxiv.contracts import project_result
from agentic_arxiv.provenance import read_protocol_metadata
from agentic_arxiv.scoring.metrics import TaskMetrics, is_strict_success
from agentic_arxiv.scoring.reward import RewardCalculator

# 回放只比对判定字段；机器、模型相关的时间与 token 不作为回归条件。
VERDICT_FIELDS = (
    "reward",
    "task_completed",
    "termination_type",
    "tool_call_accurate",
    "false_finish",
    "arg_score",
    "ref_score",
    "ref_applicable",
    "strict_success",
)

_COMPARATORS = {
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
}

STATUS_OPEN = "open"
STATUS_FIXED = "fixed"


@dataclass(frozen=True)
class BadCase:
    """一条固化的坏例。"""

    case_id: str
    task_id: str
    history: List[Dict[str, Any]]
    reproduces_when: Dict[str, Any]
    status: str = STATUS_OPEN
    source: str = ""
    note: str = ""
    captured: Dict[str, Any] = field(default_factory=dict)
    protocol_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in (STATUS_OPEN, STATUS_FIXED):
            raise ValueError(f"status 只能是 {STATUS_OPEN}/{STATUS_FIXED}，得到 {self.status!r}")
        if not self.reproduces_when:
            raise ValueError(f"用例 {self.case_id} 没写 reproduces_when，回放时无从判断")
        unknown = set(self.reproduces_when) - set(VERDICT_FIELDS)
        if unknown:
            raise ValueError(
                f"用例 {self.case_id} 的 reproduces_when 引用了不存在的字段 {sorted(unknown)}，"
                f"可用: {list(VERDICT_FIELDS)}"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BadCase":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "case_id": self.case_id,
            "task_id": self.task_id,
            "status": self.status,
            "reproduces_when": self.reproduces_when,
            "history": self.history,
        }
        if self.source:
            out["source"] = self.source
        if self.note:
            out["note"] = self.note
        if self.captured:
            out["captured"] = self.captured
        if self.protocol_metadata:
            out["protocol_metadata"] = self.protocol_metadata
        return out


@dataclass(frozen=True)
class ReplayOutcome:
    case: BadCase
    verdict: Dict[str, Any]
    reproduces: bool
    outcome: str
    reward_drift: Optional[float]
    execution: str = "historical-rescore"
    protocol_metadata: Dict[str, Any] = field(default_factory=dict)
    protocol_relation: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_regression(self) -> bool:
        return self.outcome == "regressed"


def verdict_of(
    task: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    *,
    training_step: int = 100,
    calculator: Optional[RewardCalculator] = None,
) -> Dict[str, Any]:
    """用当前打分器给一条已保存的轨迹判分。"""
    return verdict_of_result(
        task, {"history": list(history), "timing": {}, "token_usage": {},
               "iteration_count": len(history)},
        training_step=training_step, calculator=calculator,
    )


def verdict_of_result(
    task: Mapping[str, Any], result: Mapping[str, Any], *,
    training_step: int = 100, calculator: Optional[RewardCalculator] = None,
    agent_type: str = "badcase-replay",
) -> Dict[str, Any]:
    """只读实际轨迹及评分投影，不执行工具或补造结束步骤。"""
    if "history" not in result or result["history"] is None:
        raise ValueError("坏例评分缺少实际 history")
    calc = calculator or RewardCalculator()
    breakdown, metrics = calc.compute_reward_breakdown(
        dict(task), project_result(result),
        agent_type=agent_type,
        training_step=training_step,
    )
    return _verdict_from(breakdown.total, metrics)


def _verdict_from(reward: float, metrics: TaskMetrics) -> Dict[str, Any]:
    return {
        "reward": reward,
        "task_completed": metrics.task_completed,
        "termination_type": metrics.termination_type,
        "tool_call_accurate": metrics.tool_call_accurate,
        "false_finish": metrics.false_finish,
        "arg_score": metrics.arg_score,
        "ref_score": metrics.ref_score,
        "ref_applicable": metrics.ref_applicable,
        "strict_success": is_strict_success(metrics),
    }


def matches(verdict: Mapping[str, Any], condition: Mapping[str, Any]) -> bool:
    """标量按相等比较，映射按声明的阈值比较；全部条件必须成立。"""
    for key, want in condition.items():
        got = verdict.get(key)
        if isinstance(want, Mapping):
            for op, bound in want.items():
                if op not in _COMPARATORS:
                    raise ValueError(f"未知比较算子 {op!r}，可用: {sorted(_COMPARATORS)}")
                if got is None or not _COMPARATORS[op](got, bound):
                    return False
        elif got != want:
            return False
    return True


def replay(
    cases: Iterable[BadCase],
    tasks: Sequence[Mapping[str, Any]],
    *,
    training_step: int = 100,
) -> List[ReplayOutcome]:
    """历史只读重评分，不运行 Agent 或环境；任务缺失直接报错。"""
    by_id = {str(t["id"]): t for t in tasks}
    calculator = RewardCalculator()
    outcomes = []
    for case in cases:
        task = by_id.get(case.task_id)
        if task is None:
            raise KeyError(
                f"用例 {case.case_id} 指向的任务 {case.task_id!r} 不在当前任务集里。"
                "任务被删或改名时，绑在它上面的用例必须一起处理"
            )
        verdict = verdict_of(task, case.history,
                             training_step=training_step, calculator=calculator)
        outcomes.append(_replay_outcome(case, verdict, "historical-rescore"))
    return outcomes


def run_cases(
    cases: Iterable[BadCase], tasks: Sequence[Mapping[str, Any]], agent: Any, *,
    training_step: int = 100,
) -> List[ReplayOutcome]:
    """显式执行当前模型完整 episode，不把历史送给模型或改写旧用例。"""
    from agentic_arxiv.tasks.initialization import compile_initialization

    case_list = list(cases)
    by_id = {str(task["id"]): task for task in tasks}
    selected = []
    for task_id in dict.fromkeys(case.task_id for case in case_list):
        if task_id not in by_id:
            raise KeyError(f"坏例指向的任务 {task_id!r} 不在当前任务集里")
        selected.append(by_id[task_id])
    environment = getattr(agent, "env", None)
    snapshot = getattr(getattr(environment, "backend", None), "snapshot_path", None)
    if getattr(environment, "mode", None) == "replay" and snapshot:
        from agentic_arxiv.tasks.reference import attach_expected_paper_ids

        selected = attach_expected_paper_ids(selected, snapshot, task_lookup=by_id)
        by_id.update({str(task["id"]): task for task in selected})
    calculator = RewardCalculator()
    outcomes = []
    for index, case in enumerate(case_list):
        task = by_id.get(case.task_id)
        if task is None:
            raise KeyError(f"用例 {case.case_id} 指向的任务 {case.task_id!r} 不在当前任务集里")
        result = agent.run(
            task["task"],
            initialization_plan=compile_initialization(task, task_lookup=by_id),
            session_id=f"badcase_{index}",
        )
        verdict = verdict_of_result(
            task, result, training_step=training_step, calculator=calculator,
            agent_type="badcase-episode",
        )
        outcomes.append(_replay_outcome(
            case, verdict, "complete-episode",
            protocol=result.get("protocol_metadata"), relation=result.get("protocol_relation"),
        ))
    return outcomes


def _replay_outcome(case, verdict, execution, *, protocol=None, relation=None):
    reproduces = matches(verdict, case.reproduces_when)
    if case.status == STATUS_OPEN:
        outcome = "still_open" if reproduces else "newly_fixed"
    else:
        outcome = "regressed" if reproduces else "stays_fixed"
    drift = None
    if "reward" in case.captured and isinstance(case.captured["reward"], (int, float)):
        drift = round(verdict["reward"] - case.captured["reward"], 6)
    return ReplayOutcome(
        case, verdict, reproduces, outcome, drift, execution,
        read_protocol_metadata(case.protocol_metadata if protocol is None else protocol),
        dict(relation or {}),
    )


def load_cases(path: Path) -> List[BadCase]:
    cases, seen = [], set()
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno} 不是合法 JSON: {exc}") from exc
        try:
            case = BadCase.from_dict(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{lineno} 不是合法用例: {exc}") from exc
        if case.case_id in seen:
            raise ValueError(f"{path}:{lineno} case_id 重复: {case.case_id}")
        seen.add(case.case_id)
        cases.append(case)
    return cases


def dump_cases(cases: Sequence[BadCase], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(c.to_dict(), ensure_ascii=False) + "\n" for c in cases),
        encoding="utf-8",
    )


def is_badcase(verdict: Mapping[str, Any]) -> bool:
    """仅固化 FINISH 后的假完成、错误指代或错误工具序列。"""
    if verdict.get("termination_type") != "FINISH":
        return False
    return bool(
        verdict.get("false_finish")
        or (verdict.get("ref_score") is not None and verdict["ref_score"] < 1.0)
        or not verdict.get("tool_call_accurate")
    )


def capture(
    samples: Iterable[tuple],
    tasks: Sequence[Mapping[str, Any]],
    *,
    source: str = "",
    training_step: int = 100,
) -> List[BadCase]:
    """从 (task_id, history[, protocol]) 挑出坏例，保留捕获时判定。"""
    by_id = {str(t["id"]): t for t in tasks}
    calculator = RewardCalculator()
    out, counter = [], {}
    for sample in samples:
        task_id, history = sample[:2]
        metadata = read_protocol_metadata(sample[2] if len(sample) > 2 else None)
        task = by_id.get(str(task_id))
        if task is None or not history:
            continue
        verdict = verdict_of(task, history,
                             training_step=training_step, calculator=calculator)
        if not is_badcase(verdict):
            continue
        counter[task_id] = counter.get(task_id, 0) + 1
        out.append(BadCase(
            case_id=f"{task_id}#{counter[task_id]}",
            task_id=str(task_id),
            history=[dict(step) for step in history],
            reproduces_when=_condition_for(verdict),
            status=STATUS_OPEN,
            source=source,
            captured=verdict,
            protocol_metadata=metadata if metadata["metadata_status"] != "unknown" else {},
        ))
    return out


def _condition_for(verdict: Mapping[str, Any]) -> Dict[str, Any]:
    if verdict.get("false_finish"):
        return {"false_finish": True}
    if verdict.get("ref_score") is not None and verdict["ref_score"] < 1.0:
        return {"ref_score": {"lt": 1.0}}
    return {"task_completed": True, "tool_call_accurate": False}
