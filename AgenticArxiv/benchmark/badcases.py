"""坏例回放：把一次失败固化成一条永久的回归用例。

一次 benchmark 跑完，失败的轨迹只留在报告的聚合数字里。下次改了打分口径
或者改了 prompt，没有任何东西告诉你「上次那个查完缓存就 FINISH 的毛病
到底修没修好」——只能重跑一遍全量，再从均值里猜。

这里把单条轨迹连同它当时的判定一起冻住。回放不需要 LLM、不需要网络、
不需要工具：轨迹是死的，重新跑的只有打分器。所以它能回答的问题很窄，
但很确定：**同一条轨迹，今天的代码还认为它是坏的吗。**

两种用法，同一份数据：

  - 坏例回归：`status="open"` 的用例记着「这个毛病还在」。修好之后回放
    会报 newly_fixed，把 status 改成 "fixed"，从此它反过来防复发。
  - reward hacking 案例库：退化策略的轨迹（不解任务只骗分）配上
    `reproduces_when={"reward": {"ge": 0.6}}`，就是一条「这种行为不许拿
    到 0.6 分」的断言。#40 与 #47 修的两个洞都属于这一类。

判定条件写成 reproduces_when 而不是「整份判定必须一模一样」：后者会被任何
无关的口径变化（多加一个指标、换个归一化）打成 regression，那样的用例
没人会留着。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from benchmark.metrics import TaskMetrics
from rl.reward import RewardCalculator

# 回放时比对的字段。刻意只取判定类的，不取耗时 / token —— 那些取决于
# 当时那台机器和那个模型，回放里没有意义，留着只会制造假的 regression。
VERDICT_FIELDS = (
    "reward",
    "task_completed",
    "termination_type",
    "tool_call_accurate",
    "false_finish",
    "arg_score",
    "ref_score",
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
    # 捕获当时的判定。不参与「是否复现」的判断，只用来显示漂移了多少。
    captured: Dict[str, Any] = field(default_factory=dict)

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
        return out


@dataclass(frozen=True)
class ReplayOutcome:
    case: BadCase
    verdict: Dict[str, Any]
    reproduces: bool
    outcome: str          # still_open / newly_fixed / regressed / stays_fixed
    reward_drift: Optional[float]

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
    """用今天的打分器给一条死轨迹判分。"""
    calc = calculator or RewardCalculator()
    breakdown, metrics = calc.compute_reward_breakdown(
        dict(task),
        {"history": list(history), "timing": {}, "token_usage": {},
         "iteration_count": len(history)},
        agent_type="badcase-replay",
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
    }


def matches(verdict: Mapping[str, Any], condition: Mapping[str, Any]) -> bool:
    """判定条件是否成立。

    标量按相等比；`{"ge": 0.6}` 这类按比较算子比。后者是给 reward hacking
    用的：「这种轨迹不许拿到 0.6 分」是个阈值断言，写死等于某个数会在
    任何一次权重调整后失效。
    """
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
    """回放全部用例。任务 id 找不到时直接报错——静默跳过等于悄悄少测。"""
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
        reproduces = matches(verdict, case.reproduces_when)
        if case.status == STATUS_OPEN:
            outcome = "still_open" if reproduces else "newly_fixed"
        else:
            outcome = "regressed" if reproduces else "stays_fixed"
        drift = None
        if "reward" in case.captured and isinstance(case.captured["reward"], (int, float)):
            drift = round(verdict["reward"] - case.captured["reward"], 6)
        outcomes.append(ReplayOutcome(case, verdict, reproduces, outcome, drift))
    return outcomes


# ---- 读写 ----

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
            # 缺字段 / status 写错都在这里落地。不带上行号的话，几十条用例里
            # 找那一条全靠肉眼。
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


# ---- 捕获 ----

def is_badcase(verdict: Mapping[str, Any]) -> bool:
    """值不值得固化成用例。

    只挑「跑完了、判定却不对」的：异常终止本来就一眼可见，不需要用例来提醒。
    这几条各自对应一类沉默失败：
      false_finish        声称完成但没做全
      ref_score < 1       指代解析到了别的论文
      完成但工具序列错     做了别的事却算完成
    """
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
    """从 (task_id, history) 里挑出坏例，连同当时的判定一起固化。

    samples 是 (task_id, history) 的可迭代对象——benchmark 跑完的
    BenchmarkResult.raw_result["history"] 直接能喂进来。
    """
    by_id = {str(t["id"]): t for t in tasks}
    calculator = RewardCalculator()
    out, counter = [], {}
    for task_id, history in samples:
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
        ))
    return out


def _condition_for(verdict: Mapping[str, Any]) -> Dict[str, Any]:
    """挑一个能代表这条坏例的判定条件，而不是把整份判定钉死。"""
    if verdict.get("false_finish"):
        return {"false_finish": True}
    if verdict.get("ref_score") is not None and verdict["ref_score"] < 1.0:
        return {"ref_score": {"lt": 1.0}}
    return {"task_completed": True, "tool_call_accurate": False}
