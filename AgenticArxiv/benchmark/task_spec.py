"""任务声明层：`expected_tools` / `expected_tool_args` 由同一份 steps 派生。

对应 README TODO P1「任务集扩充 …… 并支持自动派生 `expected_tools` /
`expected_tool_args`，奖励才有区分度」。

## 为什么要这一层

原来每条任务手写两份平行的标准答案：

    "expected_tools":     ["download_arxiv_pdf"],
    "expected_tool_args": [{"ref": 1}],

两份列表靠人肉保持对齐，一旦漏写就会**静默**削弱奖励——`argument_match_score`
在 `expected_tool_args` 为 None 时返回 None，`RewardCalculator` 会把 argument
这一档整个踢出加权分母，任务照跑、分照打，只是参数从此不再被检查。
`benchmark/tasks.py` 里 `download_01` / `translate_01` / `cache_01` 三条
就处在这个状态。

这里把两份列表收敛成一份 `steps`，两者都从它派生，结构上不可能漂移。

## 参数化家族

`family()` 让一个模板 + 一组参数展开成多条任务：任务的自然语言描述和标准答案
都由**同一份参数**渲染，因此描述里写 `days=7`、标准答案却是 `days=30` 这种
不一致也无法发生。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Step:
    """一次工具调用。task 的标准答案由若干 Step 组成。"""

    tool: str
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskSpec:
    """一条基准任务的声明式定义。

    Attributes:
        steps: 标准解法的工具调用序列。空序列表示**正确行为是不调用任何工具**
            （见 category="infeasible"）——这不是"没写标准答案"，两者在
            `expected_tool_args` 上的区别是 `[]` 与 `None`。
        setup: 跑任务前先替会话铺好的状态（由 runner 直接执行，不计入轨迹）。
        max_iterations: 该任务允许的 ReAct 轮数上限。Agent 默认 5 轮，也就是
            最多 4 次工具调用 + 一次 FINISH；更长的链必须显式抬高，否则会被
            判成 FORCE_STOP 而非能力不足。
        depends_on: 前置任务 id。同一会话里按依赖顺序跑，供
            `tasks.get_dependency_chain` 使用。
    """

    id: str
    task: str
    steps: Tuple[Step, ...] = ()
    category: str = "misc"
    difficulty: str = "medium"
    setup: Tuple[Step, ...] = ()
    template: Optional[str] = None
    requires_offline: bool = False
    note: str = ""
    termination: str = "FINISH"
    max_iterations: Optional[int] = None
    depends_on: Optional[str] = None

    def to_task(self) -> Dict[str, Any]:
        """展开成 benchmark / RL 两侧共用的任务字典。"""
        task: Dict[str, Any] = {
            "id": self.id,
            "task": self.task,
            # 两份标准答案同源，长度必然相等
            "expected_tools": [s.tool for s in self.steps],
            "expected_tool_args": [dict(s.args) for s in self.steps],
            "expected_termination": self.termination,
            "category": self.category,
            "difficulty": self.difficulty,
        }
        if self.setup:
            task["setup"] = [{"name": s.tool, "args": dict(s.args)} for s in self.setup]
        if self.template:
            task["template"] = self.template
        if self.requires_offline:
            task["requires_offline"] = True
        if self.note:
            task["note"] = self.note
        if self.max_iterations is not None:
            task["max_iterations"] = self.max_iterations
        if self.depends_on:
            task["depends_on"] = self.depends_on
        return task


def family(
    *,
    text: Callable[[Mapping[str, Any]], str],
    steps: Callable[[Mapping[str, Any]], Sequence[Step]],
    params: Iterable[Mapping[str, Any]],
    task_id: Callable[[Mapping[str, Any]], str],
    **shared: Any,
) -> List[TaskSpec]:
    """把一个模板按参数展开成一族任务。

    `text` 和 `steps` 收到同一份参数，因此任务描述与标准答案不会各说各话。

    Args:
        text: 参数 -> 自然语言任务描述
        steps: 参数 -> 标准工具调用序列
        params: 参数组
        task_id: 参数 -> 任务 id
        **shared: 该族共享的 TaskSpec 字段（category / difficulty / template …）
    """
    out: List[TaskSpec] = []
    for p in params:
        out.append(TaskSpec(
            id=task_id(p),
            task=text(p),
            steps=tuple(steps(p)),
            **shared,
        ))
    return out


def build(specs: Iterable[TaskSpec]) -> List[Dict[str, Any]]:
    """展开成任务字典列表，顺带查一遍 id 唯一。"""
    tasks = [s.to_task() for s in specs]
    seen: Dict[str, int] = {}
    for t in tasks:
        seen[t["id"]] = seen.get(t["id"], 0) + 1
    dupes = sorted(k for k, v in seen.items() if v > 1)
    if dupes:
        raise ValueError(f"任务 id 重复: {dupes}")
    return tasks
