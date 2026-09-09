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


_BLOCKED_TERMINAL_THOUGHTS = {
    "missing_context": (
        "当前会话没有必要上下文，无法解析用户所指的论文",
        "缺少会话上下文和最近操作的论文，因此无法解析该指代",
    ),
    "invalid_reference": (
        "请求中的 ref 序号无效或超出 1-based 索引范围，无法执行",
        "该 ref 序号违反 1-based 索引范围，属于非法取值，不能执行",
    ),
    "paper_not_found": (
        "指定的 arXiv 论文 ID 不在快照记录中，无法下载该论文",
        "快照中不存在指定的 arXiv 论文 ID，因此找不到该论文，不能下载",
    ),
    "unsupported_capability": (
        "现有工具能力不支持该操作，任务超出工具范围",
        "现有工具不支持所需功能，该请求超出能力边界，无法执行",
    ),
}


def reference_terminal_thought(
    task: Mapping[str, Any], *, variant: int = 0
) -> str:
    """按任务终止契约生成可验证的专家 Thought，不暴露 task id。"""
    if task.get("expected_terminal_mode") != "blocked":
        return "所有步骤均已成功执行，任务已完成"
    reason = str(task.get("expected_terminal_reason") or "")
    choices = _BLOCKED_TERMINAL_THOUGHTS.get(reason)
    if not choices:
        raise ValueError(f"未知的 blocked terminal reason: {reason!r}")
    return choices[variant % len(choices)]


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
        terminal_mode: 终止的业务语义。``completed`` 表示任务已执行完成；
            ``blocked`` 表示任务因缺少上下文、非法参数或能力边界而无法执行，
            此时模型必须在 FINISH 前明确说明阻塞原因，不能只声称“已完成”。
        terminal_reason: ``blocked`` 的可验证原因类型；奖励只接受与该类型
            一致的解释，避免模型用一句泛化的“无法执行”刷满分。
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
    terminal_mode: str = "completed"
    terminal_reason: Optional[str] = None
    max_iterations: Optional[int] = None
    depends_on: Optional[str] = None

    def to_task(self) -> Dict[str, Any]:
        """展开成 benchmark / RL 两侧共用的任务字典。"""
        if self.terminal_mode not in {"completed", "blocked"}:
            raise ValueError(
                f"任务 {self.id!r} 的 terminal_mode={self.terminal_mode!r} 无效；"
                "只支持 'completed' 或 'blocked'"
            )
        valid_reasons = {
            "missing_context", "invalid_reference", "paper_not_found",
            "unsupported_capability",
        }
        if self.terminal_mode == "blocked" and self.terminal_reason not in valid_reasons:
            raise ValueError(
                f"阻塞任务 {self.id!r} 必须声明 terminal_reason；"
                f"可选值为 {sorted(valid_reasons)}"
            )
        if self.terminal_mode != "blocked" and self.terminal_reason is not None:
            raise ValueError(
                f"非阻塞任务 {self.id!r} 不能声明 terminal_reason"
            )
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
        # 默认值不落盘，保持既有 43 条任务及其历史评测 fixture 字节兼容。
        if self.terminal_mode != "completed":
            task["expected_terminal_mode"] = self.terminal_mode
            task["expected_terminal_reason"] = self.terminal_reason
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
