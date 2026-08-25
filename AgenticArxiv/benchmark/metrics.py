# AgenticArxiv/benchmark/metrics.py
"""从 Agent run() 结果中提取性能和准确性指标。"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


# 非工具调用的动作标记
NON_TOOL_ACTIONS = ("FINISH", "FORCE_STOP", "ERROR")


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
    # 参数级准确率（0~1）。没有 expected_tool_args 的任务恒为 1.0，不参与扣分。
    # 只看工具名的话，「下载标题含 X 的那篇」即使下错论文，
    # 工具序列仍是 [download_arxiv_pdf]，会被判为准确。
    arg_score: float = 1.0

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

    arg_score = argument_match_score(
        history, task_def.get("expected_tool_args"), expected_tools
    )
    if arg_score is None:
        arg_score = 1.0

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
        arg_score=arg_score,
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


def _parse_tool_action(action: Any) -> Optional[Dict[str, Any]]:
    """把 history 里的 action 解析成工具调用 dict；不是工具调用则返回 None。

    终止动作既可能是裸字符串 `FINISH`，也可能被模型包成和其他动作一样的
    JSON 外壳 `{"name": "FINISH", "args": {}}`。后者若不排除，会在
    tool_call_sequence 里多出一个 "FINISH"，把完全正确的轨迹判成
    accurate=False。Agent 侧已在解析阶段拦截（见
    agents/base_agent.py::is_terminal_action），这里再兜一层，
    以便正确处理历史结果文件和第三方 Agent 的轨迹。
    """
    if isinstance(action, str) and action.strip().upper() in NON_TOOL_ACTIONS:
        return None
    if isinstance(action, dict):
        parsed = action
    else:
        try:
            parsed = json.loads(action)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(parsed, dict):
        return None
    name = parsed.get("name")
    if isinstance(name, str) and name.strip().upper() in NON_TOOL_ACTIONS:
        return None
    return parsed


def _extract_tool_sequence(history: List[Dict]) -> List[str]:
    """从 history 中提取实际调用的工具名序列"""
    tools = []
    for step in history:
        parsed = _parse_tool_action(step.get("action", ""))
        if parsed is None:
            continue
        name = parsed.get("name", "")
        if name:
            tools.append(name)
    return tools


def _check_tool_sequence(actual: List[str], expected: List[str]) -> bool:
    """检查实际工具调用是否与预期序列完全一致（严格顺序、无多余/重复调用）。

    expected 为空表示**正确行为是一次工具都不调**（category="infeasible"：
    请求超出能力边界或指向不存在的对象）。此前这里无条件返回 True，于是
    幻觉调用也算「准确」——`_outcome_score` 会因为 task_completed and
    tool_call_accurate 给出满分，硬调不存在的第 20 篇反而拿到 outcome=+1.0。
    rl/reward.py 的 `_tool_score` 在同样输入下给 0.0，两边本就不一致。
    """
    if not expected:
        return not actual
    return actual == expected


def argument_match_score(history, expected_args, expected_tools=None):
    """参数级匹配度，返回 [0,1] 或 None（任务未声明 expected_tool_args）。

    每一步按「期望键里被答对的比例」打分，再对各步取平均。约定：

    - `expected_args is None`：任务没有参数标准答案，返回 None，
      `RewardCalculator` 会把 argument 这一档整个踢出加权分母。
    - `expected_args == []`：正确行为是**一次工具都不调**（category="infeasible"）。
      这与上一条不是一回事，必须能区分，否则乱调工具反而白拿满分。
    - 某一项为 `None`：该步存在但不校验参数。
    - 某个键的期望值为 `None`：该键应当缺省——省略不传、或显式传 None 都算对，
      工具会退回当前活跃论文。

    提取到此处是为了让 benchmark 报告也能反映参数准确率 ——
    此前它只存在于 RL 奖励里，TaskMetrics 中没有对应字段，
    于是「工具选对了但参数选错了」在 benchmark 里完全不可见。

    ## 为什么不再算 key_recall

    原实现是 `(键覆盖率 + 取值正确率) / 2`。键覆盖率几乎不携带独立信息：
    键缺失时 `predicted.get(k)` 就是 None，取值正确率同样判错。它唯一的
    效果是把一半的参数分白送给「照着工具签名把键填齐」的行为——而这正是
    任何能对该工具吐出合法 JSON 的模型免费拿到的。实测代价：一个无视任务、
    永远搜 cs.AI 的退化策略，在「检索 cs.CL」任务上参数分 0.833、总分 0.933。

    更糟的是期望值为 None 时它把方向搞反了：`{"ref": None}` 意为「用当前
    活跃论文」，正确写法是省略 ref，却被判键覆盖率 0；而错误地传 ref=1
    反倒拿满键覆盖率。两者最终都是 0.5——那 6 条 null 对照任务存在的唯一
    目的就是区分这两种行为，打分器却给不出任何区分。

    ## expected_tools

    传入后，第 i 步只有在**工具名也对**时才算参数分，否则该步 0 分；
    不传则退化为原来的纯按位比对（保持既有调用点兼容）。
    没有这道闸时参数分与工具名脱钩：`download_01` 期望
    `download_arxiv_pdf(ref=1)`，而 `get_paper_cache_status(ref=1)`
    照样拿满参数分——等于替调错的工具背书，权重 2 白送出去。
    """
    if expected_args is None:
        return None

    actual = []
    for step in history or []:
        parsed = _parse_tool_action(step.get("action", ""))
        if parsed is not None:
            actual.append((
                parsed.get("name"),
                parsed.get("parameters", parsed.get("args", {})) or {},
            ))

    if not expected_args:
        return 1.0 if not actual else 0.0

    scores = []
    for index, expected in enumerate(expected_args):
        if expected is None:
            continue
        name, predicted = actual[index] if index < len(actual) else (None, {})
        if (
            expected_tools is not None
            and index < len(expected_tools)
            and name != expected_tools[index]
        ):
            scores.append(0.0)
            continue
        keys = set(expected)
        if not keys:
            scores.append(1.0 if not predicted else 0.0)
            continue
        scores.append(sum(predicted.get(k) == v for k, v in expected.items()) / len(keys))
    return sum(scores) / len(scores) if scores else 1.0


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
        if action in NON_TOOL_ACTIONS:
            continue
        obs = step.get("observation", "")
        if any(marker in obs for marker in error_markers):
            failures += 1
    return failures
