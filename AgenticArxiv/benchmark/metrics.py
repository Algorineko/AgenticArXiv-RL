# AgenticArxiv/benchmark/metrics.py
"""从 Agent run() 结果中提取性能和准确性指标。"""

import json
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Sequence


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
    # 参数级准确率（0~1）。没有 expected_tool_args 时 score 用 1.0 保持中性，
    # applicable=False 让报告把它显示为 n/a，而不是虚假的 100%。
    # 只看工具名的话，「下载标题含 X 的那篇」即使下错论文，
    # 工具序列仍是 [download_arxiv_pdf]，会被判为准确。
    arg_score: float = 1.0
    arg_applicable: bool = False
    # 声称完成但期望工具没做全。与 tool_call_accurate 的区别见 is_false_finish：
    # 后者对「做多了」也判 False，这里只抓「做少了」。
    false_finish: bool = False
    # 指代解析准确率。未声明 expected_paper 时同样保持中性分，但标记为不适用。
    ref_score: float = 1.0
    ref_applicable: bool = False
    # blocked 任务需要在最终 Thought 中说明具体阻塞原因；普通任务不适用。
    terminal_semantics_accurate: bool = True
    terminal_semantics_applicable: bool = False

    # --- 原始数据 ---
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tool_call_sequence"] = ",".join(d["tool_call_sequence"])
        d["expected_tools"] = ",".join(d["expected_tools"])
        return d


def is_strict_success(metrics: TaskMetrics) -> bool:
    """任务真正成功：正常结束，且工具、参数、指代和执行过程都正确。"""
    return (
        metrics.task_completed
        and metrics.tool_call_accurate
        and metrics.arg_score == 1.0
        and metrics.ref_score == 1.0
        and metrics.parse_failures == 0
        and metrics.tool_exec_failures == 0
        and metrics.terminal_semantics_accurate
    )


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

    expected_tool_args = task_def.get("expected_tool_args")
    expected_paper_ids = task_def.get("expected_paper_ids")
    expected_paper = task_def.get("expected_paper")
    # Backward compatibility for hand-written one-paper tasks.
    if expected_paper_ids is None and expected_paper and len(expected_tools) == 1:
        expected_paper_ids = [expected_paper]
    arg_applicable = expected_tool_args is not None
    arg_score = argument_match_score(
        history, expected_tool_args, expected_tools, expected_paper_ids
    )
    if arg_score is None:
        arg_score = 1.0

    terminal_semantics_applicable = (
        task_def.get("expected_terminal_mode") == "blocked"
    )
    terminal_semantics_kind = (
        classify_blocked_terminal_semantics(task_def, history)
        if terminal_semantics_applicable
        else "not_applicable"
    )
    terminal_semantics_accurate = (
        terminal_semantics_kind == "explained_block"
        if terminal_semantics_applicable else True
    )
    false_finish = (
        is_false_finish(termination_type, tool_sequence, expected_tools)
        or terminal_semantics_kind == "false_completion"
    )

    ref_applicable = bool(expected_paper) or bool(
        expected_paper_ids and any(expected_paper_ids)
    )
    if expected_paper_ids is not None:
        ref_score = reference_resolution_score_by_step(
            history, expected_tools, expected_paper_ids
        )
    else:
        ref_score = reference_resolution_score(history, expected_paper)
    if ref_score is None:
        ref_score = 1.0

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
        arg_applicable=arg_applicable,
        false_finish=false_finish,
        ref_score=ref_score,
        ref_applicable=ref_applicable,
        terminal_semantics_accurate=terminal_semantics_accurate,
        terminal_semantics_applicable=terminal_semantics_applicable,
        error=error,
    )


def classify_blocked_terminal_semantics(
    task_def: Dict[str, Any], history: Sequence[Dict[str, Any]]
) -> str:
    """分类 blocked 任务的最终 Thought：具体拒绝、假完成或含糊终止。

    只读模型最终 FINISH 的 Thought，不读 Observation、task id、note 或标准
    工具答案。任务只提供抽象原因码，以免把参考答案文本泄漏给策略。
    """
    terminal_thought = ""
    for step in reversed(history):
        if str(step.get("action", "")).strip().upper() == "FINISH":
            terminal_thought = str(step.get("thought", "")).strip().lower()
            break

    blocked_markers = (
        "无法", "不能", "做不到", "不可", "不支持", "能力范围", "能力边界",
        "不存在", "未找到", "找不到", "无效", "非法", "超出", "越界", "缺少",
        "没有", "不在", "未提供", "请提供", "需要提供", "无从指代", "无法确定",
        "cannot", "can't", "unable", "unsupported", "not found", "does not exist",
        "invalid", "out of range", "missing", "not provided", "need context",
    )
    false_completion_markers = (
        "任务已完成", "操作已完成", "已经完成", "已成功完成", "任务完成",
        "operation completed", "task completed", "successfully completed",
        "already completed", "done successfully",
    )
    expected_reason = str(task_def.get("expected_terminal_reason") or "")
    reason_is_grounded = _terminal_reason_is_grounded(
        terminal_thought, expected_reason
    )
    if (
        any(marker in terminal_thought for marker in blocked_markers)
        and reason_is_grounded
    ):
        return "explained_block"
    if any(marker in terminal_thought for marker in false_completion_markers):
        return "false_completion"
    return "ambiguous"


def _terminal_reason_is_grounded(text: str, reason: str) -> bool:
    """要求“对象 + 错误性质”同时出现，避免只蹭一个关键词拿满分。

    例如 ``缺少必须的 ref 参数`` 对 ``ref=0`` 任务并不正确：用户已经给了
    ref，问题是 0 违反 1-based 索引约束。旧判据看到 ``ref`` 就通过，这个
    双因子判据要求同时出现“索引/ref”和“非法/越界/从1开始”。
    """
    def has_any(markers: Sequence[str]) -> bool:
        return any(marker in text for marker in markers)

    if reason == "missing_context":
        return has_any((
            "会话", "上下文", "刚才", "指代", "最近操作", "session", "context",
            "referent", "previous paper",
        )) and has_any((
            "没有", "缺少", "未提供", "无从", "无法解析", "不能确定", "找不到",
            "missing", "not provided", "no ", "cannot resolve", "unknown",
        ))
    if reason == "invalid_reference":
        return has_any((
            "索引", "序号", "ref", "第0篇", "第20篇", "index", "reference",
        )) and has_any((
            "无效", "非法", "超出", "越界", "范围", "上限", "下限", "从 1",
            "从1", "1-based", "invalid", "out of range", "outside", "zero",
        ))
    if reason == "paper_not_found":
        return has_any((
            "论文", "arxiv", "id", "快照", "记录", "paper", "snapshot", "record",
        )) and has_any((
            "不存在", "未找到", "找不到", "不在", "无记录", "unknown", "not found",
            "does not exist", "absent",
        ))
    if reason == "unsupported_capability":
        return has_any((
            "工具", "功能", "能力", "邮箱", "邮件", "tool", "capability", "email",
        )) and has_any((
            "不支持", "无法", "不能", "做不到", "超出", "范围", "边界", "unsupported",
            "cannot", "unable", "out of scope",
        ))
    return False


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


def lcs_length(left: Sequence[Any], right: Sequence[Any]) -> int:
    """最长公共子序列长度。

    与 rl/reward.py 共用一份实现。此前两处各有一份 —— 同一个 helper
    在多处复制，是 _precision_flags 那个 bug 的成因（三份里两份没跟上修复）。
    """
    row = [0] * (len(right) + 1)
    for left_item in left:
        previous = 0
        for j, right_item in enumerate(right, 1):
            saved = row[j]
            row[j] = previous + 1 if left_item == right_item else max(row[j], row[j - 1])
            previous = saved
    return row[-1]


def is_false_finish(
    termination_type: str,
    tool_sequence: Sequence[str],
    expected_tools: Sequence[str],
) -> bool:
    """声称完成，但期望的工具调用没做全。

    与 `not tool_call_accurate` 不是一回事，两者刻意区分：
    严格序列比对对「做多了」和「做少了」一视同仁，而这里只抓「做少了」。

        搜索 → 下载 → 多查一次缓存 → FINISH     召回率 1，不算假完成（只是绕路）
        查缓存 → FINISH（本该还要下载）          召回率 <1，**假完成**

    抓的是「不做事也能拿分」这条奖励漏洞。实测 state_cache_before_dl
    24 次运行里 18 次属于此类（75%）：查完缓存直接 FINISH，
    task_completed=True，任务根本没做。

    判据只用任务已声明的 expected_tools，不引入终态谓词，
    也不需要为每个任务写检查逻辑。
    """
    if termination_type != "FINISH" or not expected_tools:
        return False
    return lcs_length(tool_sequence, expected_tools) < len(expected_tools)


# observation 里 paper_id 出现过三种写法，都要认：
#   {'paper_id': '2608.14539v1', ...}      dict 的 str()，单引号
#   {"paper_id": "2608.14539v1", ...}      JSON，双引号
#   已创建翻译任务 task_id=..., paper_id=None    工具自己拼的可读文本
_PAPER_ID_PATTERNS = (
    re.compile(r"""['"]paper_id['"]\s*:\s*['"]([^'"]+)['"]"""),
    re.compile(r"""\bpaper_id\s*=\s*([A-Za-z0-9._/-]+)"""),
)
_VERSION_SUFFIX = re.compile(r"v\d+$")


def resolved_paper_id(observation: Any) -> Optional[str]:
    """从 observation 里取出工具**实际解析到**的 paper_id，取不到返回 None。"""
    text = observation if isinstance(observation, str) else str(observation or "")
    for pattern in _PAPER_ID_PATTERNS:
        match = pattern.search(text)
        if match:
            value = match.group(1).strip()
            return None if value.lower() in ("none", "null", "") else value
    return None


def _same_paper(left: str, right: str) -> bool:
    """比较论文身份时忽略版本号：v1 与 v2 是同一篇论文的不同版本。"""
    return _VERSION_SUFFIX.sub("", left.strip()) == _VERSION_SUFFIX.sub("", right.strip())


def reference_resolution_score(
    history: Sequence[Dict[str, Any]], expected_paper: Optional[str]
) -> Optional[float]:
    """指代解析准确率：工具解析到的论文，是不是任务指的那篇。

    返回 [0,1]，任务未声明 expected_paper 时返回 None。

    `ref` 支持序号 / arXiv ID / 标题子串 / null，是本 agent 真正的难点，
    而按字符串比对参数对它**同时会假阳性和假阴性**：

        假阳性  期望 {"ref":"Learning State"} 实际 {"ref":"Learning State"}
                参数分满分，但子串匹配到了另一篇 —— 下错了论文
        假阴性  期望 {"ref":"Decoding the Past"} 实际 {"ref":3}
                参数分接近 0，但两者解析到同一篇 —— 其实做对了

    所以不比参数，比工具返回值里的 paper_id：**指代形式随便用，
    但必须落到正确的那篇论文**。

    只统计观测里带 paper_id 的步骤 —— 检索类工具不解析单篇论文，不计入。
    一步都没解析到，说明 agent 压根没碰论文，记 0。
    """
    if not expected_paper:
        return None
    resolved = [
        pid for step in (history or [])
        if (pid := resolved_paper_id(step.get("observation"))) is not None
    ]
    if not resolved:
        return 0.0
    return sum(_same_paper(pid, expected_paper) for pid in resolved) / len(resolved)


def reference_resolution_score_by_step(
    history: Sequence[Dict[str, Any]],
    expected_tools: Sequence[str],
    expected_paper_ids: Sequence[Optional[str]],
) -> Optional[float]:
    """Score paper identity at the corresponding expected tool step.

    Unlike the legacy single-paper helper this supports chains that operate on
    several different papers.  Tool order remains strict; semantic equivalence
    only relaxes the representation of ``ref``.
    """
    targets = [i for i, paper_id in enumerate(expected_paper_ids) if paper_id]
    if not targets:
        return None
    actual = []
    for step in history or []:
        parsed = _parse_tool_action(step.get("action", ""))
        if parsed is not None:
            actual.append((parsed.get("name"), step.get("observation")))
    scores = []
    for index in targets:
        if index >= len(actual) or index >= len(expected_tools):
            scores.append(0.0)
            continue
        name, observation = actual[index]
        paper_id = resolved_paper_id(observation)
        scores.append(float(
            name == expected_tools[index]
            and bool(paper_id)
            and _same_paper(paper_id, str(expected_paper_ids[index]))
        ))
    return sum(scores) / len(scores)


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


def _match_arg_value(predicted_val: Any, expected_val: Any, key: str = "") -> bool:
    """Robust equivalence check between predicted and expected tool arguments."""
    if expected_val is None:
        # None 表示缺省参数，省略不传或显式传 None 均算对
        return predicted_val is None

    if predicted_val is None:
        return False

    if predicted_val == expected_val:
        return True

    # 1. aspect 字段归一化：如 "cs.AI" 与 "AI" 等价，大小写不敏感
    if key == "aspect":
        pred_norm = str(predicted_val).strip().upper()
        exp_norm = str(expected_val).strip().upper()
        if pred_norm.startswith("CS."):
            pred_norm = pred_norm[3:]
        if exp_norm.startswith("CS."):
            exp_norm = exp_norm[3:]
        return pred_norm == exp_norm

    # 2. ref 字段归一化：支持整数与字符串数字等价，支持形如 "第1篇" 的正则匹配
    if key == "ref":
        try:
            # 两边都能转成整数时，这一步就是定论，不该再往下抠数字。
            return int(predicted_val) == int(expected_val)
        except (ValueError, TypeError):
            pass
        # 只有取值不是纯数字（"第1篇"）时才退回正则。`-?` 不能省：str(-1) 里的
        # 减号不属于 \d+，写成 \d+ 会从 "-1" 里抠出 "1" 判成对。ref 是 1-based
        # 序号，-1 是越界值，正是 wrong_args 基线用来制造错误参数的取值。
        m = re.search(r"-?\d+", str(predicted_val))
        if m:
            try:
                if int(m.group(0)) == int(expected_val):
                    return True
            except (ValueError, TypeError):
                pass
        return str(predicted_val).strip().lower() == str(expected_val).strip().lower()

    # 3. 通用数值类型比对：如 7 vs "7", 5.0 vs 5
    if isinstance(expected_val, (int, float)):
        try:
            return float(predicted_val) == float(expected_val)
        except (ValueError, TypeError):
            pass

    # 4. 布尔类型：字符串 "true"/"false" 与 bool 兼容
    if isinstance(expected_val, bool):
        if isinstance(predicted_val, str):
            return predicted_val.lower() == str(expected_val).lower()

    # 5. 字符串大小写宽松比对
    if isinstance(expected_val, str) and isinstance(predicted_val, str):
        return expected_val.strip().lower() == predicted_val.strip().lower()

    return False


def argument_match_score(
    history,
    expected_args,
    expected_tools=None,
    expected_paper_ids=None,
):
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
                step.get("observation"),
            ))

    if not expected_args:
        return 1.0 if not actual else 0.0

    scores = []
    for index, expected in enumerate(expected_args):
        if expected is None:
            continue
        name, predicted, observation = (
            actual[index] if index < len(actual) else (None, {}, None)
        )
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
        matched = 0
        for key, value in expected.items():
            semantic_paper = (
                key == "ref"
                and expected_paper_ids is not None
                and index < len(expected_paper_ids)
                and expected_paper_ids[index]
            )
            if semantic_paper:
                resolved = resolved_paper_id(observation)
                matched += int(bool(resolved) and _same_paper(
                    resolved, str(expected_paper_ids[index])
                ))
            else:
                matched += int(_match_arg_value(predicted.get(key), value, key))
        scores.append(matched / len(keys))
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
