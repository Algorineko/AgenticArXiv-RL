"""只读重算已有轨迹指标，不加载模型、不执行参考答案或环境。"""

from __future__ import annotations

import json
from typing import Any, Dict, Tuple

from agentic_arxiv.config import workspace_path
from agentic_arxiv.evaluation.report import BenchmarkReport
from agentic_arxiv.scoring.metrics import extract_metrics
from agentic_arxiv.tasks.splits import load_split
from agentic_arxiv.tasks.tasks import get_all_tasks
from agentic_arxiv.tasks.tasks_expanded import get_expanded_tasks


_SEARCH_TOOLS = frozenset({"get_recently_submitted_cs_papers", "search_arxiv_papers"})


def _key(row: Dict[str, Any]) -> Tuple[str, str, int]:
    return row["task_id"], row.get("agent_type", "regex"), int(row.get("trial", 0))


def _task_with_references(task, trace, references, *, location):
    """只消费独立提供的标准答案，绝不从实际策略观测反推论文身份。"""
    tools = task.get("expected_tools") or []
    paper_steps = [index for index, name in enumerate(tools) if name not in _SEARCH_TOOLS]
    declared = task.get("expected_paper_ids")
    if declared is None and task.get("expected_paper") and len(tools) == 1:
        declared = [task["expected_paper"]]
    sources = (
        ("--references", references.get(task["id"])),
        ("轨迹 expected_paper_ids", trace.get("expected_paper_ids")),
        ("任务显式参考", declared),
    )
    expected = None
    for label, value in sources:
        if value is None:
            continue
        if (not isinstance(value, list) or len(value) != len(tools)
                or any(item is not None and (not isinstance(item, str) or not item.strip()) for item in value)
                or any(not value[index] for index in paper_steps)
                or any(value[index] is not None for index, name in enumerate(tools) if name in _SEARCH_TOOLS)):
            raise ValueError(f"{location}: 任务 {task['id']} 的 {label} 必须与 expected_tools 对齐并声明论文身份")
        if expected is not None and value != expected:
            raise ValueError(f"{location}: 任务 {task['id']} 的冻结参考冲突，无法重算相关指标")
        expected = list(value)
    if paper_steps and expected is None:
        raise ValueError(
            f"{location}: 任务 {task['id']} 缺少冻结的 expected_paper_ids，"
            "无法重算论文身份、参数及严格成功指标；请用 --references 提供已有参考。"
            "不会从策略观测推断答案或执行环境来补齐参考。"
        )
    if expected is None:
        return task
    return {**task, "expected_paper_ids": expected}


def execute(args) -> None:
    """只读冻结参考；缺少论文身份时拒绝输出不同口径的严格指标。"""
    workspace = getattr(args, "workspace", None)
    trace_path = workspace_path(args.traces, workspace=workspace)
    summary_path = workspace_path(args.summary, workspace=workspace)
    output_path = workspace_path(args.output, workspace=workspace)
    inputs = [trace_path, summary_path]
    if getattr(args, "snapshot", None):
        inputs.append(workspace_path(args.snapshot, workspace=workspace))
    reference_path = None
    if getattr(args, "references", None):
        reference_path = workspace_path(args.references, workspace=workspace)
        inputs.append(reference_path)
    for path in inputs:
        if not path.exists():
            raise SystemExit(f"缺少文件: {path}")
    references = json.loads(reference_path.read_text(encoding="utf-8")) if reference_path else {}
    if not isinstance(references, dict):
        raise ValueError(f"冻结参考必须为 task_id 到 expected_paper_ids 列表的 JSON 对象: {reference_path}")

    pool = get_expanded_tasks() if args.task_set == "expanded" else get_all_tasks()
    if args.split:
        wanted = set(load_split(args.split, workspace=workspace))
        pool = [task for task in pool if task["id"] in wanted]
        missing = wanted - {task["id"] for task in pool}
        if missing:
            raise SystemExit(f"切分中有任务不在任务池: {sorted(missing)[:3]}")
    by_id = {task["id"]: task for task in pool}

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    old_details = {_key(row): row for row in summary.get("details", [])}
    metrics = []
    unknown = set()
    with trace_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            trace = json.loads(line)
            task = by_id.get(trace.get("task_id"))
            if task is None:
                unknown.add(str(trace.get("task_id")))
                continue
            task = _task_with_references(
                task, trace, references, location=f"{trace_path}:{line_number}",
            )
            detail = old_details.get(_key(trace), {})
            trace_timing = trace.get("timing") or {}
            if "token_usage" in trace:
                # 显式未知不可被旧详情中的零覆盖。
                declared_usage = trace["token_usage"]
                trace_usage = declared_usage if isinstance(declared_usage, dict) else {}
            else:
                trace_usage = {"total_tokens": detail.get("tokens")}
            result = {
                "history": trace.get("history") or [],
                "total_time_ms": trace.get(
                    "total_time_ms", detail.get("total_ms", 0)
                ),
                "iteration_count": trace.get(
                    "iteration_count",
                    detail.get("iterations", len(trace.get("history") or [])),
                ),
                "timing": {
                    "total_router_ms": trace_timing.get(
                        "total_router_ms", detail.get("router_ms", 0)
                    ),
                    "total_llm_ms": trace_timing.get(
                        "total_llm_ms", detail.get("llm_ms", 0)
                    ),
                    "total_tool_ms": trace_timing.get(
                        "total_tool_ms", detail.get("tool_ms", 0)
                    ),
                    "framework_overhead_ms": trace_timing.get(
                        "framework_overhead_ms", detail.get("overhead_ms", 0)
                    ),
                },
                "token_usage": trace_usage,
            }
            metrics.append(extract_metrics(
                task,
                result,
                trace.get("agent_type", "regex"),
                int(trace.get("trial", 0)),
                session_id=trace.get("session_id", ""),
            ))

    if unknown:
        print(f"提示：跳过 {len(unknown)} 个不属于所选任务池/切分的 task_id")
    if not metrics:
        raise SystemExit("没有可重算的轨迹")

    # 沿用原执行及模型声明，不把本次代码读取当成新的模型运行。
    report = BenchmarkReport(
        metrics, model=summary.get("model", "unknown"),
        protocol_metadata=summary,
        training_metadata=summary.get("training_protocol_metadata"),
    )
    report.print_report()
    report.save_all(str(output_path))
    print(
        f"\n重评分完成: {len(metrics)} 条；未调用模型、工具或参考答案服务；"
        f"输出: {output_path}"
    )
