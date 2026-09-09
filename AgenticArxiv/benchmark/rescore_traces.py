"""Recompute benchmark metrics from saved trajectories without model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

from benchmark.metrics import extract_metrics
from benchmark.report import BenchmarkReport
from benchmark.semantic_oracle import attach_expected_paper_ids
from benchmark.splits import load_split
from benchmark.tasks import get_all_tasks
from benchmark.tasks_expanded import get_expanded_tasks
from tools.bootstrap import require_all_tools


def _key(row: Dict[str, Any]) -> Tuple[str, str, int]:
    return row["task_id"], row.get("agent_type", "regex"), int(row.get("trial", 0))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="用新评分规则重算已有 traces；不加载模型，只回放冻结快照 oracle"
    )
    parser.add_argument("--traces", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--snapshot", default="data/mock_arxiv_snapshot.json")
    parser.add_argument("--task-set", choices=["default", "expanded"], default="expanded")
    parser.add_argument("--split", default=None, metavar="[FILE:]NAME")
    args = parser.parse_args()
    require_all_tools("轨迹重评分")

    trace_path = Path(args.traces)
    summary_path = Path(args.summary)
    snapshot_path = Path(args.snapshot)
    for path in (trace_path, summary_path, snapshot_path):
        if not path.exists():
            raise SystemExit(f"缺少文件: {path}")

    pool = get_expanded_tasks() if args.task_set == "expanded" else get_all_tasks()
    if args.split:
        wanted = set(load_split(args.split))
        pool = [task for task in pool if task["id"] in wanted]
        missing = wanted - {task["id"] for task in pool}
        if missing:
            raise SystemExit(f"切分中有任务不在任务池: {sorted(missing)[:3]}")
    tasks = attach_expected_paper_ids(pool, snapshot_path)
    by_id = {task["id"]: task for task in tasks}

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
            detail = old_details.get(_key(trace), {})
            result = {
                "history": trace.get("history") or [],
                "total_time_ms": detail.get("total_ms", 0),
                "iteration_count": detail.get(
                    "iterations", len(trace.get("history") or [])
                ),
                "timing": {
                    "total_llm_ms": detail.get("llm_ms", 0),
                    "total_tool_ms": detail.get("tool_ms", 0),
                    "framework_overhead_ms": detail.get("overhead_ms", 0),
                },
                "token_usage": {"total_tokens": detail.get("tokens", 0)},
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

    report = BenchmarkReport(metrics, model=summary.get("model", "unknown"))
    report.print_report()
    report.save_all(args.output)
    print(
        f"\n重评分完成: {len(metrics)} 条；未调用模型，"
        f"仅回放冻结快照标准答案；输出: {args.output}"
    )


if __name__ == "__main__":
    main()
