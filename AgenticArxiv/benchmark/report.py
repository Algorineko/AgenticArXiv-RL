# AgenticArxiv/benchmark/report.py
"""Benchmark 报告生成：Markdown 对比表格 + CSV + JSON"""

import csv
import json
import os
from collections import defaultdict
from typing import List, Dict, Any, Optional

from benchmark.metrics import TaskMetrics


class BenchmarkReport:
    """从 TaskMetrics 列表生成多种格式的对比报告"""

    def __init__(self, metrics: List[TaskMetrics], model: str = "unknown",
                 errors: Optional[List[Dict[str, Any]]] = None):
        self.metrics = metrics
        self.model = model
        self.errors = errors or []

    # ---- 聚合统计 ----

    def summary_by_agent(self) -> Dict[str, Dict[str, Any]]:
        """按 Agent 类型聚合统计"""
        grouped: Dict[str, List[TaskMetrics]] = defaultdict(list)
        for m in self.metrics:
            grouped[m.agent_type].append(m)

        summary = {}
        for agent_type, items in grouped.items():
            n = len(items)
            summary[agent_type] = {
                "count": n,
                "avg_total_ms": _avg(items, "total_time_ms"),
                "avg_llm_ms": _avg(items, "total_llm_ms"),
                "avg_tool_ms": _avg(items, "total_tool_ms"),
                "avg_overhead_ms": _avg(items, "framework_overhead_ms"),
                "avg_iterations": _avg(items, "iteration_count"),
                "avg_tokens": _avg(items, "total_tokens"),
                "completion_rate": _rate(items, "task_completed"),
                "tool_accuracy": _rate(items, "tool_call_accurate"),
                "avg_parse_failures": _avg(items, "parse_failures"),
                "avg_tool_failures": _avg(items, "tool_exec_failures"),
            }
        return summary

    def summary_by_task(self) -> Dict[str, Dict[str, Any]]:
        """按任务 ID 聚合"""
        grouped: Dict[str, List[TaskMetrics]] = defaultdict(list)
        for m in self.metrics:
            grouped[m.task_id].append(m)

        summary = {}
        for task_id, items in grouped.items():
            summary[task_id] = {
                "count": len(items),
                "avg_total_ms": _avg(items, "total_time_ms"),
                "completion_rate": _rate(items, "task_completed"),
                "tool_accuracy": _rate(items, "tool_call_accurate"),
            }
        return summary

    def detail_table(self) -> List[Dict[str, Any]]:
        """返回逐条明细"""
        return [
            {
                "session_id": m.session_id,
                "task_id": m.task_id,
                "agent_type": m.agent_type,
                "trial": m.trial,
                "total_ms": m.total_time_ms,
                "llm_ms": m.total_llm_ms,
                "tool_ms": m.total_tool_ms,
                "overhead_ms": m.framework_overhead_ms,
                "iterations": m.iteration_count,
                "tokens": m.total_tokens,
                "completed": m.task_completed,
                "termination": m.termination_type,
                "tool_accurate": m.tool_call_accurate,
                "tools": ",".join(m.tool_call_sequence),
                "expected": ",".join(m.expected_tools),
                "parse_fail": m.parse_failures,
                "tool_fail": m.tool_exec_failures,
                "error": m.error or "",
            }
            for m in self.metrics
        ]

    # ---- 输出格式 ----

    def comparison_table_md(self) -> str:
        """生成 Markdown 格式的对比表格"""
        summary = self.summary_by_agent()
        agents = sorted(summary.keys())
        if not agents:
            return "无数据"

        lines = []
        lines.append(f"## Benchmark 对比报告")
        lines.append(f"模型: {self.model} | 样本数: {len(self.metrics)} | 异常: {len(self.errors)}")
        lines.append("")

        # 性能对比表
        lines.append("### 性能对比（平均值）")
        lines.append("")
        header = "| 指标 | " + " | ".join(agents) + " |"
        sep = "|---|" + "|".join(["---"] * len(agents)) + "|"
        lines.append(header)
        lines.append(sep)

        perf_rows = [
            ("总耗时(ms)", "avg_total_ms"),
            ("LLM 时间(ms)", "avg_llm_ms"),
            ("工具时间(ms)", "avg_tool_ms"),
            ("框架开销(ms)", "avg_overhead_ms"),
            ("迭代次数", "avg_iterations"),
            ("Token 用量", "avg_tokens"),
        ]
        for label, key in perf_rows:
            vals = [_fmt(summary[a].get(key, 0)) for a in agents]
            lines.append(f"| {label} | " + " | ".join(vals) + " |")

        lines.append("")

        # 准确性对比表
        lines.append("### 准确性对比")
        lines.append("")
        lines.append(header)
        lines.append(sep)

        acc_rows = [
            ("任务完成率", "completion_rate"),
            ("工具调用准确率", "tool_accuracy"),
            ("平均解析失败", "avg_parse_failures"),
            ("平均工具失败", "avg_tool_failures"),
        ]
        for label, key in acc_rows:
            vals = []
            for a in agents:
                v = summary[a].get(key, 0)
                if "rate" in key or "accuracy" in key:
                    vals.append(f"{v:.0%}")
                else:
                    vals.append(_fmt(v))
            lines.append(f"| {label} | " + " | ".join(vals) + " |")

        lines.append("")

        # 按任务对比
        task_summary = self.summary_by_task()
        if task_summary:
            lines.append("### 按任务对比")
            lines.append("")
            lines.append("| 任务 | 样本 | 平均耗时(ms) | 完成率 | 工具准确率 |")
            lines.append("|---|---|---|---|---|")
            for tid, s in sorted(task_summary.items()):
                lines.append(
                    f"| {tid} | {s['count']} | {_fmt(s['avg_total_ms'])} "
                    f"| {s['completion_rate']:.0%} | {s['tool_accuracy']:.0%} |"
                )

        return "\n".join(lines)

    def to_csv(self, path: str):
        """导出逐条明细 CSV"""
        rows = self.detail_table()
        if not rows:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def to_json(self, path: str):
        """导出 JSON 格式（汇总 + 明细）"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "model": self.model,
            "sample_count": len(self.metrics),
            "error_count": len(self.errors),
            "summary_by_agent": self.summary_by_agent(),
            "summary_by_task": self.summary_by_task(),
            "details": self.detail_table(),
            "errors": list(self.errors),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def print_report(self):
        """终端输出"""
        print("=" * 60)
        print(self.comparison_table_md())
        print("=" * 60)

    def save_all(self, output_dir: str):
        """一次性保存所有格式"""
        os.makedirs(output_dir, exist_ok=True)
        md_path = os.path.join(output_dir, "report.md")
        csv_path = os.path.join(output_dir, "raw_data.csv")
        json_path = os.path.join(output_dir, "summary.json")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.comparison_table_md())
        self.to_csv(csv_path)
        self.to_json(json_path)

        print(f"报告已保存:")
        print(f"  Markdown: {md_path}")
        print(f"  CSV:      {csv_path}")
        print(f"  JSON:     {json_path}")

        if self.errors:
            errors_csv_path = os.path.join(output_dir, "errors.csv")
            self._write_errors_csv(errors_csv_path)
            print(f"  Errors:   {errors_csv_path}")

    def _write_errors_csv(self, path: str):
        """将异常会话写入独立 CSV，便于事后分析"""
        if not self.errors:
            return
        fieldnames = ["session_id", "task_id", "agent_type", "trial", "error"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.errors)


# ---- 工具函数 ----

def _avg(items: List[TaskMetrics], attr: str) -> float:
    if not items:
        return 0.0
    vals = [getattr(m, attr, 0) or 0 for m in items]
    return round(sum(vals) / len(vals), 1)


def _rate(items: List[TaskMetrics], attr: str) -> float:
    if not items:
        return 0.0
    vals = [1 if getattr(m, attr, False) else 0 for m in items]
    return sum(vals) / len(vals)


def _fmt(v) -> str:
    if isinstance(v, float):
        if v == int(v):
            return str(int(v))
        return f"{v:.1f}"
    return str(v)
