# AgenticArxiv/benchmark/report.py
"""Benchmark 报告生成：Markdown 对比表格 + CSV + JSON"""

import csv
import json
import os
import math
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
                "arg_accuracy": _avg(items, "arg_score"),
                "ref_accuracy": _avg(items, "ref_score"),
                # 两个分母各有用途：对全部运行取率可跨 Agent 比较；
                # 对 FINISH 运行取率回答「它说完成时有多少次在撒谎」。
                "false_finish_rate": _rate(items, "false_finish"),
                "false_finish_of_completed": (
                    sum(1 for m in items if m.false_finish)
                    / sum(1 for m in items if m.task_completed)
                    if any(m.task_completed for m in items) else 0.0
                ),
                "avg_parse_failures": _avg(items, "parse_failures"),
                "avg_tool_failures": _avg(items, "tool_exec_failures"),
            }
        return summary

    def reliability_by_agent(
        self, ks=(1, 2, 3), criterion: str = "accurate"
    ) -> Dict[str, Dict[str, Any]]:
        """按 Agent 聚合 pass^k：先在每个任务内估计，再对任务取平均。

        必须先按任务估计再平均 —— 直接把所有运行混在一起算，
        「一个任务稳定成功、另一个稳定失败」会和「两个任务各成功一半」
        得到相同的数字，而这两种情况的可靠性完全不同。
        """
        if criterion not in SUCCESS_CRITERIA:
            raise ValueError(f"未知判据 {criterion}，可选 {sorted(SUCCESS_CRITERIA)}")
        is_success = SUCCESS_CRITERIA[criterion]

        grouped: Dict[str, Dict[str, List[TaskMetrics]]] = defaultdict(lambda: defaultdict(list))
        for m in self.metrics:
            grouped[m.agent_type][m.task_id].append(m)

        summary: Dict[str, Dict[str, Any]] = {}
        for agent_type, by_task in grouped.items():
            row: Dict[str, Any] = {"criterion": criterion, "tasks": len(by_task)}
            trials = [len(items) for items in by_task.values()]
            row["min_trials"] = min(trials) if trials else 0
            for k in ks:
                scores = []
                skipped = 0
                for items in by_task.values():
                    value = pass_hat_k(len(items), sum(1 for m in items if is_success(m)), k)
                    if value is None:
                        skipped += 1
                    else:
                        scores.append(value)
                row[f"pass^{k}"] = sum(scores) / len(scores) if scores else None
                # 试验次数不足的任务单独记账，不混进平均值
                row[f"pass^{k}_skipped"] = skipped
            summary[agent_type] = row
        return summary

    def cost_by_agent(self, criterion: str = "accurate") -> Dict[str, Dict[str, Any]]:
        """交付一个成功结果的代价，以及失败时的代价形态。

        分母取成功数而非运行数，分子取**全部**运行的用量：失败的尝试同样
        烧 token 和时间，把它们排除在外会让「经常失败但失败得很快」的
        Agent 显得便宜。

        另外按成功/失败分开报迭代数。对全部运行取平均会双向污染：
        撞上限的失败跑满迭代（拉高），提前放弃的失败只跑一两步（拉低），
        混在一起既不代表成功时的效率，也不代表失败的代价。
        """
        is_success = SUCCESS_CRITERIA[criterion]
        grouped: Dict[str, List[TaskMetrics]] = defaultdict(list)
        for m in self.metrics:
            grouped[m.agent_type].append(m)

        summary: Dict[str, Dict[str, Any]] = {}
        for agent_type, items in grouped.items():
            wins = [m for m in items if is_success(m)]
            losses = [m for m in items if not is_success(m)]
            n_win = len(wins)
            total_tokens = sum(m.total_tokens for m in items)
            total_calls = sum(len(m.tool_call_sequence) for m in items)
            total_ms = sum(m.total_time_ms for m in items)
            summary[agent_type] = {
                "runs": len(items),
                "successes": n_win,
                # 一次成功也没有时，「每次成功的代价」无从谈起 —— 返回 None
                # 而不是 0 或无穷大，两者都会在排序里给出错误结论。
                "tokens_per_success": total_tokens / n_win if n_win else None,
                "calls_per_success": total_calls / n_win if n_win else None,
                "ms_per_success": total_ms / n_win if n_win else None,
                "median_iterations_success": _median([m.iteration_count for m in wins]),
                "median_iterations_failure": _median([m.iteration_count for m in losses]),
                # 失败形态：撞上限（死循环）与提前放弃是两种不同的病
                "failed_at_limit": sum(1 for m in losses if m.termination_type == "FORCE_STOP"),
                "failed_claiming_done": sum(1 for m in losses if m.termination_type == "FINISH"),
                "failed_with_error": sum(1 for m in losses if m.termination_type == "ERROR"),
            }
        return summary

    def difficulty_bands(self, criterion: str = "accurate") -> Dict[str, List[str]]:
        """按成功率把任务分三档，跨 Agent 合并统计。

        GRPO 的梯度来自组内奖励方差：成功率贴近 0 或 1 的任务，
        同一 prompt 采样出的轨迹奖励一致，优势为零、不产生梯度。
        中间带才是有效的训练样本，两端更适合留作评测的上下限。
        """
        is_success = SUCCESS_CRITERIA[criterion]
        by_task: Dict[str, List[TaskMetrics]] = defaultdict(list)
        for m in self.metrics:
            by_task[m.task_id].append(m)

        bands: Dict[str, List[str]] = {"floor": [], "middle": [], "ceiling": []}
        for task_id, items in sorted(by_task.items()):
            rate = sum(1 for m in items if is_success(m)) / len(items)
            key = "floor" if rate < 0.2 else "ceiling" if rate > 0.8 else "middle"
            bands[key].append(task_id)
        return bands

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
                "arg_score": m.arg_score,
                "false_finish": m.false_finish,
                "ref_score": m.ref_score,
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

        # 可靠性对比表
        rel = self.reliability_by_agent()
        if rel:
            lines.append("### 可靠性（pass^k：k 次试验全部成功）")
            lines.append("")
            lines.append(header)
            lines.append(sep)
            for k in (1, 2, 3):
                vals = []
                for a in agents:
                    v = rel.get(a, {}).get(f"pass^{k}")
                    vals.append("n/a" if v is None else f"{v:.0%}")
                lines.append(f"| pass^{k} | " + " | ".join(vals) + " |")
            skipped = max(rel[a].get("pass^3_skipped", 0) for a in agents)
            min_trials = min(rel[a].get("min_trials", 0) for a in agents)
            if skipped:
                lines.append("")
                lines.append(
                    f"> {skipped} 个任务的试验次数不足 3，未计入 pass^3"
                    f"（最少的任务只有 {min_trials} 次）。"
                )
            lines.append("")

        # 准确性对比表
        lines.append("### 准确性对比")
        lines.append("")
        lines.append(header)
        lines.append(sep)

        acc_rows = [
            ("任务完成率", "completion_rate"),
            ("工具调用准确率", "tool_accuracy"),
            ("参数准确率", "arg_accuracy"),
            ("指代解析准确率", "ref_accuracy"),
            ("假完成率", "false_finish_rate"),
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

        # 代价（按成功归一化）
        cost = self.cost_by_agent()
        if cost:
            lines.append("### 代价（按成功次数归一化）")
            lines.append("")
            lines.append(header)
            lines.append(sep)
            cost_rows = [
                ("Token / 成功", "tokens_per_success"),
                ("工具调用 / 成功", "calls_per_success"),
                ("耗时(ms) / 成功", "ms_per_success"),
                ("迭代数中位·成功", "median_iterations_success"),
                ("迭代数中位·失败", "median_iterations_failure"),
            ]
            for label, key in cost_rows:
                vals = []
                for a in agents:
                    v = cost.get(a, {}).get(key)
                    vals.append("n/a" if v is None else _fmt(v))
                lines.append(f"| {label} | " + " | ".join(vals) + " |")
            lines.append("")
            lines.append("失败形态：")
            lines.append("")
            lines.append(header)
            lines.append(sep)
            for label, key in [("撞迭代上限", "failed_at_limit"),
                               ("声称完成但错", "failed_claiming_done"),
                               ("异常终止", "failed_with_error")]:
                vals = [str(cost.get(a, {}).get(key, 0)) for a in agents]
                lines.append(f"| {label} | " + " | ".join(vals) + " |")
            lines.append("")

        # 难度分档
        bands_md = self.difficulty_bands_md()
        if bands_md:
            lines.append(bands_md)
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

    def difficulty_bands_md(self) -> str:
        bands = self.difficulty_bands()
        total = sum(len(v) for v in bands.values())
        if not total:
            return ""
        lines = ["### 任务难度分档（跨 Agent 合并成功率）", ""]
        lines.append("| 档位 | 成功率 | 任务数 | 用途 |")
        lines.append("|---|---|---|---|")
        lines.append(f"| floor | < 20% | {len(bands['floor'])} | 评测下限；GRPO 无梯度 |")
        lines.append(f"| middle | 20~80% | {len(bands['middle'])} | **GRPO 训练集**；奖励方差最大 |")
        lines.append(f"| ceiling | > 80% | {len(bands['ceiling'])} | 评测上限；GRPO 无梯度 |")
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

# 可靠性判据。pass^k 对「成功」的定义敏感，所以显式列出而不是写死一个。
SUCCESS_CRITERIA = {
    "completed": lambda m: m.task_completed,
    "accurate": lambda m: m.task_completed and m.tool_call_accurate,
}


def pass_hat_k(n: int, c: int, k: int) -> Optional[float]:
    """pass^k：从 n 次试验里随机抽 k 次，全部成功的概率。

    采用 τ-bench 的 pass^k（**全部**成功），而非 HumanEval 的 pass@k
    （**至少一次**成功）。前者衡量可靠性，k 越大越低；后者衡量能力上界，
    k 越大越高。Agent 关心的是前者 —— 一个「三次里对一次」的 Agent
    不能用，哪怕它的单次成功率看着不错。

        pass^k = C(c, k) / C(n, k)

    这是无偏估计，用上了全部 n 次试验，而不是只取前 k 次。
    pass^1 恒等于朴素成功率 c/n，可作自检。

    n < k 时无法估计，返回 None（由调用方决定跳过还是报缺失），
    静默按 0 计会把「样本不够」和「真的做不到」混为一谈。
    """
    if k <= 0:
        raise ValueError("k 必须为正")
    if n < k:
        return None
    if c < k:
        return 0.0
    return math.comb(c, k) / math.comb(n, k)


def _median(values: List[float]) -> Optional[float]:
    """中位数。用中位而非均值：迭代数的分布被撞上限的样本拉出长尾。"""
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


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
