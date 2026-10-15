"""展示已有 GRPO rollout 的只读诊断，可显式另存汇总。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from agentic_arxiv.evaluation.rollout_audit import load_records, summarize


def print_report(summary: Dict[str, Any]) -> None:
    print("=== GRPO 原始 rollout 审计 ===")
    print(f"样本数: {summary['sample_count']}")
    print(f"分组数: {summary['group_count']}")
    print(
        "异常样本: "
        f"{summary['anomalous_sample_count']} "
        f"({summary['anomalous_sample_fraction']:.1%})"
    )
    print(
        "格式异常: "
        f"{summary['format_anomalous_sample_count']} "
        f"({summary['format_anomalous_sample_fraction']:.1%})"
    )
    print(
        "奖励标准差为零的分组: "
        f"{summary['zero_reward_std_group_count']} "
        f"({summary['zero_reward_std_group_fraction']:.1%})"
    )
    print(
        "完全重复的分组: "
        f"{summary['exact_duplicate_group_count']} "
        f"({summary['exact_duplicate_group_fraction']:.1%})"
    )
    print("异常计数:")
    if not summary["anomaly_counts"]:
        print("  无")
    for name, count in summary["anomaly_counts"].items():
        print(f"  {name}: {count}")
    semantics = summary.get("terminal_semantics") or {}
    if semantics.get("applicable_sample_count"):
        print("阻塞终止语义:")
        for name, count in semantics["counts"].items():
            rate = semantics["rates"][name]
            mean_reward = semantics["mean_reward_by_class"][name]
            print(f"  {name}: {count} ({rate:.1%}), 平均奖励={mean_reward:.4f}")


def print_examples(records: List[Dict[str, Any]], limit: int) -> None:
    anomalous = [row for row in records if row.get("active_anomalies")]
    anomalous.sort(key=lambda row: (float(row.get("reward") or 0.0), row.get("task_id", "")))
    if not anomalous or limit <= 0:
        return
    print(f"\n=== 前 {min(limit, len(anomalous))} 个异常样本 ===")
    for row in anomalous[:limit]:
        print(
            f"\n任务={row.get('task_id')} 分组={row.get('group_index')} "
            f"生成序号={row.get('generation_index')} 奖励={row.get('reward')}"
        )
        print(f"异常={row.get('active_anomalies')}")
        for turn_index, turn in enumerate(row.get("raw_assistant_turns") or [], 1):
            print(f"--- assistant 第 {turn_index} 轮 ---")
            print(turn)


def run(args: Any) -> int:
    """消费 CLI 已解析的路径与参数，不查询源码位置或执行模型。"""
    trace = Path(args.trace)
    summary_out = Path(args.summary_out) if args.summary_out is not None else None
    if summary_out is not None and summary_out.resolve() == trace.resolve():
        raise SystemExit(f"诊断汇总必须使用独立输出路径，不能覆盖输入: {trace}")
    records = load_records(trace)
    summary = summarize(records)
    print_report(summary)
    print_examples(records, args.show)
    if summary_out is not None:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\n汇总已写入: {summary_out}")
    return 0
