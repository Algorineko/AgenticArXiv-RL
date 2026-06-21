# AgenticArxiv/benchmark/run_benchmark.py
"""
Benchmark CLI 入口。

用法:
  cd AgenticArxiv
  python -m benchmark.run_benchmark                              # 全部
  python -m benchmark.run_benchmark --agents regex mcp     # 指定 Agent
  python -m benchmark.run_benchmark --repeat 5                   # 重复次数
  python -m benchmark.run_benchmark --tasks search               # 按类别
  python -m benchmark.run_benchmark --task-ids search_01 cache_01 # 按 ID
  python -m benchmark.run_benchmark --output benchmark/output/   # 输出目录
  python -m benchmark.run_benchmark --model gpt-4-turbo          # 指定模型
"""

import argparse
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark.tasks import get_all_tasks, get_tasks_by_category, get_task_by_id
from benchmark.runner import BenchmarkRunner
from benchmark.report import BenchmarkReport
from config import settings


def main():
    parser = argparse.ArgumentParser(description="AgenticArxiv Benchmark — 三种 Agent 模式对比测试")
    parser.add_argument(
        "--agents", nargs="+",
        default=["regex", "mcp", "skill_cli"],
        choices=["regex", "mcp", "skill_cli"],
        help="要测试的 Agent 类型 (默认全部)",
    )
    parser.add_argument(
        "--repeat", type=int, default=3,
        help="每个任务重复次数 (默认 3)",
    )
    parser.add_argument(
        "--tasks", type=str, default=None,
        choices=["search", "download", "translate", "cache", "composite"],
        help="按类别筛选测试任务",
    )
    parser.add_argument(
        "--task-ids", nargs="+", default=None,
        help="按 ID 指定测试任务",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(os.path.dirname(PROJECT_ROOT), "data"),
        help="报告输出目录 (默认 项目根目录/data)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="LLM 模型名 (默认使用 .env 中的 MODEL)",
    )
    parser.add_argument(
        "--prefix", type=str, default=None,
        help="Session ID 前缀，用于区分不同测试轮次 (默认: bench_r<timestamp>)",
    )
    args = parser.parse_args()

    # 筛选任务
    if args.task_ids:
        task_list = [t for tid in args.task_ids if (t := get_task_by_id(tid)) is not None]
        if not task_list:
            print(f"错误: 未找到任务 {args.task_ids}")
            sys.exit(1)
    elif args.tasks:
        task_list = get_tasks_by_category(args.tasks)
        if not task_list:
            print(f"错误: 类别 '{args.tasks}' 无任务")
            sys.exit(1)
    else:
        task_list = get_all_tasks()

    model = args.model or settings.models.agent_model

    runner = BenchmarkRunner(
        agent_types=args.agents,
        repeat=args.repeat,
        model=model,
        session_prefix=args.prefix,
    )

    print("=" * 60)
    print("AgenticArxiv Benchmark")
    print(f"  模型:     {model}")
    print(f"  前缀:     {runner.session_prefix}")
    print(f"  Agent:    {', '.join(args.agents)}")
    print(f"  任务数:   {len(task_list)}")
    print(f"  重复次数: {args.repeat}")
    print(f"  总运行数: {len(task_list) * len(args.agents) * args.repeat}")
    print(f"  输出目录: {args.output}")
    print("=" * 60)

    results = runner.run_all(task_list)

    # 提取有效 metrics
    all_metrics = [r.metrics for r in results if r.metrics is not None]
    error_dicts = [
        {
            "session_id": r.session_id,
            "task_id": r.task_id,
            "agent_type": r.agent_type,
            "trial": r.trial,
            "error": r.error,
        }
        for r in results if r.error
    ]

    if error_dicts:
        print(f"\n共 {len(error_dicts)} 个任务执行异常:")
        for e in error_dicts[:10]:
            print(f"  [{e['session_id']}] {e['error'][:100]}")

    if not all_metrics:
        print("无有效指标数据，退出")
        sys.exit(1)

    report = BenchmarkReport(all_metrics, model=model, errors=error_dicts)
    report.print_report()
    report.save_all(args.output)

    print(f"\nBenchmark 完成: {len(all_metrics)} 条有效数据, {len(error_dicts)} 个异常")


if __name__ == "__main__":
    main()
