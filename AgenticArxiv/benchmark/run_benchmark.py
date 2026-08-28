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

from tools.bootstrap import require_all_tools
from benchmark.tasks import get_all_tasks
from benchmark.tasks_expanded import get_expanded_tasks
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
        choices=["search", "download", "translate", "cache", "composite",
                 "ref_form", "optional", "state", "long_chain", "constraint",
                 "infeasible"],
        help="按类别筛选测试任务。扩展类别只存在于 --task-set expanded",
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
    parser.add_argument(
        "--no-thinking", action="store_true",
        help="关闭 Qwen3 等推理模型的思维链。思维链会让生成 token 数翻倍，"
             "而 token 用量是本 benchmark 的核心对比指标之一，"
             "混入后三种范式之间的差异会被淹没。",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="用 data/mock_arxiv_snapshot.json 回放工具调用，不请求真实 arXiv。"
             "结果可复现，且去掉网络耗时后「框架开销」才真正可比；"
             "代价是不再覆盖真实 API 集成。",
    )
    parser.add_argument("--snapshot", default=None, help="指定快照路径（默认 data/mock_arxiv_snapshot.json）")
    parser.add_argument(
        "--task-set", choices=["default", "expanded"], default="default",
        help=(
            f"default=benchmark/tasks.py 的 {len(get_all_tasks())} 条；"
            f"expanded=扩充后的 {len(get_expanded_tasks())} 条"
            "（指代形态、可选参数、跨步状态、多跳链路、负向约束、不可行请求）"
        ),
    )
    args = parser.parse_args()

    # 工具没注册齐就别开跑：registry 不全时模型不会报错，它会编造工具名，
    # benchmark 照样跑完并给出成功率 —— 那种数字比没有数字更危险。
    require_all_tools("Benchmark")

    llm_extra = {}
    if args.no_thinking:
        llm_extra["chat_template_kwargs"] = {"enable_thinking": False}

    # 任务池
    if args.task_set == "expanded":
        from benchmark.tasks_expanded import offline_only_ids
        pool = get_expanded_tasks()
        if not args.offline:
            # 这些任务的标准答案绑定 data/mock_arxiv_snapshot.json（例如标题子串
            # 指向快照里的特定论文），联网跑会拿到另一批论文，判分没有意义
            skip = set(offline_only_ids())
            dropped = [t["id"] for t in pool if t["id"] in skip]
            pool = [t for t in pool if t["id"] not in skip]
            if dropped:
                print(f"提示：{len(dropped)} 条任务的标准答案绑定快照，"
                      f"未加 --offline 故跳过，例如 {dropped[:3]}")
    else:
        pool = get_all_tasks()

    # 筛选任务
    if args.task_ids:
        by_id = {t["id"]: t for t in pool}
        task_list = [by_id[tid] for tid in args.task_ids if tid in by_id]
        if not task_list:
            print(f"错误: 未找到任务 {args.task_ids}")
            sys.exit(1)
    elif args.tasks:
        task_list = [t for t in pool if t.get("category") == args.tasks]
        if not task_list:
            print(f"错误: 类别 '{args.tasks}' 在当前任务集里无任务")
            sys.exit(1)
    else:
        task_list = pool

    model = args.model or settings.models.agent_model

    runner = BenchmarkRunner(
        agent_types=args.agents,
        repeat=args.repeat,
        model=model,
        session_prefix=args.prefix,
        llm_extra=llm_extra,
        offline=args.offline,
        snapshot=args.snapshot,
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
    print(f"  工具执行: {'离线快照回放' if args.offline else '真实 arXiv API'}")
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
