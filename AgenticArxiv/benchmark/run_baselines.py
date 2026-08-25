"""CLI for deterministic benchmark-score sensitivity checks.

Run from ``AgenticArxiv/``:

    python -m benchmark.run_baselines --task-set expanded --output /tmp/baselines
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark.baselines import (  # noqa: E402
    ALL_POLICIES,
    category_gap_failures,
    evaluate_baselines,
    highest_scoring_tasks,
    reference_gap_failures,
    render_markdown,
    resolve_policies,
    summarize_baselines,
)


def _task_pool(task_set: str):
    if task_set == "expanded":
        from benchmark.tasks_expanded import get_expanded_tasks
        return get_expanded_tasks()
    from benchmark.tasks import get_all_tasks
    return get_all_tasks()


def _save_report(output_dir: Path, report: dict, markdown: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "baseline_report.md").write_text(markdown, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline diagnostic for benchmark reward sensitivity."
    )
    parser.add_argument(
        "--task-set", choices=["default", "expanded"], default="expanded",
        help="default=8 seed tasks; expanded=58 tasks including constraints and infeasible requests",
    )
    parser.add_argument(
        "--policies", nargs="+", choices=sorted(ALL_POLICIES),
        default=list(ALL_POLICIES),
        help="Synthetic policies to score (default: all, including reference)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="First stable seed used by random_tool (default: 42)",
    )
    parser.add_argument(
        "--random-samples", type=int, default=20,
        help="Number of consecutive seeds sampled by random_tool (default: 20)",
    )
    parser.add_argument(
        "--min-reference-gap", type=float, default=0.3,
        help="Minimum mean-reward gap required for every weak policy (default: 0.3)",
    )
    parser.add_argument(
        "--min-category-gap", type=float, default=0.3,
        help="Minimum mean-reward gap required per task category (default: 0.3). "
             "The aggregate check dilutes a single leaky category.",
    )
    parser.add_argument(
        "--top", type=int, default=5,
        help="Highest-scoring tasks shown per weak policy (default: 5)",
    )
    parser.add_argument(
        "--training-step", type=int, default=100,
        help="Reward curriculum step; 100 uses full correctness weights (default: 100)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional directory for baseline_report.json and baseline_report.md",
    )
    args = parser.parse_args()
    if args.random_samples < 1:
        parser.error("--random-samples must be at least 1")
    if args.min_reference_gap < 0:
        parser.error("--min-reference-gap must be non-negative")
    if args.min_category_gap < 0:
        parser.error("--min-category-gap must be non-negative")
    if args.top < 1:
        parser.error("--top must be at least 1")

    tasks = _task_pool(args.task_set)
    results = evaluate_baselines(
        tasks,
        resolve_policies(args.policies),
        seed=args.seed,
        random_samples=args.random_samples,
        training_step=args.training_step,
    )
    summaries = summarize_baselines(results)
    top_tasks = highest_scoring_tasks(results, limit_per_policy=args.top)
    health_failures = reference_gap_failures(
        summaries, min_gap=args.min_reference_gap
    )
    category_failures = category_gap_failures(
        results, min_gap=args.min_category_gap
    )
    markdown = render_markdown(
        summaries,
        task_set=args.task_set,
        seed=args.seed,
        training_step=args.training_step,
        top_tasks=top_tasks,
        min_reference_gap=args.min_reference_gap,
        health_failures=health_failures,
        min_category_gap=args.min_category_gap,
        category_failures=category_failures,
    )
    print(markdown, end="")

    if args.output is not None:
        report = {
            "task_set": args.task_set,
            "seed": args.seed,
            "random_samples": args.random_samples,
            "training_step": args.training_step,
            "health_check": {
                "passed": not health_failures,
                "min_reference_gap": args.min_reference_gap,
                "failures": health_failures,
            },
            "category_check": {
                "passed": not category_failures,
                "min_category_gap": args.min_category_gap,
                "failures": category_failures,
            },
            "note": (
                "Synthetic trajectories are a scorer-sensitivity diagnostic; "
                "they do not execute tools or measure LLM quality."
            ),
            "summary": [summary.to_dict() for summary in summaries],
            "highest_scoring_tasks": [task.to_dict() for task in top_tasks],
            "results": [result.to_dict() for result in results],
        }
        _save_report(args.output, report, markdown)
        print(f"Saved JSON and Markdown reports to: {args.output}")

    for failure in health_failures:
        print(f"Health check failed: {failure}", file=sys.stderr)
    for failure in category_failures:
        print(f"Per-category check failed: {failure}", file=sys.stderr)
    if health_failures or category_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
