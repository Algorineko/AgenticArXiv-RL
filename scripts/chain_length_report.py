#!/usr/bin/env python3
"""按链长分层统计严格成功率与首错类型。

读取 ``run_benchmark --save-traces`` 写出的 traces.jsonl（可传多个），按任务的
标准链长分组统计 strict success，并用 ``rl.step_credit.label_trajectory`` 给出
每条失败轨迹第一次偏离的位置与类型。dev 切分最长只有 2 步，长链结论必须
按链长分层看。

用法（仓库根目录）::

    python scripts/chain_length_report.py eval_results/run_iid_test/traces.jsonl \\
        eval_results/run_ood_test/traces.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "AgenticArxiv"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from benchmark.metrics import extract_metrics, is_strict_success  # noqa: E402
from benchmark.task_io import load_tasks  # noqa: E402
from rl.step_credit import label_trajectory  # noqa: E402


def chain_length_report(paths: Sequence[str], task_set: str = "expanded") -> Dict[str, object]:
    tasks = {task["id"]: task for task in load_tasks(task_set, None)}
    by_length: Dict[int, List[int]] = defaultdict(lambda: [0, 0])
    failures: Counter = Counter()
    skipped = 0
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            task = tasks.get(record.get("task_id"))
            if task is None:
                skipped += 1
                continue
            history = record.get("history") or []
            metrics = extract_metrics(task, {"history": history}, record.get("agent_type", "regex"),
                                      record.get("trial", 0))
            ok = is_strict_success(metrics)
            length = len(task.get("expected_tools") or [])
            by_length[length][0] += int(ok)
            by_length[length][1] += 1
            if not ok:
                labels = label_trajectory(task, history)
                position = (labels.first_error_index or 0) + 1
                failures[(length, labels.first_error_kind, position)] += 1
    return {"by_length": dict(by_length), "failures": failures, "skipped": skipped}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="按链长分层的严格成功率与首错类型")
    parser.add_argument("traces", nargs="+", help="run_benchmark --save-traces 产出的 traces.jsonl")
    parser.add_argument("--task_set", default="expanded", choices=["default", "expanded"])
    args = parser.parse_args(argv)
    report = chain_length_report(args.traces, args.task_set)
    for length in sorted(report["by_length"]):
        ok, total = report["by_length"][length]
        print(f"链长 {length}: {ok}/{total} = {ok / total:.3f}")
    print("失败（链长、首错类型、位置）:")
    for (length, kind, position), count in sorted(report["failures"].items(),
                                                  key=lambda item: (-item[0][0], -item[1])):
        print(f"  链长 {length}  首错={kind:<13} 第 {position} 步: {count}")
    if report["skipped"]:
        print(f"（跳过 {report['skipped']} 条不在任务集里的记录）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
