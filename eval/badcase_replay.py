#!/usr/bin/env python3
"""坏例回放 CLI。

不需要 LLM、不需要网络、不需要工具：用例里存的是死轨迹，回放重跑的只有
打分器。所以它能进 CI，也能在改完 reward 之后立刻回答一个很窄但很确定的
问题——**同一条轨迹，今天的代码还认为它是坏的吗。**

    # 回放全部用例；有 regression 时退出码 1
    python eval/badcase_replay.py

    # 从一次 benchmark 跑的轨迹里挑坏例，追加进用例库
    python -m benchmark.run_benchmark --task-set expanded --offline --save-traces
    python eval/badcase_replay.py capture --traces data/traces.jsonl

用例状态两种：
    open    毛病还在。回放报 newly_fixed 说明修好了，把 status 改成 fixed。
    fixed   已修。再复现就是 regression，退出码 1。
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "AgenticArxiv"
sys.path.insert(0, str(PACKAGE_ROOT))

os.environ.setdefault("STORE_BACKEND", "memory")

from benchmark.badcases import (  # noqa: E402
    STATUS_FIXED,
    STATUS_OPEN,
    capture,
    dump_cases,
    load_cases,
    replay,
)

DEFAULT_CASES = REPO_ROOT / "eval" / "eval_cases.jsonl"

_LABEL = {
    "still_open":  ("· ", "仍复现"),
    "newly_fixed": ("✓ ", "已修复 —— 把 status 改成 fixed，它就反过来防复发"),
    "regressed":   ("✗ ", "回归 —— 修好过的毛病又出现了"),
    "stays_fixed": ("  ", "守住"),
}


def _load_tasks(task_set: str):
    if task_set == "expanded":
        from benchmark.tasks_expanded import get_expanded_tasks
        return get_expanded_tasks()
    from benchmark.tasks import get_all_tasks
    return get_all_tasks()


def cmd_replay(args) -> int:
    cases = load_cases(args.cases)
    if not cases:
        print(f"{args.cases} 里没有用例")
        return 0

    outcomes = replay(cases, _load_tasks(args.task_set), training_step=args.training_step)

    width = max(len(o.case.case_id) for o in outcomes)
    print(f"回放 {len(outcomes)} 条用例（{args.cases}）\n")
    for o in sorted(outcomes, key=lambda x: (x.outcome != "regressed", x.case.case_id)):
        mark, label = _LABEL[o.outcome]
        drift = ""
        if o.reward_drift:
            drift = f"  奖励较捕获时 {o.reward_drift:+.3f}"
        print(f"{mark}{o.case.case_id:<{width}}  reward={o.verdict['reward']:+.3f}  {label}{drift}")
        if args.verbose:
            print(f"    条件 {o.case.reproduces_when} -> {'成立' if o.reproduces else '不成立'}")
            if o.case.note:
                print(f"    {o.case.note}")

    tally = {}
    for o in outcomes:
        tally[o.outcome] = tally.get(o.outcome, 0) + 1
    print("\n" + "  ".join(f"{_LABEL[k][1].split(' ')[0]}: {v}" for k, v in sorted(tally.items())))

    regressions = [o for o in outcomes if o.is_regression]
    if regressions:
        print(f"\n❌ {len(regressions)} 条回归：" + "、".join(o.case.case_id for o in regressions))
        return 1
    newly = [o for o in outcomes if o.outcome == "newly_fixed"]
    if newly:
        print(f"\n提示：{len(newly)} 条用例不再复现，确认属实后把 status 改成 "
              f"{STATUS_FIXED!r}：" + "、".join(o.case.case_id for o in newly))
    return 0


def cmd_capture(args) -> int:
    import json

    samples = []
    for line in Path(args.traces).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            samples.append((row["task_id"], row.get("history") or []))
    if not samples:
        print(f"{args.traces} 里没有轨迹")
        return 0

    found = capture(samples, _load_tasks(args.task_set),
                    source=args.source or Path(args.traces).name,
                    training_step=args.training_step)
    if not found:
        print(f"从 {len(samples)} 条轨迹里没挑出坏例")
        return 0

    existing = load_cases(args.cases) if Path(args.cases).exists() else []
    taken = {c.case_id for c in existing}
    fresh, renamed = [], 0
    for case in found:
        case_id = case.case_id
        while case_id in taken:
            renamed += 1
            case_id = f"{case.case_id}+{renamed}"
        taken.add(case_id)
        fresh.append(case if case_id == case.case_id
                     else type(case)(**{**case.to_dict(), "case_id": case_id,
                                        "captured": case.captured}))

    if args.dry_run:
        print(f"会新增 {len(fresh)} 条（--dry-run，未写入）:")
    else:
        dump_cases(existing + fresh, args.cases)
        print(f"新增 {len(fresh)} 条 -> {args.cases}（原有 {len(existing)} 条）:")
    for case in fresh:
        print(f"  {case.case_id:<40} {case.reproduces_when}  status={STATUS_OPEN}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="坏例回放：把一次失败固化成一条永久的回归用例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="用例库路径")
    parser.add_argument("--task-set", choices=["default", "expanded"], default="expanded")
    parser.add_argument("--training-step", type=int, default=100,
                        help="课程权重按第几步算。前 30 步会缩放正确性分量，"
                             "用例之间要可比就得固定它")
    parser.add_argument("-v", "--verbose", action="store_true", help="打印判定条件与说明")
    sub = parser.add_subparsers(dest="command")

    p_replay = sub.add_parser("replay", help="回放全部用例（默认）")
    p_replay.add_argument("-v", "--verbose", action="store_true", help="打印判定条件与说明")
    p_replay.set_defaults(func=cmd_replay)

    p_cap = sub.add_parser("capture", help="从轨迹里挑坏例追加进用例库")
    p_cap.add_argument("--traces", type=Path, required=True,
                       help="run_benchmark.py --save-traces 产出的 JSONL")
    p_cap.add_argument("--source", default="", help="来源标记，默认取轨迹文件名")
    p_cap.add_argument("--dry-run", action="store_true", help="只看会挑出什么，不写入")
    p_cap.set_defaults(func=cmd_capture)

    args = parser.parse_args()
    if not getattr(args, "command", None):
        args.func = cmd_replay
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
