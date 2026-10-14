"""坏例命令业务：历史 replay/capture 与实际 run 不互相降级。"""

from pathlib import Path

from agentic_arxiv.config import resolve_model_source, workspace_path
from agentic_arxiv.evaluation.badcases import (
    STATUS_FIXED,
    STATUS_OPEN,
    capture,
    dump_cases,
    load_cases,
    replay,
    run_cases,
)
from agentic_arxiv.tasks.task_io import load_tasks

_LABEL = {
    "still_open":  ("· ", "仍复现"),
    "newly_fixed": ("✓ ", "已修复 —— 把 status 改成 fixed，它就反过来防复发"),
    "regressed":   ("✗ ", "回归 —— 修好过的毛病又出现了"),
    "stays_fixed": ("  ", "守住"),
}


def _path(args, value):
    return workspace_path(value, workspace=getattr(args, "workspace", None))


def cmd_replay(args) -> int:
    args.cases = _path(args, args.cases)
    cases = load_cases(args.cases)
    if not cases:
        print(f"{args.cases} 里没有用例")
        return 0
    outcomes = replay(cases, load_tasks(args.task_set), training_step=args.training_step)
    return _print_outcomes(args, outcomes)


def cmd_run(args) -> int:
    from agentic_arxiv.llm.client import TransformersLLMClient
    from agentic_arxiv.runtime.agent_engine import ReActAgent

    if args.snapshot and not args.offline:
        raise ValueError("--snapshot 仅用于显式 --offline 回放，不静默切换运行模式")
    if args.max_iterations < 1:
        raise ValueError("完整 episode 轮数必须为正数")
    snapshot = _path(args, args.snapshot) if args.snapshot else None
    if args.offline:
        snapshot = snapshot or _path(args, "data/mock_arxiv_snapshot.json")
        if not snapshot.is_file():
            raise FileNotFoundError(f"离线快照缺失，禁止自动联网: {snapshot}")
    args.cases = _path(args, args.cases)
    cases = load_cases(args.cases)
    if not cases:
        print(f"{args.cases} 里没有用例")
        return 0
    model_path = resolve_model_source(
        str(args.model), local_only=True, workspace=getattr(args, "workspace", None)
    )
    client = TransformersLLMClient(model_path)
    agent = ReActAgent(
        client, max_iterations=args.max_iterations,
        mode="replay" if args.offline else "live", snapshot_path=snapshot,
    )
    try:
        outcomes = run_cases(
            cases, load_tasks(args.task_set), agent, training_step=args.training_step
        )
    finally:
        agent.close()
    return _print_outcomes(args, outcomes)


def _print_outcomes(args, outcomes) -> int:
    width = max(len(o.case.case_id) for o in outcomes)
    execution = outcomes[0].execution
    label = "历史只读重评分，未执行模型或工具" if execution == "historical-rescore" else "当前模型完整 episode"
    print(f"{label}: {len(outcomes)} 条用例（{args.cases}）\n")
    if execution == "complete-episode":
        relations = sorted({out.protocol_relation.get("relation", "unknown") for out in outcomes})
        print(f"模型训练/本次评测协议关系: {', '.join(relations)}；未知或跨协议不代表同口径复现")
    else:
        statuses = sorted({out.protocol_metadata["metadata_status"] for out in outcomes})
        print(f"历史轨迹协议元数据: {', '.join(statuses)}；缺字段不补为当前协议")
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

    args.traces = _path(args, args.traces)
    args.cases = _path(args, args.cases)
    samples = []
    for line in Path(args.traces).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            samples.append((row["task_id"], row.get("history") or [], row))
    if not samples:
        print(f"{args.traces} 里没有轨迹")
        return 0

    found = capture(samples, load_tasks(args.task_set),
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
