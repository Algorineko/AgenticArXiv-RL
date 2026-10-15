"""评测运行、轨迹保存和报告交付；参数声明位于 CLI。"""

import json
import sys
from pathlib import Path
from collections.abc import Mapping

from agentic_arxiv.tools.bootstrap import require_all_tools
from agentic_arxiv.tasks.tasks import get_all_tasks
from agentic_arxiv.tasks.tasks_expanded import get_expanded_tasks
from agentic_arxiv.tasks.splits import DEFAULT_SPLIT_PATH, load_split
from agentic_arxiv.evaluation.runner import BenchmarkRunner
from agentic_arxiv.evaluation.episode import EvaluationCancelled
from agentic_arxiv.evaluation.report import BenchmarkReport
from agentic_arxiv.provenance import read_protocol_metadata
from agentic_arxiv.config import settings


def _trace_record(result) -> dict:
    """保留任务评分、耗时与 token 用量的实际运行记录。"""
    raw = result.raw_result or {}
    record = {
        "task_id": result.task_id,
        "agent_type": result.agent_type,
        "trial": result.trial,
        "session_id": result.session_id,
        "history": raw.get("history") or [],
        "final_observation": raw.get("final_observation", ""),
        "total_time_ms": raw.get("total_time_ms", 0),
        "iteration_count": raw.get(
            "iteration_count", len(raw.get("history") or [])
        ),
        "timing": raw.get("timing") or {},
        "token_usage": raw.get("token_usage") or {},
        "termination_reason": raw.get("termination_reason"),
        "runtime_version": raw.get("runtime_version"),
        "context_version": raw.get("context_version"),
        "prompt_template_hash": raw.get("prompt_template_hash"),
        "chat_template_hash": raw.get("chat_template_hash"),
        "chat_template_hashes": raw.get("chat_template_hashes"),
        "observation_body_limit": raw.get("observation_body_limit"),
        "protocol_metadata": raw.get("protocol_metadata"),
        "training_protocol_metadata": raw.get("training_protocol_metadata"),
        "protocol_relation": raw.get("protocol_relation"),
    }
    # 只冻结实际提供给评分器的参考，不向模型 history 注入标准答案。
    if result.expected_paper_ids is not None:
        record["expected_paper_ids"] = list(result.expected_paper_ids)
    return record


def _common_protocol_metadata(results, field: str) -> dict:
    """报告复用实际 episode 事实；缺报或不同值保持未知，不重算默认模板。"""
    documents = []
    for result in results:
        if result.metrics is None:
            continue
        source = (result.raw_result or {}).get(field)
        documents.append(read_protocol_metadata(source if isinstance(source, Mapping) else None))
    if not documents:
        return read_protocol_metadata(None)
    first = documents[0]
    common = {
        key: value if all(document.get(key) == value for document in documents) else None
        for key, value in first.items() if key != "metadata_status"
    }
    return read_protocol_metadata(common)


def execute(args, *, cancel_requested=None, model_artifact=False) -> dict:
    """执行已解析参数；命令参数声明由 CLI 单独维护。"""
    if args.backend == "transformers" and not args.model:
        raise ValueError("--backend transformers 必须同时提供 --model 本地模型目录")

    # 工具没注册齐就别开跑：registry 不全时模型不会报错，它会编造工具名，
    # benchmark 照样跑完并给出成功率 —— 那种数字比没有数字更危险。
    require_all_tools("Benchmark")

    llm_extra = {}
    if args.no_thinking:
        llm_extra["chat_template_kwargs"] = {"enable_thinking": False}

    # 任务池
    if args.task_set == "expanded":
        from agentic_arxiv.tasks.tasks_expanded import offline_only_ids
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
    elif args.split:
        try:
            wanted = set(load_split(args.split))
        except (OSError, ValueError, KeyError) as exc:
            print(f"错误: {exc}")
            sys.exit(1)
        by_id = {t["id"]: t for t in pool}
        task_list = [by_id[tid] for tid in sorted(wanted) if tid in by_id]
        missing = wanted - set(by_id)
        if missing:
            # 切分文件是按完整任务集划的。少了任务通常意味着任务池被别的
            # 开关缩过（--task-set default 只有 8 条，或未加 --offline
            # 跳掉了绑定快照的那些），此时跑出来的数字不能与别的阶段比。
            print(f"错误: 切分 '{args.split}' 里有 {len(missing)} 条任务不在当前任务池中，"
                  f"例如 {sorted(missing)[:3]}")
            print("      切分按 --task-set expanded 的完整任务集划定；"
                  "绑定快照的任务还需要 --offline")
            sys.exit(1)
        print(f"使用切分 {args.split}（{len(task_list)} 条）")
    elif args.tasks:
        task_list = [t for t in pool if t.get("category") == args.tasks]
        if not task_list:
            print(f"错误: 类别 '{args.tasks}' 在当前任务集里无任务")
            sys.exit(1)
    else:
        task_list = pool

    model = args.model or settings.models.agent_model

    runner = BenchmarkRunner(
        repeat=args.repeat,
        model=model,
        session_prefix=args.prefix,
        llm_extra=llm_extra,
        offline=args.offline,
        snapshot=args.snapshot,
        llm_backend=args.backend,
        local_device=args.local_device,
        local_dtype=args.local_dtype,
        generation_seed=args.seed,
        **({"cancel_requested": cancel_requested} if cancel_requested is not None else {}),
        **({"model_artifact": True} if model_artifact else {}),
    )

    print("=" * 60)
    print("AgenticArxiv Benchmark")
    print(f"  模型:     {model}")
    print(f"  后端:     {args.backend}")
    if args.backend == "transformers":
        print(f"  设备/精度: {args.local_device} / {args.local_dtype}")
    print(f"  前缀:     {runner.session_prefix}")
    print("  Agent:    文本 ReAct")
    print(f"  任务数:   {len(task_list)}")
    print(f"  重复次数: {args.repeat}")
    print(f"  总运行数: {len(task_list) * args.repeat}")
    print(f"  输出目录: {args.output}")
    print(f"  工具执行: {'离线快照回放' if args.offline else '真实 arXiv API'}")
    print("=" * 60)

    interrupted = None
    try:
        results = runner.run_all(task_list)
    except EvaluationCancelled as error:
        interrupted = error
        results = error.partial_results

    if args.save_traces is not None:
        traces_path = Path(args.save_traces) if args.save_traces else Path(args.output) / "traces.jsonl"
        traces_path.parent.mkdir(parents=True, exist_ok=True)
        with open(traces_path, "w", encoding="utf-8") as handle:
            written = 0
            for r in results:
                history = (r.raw_result or {}).get("history")
                if not history:
                    continue
                handle.write(json.dumps(
                    _trace_record(r), ensure_ascii=False
                ) + "\n")
                written += 1
        print(f"轨迹已写出: {traces_path}（{written} 条）")

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

    if interrupted is not None:
        raise interrupted

    if not all_metrics:
        print("无有效指标数据，退出")
        sys.exit(1)

    execution_protocol = _common_protocol_metadata(results, "protocol_metadata")
    training_protocol = _common_protocol_metadata(results, "training_protocol_metadata")
    report = BenchmarkReport(
        all_metrics, model=model, errors=error_dicts,
        protocol_metadata=execution_protocol, training_metadata=training_protocol,
    )
    report.print_report()
    report.save_all(args.output)

    print(f"\nBenchmark 完成: {len(all_metrics)} 条有效数据, {len(error_dicts)} 个异常")
    return {"execution": "failed" if error_dicts else "complete",
            "expected_episodes": len(task_list) * args.repeat,
            "completed_episodes": len(results), "error_count": len(error_dicts)}
