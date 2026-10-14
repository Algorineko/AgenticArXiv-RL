"""只读核对仓库既有报告与 traces；不构造动作、响应或环境执行轨迹。

输出是原文件的索引和检查报告，不是新的行为 fixture 或模型评测结果。
未知历史字段保持 None；当前评分只用于读取契约检查，不覆盖历史报告。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import socket
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "AgenticArxiv"))


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_index(names):
    """保留退出源码的身份；删除不意味着有可读取内容或可猜测的新哈希。"""
    index = {}
    for name in names:
        path = ROOT / name
        exists = path.is_file()
        index[name] = {"source": name if exists else None,
                       "status": "present" if exists else "removed",
                       "sha256": _hash(path) if exists else None}
    return index


def _reject_execution(*args, **kwargs):
    raise AssertionError("历史只读检查不得联网或执行工具")


def audit(phase="S0"):
    from benchmark.metrics import is_strict_success
    from benchmark.report import BenchmarkReport
    from benchmark.tasks import get_all_tasks
    from benchmark.tasks_expanded import get_expanded_tasks
    from rl.reward import RewardCalculator, _parse_action
    from rl.trajectory import load_trajectories
    from tools.tool_registry import registry

    sources = (
        "AgenticArxiv/benchmark/metrics.py", "AgenticArxiv/benchmark/report.py",
        "AgenticArxiv/benchmark/tasks.py", "AgenticArxiv/benchmark/tasks_expanded.py",
        "AgenticArxiv/rl/reward.py", "AgenticArxiv/rl/trajectory.py",
        "AgenticArxiv/agents/agent_engine.py", "AgenticArxiv/agents/base_agent.py",
        "AgenticArxiv/agents/side_effects.py", "AgenticArxiv/rl/grpo_reward.py",
        "AgenticArxiv/rl/multiturn_env.py", "scripts/generate_sft_data.py",
        "AgenticArxiv/benchmark/semantic_oracle.py", "AgenticArxiv/benchmark/runner.py",
        "scripts/audit_historical_records.py",
    )
    source_index = _source_index(sources)
    source_hashes = {name: record["sha256"] for name, record in source_index.items()}
    tasks = {task["id"]: task for task in get_expanded_tasks() + get_all_tasks()}
    calculator = RewardCalculator(curriculum_steps=0)
    indexed = []
    trace_results = []
    totals = Counter()
    terminal_counts = Counter()
    seen_tasks = set()
    action_counts = Counter()
    observation_markers = Counter()
    all_comparisons = Counter()
    marker_examples = {}
    limits = ("runtime_version", "context_version", "template_hash", "training_protocol")
    field_counts = Counter()
    errors = []

    paths = []
    for directory in (ROOT / "artifacts", ROOT / "eval_results"):
        paths.extend(path for path in directory.rglob("*") if path.is_file()
                     and "mainline_baseline" not in path.parts
                     and (path.name in {"summary.json", "report.md", "traces.jsonl", "raw_data.csv"}
                          or path.name.endswith("_summary.json")
                          or path.name in {"baseline_report.json", "baseline_report.md",
                                           "jev_e2e_pilot_10.json", "jev_project_integration_smoke.json"}))
    paths.sort()
    before = {str(path.relative_to(ROOT)): _hash(path) for path in paths}
    summaries = {}
    for path in paths:
        relative = str(path.relative_to(ROOT))
        item = {"path": relative, "sha256": before[relative], "bytes": path.stat().st_size}
        if path.suffix == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                item["json_read"] = "通过"
                if isinstance(data, dict):
                    item["top_level_fields"] = sorted(data)
                    item["historical_protocol"] = {key: data.get(key) for key in limits}
                    item["declared_scope"] = data.get("scope", data.get("note"))
                    if path.name == "summary.json":
                        summaries[path.parent] = data
                        item["sample_count"] = data.get("sample_count")
                        item["detail_count"] = len(data.get("details", []))
                        item["model"] = data.get("model")
            except Exception as exc:
                errors.append({"path": relative, "check": "JSON 读取", "error": str(exc)})
                item["json_read"] = "失败"
        indexed.append(item)

    with patch.object(socket.socket, "connect", _reject_execution), \
            patch.object(socket, "getaddrinfo", _reject_execution), \
            patch.object(registry, "execute_tool", _reject_execution):
        for path in paths:
            if path.name != "traces.jsonl":
                continue
            relative = str(path.relative_to(ROOT))
            rows = 0
            scored = 0
            rewards = []
            signatures = []
            comparison = Counter()
            examples = []
            metrics_list = []
            local_fields = Counter()
            details = {
                (row.get("task_id"), row.get("agent_type"), row.get("trial")): row
                for row in summaries.get(path.parent, {}).get("details", [])
            }
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception as exc:
                    errors.append({"path": relative, "line": line_number,
                                   "check": "JSONL 读取", "error": str(exc)})
                    continue
                rows += 1
                totals["historical_rows"] += 1
                for field in limits + ("reward_components", "final_reward", "raw_assistant_turns",
                                       "timing", "token_usage", "setup_actions"):
                    if field in row:
                        field_counts[field] += 1
                        local_fields[field] += 1
                history = row.get("history")
                if not isinstance(history, list):
                    errors.append({"path": relative, "line": line_number,
                                   "check": "history 读取", "error": "缺少已有 history 列表"})
                    continue
                for step_number, step in enumerate(history, 1):
                    action = step.get("action")
                    totals["historical_steps"] += 1
                    if action in ("FINISH", "FORCE_STOP", "ERROR", "PARSE_ERROR"):
                        action_counts[action] += 1
                    else:
                        parsed = _parse_action(action)
                        if parsed is None:
                            totals["unparsed_recorded_actions"] += 1
                        else:
                            totals["parsed_recorded_actions"] += 1
                            action_counts[str(parsed.get("name"))] += 1
                            if not isinstance(parsed.get("args"), dict):
                                totals["non_object_recorded_args"] += 1
                            elif "session_id" in parsed["args"]:
                                totals["actions_containing_session_id"] += 1
                    observation = str(step.get("observation", ""))
                    for marker in ("PENDING", "READY", "mocktask", "SSE", "offline",
                                   "工具执行失败", "错误: 工具"):
                        if marker in observation:
                            observation_markers[marker] += 1
                            marker_examples.setdefault(marker, {
                                "path": relative, "line": line_number, "step": step_number})
                task_id = row.get("task_id")
                seen_tasks.add(task_id)
                if task_id not in tasks:
                    totals["unknown_task_rows"] += 1
                    continue
                unchanged = copy.deepcopy(row)
                try:
                    breakdown, metrics = calculator.compute_reward_breakdown(
                        tasks[task_id], row, agent_type=row.get("agent_type", "unknown"),
                        trial=row.get("trial", 0), session_id=row.get("session_id", ""))
                    second, _ = calculator.compute_reward_breakdown(
                        tasks[task_id], row, agent_type=row.get("agent_type", "unknown"),
                        trial=row.get("trial", 0), session_id=row.get("session_id", ""))
                    if breakdown.to_dict() != second.to_dict() or unchanged != row:
                        raise AssertionError("重复评分结果不稳定或修改输入")
                except Exception as exc:
                    errors.append({"path": relative, "line": line_number,
                                   "check": "原 history 只读评分", "error": str(exc)})
                    continue
                scored += 1
                totals["read_only_scored_rows"] += 1
                totals["current_strict_success_on_recorded_history"] += int(is_strict_success(metrics))
                totals["current_ref_applicable_rows"] += int(metrics.ref_applicable)
                totals["current_ref_not_applicable_rows"] += int(not metrics.ref_applicable)
                terminal_counts[metrics.termination_type] += 1
                rewards.append(breakdown.total)
                signatures.append({"line": line_number, "task_id": task_id,
                                   "components": breakdown.to_dict()})
                metrics_list.append(metrics)
                detail = details.get((task_id, row.get("agent_type"), row.get("trial")))
                if detail is None:
                    comparison["missing_historical_detail"] += 1
                    continue
                comparisons = {
                    "termination": metrics.termination_type,
                    "tools": ",".join(metrics.tool_call_sequence),
                    "tool_accurate": metrics.tool_call_accurate,
                    "arg_score": metrics.arg_score,
                    "ref_score": metrics.ref_score,
                    "parse_fail": metrics.parse_failures,
                    "tool_fail": metrics.tool_exec_failures,
                    "false_finish": metrics.false_finish,
                }
                for key, current in comparisons.items():
                    if key not in detail:
                        comparison[f"unknown_{key}"] += 1
                    elif detail[key] == current:
                        comparison[f"equal_{key}"] += 1
                    else:
                        comparison[f"different_{key}"] += 1
                        if len(examples) < 5:
                            examples.append({"line": line_number, "task_id": task_id,
                                             "field": key, "historical": detail[key],
                                             "current_read": current,
                                             "current_ref_applicable": metrics.ref_applicable,
                                             "current_blocked_semantics_applicable": metrics.terminal_semantics_applicable})
            # 原 traces 是 benchmark history schema，不是训练 trajectory schema。
            # 只测试读取入口是否接受，不转换、不补造缺失步骤/奖励/时间字段。
            try:
                loaded = load_trajectories(path)
                loader = {"accepted": True, "count": len(loaded)}
            except Exception as exc:
                loader = {"accepted": False, "error_type": type(exc).__name__, "error": str(exc),
                          "note": "已有 benchmark history schema 与训练 Trajectory schema 不同；直接 JSONL 读取已检查。"}
            report = BenchmarkReport(metrics_list)
            report_summary = report.summary_by_agent()
            report_count = sum(row["count"] for row in report_summary.values())
            if report_count != scored:
                errors.append({"path": relative, "check": "BenchmarkReport 读取投影",
                               "error": "报告聚合条数与评分读取条数不一致"})
            all_comparisons.update(comparison)
            trace_results.append({
                "path": relative, "rows": rows, "read_only_scored": scored,
                "historical_fields_present": dict(local_fields),
                "historical_missing_fields": [field for field in limits + (
                    "reward_components", "final_reward", "raw_assistant_turns", "timing", "token_usage")
                    if local_fields[field] == 0],
                "missing_field_value": None,
                "current_reward_min": min(rewards) if rewards else None,
                "current_reward_max": max(rewards) if rewards else None,
                "current_reward_components_sha256": hashlib.sha256(json.dumps(
                    signatures, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                "report_read_projection_count": report_count,
                "current_history_usage_projection": {
                    agent: {"avg_tokens": values.get("avg_tokens"),
                            "token_usage_coverage": values.get("token_usage_coverage")}
                    for agent, values in report_summary.items()
                },
                "current_history_usage_missing_rows": {
                    field: sum(getattr(metrics, field) is None for metrics in metrics_list)
                    for field in ("prompt_tokens", "completion_tokens", "total_tokens")
                },
                "historical_explicit_avg_tokens": {
                    agent: values["avg_tokens"]
                    for agent, values in summaries.get(path.parent, {}).get(
                        "summary_by_agent", {}).items() if "avg_tokens" in values
                },
                "usage_projection_note": (
                    "这里只统计原history经当前读取器的用量字段；缺字段不据历史summary补写。"
                    "historical_explicit_avg_tokens直接读取原summary明确数字，保持原样，"
                    "与缺token_usage的history投影分开记录。"
                ),
                "historical_detail_comparison": dict(comparison),
                "comparison_examples": examples,
                "training_trajectory_loader": loader,
            })

    after = {str(path.relative_to(ROOT)): _hash(path) for path in paths}
    return {
        "schema_version": 1, "phase": phase, "description": "已有历史记录的只读索引与读取/评分契约核对。",
        "command": f".venv/bin/python scripts/audit_historical_records.py --json --phase {phase}",
        "source_index": source_index,
        "source_sha256": source_hashes,
        "source_unchanged_during_check": source_index == _source_index(sources),
        "historical_files_unchanged": before == after,
        "boundaries": [
            "只读已有文件，不执行环境、工具、模型、数据生成或训练；禁止 socket 和 registry 工具执行。",
            "不构造缺少的原始模型响应；直接解析 history 中已有 JSON action。",
            "评分是当前 RewardCalculator 对原 history 和当前仓库 TaskSpec 的显式核对，不是历史奖励复现。",
            "使用 curriculum_steps=0 的完整评分权重；历史训练步数/权重未知，不反推历史值。",
            "原 traces 缺失的状态、原始响应、奖励分解、耗时、token、协议版本和模板保持未知。",
            "历史报告指标差异可能来自历史任务/评分版本与语义判定；只列结果，不更改旧文件。",
            "现存 mainline_baseline/behavior.json 及其测试未读取或运行，不用于本次验收。",
            "已有 reward_baseline/jev 等模拟报告仅索引原 scope/note，不宣称真实执行或模型质量。",
        ],
        "inventory": indexed, "trace_checks": trace_results,
        "totals": dict(totals), "task_ids_seen": sorted(seen_tasks),
        "termination_on_recorded_histories": dict(terminal_counts),
        "recorded_action_counts": dict(action_counts),
        "observation_marker_counts": dict(observation_markers),
        "observation_marker_examples": marker_examples,
        "historical_field_presence": dict(field_counts), "errors": errors,
        "historical_detail_comparison_totals": dict(all_comparisons),
        "score_comparison_limits": [
            "arg/ref 差异不能当作历史分数回归：benchmark.runner 离线运行调用 semantic_oracle 追加 expected_paper_ids，但 traces 不保存这些 oracle 目标，本次不执行 oracle 或重建快照。",
            "当前读取 ref_score=1 且 ref_applicable=False 只是原函数的中性值，不是已核验指代准确；指标适用性明确计数，不将其转写成历史准确率。",
            "当前 false_finish 包含 blocked 终止语义检查；既有旧报告未保存终止语义版本。差异只列源记录与当前判定，不能宣称原模型能力变化。",
            "缺原 reward_components/final_reward 时历史奖励未知；仅保存当前分解摘要和哈希，不往原轨迹补写奖励，也不与发布 GRPO 训练奖励混算。",
        ],
        "known_path_differences_basis": "以下固定说明来自 S0 源码审查；位置对应 S0 historical_records.json 内的源文件哈希，不是本阶段重新审查或执行的结论。",
        "known_path_differences": [
            {
                "difference": "普通解析含猜工具与文本降级；未解析返回 None 后循环记为 FINISH；GRPO 明确区分 parse_error。",
                "evidence_type": "S0 源码审查；既有 traces 无原始响应，不能动态重测响应解析。",
                "source_locations": ["AgenticArxiv/agents/agent_engine.py:133-166",
                                     "AgenticArxiv/agents/base_agent.py:414-420",
                                     "AgenticArxiv/rl/grpo_reward.py:176-213"],
            },
            {
                "difference": "GRPO 使用终止字符串前缀识别，而普通 is_terminal_action 使用精确标记集合；两者均接受 FORCE_STOP/ERROR 为结束。",
                "evidence_type": "S0 源码审查，不补造 FINISH 前缀模型响应。",
                "source_locations": ["AgenticArxiv/agents/base_agent.py:34-36",
                                     "AgenticArxiv/rl/grpo_reward.py:194-195"],
            },
            {
                "difference": "普通和专家翻译走 enqueue 的模拟 PENDING；GRPO 将局部 downloaded/translated 集合置 READY，缓存真值来源不同。",
                "evidence_type": "S0 源码审查；PENDING/mocktask/SSE 的已有观测只证明历史记录含模拟标记。",
                "source_locations": ["AgenticArxiv/agents/base_agent.py:701-719",
                                     "AgenticArxiv/agents/side_effects.py:148-171",
                                     "scripts/generate_sft_data.py:151-159",
                                     "AgenticArxiv/rl/multiturn_env.py:331-370"],
                "historical_observation_reference": marker_examples.get("PENDING"),
            },
            {
                "difference": "普通搜索观测显示标题，专家只显示数量；GRPO 使用 str(result) 并截断 1000 字符，专家截断 500 字符。",
                "evidence_type": "S0 源码审查；历史记录只有普通 regex history，没有三入口相同世界执行记录。",
                "source_locations": ["AgenticArxiv/agents/base_agent.py:767-791",
                                     "scripts/generate_sft_data.py:264-269",
                                     "AgenticArxiv/rl/grpo_reward.py:603"],
            },
            {
                "difference": "普通/专家拒绝 fallback 写候选，GRPO keyword search 方法将返回列表写入自身 store。",
                "evidence_type": "S0 源码审查；历史 traces 没有完整 store，不能重建状态或声称状态差分已动态验证。",
                "source_locations": ["AgenticArxiv/agents/base_agent.py:744-765",
                                     "scripts/generate_sft_data.py:162-177",
                                     "AgenticArxiv/rl/multiturn_env.py:122-133"],
            },
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="向 stdout 输出完整只读检查报告")
    parser.add_argument("--phase", choices=("S0", "S1", "S2", "S3"), default="S0",
                        help="标注当前检查阶段，不改写 S0 原记录")
    args = parser.parse_args()
    result = audit(phase=args.phase)
    print(json.dumps(result if args.json else {
        "phase": result["phase"],
        "files": len(result["inventory"]), "trace_files": len(result["trace_checks"]),
        "totals": result["totals"], "errors": result["errors"],
        "historical_files_unchanged": result["historical_files_unchanged"],
        "source_unchanged_during_check": result["source_unchanged_during_check"],
    }, ensure_ascii=False, indent=2))
    return 1 if (result["errors"] or not result["historical_files_unchanged"]
                 or not result["source_unchanged_during_check"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
