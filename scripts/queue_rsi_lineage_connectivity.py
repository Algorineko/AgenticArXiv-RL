"""Conditionally exercise the real-model improver lineage without scoring it."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from AgenticArxiv.rsi.improver import make_request, propose
from rsi_api_pilot import API, utc


def eligible(status, summary, replay):
    """Only a completed, replayed, non-saturated pair may trigger live calls."""
    episodes = summary.get("episodes", [])
    source_checks = replay.get("source_digests_match") or {}
    protocol = summary.get("protocol") or {}
    return (status.get("state") == "complete" and summary.get("complete") is True
            and len(episodes) == 2 and all(type(e.get("success")) is bool for e in episodes)
            and sum(e["success"] for e in episodes) == 1
            and len({e.get("task_id_sha256") for e in episodes}) == 2
            and len(protocol.get("task_id_sha256", [])) == 2
            and {e.get("task_id_sha256") for e in episodes} == set(protocol.get("task_id_sha256", []))
            and status.get("requested_model") == protocol.get("model")
            and (protocol.get("training_pool") or {}).get("selection") == "Frozen meta_train prefix, inner_train only"
            and replay.get("all_observations_match") is True
            and replay.get("all_goal_verdicts_match") is True
            and bool(source_checks) and all(v is True for v in source_checks.values()))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prior-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--private-output", type=Path, required=True)
    p.add_argument("--local-findings", type=Path, required=True)
    p.add_argument("--wait-seconds", type=int, default=2400)
    args = p.parse_args()
    if not 1 <= args.wait_seconds <= 3600 or not os.environ.get("RSI_API_KEY"):
        p.error("A bounded wait and API key environment variable are required")
    if args.output.exists() or args.private_output.exists() or args.local_findings.exists():
        raise SystemExit("Refusing to overwrite an existing smoke run")
    args.output.mkdir(parents=True)
    status_path = args.output / "status.json"
    write_json(status_path, {"state": "waiting_for_replayed_development_pair"})
    deadline = time.clock_gettime(time.CLOCK_MONOTONIC) + args.wait_seconds
    while time.clock_gettime(time.CLOCK_MONOTONIC) < deadline:
        prior_status = args.prior_dir / "queue_status.json"
        if prior_status.exists():
            status = json.loads(prior_status.read_text())
            if status.get("state") in {"complete", "skipped_preregistered_condition_not_met",
                                       "driver_failed", "replay_or_finalization_failed",
                                       "timeout_waiting_for_previous_replay"}:
                break
        time.sleep(15)
    else:
        write_json(status_path, {"state": "timeout_waiting_for_development_pair"})
        return
    if status.get("state") != "complete":
        write_json(status_path, {"state": "skipped_prior_stage_incomplete"})
        args.local_findings.parent.mkdir(parents=True, exist_ok=True)
        args.local_findings.write_text("# 真实模型谱系连通性检查\n\n前一阶段未完成，已按预案跳过；没有发生谱系模型调用。\n")
        return
    summary = json.loads((args.prior_dir / "summary.json").read_text())
    replay = json.loads((args.prior_dir / "replay_analysis.json").read_text())
    if not eligible(status, summary, replay):
        write_json(status_path, {"state": "skipped_preregistered_gate", "no_model_calls": True})
        successes = sum(e.get("success") is True for e in summary.get("episodes", []))
        args.local_findings.parent.mkdir(parents=True, exist_ok=True)
        args.local_findings.write_text(f"# 真实模型谱系连通性检查\n\n前置两题成功 {successes}/{len(summary.get('episodes', []))}，或回放未满足完整核验；按预案跳过，没有发生谱系模型调用。\n")
        return
    model = status["requested_model"]
    if model not in {"doubao-seed-2.0-mini", "qwen/qwen3.6-27b"}:
        raise ValueError("Unexpected model alias")
    args.private_output.mkdir(parents=True)
    solver = Path("configs/rsi/same_api_batch_development.txt").read_text()
    seed = ("Use observed development failures to propose a clearer, reusable improvement procedure. "
            "Prefer concrete testable behavior, preserve the tool interface and independent evaluation, "
            "and avoid memorizing task-specific names or identifiers.")
    experience = [{k: row[k] for k in ("task_id_sha256", "success", "stop", "model_steps", "primitive_attempts")}
                  for row in summary["episodes"]]
    limits = {"calls": 6, "input_tokens": 100000, "output_tokens": 8192}
    api = API("https://api.qnaigc.com/v1", model, args.private_output, limits,
              min_interval=15, max_completion_tokens=1024)
    paths = ["AgenticArxiv/rsi/improver.py", "scripts/rsi_api_pilot.py", "scripts/rsi_cpu_tasks.py",
             "scripts/queue_rsi_lineage_connectivity.py", "configs/rsi/same_api_batch_development.txt"]
    protocol = {"created_utc": utc(), "model": model, "limits": limits,
                "source_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in paths},
                "prior_summary_sha256": hashlib.sha256((args.prior_dir / "summary.json").read_bytes()).hexdigest(),
                "stage": "real-model lineage plumbing; no performance evaluation or candidate acceptance"}
    write_json(args.output / "protocol.json", protocol)
    for name in paths:
        destination = args.private_output / "frozen_sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(name).read_bytes())
    records, candidates = [], []
    write_json(status_path, {"state": "running", "requested_model": model})
    # The preceding stage may have just sent its last request.
    time.sleep(15)
    state = "completed"
    try:
        first = make_request("recursive", "improver", solver, seed, seed, experience)
        record, first_candidate = propose(api, first, "lineage-shared-first")
        records.append(record)
        if first_candidate is not None:
            candidates.append({"stage": "shared_provisional_m1", "text": first_candidate})
        if not record["valid_syntax"] or record.get("unchanged"):
            state = "first_candidate_invalid_or_unchanged"
        else:
            fixed = make_request("fixed_outer", "improver", solver, first_candidate, seed, experience)
            recursive = make_request("recursive", "improver", solver, first_candidate, seed, experience)
            assert fixed.messages[1] == recursive.messages[1]
            assert fixed.metadata["experience_sha256"] == recursive.metadata["experience_sha256"]
            assert fixed.metadata["parent_text_sha256"] == recursive.metadata["parent_text_sha256"]
            for request in (fixed, recursive):
                record, candidate = propose(api, request, f"lineage-{request.metadata['arm']}")
                records.append(record)
                if candidate is not None:
                    candidates.append({"stage": request.metadata["arm"], "text": candidate})
    except RuntimeError:
        state = "api_or_budget_failure"
    with (args.private_output / "candidates.jsonl").open("x") as stream:
        for candidate in candidates:
            stream.write(json.dumps(api.clean(candidate), ensure_ascii=False) + "\n")
    result = {"state": state, "stage": protocol["stage"], "records": records, "usage": api.totals(),
              "initial_candidate_performance_evaluated": False, "candidates_accepted": 0,
              "meta_final_tasks_read": 0, "private_candidate_location": str(args.private_output),
              "limitation": "Syntactic lineage plumbing only; no evidence of improved performance."}
    write_json(args.output / "summary.json", api.clean(result))
    write_json(status_path, {"state": "complete", "result_state": state, "model_calls": api.totals()["calls"]})
    args.local_findings.parent.mkdir(parents=True, exist_ok=True)
    args.local_findings.write_text(
        "# 真实模型谱系连通性检查\n\n"
        f"结果状态：{state}；候选请求记录 {len(records)}，其中语法有效 {sum(r['valid_syntax'] for r in records)}。"
        f"API计量：{api.totals()}。\n\n"
        "M1仅为未经任务评估的临时候选，没有候选被接受；本检查不证明性能改进，也没有访问最终任务。"
        "详细哈希与边界见 RSI 工作树 `artifacts/rsi_lineage_connectivity_v1/summary.json`。\n")
    print(json.dumps({"state": state, "model_calls": api.totals()["calls"], "proposals": len(records)}), flush=True)


if __name__ == "__main__":
    main()
