"""Finish a bounded calibration: wait, independently replay, write real findings.

Never starts task optimization, touches locked tasks or commits unrelated files.
An optional neutral model probe is capped separately and runs after calibration.
"""
import argparse
import collections
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--private-run", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--replay-output", type=Path, required=True)
    p.add_argument("--findings", type=Path, required=True)
    p.add_argument("--local-findings", type=Path, required=True)
    p.add_argument("--probe-output", type=Path)
    p.add_argument("--wait-seconds", type=int, default=2100)
    args = p.parse_args()
    if not 1 <= args.wait_seconds <= 3600:
        p.error("Maximum wait is one hour")
    deadline = time.clock_gettime(time.CLOCK_MONOTONIC) + args.wait_seconds
    summary = None
    while time.clock_gettime(time.CLOCK_MONOTONIC) < deadline:
        if args.summary.exists():
            try:
                candidate = json.loads(args.summary.read_text())
            except json.JSONDecodeError:
                candidate = None  # Writer may be replacing a checkpoint.
            if candidate and candidate.get("episodes"):
                fatal = candidate["episodes"][-1]["stop"] in ("api_error", "global_budget")
                if candidate.get("complete") or fatal:
                    summary = candidate
                    break
        time.sleep(15)
    if summary is None:
        raise SystemExit("Calibration not terminal within watcher deadline; no result invented")
    if args.replay_output.exists() or args.findings.exists() or args.local_findings.exists():
        raise SystemExit("Refusing to overwrite prior finalization")
    subprocess.run([sys.executable, str(Path(__file__).with_name("analyze_rsi_appworld_probe.py")),
                    "--root", args.root, "--private-run", str(args.private_run),
                    "--output", str(args.replay_output)], check=True)
    replay = json.loads(args.replay_output.read_text())
    assert replay["all_observations_match"] and replay["all_goal_verdicts_match"]
    episodes = summary["episodes"]
    planned = len(summary["protocol"]["task_id_sha256"])
    if not planned:
        raise ValueError("No planned task identifiers")
    counts = collections.Counter(row["stop"] for row in episodes)
    successes = sum(row["success"] for row in episodes)
    normal_successes = sum(row["success"] and row["stop"] == "assistant_final" for row in episodes)
    usage = summary["usage"]
    probe = None
    if (args.probe_output and summary["complete"] and successes == 0):
        # Separate post-calibration connectivity budget, never mixed into scores.
        time.sleep(15)
        subprocess.run([sys.executable, str(Path(__file__).with_name("rsi_model_probe.py")),
                        "--model", "qwen/qwen3.6-27b", "--output", str(args.probe_output)], check=True)
        probe = json.loads((args.probe_output / "summary.json").read_text())
    batch_calls = batch_children = 0
    for line in (args.private_run / "traces.jsonl").read_text().splitlines():
        trace = json.loads(line)
        for step in trace["steps"]:
            for operation, observed in zip(step["assistant"].get("tool_calls", []), step["observations"]):
                function = operation.get("function") or {}
                if function.get("name") in ("batch_tools", "batch_api_calls"):
                    batch_calls += 1
                    batch_children += len(observed["result"].get("results", []))
    lines = ["# 固定批量接口训练校准：实际结果（2026-10-31）", "",
             f"完成状态：{summary['complete']}；已结束 {len(episodes)}/{planned} 题，独立终局成功 {successes}/{len(episodes)}，正常 final 且成功 {normal_successes}。",
             f"这是预定的 {planned} 个 meta_train/inner_train 开发实例，不是最终测试或 RSI 方法实验。", "",
             "| 训练实例摘要前缀 | 成功 | 停止原因 | 模型响应 | 顶层工具调用 | 基础调用尝试 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in episodes:
        lines.append(f"| {row['task_id_sha256'][:12]} | {row['success']} | {row['stop']} | {row['model_steps']} | {row['tool_calls']} | {row['primitive_attempts']} |")
    lines += ["", f"API 请求：{usage['calls']}；计量 input：{usage['accounted_input_tokens']}，output：{usage['accounted_output_tokens']}；缺失提供方用量：{usage['calls_without_provider_usage']} 次。缺失时采用预算预留估计，不能冒称实际账单。",
              f"停止原因：{dict(counts)}。实际分发批量调用 {batch_calls} 次，其已执行子调用 {batch_children} 次；基础操作单独计数。",
              f"独立回放 {replay['observations_replayed']} 条顶层观察，观察与最终判定均一致。源码摘要核验通过。", "",
              "未读取 meta_dev/锁定最终任务内容，未给模型参考源码或解答，没有使用 GPU。"]
    if not summary["complete"]:
        decision = "全套未完成，先处理预算/网络中断；不计算完整总分或开始递归效果对照。"
    elif successes == 0:
        decision = f"0/{planned}，未通过健康基线门槛；继续诊断模型/工具使用与响应预算。不能据此否定 RSI。"
    elif successes == planned:
        decision = f"{planned}/{planned}，本前缀已饱和；需在训练范围扩展标定，不能据此宣称改进。"
    else:
        decision = f"本前缀有初步非饱和空间，但{planned}题不足以证明稳定性；下一步冻结更完整开发/选择协议与重复基线，再进入等经验/等预算三组对照。"
    lines += ["", "推进决策：" + decision]
    if probe:
        lines += ["", f"独立中性 Qwen 连通性检查：status={probe['status']}，native_tool_call_valid={probe['native_tool_call_valid']}，usage={probe['usage']}。此费用独立于上述四题校准，未运行 Qwen 任务评测；模型别名不证明实际后端权重身份。"]
    content = "\n".join(lines) + "\n"
    for path in (args.findings, args.local_findings):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(content)
    print(json.dumps({"finalized": True, "complete": summary["complete"], "episodes": len(episodes),
                      "successes": successes, "replayed_observations": replay["observations_replayed"],
                      "findings": str(args.findings)}), flush=True)


if __name__ == "__main__":
    main()
