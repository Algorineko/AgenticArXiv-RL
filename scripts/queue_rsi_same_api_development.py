"""Run a preregistered two-task development diagnostic after a 0/4 baseline."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write_status(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--previous-summary", type=Path, required=True)
    parser.add_argument("--previous-findings", type=Path, required=True)
    parser.add_argument("--probe-summary", type=Path, required=True)
    parser.add_argument("--private-pool", type=Path, required=True)
    parser.add_argument("--public-pool", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--findings", type=Path, required=True)
    parser.add_argument("--local-findings", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=2400)
    args = parser.parse_args()
    if not 1 <= args.wait_seconds <= 3600 or not os.environ.get("RSI_API_KEY"):
        parser.error("API key environment variable and wait of at most one hour required")
    status = args.public_dir / "queue_status.json"
    if status.exists() or args.private_output.exists() or (args.public_dir / "summary.json").exists():
        raise SystemExit("Refusing to overwrite an existing diagnostic")
    record = {"state": "waiting_for_completed_baseline_and_replay",
              "protocol": "docs/research/rsi_same_api_development_protocol_20261031.md",
              "selection": "First two frozen meta_train groups, inner_train only"}
    write_status(status, record)
    deadline = time.clock_gettime(time.CLOCK_MONOTONIC) + args.wait_seconds
    while time.clock_gettime(time.CLOCK_MONOTONIC) < deadline:
        if args.previous_findings.exists() and args.previous_summary.exists():
            break
        time.sleep(15)
    else:
        record["state"] = "timeout_waiting_for_previous_replay"
        write_status(status, record)
        raise SystemExit("Previous findings did not appear within the bounded wait")
    previous = json.loads(args.previous_summary.read_text())
    successes = sum(e["success"] for e in previous["episodes"])
    record["previous_complete"] = previous["complete"]
    record["previous_successes"] = successes
    if not previous["complete"] or len(previous["episodes"]) != 4 or successes != 0:
        record["state"] = "skipped_preregistered_condition_not_met"
        write_status(status, record)
        return
    probe = json.loads(args.probe_summary.read_text()) if args.probe_summary.exists() else {}
    model = "qwen/qwen3.6-27b" if probe.get("native_tool_call_valid") is True else "doubao-seed-2.0-mini"
    record.update(state="running", requested_model=model,
                  model_selection="Qwen if neutral tool probe passed, otherwise the original mini model",
                  previous_probe_passed=probe.get("native_tool_call_valid") is True)
    write_status(status, record)
    # Previous connectivity probe might have just finished; respect the shared RPM limit.
    time.sleep(15)
    scripts = Path(__file__).resolve().parent
    repo = scripts.parent
    command = [sys.executable, str(scripts / "rsi_appworld_api_probe.py"),
               "--root", args.root, "--private-output", str(args.private_output),
               "--public-output", str(args.public_dir / "summary.json"),
               "--harness-file", "configs/rsi/same_api_batch_development.txt", "--registry", "--batch",
               "--same-api-batch", "--primitive-budget", "128", "--max-steps", "24", "--max-calls", "50",
               "--max-input-tokens", "800000", "--max-output-tokens", "50000", "--completion-tokens", "2048",
               "--train-pool-file", str(args.private_pool), "--train-pool-manifest", str(args.public_pool),
               "--train-count", "2", "--model", model, "--experiment-prefix", "rsi_same_api_dev_v1"]
    process = subprocess.run(command, cwd=repo)
    if process.returncode:
        record.update(state="driver_failed", exit_code=process.returncode)
        write_status(status, record)
        raise SystemExit(process.returncode)
    command = [sys.executable, str(scripts / "finalize_rsi_train_stage.py"), "--root", args.root,
               "--private-run", str(args.private_output), "--summary", str(args.public_dir / "summary.json"),
               "--replay-output", str(args.public_dir / "replay_analysis.json"), "--findings", str(args.findings),
               "--local-findings", str(args.local_findings), "--wait-seconds", "60"]
    process = subprocess.run(command, cwd=repo)
    if process.returncode:
        record.update(state="replay_or_finalization_failed", exit_code=process.returncode)
        write_status(status, record)
        raise SystemExit(process.returncode)
    result = json.loads((args.public_dir / "summary.json").read_text())
    record.update(state="complete", complete=result["complete"],
                  episodes=len(result["episodes"]), successes=sum(e["success"] for e in result["episodes"]))
    write_status(status, record)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
