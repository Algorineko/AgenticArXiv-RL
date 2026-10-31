"""Run the preregistered local-model diagnostic only after all resource gates."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from rsi_qwen_appworld_probe import neutral_ready, select_gpu


def eligible(previous, previous_replay, neutral_status, neutral_probe, manifest):
    episodes = previous.get("episodes", [])
    checks = previous_replay.get("source_digests_match") or {}
    return (previous.get("complete") is True and len(episodes) == 2
            and all(e.get("success") is False for e in episodes)
            and previous_replay.get("all_observations_match") is True
            and previous_replay.get("all_goal_verdicts_match") is True
            and bool(checks) and all(value is True for value in checks.values())
            and neutral_ready(neutral_status, neutral_probe, manifest))


def write_status(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--previous-dir", type=Path, required=True)
    p.add_argument("--neutral-dir", type=Path, required=True)
    p.add_argument("--private-pool", type=Path, required=True)
    p.add_argument("--public-pool", type=Path, required=True)
    p.add_argument("--remote", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--remote-root", required=True)
    p.add_argument("--private-output", type=Path, required=True)
    p.add_argument("--public-dir", type=Path, required=True)
    p.add_argument("--findings", type=Path, required=True)
    p.add_argument("--local-findings", type=Path, required=True)
    p.add_argument("--wait-seconds", type=int, default=5400)
    args = p.parse_args()
    if not 1 <= args.wait_seconds <= 7200:
        p.error("Wait bound must not exceed two hours")
    status_file = args.public_dir / "queue_status.json"
    if status_file.exists() or args.private_output.exists() or (args.public_dir / "summary.json").exists():
        raise SystemExit("Refusing to overwrite a run")
    state = {"state": "waiting_for_verified_neutral_probe",
             "protocol": "docs/research/rsi_a40_local_calibration_protocol_20261031.md"}
    write_status(status_file, state)
    deadline = time.clock_gettime(time.CLOCK_MONOTONIC) + args.wait_seconds
    terminal = {"complete", "no_sufficient_gpu_memory", "model_download_failed", "environment_timeout",
                "probe_failed", "runner_failed", "verification_failed", "collector_timeout"}
    while time.clock_gettime(time.CLOCK_MONOTONIC) < deadline:
        path = args.neutral_dir / "status.json"
        if path.exists():
            neutral_status = json.loads(path.read_text())
            if neutral_status.get("state") in terminal:
                break
        time.sleep(20)
    else:
        state["state"] = "timeout_waiting_for_neutral_probe"
        write_status(status_file, state)
        return
    if neutral_status.get("state") != "complete":
        state.update(state="skipped_neutral_probe_not_available", neutral_state=neutral_status.get("state"))
        write_status(status_file, state)
        return
    previous = json.loads((args.previous_dir / "summary.json").read_text())
    replay = json.loads((args.previous_dir / "replay_analysis.json").read_text())
    probe = json.loads((args.neutral_dir / "probe.json").read_text())
    manifest = json.loads((args.neutral_dir / "model_manifest.json").read_text())
    if not eligible(previous, replay, neutral_status, probe, manifest):
        state["state"] = "skipped_preregistered_gates"
        write_status(status_file, state)
        return
    state["state"] = "waiting_for_eligible_gpu"
    write_status(status_file, state)
    while time.clock_gettime(time.CLOCK_MONOTONIC) < deadline:
        try:
            gpu = select_gpu(args.remote, args.port)
        except (OSError, subprocess.SubprocessError):
            gpu = None
        if gpu is not None:
            break
        time.sleep(30)
    else:
        state["state"] = "timeout_waiting_for_gpu"
        write_status(status_file, state)
        return
    state.update(state="running", selected_gpu=gpu)
    write_status(status_file, state)
    scripts = Path(__file__).resolve().parent
    repo = scripts.parent
    command = [sys.executable, str(scripts / "rsi_qwen_appworld_probe.py"), "--root", args.root,
               "--private-pool", str(args.private_pool), "--public-pool", str(args.public_pool),
               "--neutral-result", str(args.neutral_dir), "--remote", args.remote, "--port", str(args.port),
               "--remote-root", args.remote_root, "--private-output", str(args.private_output),
               "--public-output", str(args.public_dir / "summary.json")]
    process = subprocess.run(command, cwd=repo)
    if process.returncode:
        state.update(state="driver_failed", exit_code=process.returncode)
        write_status(status_file, state)
        return
    command = [sys.executable, str(scripts / "finalize_rsi_train_stage.py"), "--root", args.root,
               "--private-run", str(args.private_output), "--summary", str(args.public_dir / "summary.json"),
               "--replay-output", str(args.public_dir / "replay_analysis.json"), "--findings", str(args.findings),
               "--local-findings", str(args.local_findings), "--wait-seconds", "60"]
    process = subprocess.run(command, cwd=repo)
    if process.returncode:
        state.update(state="replay_or_finalization_failed", exit_code=process.returncode)
        write_status(status_file, state)
        return
    result = json.loads((args.public_dir / "summary.json").read_text())
    state.update(state="complete", complete=result["complete"], episodes=len(result["episodes"]),
                 successes=sum(e["success"] for e in result["episodes"]))
    write_status(status_file, state)
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
