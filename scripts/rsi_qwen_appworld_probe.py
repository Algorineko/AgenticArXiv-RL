"""Two train-only AppWorld diagnostics with a bounded remote open-weight model."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from AgenticArxiv.rsi.api_batch_environment import SameAPIBatchEnvironment
from AgenticArxiv.rsi.tool_environment import RegisteredToolEnvironment
from rsi_api_pilot import episode, utc
from rsi_appworld_api_probe import AppWorldEnvironment
from rsi_qwen_stdio_client import LocalQwenClient


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_gpu(remote, port):
    command = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=8",
               "-p", str(port), remote,
               "nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits"]
    result = subprocess.check_output(command, text=True, timeout=12)
    rows = {int(a.strip()): (int(b.strip()), int(c.strip())) for a, b, c in
            (line.split(",") for line in result.splitlines())}
    return next((i for i in (1, 0, 3, 2) if i in rows and rows[i][0] >= 40000 and rows[i][1] < 10), None)


def neutral_ready(status, probe, manifest):
    checks = status.get("checks") or {}
    return (status.get("state") == "complete" and bool(checks) and all(v is True for v in checks.values())
            and probe.get("native_tool_call_valid") is True and manifest.get("model_id") == "Qwen/Qwen3-8B")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--private-pool", type=Path, required=True)
    p.add_argument("--public-pool", type=Path, required=True)
    p.add_argument("--neutral-result", type=Path, required=True)
    p.add_argument("--remote", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--remote-root", required=True)
    p.add_argument("--private-output", type=Path, required=True)
    p.add_argument("--public-output", type=Path, required=True)
    args = p.parse_args()
    if (not re.fullmatch(r"[A-Za-z0-9_-]+@[A-Za-z0-9.:-]+", args.remote)
            or not re.fullmatch(r"/tmp/agentic-rsi-[A-Za-z0-9_-]+", args.remote_root)
            or not 1 <= args.port <= 65535):
        p.error("Invalid remote or temporary model directory")
    if args.private_output.exists() or args.public_output.exists():
        raise SystemExit("Refusing to overwrite a run")
    neutral_status = json.loads((args.neutral_result / "status.json").read_text())
    probe = json.loads((args.neutral_result / "probe.json").read_text())
    manifest_file = args.neutral_result / "model_manifest.json"
    manifest = json.loads(manifest_file.read_text())
    if not neutral_ready(neutral_status, probe, manifest):
        raise SystemExit("Verified successful neutral tool probe is required")
    gpu = select_gpu(args.remote, args.port)
    if gpu is None:
        raise SystemExit("No eligible single GPU currently has enough free memory")
    from appworld import load_task_ids
    from appworld.common.path_store import path_store
    path_store.update_root(args.root)
    pool = json.loads(args.private_pool.read_text())
    public = json.loads(args.public_pool.read_text())
    allowed_train = set(load_task_ids("train"))
    selected = []
    for index, entry in enumerate(pool["meta_train"][:2]):
        task_id = entry["tasks"]["inner_train"]
        assert task_id in allowed_train
        assert hashlib.sha256(task_id.encode()).hexdigest() == public["pools"]["meta_train"][index]["task_sha256_by_role"]["inner_train"]
        selected.append(task_id)
    if len(selected) != 2 or len(set(selected)) != 2:
        raise ValueError("Expected the fixed two-task training prefix")
    args.private_output.mkdir(parents=True)
    args.public_output.parent.mkdir(parents=True, exist_ok=True)
    remote_manifest = shlex.quote(f"{args.remote_root}/model_manifest.json")
    remote_digest_output = subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=8",
         "-p", str(args.port), args.remote, f"sha256sum {remote_manifest}"], text=True, timeout=12)
    remote_manifest_sha = remote_digest_output.split()[0]
    if remote_manifest_sha != sha(manifest_file):
        raise RuntimeError("Collected and remote model manifests differ")
    server_file = Path("scripts/rsi_qwen_stdio_server.py")
    server_hash = sha(server_file)
    remote_server = f"{args.remote_root}/rsi_qwen_stdio_server_{server_hash[:12]}.py"
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                    "-o", "ConnectTimeout=8", "-P", str(args.port), str(server_file),
                    f"{args.remote}:{remote_server}"], check=True, timeout=30)
    remote_command = shlex.join(["timeout", "3600", "env", f"CUDA_VISIBLE_DEVICES={gpu}",
                                f"{args.remote_root}/venv/bin/python", "-u", remote_server,
                                "--model-dir", f"{args.remote_root}/model", "--physical-gpu", str(gpu),
                                "--manifest", f"{args.remote_root}/model_manifest.json",
                                "--max-context-tokens", "8192", "--max-new-tokens", "1024"])
    command = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
               "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
               "-p", str(args.port), args.remote, remote_command]
    limits = {"calls": 50, "input_tokens": 800000, "output_tokens": 50000}
    sources = ["rsi_api_pilot.py", "rsi_cpu_tasks.py", "rsi_appworld_api_probe.py"]
    framework = ["AgenticArxiv/tools/tool_registry.py", "AgenticArxiv/rsi/tool_environment.py",
                 "AgenticArxiv/rsi/batch_environment.py", "AgenticArxiv/rsi/api_batch_environment.py"]
    backend = ["scripts/rsi_qwen_stdio_client.py", "scripts/rsi_qwen_stdio_server.py",
               "scripts/rsi_qwen_appworld_probe.py", "configs/rsi/same_api_batch_development.txt"]
    protocol = {"stage": "local model train-only development diagnostic; no RSI effect", "created_utc": utc(),
                "task_id_sha256": [hashlib.sha256(x.encode()).hexdigest() for x in selected],
                "training_pool": {"selection": "Frozen meta_train prefix, inner_train only", "count": 2,
                                  "public_manifest_sha256": sha(args.public_pool)},
                "model": "Qwen/Qwen3-8B", "model_manifest_sha256": sha(manifest_file),
                "max_steps": 24, "max_parallel_tools": 4, "limits": limits,
                "isolated_framework_registry": True, "bounded_batch_interface": True,
                "same_api_batch_interface": True, "primitive_budget": 128, "max_environment_interactions": 128,
                "max_completion_tokens": 1024, "max_prompt_tokens": 8192, "physical_gpu": gpu,
                "harness_sha256": sha("configs/rsi/same_api_batch_development.txt"),
                "sources": {name: sha(Path("scripts") / name) for name in sources},
                "framework_sources": {name: sha(name) for name in framework},
                "backend_sources": {name: sha(name) for name in backend}}
    (args.private_output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for name in sources:
        destination = args.private_output / "frozen_sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((Path("scripts") / name).read_bytes())
    for base, names in (("frozen_framework_sources", framework), ("frozen_backend_sources", backend)):
        for name in names:
            destination = args.private_output / base / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(Path(name).read_bytes())
    expected = {"model_id": "Qwen/Qwen3-8B", "physical_gpu": gpu, "server_source_sha256": server_hash,
                "model_manifest_sha256": sha(manifest_file), "config_sha256": probe["config_sha256"],
                "max_memory_fraction": 0.60, "max_context_tokens": 8192, "max_new_tokens": 1024}
    api = LocalQwenClient(command, args.private_output, limits, expected, max_completion_tokens=1024)
    if "a40" not in api.ready.get("gpu_name", "").lower() or api.ready.get("memory_before_mib", 0) < 40000:
        api.close()
        raise RuntimeError("Unexpected GPU or inadequate pre-load free memory")
    harness = Path("configs/rsi/same_api_batch_development.txt").read_text()
    records = []
    try:
        for index, task_id in enumerate(selected):
            raw = AppWorldEnvironment(task_id, f"rsi_local_qwen_v1_{index}", max_interactions=128)
            task = {"id": hashlib.sha256(task_id.encode()).hexdigest(), "domain": "appworld",
                    "prompt": raw.world.task.instruction + "\nSupervisor identity: " + str(raw.world.task.supervisor) +
                    "\nUse discovery tools to find the APIs and their parameters. Perform the actual requested "
                    "operations, then call supervisor.complete_task with the appropriate answer if required."}
            primitive = SameAPIBatchEnvironment(raw, primitive_budget=128)
            trace = episode(api, task, repeat=0, max_steps=24, environment=RegisteredToolEnvironment(primitive),
                            harness_instruction=harness)
            records.append({"task_id_sha256": task["id"], "success": trace["verdict"]["success"],
                            "stop": trace["stop"], "model_steps": len(trace["steps"]),
                            "tool_calls": trace["tool_calls"], "tool_errors": trace["tool_errors"],
                            "primitive_attempts": primitive.primitive_attempts})
            calls_file = args.private_output / "api_calls.jsonl"
            calls = [json.loads(x) for x in calls_file.read_text().splitlines()] if calls_file.exists() else []
            peak_reserved = max([api.ready.get("peak_reserved_mib", 0)] + [x.get("peak_reserved_mib") or 0 for x in calls])
            result = {"protocol": protocol, "episodes": records, "usage": api.totals(),
                      "complete": len(records) == len(selected), "protected_trace_location": str(args.private_output),
                      "local_backend": {"physical_gpu": gpu, "gpu_name": api.ready["gpu_name"],
                                        "peak_reserved_mib": peak_reserved, "memory_fraction": 0.60},
                      "limitation": "Two exposed train instances, local context/output limits; not an RSI or official benchmark score."}
            temporary = args.public_output.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            temporary.replace(args.public_output)
            print(json.dumps(records[-1]), flush=True)
            if trace["stop"] in ("global_budget", "api_error", "model_resource_unavailable"):
                break
    finally:
        api.close()


if __name__ == "__main__":
    main()
