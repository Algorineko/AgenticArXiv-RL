"""Collect only aggregate, fingerprinted results from the isolated A40 probe."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import time


def source_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(manifest, probe):
    files = manifest.get("files") or {}
    weights = [value for name, value in files.items() if name.endswith(".safetensors")]
    valid_hashes = bool(weights) and all(re.fullmatch(r"[a-f0-9]{64}", v.get("sha256", "")) for v in files.values())
    return {"expected_model_and_type": manifest.get("model_id") == "Qwen/Qwen3-8B" and manifest.get("model_type") == "qwen3",
            "expected_weight_size_and_hashes": valid_hashes and sum(v.get("bytes", 0) for v in weights) >= 15_000_000_000,
            "download_script_matches": manifest.get("source_sha256") == source_sha("scripts/rsi_prepare_qwen_model.py"),
            "probe_script_matches": probe.get("probe_source_sha256") == source_sha("scripts/rsi_qwen_local_probe.py"),
            "model_config_matches": probe.get("config_sha256") == files.get("config.json", {}).get("sha256"),
            "single_a40_with_headroom": "a40" in probe.get("gpu_name", "").lower()
                                        and probe.get("physical_gpu") in (0, 1, 2, 3)
                                        and probe.get("memory_before_mib", 0) >= 40000,
            "bounded_peak_memory": probe.get("max_memory_fraction") == 0.60
                                   and 0 < probe.get("peak_allocated_mib", 0) <= probe.get("peak_reserved_mib", 0)
                                   <= 0.60 * probe.get("gpu_total_memory_mib", 0)}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--remote", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--remote-root", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--local-findings", type=Path, required=True)
    p.add_argument("--wait-seconds", type=int, default=5400)
    args = p.parse_args()
    if (not re.fullmatch(r"[a-zA-Z0-9_-]+@[a-zA-Z0-9.:-]+", args.remote)
            or not re.fullmatch(r"/tmp/agentic-rsi-[a-zA-Z0-9_-]+", args.remote_root)
            or not 1 <= args.port <= 65535 or not 1 <= args.wait_seconds <= 7200):
        p.error("Invalid remote, temporary directory, port or wait bound")
    if args.output.exists() or args.local_findings.exists():
        raise SystemExit("Refusing to overwrite existing results")
    args.output.mkdir(parents=True)
    status_path = args.output / "status.json"
    write_json(status_path, {"state": "waiting_for_remote_probe"})
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=8",
           "-p", str(args.port), args.remote]
    end = time.clock_gettime(time.CLOCK_MONOTONIC) + args.wait_seconds
    terminal = {"complete", "no_sufficient_gpu_memory", "model_download_failed", "environment_timeout",
                "probe_failed", "runner_failed"}
    state = None
    while time.clock_gettime(time.CLOCK_MONOTONIC) < end:
        try:
            result = subprocess.run(ssh + ["cat " + shlex.quote(args.remote_root + "/stage.json")],
                                    capture_output=True, text=True, timeout=12)
            if result.returncode == 0:
                state = json.loads(result.stdout).get("state")
                write_json(status_path, {"state": state})
                if state in terminal:
                    break
        except (subprocess.TimeoutExpired, ValueError):
            pass
        time.sleep(20)
    else:
        state = "collector_timeout"
    if state != "complete":
        write_json(status_path, {"state": state})
        args.local_findings.parent.mkdir(parents=True, exist_ok=True)
        args.local_findings.write_text(f"# A40 本地模型工具调用检查\n\n当前状态：{state}。没有可报告的模型能力结果；原始准备日志保留在服务器临时目录。\n")
        return
    scp = ["scp", "-q", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
           "-o", "ConnectTimeout=8", "-P", str(args.port)]
    with tempfile.TemporaryDirectory(prefix="rsi-a40-collect-") as tmp:
        dest = Path(tmp)
        for name in ("model_manifest.json", "probe.json"):
            subprocess.run(scp + [f"{args.remote}:{args.remote_root}/{name}", str(dest / name)], check=True, timeout=60)
        manifest = json.loads((dest / "model_manifest.json").read_text())
        probe = json.loads((dest / "probe.json").read_text())
        checks = validate(manifest, probe)
        if not all(checks.values()):
            write_json(status_path, {"state": "verification_failed", "checks": checks})
            raise SystemExit("Remote aggregate failed provenance or resource verification")
        write_json(args.output / "model_manifest.json", manifest)
        write_json(args.output / "probe.json", probe)
        write_json(status_path, {"state": "complete", "checks": checks})
    args.local_findings.parent.mkdir(parents=True, exist_ok=True)
    args.local_findings.write_text(
        "# A40 本地模型工具调用检查\n\n"
        f"Qwen3-8B 使用 A40 {probe['physical_gpu']} 号卡，加载前可用 {probe['memory_before_mib']} MiB，"
        f"PyTorch 峰值分配 {probe['peak_allocated_mib']} MiB、峰值保留 {probe['peak_reserved_mib']} MiB"
        f"（分配器上限为单卡总显存的60%，另有CUDA上下文等开销）。"
        f"一次中性工具调用格式有效：{probe['native_tool_call_valid']}。\n\n"
        "模型文件已逐一计算哈希，下载实现和探针源码与本地一致；这些哈希标识实际下载内容，"
        "并非对上游签名的认证。没有运行任务基线或 RSI 效果实验。"
        f"权重在服务器临时目录 {args.remote_root}，不是持久备份。\n")
    print(json.dumps({"state": "complete", "native_tool_call_valid": probe["native_tool_call_valid"],
                      "physical_gpu": probe["physical_gpu"], "peak_allocated_mib": probe["peak_allocated_mib"]}), flush=True)


if __name__ == "__main__":
    main()
