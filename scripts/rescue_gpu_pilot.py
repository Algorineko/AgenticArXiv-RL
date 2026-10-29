"""Bounded single-GPU infrastructure pilot, explicitly separate from research.

Without --snapshot the fabricated paper pools exercise only the eight default
smoke tasks. They are not the historical snapshot or a generalization benchmark.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
NOTE = (
    "本轮行动前，请核对原始用户请求中的每个操作是否已有成功的工具观察。"
    "尚未执行完的操作需要继续调用相应工具；只有请求全部完成，或确实无法执行且已说明原因，"
    "才输出 FINISH。不要把已有会话状态误认为当前请求已经执行。"
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(path):
    pools = {}
    for offset, aspect in enumerate(("*", "AI", "LG", "CL", "CV", "RO", "CR")):
        args = {"aspect": aspect, "days": 7, "max_results": 10}
        papers = [
            {"id": f"2401.{offset * 100 + i:05d}v1",
             "title": f"SYNTHETIC infrastructure smoke paper {aspect} {i}",
             "authors": ["Synthetic Fixture"], "summary": "Fabricated test fixture.",
             "pdf_url": f"https://example.invalid/{aspect}/{i}.pdf",
             "primary_category": "cs." + ("AI" if aspect == "*" else aspect)}
            for i in range(1, 11)
        ]
        key = json.dumps(args, sort_keys=True, ensure_ascii=False)
        pools[key] = {"args": args, "result": papers}
    path.write_text(json.dumps({"get_recently_submitted_cs_papers": pools},
                               ensure_ascii=False, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--snapshot", type=Path)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--variant", choices=("baseline", "completion_check"), default="baseline")
    p.add_argument("--timeout", type=int, default=1200)
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    # Serialize our own jobs; this does not reserve the GPU against other users.
    lock = open(f"/tmp/agenticarxiv-rescue-{os.getuid()}-gpu{args.gpu}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    query = subprocess.check_output([
        "nvidia-smi", f"--id={args.gpu}",
        "--query-gpu=index,uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"
    ], text=True).strip()
    free_mib = int(query.split(",")[2].strip())
    if free_mib < 16000:
        raise SystemExit(f"Insufficient free memory: {query}")
    snapshot = args.snapshot.resolve() if args.snapshot else args.output / "SYNTHETIC_smoke_snapshot.json"
    if not args.snapshot:
        fixture(snapshot)
    if not snapshot.is_file():
        raise SystemExit(f"Missing snapshot: {snapshot}")
    model = Path(args.model).resolve()
    model_config = model / "config.json"
    if not model_config.is_file():
        raise SystemExit(f"Missing model config: {model_config}")
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": str(args.gpu), "STORE_BACKEND": "memory",
                "TOOL_ROUTER": "policy", "AA_EVAL_GREEDY": "1",
                "AA_MAX_ITERATIONS_OVERRIDE": "6", "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
                "TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1",
                "PYTHONPATH": str(ROOT / "AgenticArxiv") + os.pathsep + str(ROOT)})
    for name in ("PDF_RAW_PATH", "PDF_TRANSLATED_PATH", "PDF_TRANSLATED_LOG_PATH",
                 "PDF_CACHE_PATH", "TRANSLATE_CACHE_PATH", "PDF_FIGURES_PATH"):
        env[name] = str(args.output / "runtime" / name.lower())
    command = [sys.executable, "-m", "AgenticArxiv.benchmark.run_benchmark",
               "--backend", "transformers", "--model", str(model), "--agents", "regex",
               "--repeat", "1", "--seed", "45", "--local-device", "cuda",
               "--local-dtype", "bfloat16", "--offline", "--task-set", "default",
               "--snapshot", str(snapshot), "--output", str(args.output / "benchmark"),
               "--prefix", "rescue_smoke", "--save-traces"]
    # In-process setup only controls memory and the visible prompt; it never sees gold steps.
    bootstrap = (
        "import runpy,sys,torch; torch.set_num_threads(4); "
        "torch.cuda.set_per_process_memory_fraction(0.35,0); "
    )
    if args.variant == "completion_check":
        bootstrap += (
            "import agents.agent_engine as ae; original=ae.get_react_prompt; "
            f"ae.get_react_prompt=lambda *a,**k: original(*a,**k)+{NOTE!r}; "
        )
    bootstrap += "sys.argv=" + repr(command[2:]) + "; runpy.run_module('AgenticArxiv.benchmark.run_benchmark',run_name='__main__')"
    actual = [sys.executable, "-c", bootstrap]
    manifest = {"kind": "infrastructure_smoke_only", "variant": args.variant,
                "historical_reproduction": False, "synthetic_snapshot": not bool(args.snapshot),
                "gpu_preflight": query, "physical_gpu": args.gpu, "seed": 45,
                "task_set": "default", "n_tasks": 8, "repeat": 1, "decode": "greedy",
                "max_iterations_cap": 6, "timeout_seconds": args.timeout,
                "model": str(model), "model_config_sha256": digest(model_config),
                "snapshot_sha256": digest(snapshot), "launcher_sha256": digest(__file__),
                "client_sha256": digest(ROOT / "AgenticArxiv/utils/llm_client.py"),
                "prompt_addition": NOTE if args.variant == "completion_check" else "",
                "started_at_unix": time.time(), "status": "running"}
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    try:
        with (args.output / "run.log").open("w") as log:
            result = subprocess.run(actual, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=args.timeout)
        manifest.update(status="completed" if result.returncode == 0 else "failed",
                        exit_code=result.returncode)
    except subprocess.TimeoutExpired:
        manifest.update(status="timeout", exit_code=124)
    finally:
        manifest["ended_at_unix"] = time.time()
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return manifest["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
