"""Evaluate a frozen new catalog, with full traces and single-GPU bounds.

The final split is accessible only to the scripted reference preflight, never
to a model through this development entry point.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "AgenticArxiv"), str(ROOT / "scripts")]
from benchmark.rescue_catalog import sha256, verify_catalog
from rescue_gpu_pilot import NOTE


class ReferenceClient:
    """Gold actions are available only in the explicitly scripted preflight."""
    def __init__(self, tasks, finish_only=False):
        self.tasks = {t["id"]: t for t in tasks}
        self.finish_only = finish_only

    def start_generation_stream(self, key):
        task = self.tasks[key.rsplit(":", 1)[0]]
        self.actions = [] if self.finish_only else [json.dumps({"name": n, "args": a}, ensure_ascii=False)
            for n, a in zip(task["expected_tools"], task["expected_tool_args"])]
        self.actions.append("FINISH")
        self.position = 0

    def chat_completions(self, **kwargs):
        action = self.actions[self.position]
        self.position += 1
        return {"choices": [{"message": {"content": "Thought: 执行当前步骤\nAction: " + action}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


def set_runtime(output, gpu):
    os.environ.update({"CUDA_VISIBLE_DEVICES": str(gpu), "STORE_BACKEND": "memory",
                       "TOOL_ROUTER": "policy", "AA_EVAL_GREEDY": "1",
                       "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                       "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1",
                       "TRANSFORMERS_OFFLINE": "1"})
    # Explicitly remove inherited sampling/budget overrides so paired arms agree.
    os.environ.pop("AA_EVAL_TEMPERATURE", None)
    os.environ.pop("AA_MAX_ITERATIONS_OVERRIDE", None)
    os.environ.pop("VLM_ADAPTER_PATH", None)
    for key in ("PDF_RAW_PATH", "PDF_TRANSLATED_PATH", "PDF_TRANSLATED_LOG_PATH",
                "PDF_CACHE_PATH", "TRANSLATE_CACHE_PATH", "PDF_FIGURES_PATH"):
        os.environ[key] = str(output / "runtime" / key.lower())


def _timeout(signum, frame):
    raise TimeoutError("Development arm reached its declared wall-time limit")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--catalog-sha256", required=True)
    p.add_argument("--split", choices=("all", "train", "search_dev", "promotion_val", "final_test"), default="search_dev")
    p.add_argument("--policy", choices=("model", "reference", "finish_only"), default="model")
    p.add_argument("--model", type=Path)
    p.add_argument("--model-id", default="")
    p.add_argument("--variant", choices=("baseline", "completion_check"), default="baseline")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--timeout", type=int, default=900)
    args = p.parse_args()
    if args.policy != "reference" and args.split not in ("search_dev", "promotion_val"):
        p.error("Train/final/all are reserved for scripted reference validation")
    if args.policy == "model" and (not args.model or not args.model_id):
        p.error("Model evaluation requires --model and --model-id")
    if sha256(args.catalog / "manifest.json") != args.catalog_sha256:
        p.error("Catalog does not match the pre-evaluation hash")
    catalog_manifest, tasks, splits = verify_catalog(args.catalog)
    if args.split != "all":
        ids = set(splits["split"][args.split])
        tasks = [t for t in tasks if t["id"] in ids]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    set_runtime(args.output, args.gpu)
    manifest = {"kind": "real_metadata_development_screen", "policy": args.policy,
                "model_id": args.model_id, "variant": args.variant, "split": args.split,
                "catalog_manifest_sha256": args.catalog_sha256,
                "split_sha256": sha256(args.catalog / "splits.json"),
                "task_ids": [t["id"] for t in tasks], "n_tasks": len(tasks),
                "seed": 45, "repeat": 1, "decode": "greedy", "max_new_tokens": 384,
                "max_iterations": 6, "timeout_seconds": args.timeout,
                "prompt_addition": NOTE if args.variant == "completion_check" else "",
                "source_sha256": {name: sha256(ROOT / name) for name in (
                    "scripts/rescue_replay_eval.py", "scripts/rescue_gpu_pilot.py",
                    "AgenticArxiv/benchmark/rescue_catalog.py", "AgenticArxiv/benchmark/runner.py",
                    "AgenticArxiv/benchmark/metrics.py", "AgenticArxiv/utils/llm_client.py",
                    "AgenticArxiv/agents/agent_engine.py", "AgenticArxiv/agents/prompt_templates.py")},
                "started_at": datetime.now(timezone.utc).isoformat(), "status": "running",
                "final_test_model_evaluated": False, "completed_task_ids": []}
    manifest_path = args.output / "run_manifest.json"
    def save():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    save()
    lock = None
    results = []
    started = time.monotonic()
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(args.timeout)
    try:
        from tools.bootstrap import require_all_tools
        from benchmark.runner import BenchmarkRunner
        from benchmark.report import BenchmarkReport
        from benchmark.metrics import is_strict_success
        require_all_tools("rescue frozen development screen")
        runner = BenchmarkRunner(agent_types=["regex"], repeat=1, model=args.model_id or args.policy,
                                 session_prefix="rescue_v1", offline=True,
                                 snapshot=str(args.catalog / "snapshot.json"), generation_seed=45)
        if args.policy == "model":
            lock = open(f"/tmp/agenticarxiv-rescue-{os.getuid()}-gpu{args.gpu}.lock", "w")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            gpu = subprocess.check_output(["nvidia-smi", f"--id={args.gpu}",
                "--query-gpu=index,uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
            if int(gpu.split(",")[2]) < 16000:
                raise RuntimeError("GPU has fewer than 16000MiB free; no model loaded")
            manifest["gpu_preflight"] = gpu
            manifest["physical_gpu"] = args.gpu
            model_path = args.model.resolve()
            manifest["model_config_sha256"] = sha256(model_path / "config.json")
            manifest["weights_sha256"] = {f.name: sha256(f) for f in sorted(model_path.glob("*.safetensors"))}
            if not manifest["weights_sha256"]:
                raise ValueError("Missing local safetensors checkpoint")
            import torch
            from utils.llm_client import TransformersLLMClient
            torch.set_num_threads(4)
            torch.cuda.set_per_process_memory_fraction(0.35, 0)
            class BoundedClient(TransformersLLMClient):
                def chat_completions(self, *pos, **kw):
                    kw["max_tokens"] = 384
                    return super().chat_completions(*pos, **kw)
            runner._llm_client = BoundedClient(str(model_path), device="cuda", dtype="bfloat16", seed=45)
            if args.variant == "completion_check":
                import agents.agent_engine as ae
                original = ae.get_react_prompt
                ae.get_react_prompt = lambda *a, **k: original(*a, **k) + NOTE
        else:
            runner._llm_client = ReferenceClient(tasks, finish_only=args.policy == "finish_only")
        manifest["versions"] = {x: importlib.metadata.version(x) for x in ("torch", "transformers", "pydantic")}
        save()
        with (args.output / "traces.jsonl").open("x") as traces:
            for task in tasks:
                result = runner.run_all([task])[0]
                results.append(result)
                traces.write(json.dumps({"task_id": result.task_id, "family": task["rescue_family"],
                    "trial": result.trial, "agent_type": result.agent_type, "error": result.error,
                    "raw_result": result.raw_result}, ensure_ascii=False) + "\n")
                traces.flush()
                manifest["completed_task_ids"].append(result.task_id)
                save()
                if result.error:
                    raise RuntimeError(f"Execution error on {result.task_id}: {result.error}")
        metrics = [r.metrics for r in results if r.metrics is not None]
        report = BenchmarkReport(metrics, model=args.model_id or args.policy)
        report.save_all(str(args.output / "benchmark"))
        if len(metrics) != len(tasks):
            raise RuntimeError("Incomplete metric coverage")
        strict = sum(is_strict_success(m) for m in metrics)
        if args.policy == "reference" and strict != len(tasks):
            raise RuntimeError(f"Reference pipeline failed: {strict}/{len(tasks)}")
        if args.policy == "finish_only" and strict:
            raise RuntimeError("Trivial FINISH policy passed a non-empty task")
        manifest.update(status="completed", strict_successes=strict, n_errors=0,
                        traces_sha256=sha256(args.output / "traces.jsonl"))
    except Exception as exc:
        manifest.update(status="timeout" if isinstance(exc, TimeoutError) else "failed",
                        error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        signal.alarm(0)
        manifest["wall_seconds"] = round(time.monotonic() - started, 3)
        save()
        if lock is not None:
            lock.close()
    print(json.dumps({k: manifest[k] for k in ("status", "policy", "model_id", "variant", "split", "n_tasks", "strict_successes", "wall_seconds")}))


if __name__ == "__main__":
    main()
