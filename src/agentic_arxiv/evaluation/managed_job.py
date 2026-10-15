"""Container-side independent evaluation execution; never invokes a trainer."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agentic_arxiv.experiments.store import atomic_json, now, read_json


def execute_job(job):
    from agentic_arxiv.config import initialize_operation
    from agentic_arxiv.training.control import read_stop_request
    from agentic_arxiv.evaluation.episode import EvaluationCancelled

    initialize_operation(job.get("workspace"), load_env=False)
    output = Path(job["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "evaluation-result.json"
    if result_path.exists():
        raise ValueError("evaluation result already exists; retry in a new record")
    started = time.monotonic()
    result = {"schema_version": 1, "kind": job["kind"], "model_path": job["model_path"],
              "source": job["source"], "configuration": job["config"],
              "run_id": job.get("run_id"), "attempt_id": job.get("attempt_id"),
              "started_at": now(), "execution": "running", "passed": False,
              "diagnostic": bool(job.get("diagnostic")),
              "qualification_eligible": not bool(job.get("diagnostic")),
              "training_executed": False, "training_resume_supported": False}
    # Persist provenance before model/backend calls, including failed executions.
    atomic_json(result_path, result)

    def cancelled():
        if not job.get("run_id") or not job.get("attempt_id"):
            return False
        return read_stop_request(job["control_dir"], job["run_id"], job["attempt_id"]) is not None

    try:
        if cancelled():
            raise EvaluationCancelled()
        from agentic_arxiv.provenance import file_fingerprint, model_input_fingerprint
        source = job["source"]
        if source.get("fingerprint") != model_input_fingerprint(job["model_path"]):
            raise ValueError("evaluation model identity changed after planning")
        base = source.get("base_model")
        if base and base["fingerprint"] != model_input_fingerprint(base["path"]):
            raise ValueError("evaluation base identity changed after planning")
        for identity in source.get("evaluation_inputs", {}).values():
            if file_fingerprint(identity["path"]) != identity:
                raise ValueError("evaluation input changed after planning")
        if job["kind"] == "benchmark":
            from agentic_arxiv.evaluation.run_benchmark import execute
            config = dict(repeat=3, tasks=None, task_ids=None, backend="transformers",
                          local_device="auto", local_dtype="auto", seed=42, prefix=None,
                          no_thinking=False, offline=False, snapshot=None, save_traces=None,
                          split=None, task_set="default")
            config.update(job["config"])
            if config["backend"] != "transformers":
                raise ValueError("managed benchmark requires the explicitly selected local model")
            config.update(model=job["model_path"], output=str(output))
            config["save_traces"] = str(output / "traces.jsonl") if config["save_traces"] else None
            report = execute(argparse.Namespace(**config), cancel_requested=cancelled,
                             model_artifact=source.get("metadata", {}).get("stage") != "base")
            result.update(report, passed=report["execution"] == "complete")
        elif job["kind"] == "stage":
            from agentic_arxiv.training.stage_verifier import StageVerifier
            from dataclasses import asdict
            config = job["config"]
            verifier = StageVerifier(cancel_requested=cancelled, **{
                key: config[key] for key in ("sft_min_parse_rate", "dpo_min_reward", "grpo_min_reward") if key in config})
            method = getattr(verifier, "verify_" + job["stage"])
            kwargs = {key: config[key] for key in ("max_iterations",) if key in config}
            if job["stage"] == "sft" and "num_samples" in config:
                kwargs["num_samples"] = config["num_samples"]
            if config.get("snapshot"):
                from agentic_arxiv.runtime.env import MockArxivEnv
                kwargs["env"] = MockArxivEnv(snapshot_path=config["snapshot"], mode="replay")
            report = method(job["model_path"], **kwargs)
            verifier.save_report(report, output)
            interrupted = report.coverage.get("execution") == "interrupted"
            result.update(execution="interrupted" if interrupted else ("complete" if report.passed else "failed"),
                          passed=report.passed and not interrupted, report=asdict(report))
        else:
            raise ValueError("unknown evaluation kind")
    except EvaluationCancelled as error:
        result.update(execution="interrupted", passed=False, completed_episodes=error.completed_episodes,
                      reason=str(error))
    except (Exception, SystemExit) as error:
        result.update(execution="failed", passed=False, reason=str(error))
    finally:
        result.update(finished_at=now(), elapsed_seconds=time.monotonic() - started)
        atomic_json(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    result = execute_job(read_json(args.job))
    return 0 if result["execution"] == "complete" else (130 if result["execution"] == "interrupted" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
