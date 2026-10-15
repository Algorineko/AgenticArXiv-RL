"""Lightweight planning and read-only accounting for independent evaluation jobs."""
from __future__ import annotations

import math
from pathlib import Path

from .store import atomic_json, parse_record_time


def prepare_evaluation(store, reference, kind, config, workspace=None, *, diagnostic=False, persist=True):
    """Resolve one explicit model and create an independent evaluation run.

    Stage retry intentionally relaxes only the prior verification requirement,
    not the final-model identity, stage, protocol or source checks.
    """
    from .artifacts import _validate_artifact_input, resolve_model_reference
    from .config import EVALUATION_TYPES, _typed, read_json as read_config
    from agentic_arxiv.provenance import file_fingerprint, model_input_fingerprint
    from agentic_arxiv.llm.artifacts import _require_full_weights

    config = dict(config or {})
    configured_kind = config.pop("kind", kind)
    if configured_kind != kind:
        raise ValueError("evaluation kind conflicts with fixed configuration")
    config_path = config.pop("config", None)
    if config_path:
        if config:
            raise ValueError("fixed evaluation config cannot be mixed with inline options")
        location = Path(config_path)
        if not location.is_absolute():
            location = Path(workspace or Path.cwd()) / location
        config = read_config(location)
    allowed = set(EVALUATION_TYPES) - {"config", "kind"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("unknown evaluation configuration fields")
    for key, value in config.items():
        _typed(value, EVALUATION_TYPES[key], f"evaluation.{key}",
               nullable=key in {"tasks", "snapshot", "split", "prefix", "save_traces"})
    if kind not in {"benchmark", "stage"}:
        raise ValueError("unknown evaluation kind")
    stage_fields = {"num_samples", "max_iterations", "sft_min_parse_rate", "dpo_min_reward", "grpo_min_reward"}
    if kind == "benchmark" and set(config) & stage_fields:
        raise ValueError("stage verification options do not apply to benchmark")
    if kind == "stage" and set(config) - stage_fields - {"snapshot", "offline"}:
        raise ValueError("benchmark options do not apply to stage verification")
    for key in {"repeat", "num_samples", "max_iterations"} & set(config):
        if config[key] < 1:
            raise ValueError(f"evaluation.{key} must be positive")
    if kind == "benchmark":
        if config.get("offline") is not True:
            raise ValueError("managed benchmark requires explicit offline=true; containers have no network")
        if not config.get("snapshot"):
            raise ValueError("managed benchmark requires an explicit existing snapshot")
    if config.get("snapshot") and not config.get("offline"):
        raise ValueError("snapshot requires explicit offline evaluation")
    if kind == "stage" and config.get("offline") is False:
        raise ValueError("stage verification remains offline")
    if config.get("backend", "transformers") != "transformers":
        raise ValueError("managed evaluation requires the selected local model, not an API replacement")
    if config.get("local_dtype", "auto") not in {"auto", "float16", "bfloat16", "float32"}:
        raise ValueError("unknown evaluation local_dtype")
    if config.get("task_set", "default") not in {"default", "expanded"}:
        raise ValueError("unknown evaluation task_set")
    if "task_ids" in config and (not config["task_ids"] or any(type(item) is not str or not item for item in config["task_ids"])):
        raise ValueError("task_ids must be a nonempty list of explicit task names")
    path = None
    path_reference = reference if isinstance(reference, str) else (
        reference.get("path") if isinstance(reference, dict) and set(reference) == {"kind", "path"}
        and reference.get("kind") == "path" else None)
    if path_reference is not None:
        candidate = Path(path_reference).expanduser()
        if not candidate.is_absolute():
            candidate = Path(workspace or Path.cwd()) / candidate
        if candidate.is_dir():
            path = candidate.resolve(strict=True)
    if path is not None and not (path / "training_manifest.json").exists():
        if kind != "benchmark" or (path / "adapter_config.json").exists():
            raise ValueError("stage verification requires a declared final training artifact")
        _validate_artifact_input(_require_full_weights, path)
        identity = model_input_fingerprint(path)
        if identity["weights_status"] != "known":
            raise ValueError("Base model has no verifiable local weights")
        source = {"path": str(path), "metadata": {"stage": "base"}, "fingerprint": identity,
                  "reference": reference, "verification": {"status": "not_applicable"}}
    else:
        source = resolve_model_reference(reference, store_root=store.root,
            require_verified=kind == "benchmark" and not diagnostic, workspace=workspace)
    stage = source["metadata"]["stage"].split("_", 1)[0]
    if kind == "stage" and stage != "sft" and "num_samples" in config:
        raise ValueError("num_samples only applies to SFT verification")
    paths = [source["path"]]
    if source.get("base_model"):
        paths.append(source["base_model"]["path"])
    fixed_files = {}
    for key in ("snapshot", "split"):
        value = config.get(key)
        if not value:
            continue
        # Named splits still use the existing CLI's workspace contract.
        name = value.rsplit(":", 1)[0] if key == "split" and ":" in value else value
        if key == "split" and ":" not in value:
            from agentic_arxiv.tasks.splits import DEFAULT_SPLIT_PATH
            name = str(DEFAULT_SPLIT_PATH)
        candidate = Path(name).expanduser()
        candidate = candidate if candidate.is_absolute() else Path(workspace or Path.cwd()) / candidate
        if key in {"snapshot", "split"}:
            candidate = candidate.resolve(strict=True)
            if key == "snapshot" and kind == "benchmark":
                if not candidate.is_file():
                    raise ValueError("evaluation snapshot must be an existing JSON file")
                snapshot = read_config(candidate)
                # Replay consumes tool -> cache key -> {result: ...}. Validate
                # that shape without importing tools or executing episodes.
                if (not isinstance(snapshot, dict) or not snapshot
                        or not all(isinstance(entries, dict) for entries in snapshot.values())
                        or not any(snapshot.values())
                        or any(not isinstance(entry, dict) or "result" not in entry
                               for entries in snapshot.values() for entry in entries.values())):
                    raise ValueError("evaluation snapshot must contain replay tool records with result fields")
            paths.append(str(candidate))
            fixed_files[key] = file_fingerprint(candidate)
            config[key] = str(candidate) + (":" + value.rsplit(":", 1)[-1] if key == "split" else "")
    if config.get("offline") and not config.get("snapshot"):
        candidate = (Path(workspace or Path.cwd()) / "data/mock_arxiv_snapshot.json").resolve(strict=True)
        paths.append(str(candidate))
        fixed_files["snapshot"] = file_fingerprint(candidate)
        config["snapshot"] = str(candidate)
    if kind == "stage" and not config.get("snapshot"):
        candidate = (Path(workspace or Path.cwd()) / "data/mock_arxiv_snapshot.json").resolve(strict=True)
        paths.append(str(candidate))
        fixed_files["snapshot"] = file_fingerprint(candidate)
        config.update(snapshot=str(candidate), offline=True)
    source["evaluation_inputs"] = fixed_files
    evaluation = dict(kind=kind, model_path=source["path"], config=config,
                      stage=stage if kind == "stage" else None, source=source,
                      workspace=str(workspace) if workspace else None, input_files=fixed_files,
                      diagnostic=diagnostic, qualification_eligible=not diagnostic)
    if not persist:
        return {"run": None, "command": None, "inputs": paths,
                "context": {"evaluation": evaluation}}
    run = store.create_run({"group": "M0" if stage == "base" else None,
                            "stage": "evaluate", "evaluation": evaluation},
                           {"evaluation": evaluation}, {"model": source, "files": fixed_files})
    record = store.create_evaluation(run["run_id"], dict(evaluation, execution="not_started"))
    evaluation["evaluation_id"] = record["evaluation_id"]
    return {"run": run, "command": None, "inputs": paths,
            "context": {"evaluation": evaluation}}


def build_evaluation_command(model_path, kind, config, output_dir, control_dir, *,
                             stage=None, source=None, workspace=None,
                             run_id=None, attempt_id=None, diagnostic=False):
    """Write an attempt-local job, never a report into the model directory."""
    if kind not in {"benchmark", "stage"}:
        raise ValueError("evaluation kind must be benchmark or stage")
    if kind == "stage" and stage not in {"sft", "dpo", "grpo"}:
        raise ValueError("stage verification requires an explicit final-model stage")
    output = Path(output_dir).resolve()
    model = Path(model_path).resolve(strict=True)
    if output == model or model in output.parents or output in model.parents:
        raise ValueError("evaluation output must not overlap its read-only model")
    job_path = Path(control_dir) / "evaluation-job.json"
    if job_path.exists():
        raise ValueError("evaluation job already exists; create a new attempt")
    payload = dict(schema_version=1, kind=kind, model_path=str(model), stage=stage,
                   config=dict(config), output_dir=str(output), control_dir=str(control_dir),
                   source=source or {}, workspace=str(workspace) if workspace else None,
                   run_id=run_id, attempt_id=attempt_id, diagnostic=diagnostic)
    atomic_json(job_path, payload)
    return ["python", "-m", "agentic_arxiv.evaluation.managed_job", "--job", str(job_path)]


def _number(value):
    return value if type(value) in {int, float} and math.isfinite(value) and value >= 0 else None


def _duration(record):
    explicit = _number(record.get("elapsed_seconds"))
    if explicit is not None:
        return explicit
    start = parse_record_time(record.get("started_at"))
    end = parse_record_time(record.get("finished_at"))
    if start is None or end is None:
        return None
    return _number((end - start).total_seconds())


def _total(rows, key):
    values = [row.get(key) for row in rows]
    return sum(values) if values and all(value is not None for value in values) else None


def summarize_costs(attempts, evaluations=(), upstream_costs=None):
    """Physical compute is additive; logical progress is not summed across retries.

    GPU-hours are allocated-device count multiplied by elapsed wall hours, not
    utilization-adjusted measurements. Missing endpoints/metrics stay unknown.
    """
    rows = []
    for attempt in attempts:
        metrics = attempt.get("costs") or {}
        elapsed = _duration(metrics) if metrics else None
        if elapsed is None:
            elapsed = _duration(attempt)
        training_elapsed = _number(attempt.get("training_elapsed_seconds", metrics.get("training_elapsed_seconds")))
        validation_elapsed = _number(attempt.get("stage_validation_elapsed_seconds", metrics.get("stage_validation_elapsed_seconds")))
        devices = attempt.get("gpu_uuids")
        count = len(devices) if isinstance(devices, list) else None
        start = _number(attempt.get("source_checkpoint_step", metrics.get("start_step")))
        actual = _number(metrics.get("actual_step", attempt.get("global_step")))
        committed = _number(metrics.get("committed_step", attempt.get("checkpoint_step")))
        rows.append({"attempt_id": attempt.get("attempt_id"), "elapsed_seconds": elapsed,
                     "gpu_count": count,
                     "gpu_hours": training_elapsed * count / 3600 if training_elapsed is not None and count is not None else None,
                     "allocated_attempt_gpu_hours": elapsed * count / 3600 if elapsed is not None and count is not None else None,
                     "training_elapsed_seconds": training_elapsed,
                     "stage_validation_elapsed_seconds": validation_elapsed,
                     "stage_validation_gpu_hours": validation_elapsed * count / 3600 if validation_elapsed is not None and count is not None else None,
                     "start_step": start, "actual_step": actual, "committed_step": committed,
                     "uncommitted_steps": max(0, actual - committed) if actual is not None and committed is not None else None,
                     "executed_updates": actual - start if actual is not None and start is not None and actual >= start else None,
                     "generated_tokens": _number(metrics.get("generated_tokens", attempt.get("generated_tokens"))),
                     "peak_memory_bytes": _number(metrics.get("peak_memory_bytes", attempt.get("peak_memory_bytes"))),
                     "stop_reason": attempt.get("stop_reason", attempt.get("reason")),
                     "stage_validation_cost": attempt.get("stage_validation_cost")})
    evaluation_rows = []
    for item in evaluations:
        elapsed = _duration(item)
        devices = item.get("gpu_uuids")
        evaluation_rows.append(dict(item, elapsed_seconds=elapsed,
            gpu_hours=elapsed * len(devices) / 3600 if elapsed is not None and isinstance(devices, list) else None))
    committed_steps = [row["committed_step"] for row in rows if row["committed_step"] is not None]
    actual = _total(rows, "executed_updates")
    logical = max(committed_steps) if committed_steps else None
    return {"attempts": rows, "evaluations": evaluation_rows,
            "training_gpu_hours": _total(rows, "gpu_hours"),
            "allocated_attempt_gpu_hours": _total(rows, "allocated_attempt_gpu_hours"),
            "stage_validation_gpu_hours": _total(rows, "stage_validation_gpu_hours"),
            "evaluation_gpu_hours": _total(evaluation_rows, "gpu_hours"),
            "logical_committed_step": logical, "physical_executed_updates": actual,
            "recomputed_updates": (max(0, actual - max(row["actual_step"] for row in rows))
                                   if actual is not None and rows and all(row["actual_step"] is not None for row in rows) else None),
            "upstream_costs": upstream_costs or {"preference_generation": None, "dpo": None},
            "strict_equal_budget": None,
            "gpu_hours_basis": "allocated GPU count * elapsed wall seconds / 3600; not utilization-adjusted",
            "unknown_policy": "missing is unknown, never zero; reused upstream work is retained"}


def attempt_observability(attempt_dir, *, read_tensorboard=False):
    """Read existing files only; optional TensorBoard support needs [logging]."""
    directory = Path(attempt_dir)
    logs = sorted(str(path) for path in (directory / "logs").glob("**/*") if path.is_file())
    # SFT/DPO use managed logs; GRPO and older attempts keep output/logs.
    events = sorted({str(path) for root in (directory / "logs", directory / "output")
                     for path in root.glob("**/events.out.tfevents.*") if path.is_file()})
    result = {"logs": logs, "tensorboard_events": events, "tensorboard_scalars": None,
              "tensorboard_status": "not_requested"}
    if read_tensorboard:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError:
            result["tensorboard_status"] = "logging_extra_not_installed"
        else:
            scalars = {}
            for filename in events:
                accumulator = EventAccumulator(filename, size_guidance={"scalars": 0})
                accumulator.Reload()
                scalars[filename] = {
                    tag: [{"step": event.step, "value": event.value, "wall_time": event.wall_time}
                          for event in accumulator.Scalars(tag)]
                    for tag in accumulator.Tags().get("scalars", [])}
            result.update(tensorboard_status="read", tensorboard_scalars=scalars)
    return result


def read_attempt_log(attempt_dir, *, rank=None, tail=200):
    """Read an exact attempt/rank log without starting any process."""
    if rank is not None and (type(rank) is not int or rank < 0):
        raise ValueError("rank must be a nonnegative integer")
    if type(tail) is not int or tail < 1:
        raise ValueError("tail must be positive")
    directory = Path(attempt_dir) / "logs"
    names = ["main.log", "stdout.log"] if rank is None else [f"rank-{rank}.log", f"rank{rank}.log"]
    for name in names:
        path = directory / name
        if path.is_file():
            from collections import deque
            with path.open(encoding="utf-8", errors="replace") as stream:
                return "".join(deque(stream, maxlen=tail))
    raise FileNotFoundError("requested attempt/rank log is unavailable")
