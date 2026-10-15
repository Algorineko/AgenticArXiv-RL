"""Control metadata regressions, not evidence of real model or GPU evaluation."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agentic_arxiv.evaluation.episode import EvaluationCancelled, evaluate_episodes
from agentic_arxiv.experiments.evaluation import (
    attempt_observability, build_evaluation_command, prepare_evaluation,
    read_attempt_log, summarize_costs,
)
from agentic_arxiv.experiments.store import FileStore, read_json


def benchmark_config():
    return {"offline": True, "snapshot": str(Path(__file__).parent / "fixtures/t4_snapshot_for_t5.json")}


def test_pre_episode_cancellation_never_runs_backend():
    client = Mock()
    with pytest.raises(EvaluationCancelled) as caught:
        evaluate_episodes(client, [{"id": "search_01", "task": "unused"}], Mock(), Mock(),
                          cancel_requested=lambda: True)
    assert caught.value.completed_episodes == 0
    client.assert_not_called()


def test_benchmark_cancellation_occurs_between_episodes(monkeypatch):
    from agentic_arxiv.evaluation.runner import BenchmarkRunner
    runner = BenchmarkRunner(repeat=2, cancel_requested=lambda: len(completed) == 1)
    completed = []
    monkeypatch.setattr(runner, "_start_generation_stream", lambda *args: None)
    runner._llm_client = Mock()
    monkeypatch.setattr("agentic_arxiv.evaluation.runner.execute_episode",
                        lambda *args, **kwargs: completed.append("complete") or {"history": []})
    monkeypatch.setattr("agentic_arxiv.evaluation.runner.extract_metrics", lambda *args: None)
    with pytest.raises(EvaluationCancelled) as caught:
        runner.run_all([{"id": "search_01", "task": "unused"}])
    assert caught.value.completed_episodes == 1
    assert len(caught.value.partial_results) == 1
    assert completed == ["complete"]


def test_job_command_is_separate_and_never_overwrites(tmp_path):
    model = Path(__file__).parent  # Existing directory, not a synthetic model.
    output, control = tmp_path / "output", tmp_path / "control"
    command = build_evaluation_command(model, "stage", {}, output, control, stage="sft")
    assert command[:3] == ["python", "-m", "agentic_arxiv.evaluation.managed_job"]
    payload = read_json(control / "evaluation-job.json")
    assert payload["kind"] == "stage"
    assert Path(payload["output_dir"]) != model
    with pytest.raises(ValueError, match="already exists"):
        build_evaluation_command(model, "stage", {}, output, control, stage="sft")
    with pytest.raises(ValueError, match="overlap"):
        build_evaluation_command(model, "benchmark", {}, model, tmp_path / "other")


def test_stage_retry_creates_independent_records(monkeypatch, tmp_path):
    source = {"path": str(Path(__file__).parent), "metadata": {"stage": "sft_full"},
              "fingerprint": {"status": "control-test-only"}}
    resolve = Mock(return_value=source)
    monkeypatch.setattr("agentic_arxiv.experiments.artifacts.resolve_model_reference", resolve)
    store = FileStore(tmp_path)
    config = {"offline": True, "snapshot": str(Path(__file__).parent / "fixtures/t4_snapshot_for_t5.json")}
    first = prepare_evaluation(store, {"kind": "registration"}, "stage", config, workspace=Path.cwd())
    second = prepare_evaluation(store, {"kind": "registration"}, "stage", config, workspace=Path.cwd())
    assert first["run"]["run_id"] != second["run"]["run_id"]
    assert first["run"]["stage"] == "evaluate"
    assert resolve.call_args.kwargs["require_verified"] is False
    assert len(list(tmp_path.glob("runs/*/evaluations/*/evaluation.json"))) == 2


def test_m0_path_reference_resolves_only_benchmark(monkeypatch, tmp_path):
    existing = Path(__file__).parent
    monkeypatch.setattr("agentic_arxiv.llm.artifacts._require_full_weights", lambda _: None)
    monkeypatch.setattr("agentic_arxiv.provenance.model_input_fingerprint",
                        lambda _: {"weights_status": "known", "scope": "control-metadata-test-only"})
    check_root = tmp_path / "check-only"
    checked = prepare_evaluation(FileStore(check_root), {"kind": "path", "path": str(existing)},
                                 "benchmark", benchmark_config(), workspace=Path.cwd(), persist=False)
    assert checked["run"] is None and not check_root.exists()
    prepared = prepare_evaluation(FileStore(tmp_path), {"kind": "path", "path": str(existing)},
                                  "benchmark", benchmark_config(), workspace=Path.cwd())
    assert prepared["run"]["group"] == "M0"
    assert prepared["run"]["stage"] == "evaluate"
    assert prepared["context"]["evaluation"]["source"]["metadata"]["stage"] == "base"
    assert prepared["context"]["evaluation"]["config"]["offline"] is True
    assert prepared["context"]["evaluation"]["input_files"]["snapshot"]
    assert str(Path(benchmark_config()["snapshot"]).resolve()) in prepared["inputs"]
    with pytest.raises(ValueError, match="final training artifact"):
        prepare_evaluation(FileStore(tmp_path), {"kind": "path", "path": str(existing)},
                           "stage", {}, workspace=Path.cwd())


def test_missing_base_weights_are_a_management_input_error(tmp_path):
    model = tmp_path / "base"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    store = FileStore(tmp_path / "managed")
    with pytest.raises(ValueError, match="权重或分片索引缺失"):
        prepare_evaluation(store, str(model), "benchmark", benchmark_config(), persist=False)
    assert not store.root.exists()


def test_m0_and_independent_evaluate_bind_the_same_explicit_offline_input(monkeypatch, tmp_path):
    import json
    from agentic_arxiv.experiments.service import ExperimentService

    model = str(Path(__file__).parent)
    config = benchmark_config()
    config_path = tmp_path / "evaluation.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr("agentic_arxiv.llm.artifacts._require_full_weights", lambda _: None)
    monkeypatch.setattr("agentic_arxiv.provenance.model_input_fingerprint",
                        lambda _: {"weights_status": "known", "scope": "control-metadata-test-only"})
    service = ExperimentService({"workspace": str(tmp_path), "run_root": str(tmp_path / "managed")})
    launch = Mock(side_effect=lambda prepared, **_: prepared)
    monkeypatch.setattr(service, "_launch_evaluation", launch)
    monkeypatch.setattr(service.runtime, "call", Mock(side_effect=AssertionError("planning only")))
    m0 = service._prepare_m0({"recipe": {"evaluation": {"offline": True}}},
        {"audit": {"inputs": {"model": {"path": model}, "snapshot": {"path": config["snapshot"]}}}})
    independent = service.evaluate(SimpleNamespace(evaluation_config=str(config_path), reference=model,
                                                  kind="benchmark", diagnostic=False, gpus="GPU-a"))
    for prepared in (m0, independent):
        assert prepared["run"]["group"] == "M0"
        assert prepared["context"]["evaluation"]["config"] == config
        assert prepared["context"]["evaluation"]["input_files"]["snapshot"]
    assert m0["run"]["run_id"] != independent["run"]["run_id"]
    launch.assert_called_once()
    service.runtime.call.assert_not_called()


def test_diagnostic_is_explicit_and_cannot_qualify(monkeypatch, tmp_path):
    source = {"path": str(Path(__file__).parent), "metadata": {"stage": "sft_full"},
              "fingerprint": {"status": "control-test-only"}}
    resolver = Mock(return_value=source)
    monkeypatch.setattr("agentic_arxiv.experiments.artifacts.resolve_model_reference", resolver)
    prepared = prepare_evaluation(FileStore(tmp_path), {"kind": "registration"},
                                  "benchmark", benchmark_config(), diagnostic=True)
    assert resolver.call_args.kwargs["require_verified"] is False
    context = prepared["context"]["evaluation"]
    assert context["diagnostic"] is True and context["qualification_eligible"] is False


@pytest.mark.parametrize("config", [{}, {"offline": False}, {"offline": True},
                                    {"offline": True, "snapshot": None},
                                    {"snapshot": "unused.json"}])
@pytest.mark.parametrize("persist", [False, True])
def test_benchmark_requires_explicit_offline_inputs_before_model_resolution(monkeypatch, tmp_path, config, persist):
    resolver = Mock(side_effect=AssertionError("invalid config must not resolve models"))
    monkeypatch.setattr("agentic_arxiv.experiments.artifacts.resolve_model_reference", resolver)
    store = FileStore(tmp_path / "managed")
    with pytest.raises(ValueError, match="managed benchmark requires"):
        prepare_evaluation(store, {"kind": "registration"}, "benchmark", config, persist=persist)
    resolver.assert_not_called()
    assert not store.root.exists()


@pytest.mark.parametrize("contents", ["invalid JSON", "[]", "{}", '{"search": []}',
                                      '{"search": {}}', '{"search": {"key": {"args": {}}}}'])
def test_benchmark_rejects_unusable_snapshot_without_creating_records(monkeypatch, tmp_path, contents):
    snapshot = tmp_path / "broken.json"
    snapshot.write_text(contents, encoding="utf-8")
    source = {"path": str(Path(__file__).parent), "metadata": {"stage": "sft_full"}}
    monkeypatch.setattr("agentic_arxiv.experiments.artifacts.resolve_model_reference", Mock(return_value=source))
    store = FileStore(tmp_path / "managed")
    with pytest.raises(ValueError):
        prepare_evaluation(store, {"kind": "registration"}, "benchmark",
                           {"offline": True, "snapshot": str(snapshot)})
    assert not store.root.exists()
    assert snapshot.read_text(encoding="utf-8") == contents


@pytest.mark.parametrize("is_directory", [False, True])
def test_benchmark_rejects_missing_snapshot_or_directory(monkeypatch, tmp_path, is_directory):
    snapshot = tmp_path / "snapshot"
    if is_directory:
        snapshot.mkdir()
    monkeypatch.setattr("agentic_arxiv.experiments.artifacts.resolve_model_reference", Mock(return_value={
        "path": str(Path(__file__).parent), "metadata": {"stage": "sft_full"}}))
    store = FileStore(tmp_path / "managed")
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare_evaluation(store, {"kind": "registration"}, "benchmark",
                           {"offline": True, "snapshot": str(snapshot)})
    assert not store.root.exists()


def test_cancelled_stage_report_is_not_a_pass(monkeypatch):
    from agentic_arxiv.training.stage_verifier import StageVerifier
    monkeypatch.setattr("agentic_arxiv.training.stage_verifier._evaluation_protocol", lambda *_: {})
    report = StageVerifier()._evaluation_failure("sft", "unused", EvaluationCancelled(2),
                                                ["search_01"], 6, None)
    assert report.passed is False
    assert report.coverage["execution"] == "interrupted"
    assert report.coverage["completed_episodes"] == 2


def test_managed_failure_is_not_pass_and_does_not_train(monkeypatch, tmp_path):
    from agentic_arxiv.evaluation.managed_job import execute_job
    monkeypatch.setattr("agentic_arxiv.provenance.model_input_fingerprint", lambda _: {"test": "metadata"})
    run = Mock(return_value={"execution": "failed", "error_count": 1})
    monkeypatch.setattr("agentic_arxiv.evaluation.run_benchmark.execute", run)
    job = {"kind": "benchmark", "output_dir": str(tmp_path / "result"),
           "model_path": "/not-loaded/control-test", "config": {}, "control_dir": str(tmp_path),
           "source": {"fingerprint": {"test": "metadata"}}}
    result = execute_job(job)
    assert result["execution"] == "failed" and result["passed"] is False
    assert result["training_executed"] is False
    assert result["training_resume_supported"] is False
    with pytest.raises(ValueError, match="already exists"):
        execute_job(job)


def test_managed_interruption_never_returns_partial_pass(monkeypatch, tmp_path):
    from agentic_arxiv.evaluation.managed_job import execute_job
    monkeypatch.setattr("agentic_arxiv.provenance.model_input_fingerprint", lambda _: {})
    monkeypatch.setattr("agentic_arxiv.evaluation.run_benchmark.execute",
                        Mock(side_effect=EvaluationCancelled(2)))
    result = execute_job({"kind": "benchmark", "output_dir": str(tmp_path / "result"),
                          "model_path": "unused", "config": {}, "control_dir": str(tmp_path),
                          "source": {"fingerprint": {}}})
    assert result["execution"] == "interrupted"
    assert result["passed"] is False and result["completed_episodes"] == 2


def test_failed_retry_cost_is_physical_not_logical_progress():
    rows = [{"attempt_id": "one", "gpu_uuids": ["GPU-one"], "elapsed_seconds": 3600,
             "training_elapsed_seconds": 3500, "stage_validation_elapsed_seconds": 100,
             "source_checkpoint_step": 0, "global_step": 8, "checkpoint_step": 5},
            {"attempt_id": "two", "gpu_uuids": ["GPU-one"], "elapsed_seconds": 3600,
             "training_elapsed_seconds": 3500, "stage_validation_elapsed_seconds": 100,
             "source_checkpoint_step": 5, "global_step": 10, "checkpoint_step": 10}]
    result = summarize_costs(rows, upstream_costs={"dpo_gpu_hours": 4, "preference_generation_gpu_hours": None})
    assert result["allocated_attempt_gpu_hours"] == 2
    assert result["training_gpu_hours"] == pytest.approx(7000 / 3600)
    assert result["stage_validation_gpu_hours"] == pytest.approx(200 / 3600)
    assert result["logical_committed_step"] == 10
    assert result["physical_executed_updates"] == 13
    assert result["recomputed_updates"] == 3
    assert result["attempts"][0]["uncommitted_steps"] == 3
    assert result["upstream_costs"]["dpo_gpu_hours"] == 4
    assert result["strict_equal_budget"] is None


def test_missing_cost_is_unknown_not_zero():
    result = summarize_costs([{"attempt_id": "unknown"}])
    assert result["training_gpu_hours"] is None
    assert result["attempts"][0]["generated_tokens"] is None
    assert result["logical_committed_step"] is None


@pytest.mark.parametrize("endpoint", ["started_at", "finished_at"])
@pytest.mark.parametrize("invalid", [None, "", "invalid", 0, "0001-01-01T00:00:00Z",
                                     "0001-01-01T00:00:00.000000000Z", "2026-10-15T02:00:00"])
def test_invalid_container_time_is_unknown_for_attempts_and_evaluations(endpoint, invalid):
    record = {"gpu_uuids": ["GPU-a"], "started_at": "2026-10-15T02:00:00Z",
              "finished_at": "2026-10-15T03:00:00Z", endpoint: invalid}
    result = summarize_costs([record], [record])
    assert result["allocated_attempt_gpu_hours"] is None
    assert result["evaluation_gpu_hours"] is None


def test_container_times_preserve_timezone_and_fractional_seconds():
    record = {"gpu_uuids": ["GPU-a", "GPU-b"], "started_at": "2026-10-15T02:00:00.123456789Z",
              "finished_at": "2026-10-15T11:00:00.123456789+08:00"}
    result = summarize_costs([record], [record])
    assert result["allocated_attempt_gpu_hours"] == 2
    assert result["evaluation_gpu_hours"] == 2


@pytest.mark.parametrize("event_directory", ["logs", "logs/rank-0", "output", "output/logs"])
def test_rank_logs_and_tensorboard_inventory_are_read_only(tmp_path, event_directory):
    logs = tmp_path / "logs"
    logs.mkdir()
    rank = logs / "rank-2.log"
    rank.write_text("old\nnew\n", encoding="utf-8")
    destination = tmp_path / event_directory
    destination.mkdir(parents=True, exist_ok=True)
    event = destination / "events.out.tfevents.existing-control-name"
    event.touch()
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert read_attempt_log(tmp_path, rank=2, tail=1) == "new\n"
    observed = attempt_observability(tmp_path)
    assert observed["tensorboard_events"] == [str(event)]
    assert observed["tensorboard_scalars"] is None
    assert observed["tensorboard_status"] == "not_requested"
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert rank.read_text(encoding="utf-8") == "old\nnew\n"
    with pytest.raises(ValueError):
        read_attempt_log(tmp_path, rank=-1)


def test_tensorboard_reads_unique_files_from_both_log_locations(monkeypatch, tmp_path):
    events = []
    for location in ("logs", "output/logs"):
        directory = tmp_path / location
        directory.mkdir(parents=True)
        # Identical basenames in distinct directories are distinct event files.
        event = directory / "events.out.tfevents.control"
        event.touch()
        events.append(str(event))
        (directory / "events.out.tfevents.not-a-file").mkdir()
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    glob = Path.glob

    def repeated_entries(path, pattern):
        entries = list(glob(path, pattern))
        return iter(entries + entries if pattern == "**/events.out.tfevents.*" else entries)

    monkeypatch.setattr(Path, "glob", repeated_entries)
    accumulator = Mock()
    accumulator.Tags.return_value = {"scalars": ["control/count"]}
    accumulator.Scalars.return_value = [SimpleNamespace(step=2, value=3.0, wall_time=4.0)]
    factory = Mock(return_value=accumulator)
    monkeypatch.setitem(sys.modules, "tensorboard.backend.event_processing.event_accumulator",
                        SimpleNamespace(EventAccumulator=factory))

    result = attempt_observability(tmp_path, read_tensorboard=True)

    assert result["tensorboard_events"] == sorted(events)
    assert result["tensorboard_status"] == "read"
    assert result["tensorboard_scalars"] == {
        event: {"control/count": [{"step": 2, "value": 3.0, "wall_time": 4.0}]} for event in events
    }
    assert [call.args[0] for call in factory.call_args_list] == sorted(events)
    assert all(call.kwargs == {"size_guidance": {"scalars": 0}} for call in factory.call_args_list)
    assert accumulator.Reload.call_count == len(events)
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("read_tensorboard", [False, True])
def test_tensorboard_inventory_without_logging_extra(monkeypatch, tmp_path, read_tensorboard):
    event = tmp_path / "logs/events.out.tfevents.control"
    event.parent.mkdir()
    event.touch()
    monkeypatch.setitem(sys.modules, "tensorboard.backend.event_processing.event_accumulator", None)
    result = attempt_observability(tmp_path, read_tensorboard=read_tensorboard)
    assert result["tensorboard_events"] == [str(event)]
    assert result["tensorboard_scalars"] is None
    assert result["tensorboard_status"] == ("logging_extra_not_installed" if read_tensorboard else "not_requested")


def test_tensorboard_inventory_does_not_create_missing_attempt(tmp_path):
    attempt = tmp_path / "missing-attempt"
    assert attempt_observability(attempt) == {
        "logs": [], "tensorboard_events": [], "tensorboard_scalars": None,
        "tensorboard_status": "not_requested",
    }
    assert not list(tmp_path.iterdir())
