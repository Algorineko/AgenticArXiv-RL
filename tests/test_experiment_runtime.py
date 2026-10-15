"""只使用控制元数据与命令替身，不创建模型、研究数据或真实容器。"""
import concurrent.futures
import json
import os
import subprocess

import pytest

from agentic_arxiv.experiments.docker import DockerRuntime, IdentityMismatch, RuntimeUnavailable
from agentic_arxiv.experiments.paths import input_path, mounts, validate_layout
from agentic_arxiv.experiments.service import ExperimentService
from agentic_arxiv.experiments.store import FileStore, atomic_json


@pytest.fixture
def setup(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "control-metadata.json").write_text("{}")
    profile = dict(workspace=str(tmp_path), run_root=str(tmp_path / "runs"), input_roots=[str(inputs)],
                   image="local/train@sha256:" + "a" * 64, uid=1000, gid=1000, gpus=["0", "1"], shm_size="1g")
    service = ExperimentService(profile)
    run = service.store.create_run({"stage": "sft", "total_budget": 10}, {}, {})
    attempt = service.store.create_attempt(run["run_id"], {"gpu_uuids": ["GPU-a"]})
    attempt = service.store.update_attempt(run["run_id"], attempt["attempt_id"], {"container_name": "exact-container", "container_id": "container-id"})
    return service, run, attempt, inputs


def container(service, attempt, *, running=True, exit_code=0):
    return {"Id": "container-id", "Name": "/exact-container", "Config": {
        "User": "1000:1000", "Image": service.profile["image"], "Env": ["CUDA_VISIBLE_DEVICES=GPU-a"], "Labels": {
            "org.agentic-arxiv.managed": "1", "org.agentic-arxiv.run": attempt["run_id"],
            "org.agentic-arxiv.attempt": attempt["attempt_id"], "org.agentic-arxiv.owner": "1000:1000"}},
        "HostConfig": {"DeviceRequests": [{"DeviceIDs": ["GPU-a"]}]},
        "State": {"Running": running, "Status": "running" if running else "exited", "ExitCode": exit_code}}


def test_concurrent_run_and_attempt_records_are_unique(tmp_path):
    store = FileStore(tmp_path / "store")
    def create(_):
        run = store.create_run({"group": "M1"}, {"seed": 1}, {"identity": "control"})
        attempt = store.create_attempt(run["run_id"], {"gpu_uuids": []})
        return run, attempt
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(create, range(24)))
    assert len({run["run_id"] for run, _ in records}) == 24
    assert len({attempt["attempt_id"] for _, attempt in records}) == 24
    for run, attempt in records:
        store.update_attempt(run["run_id"], attempt["attempt_id"], {"execution": "exited"})
        fresh = store.create_attempt(run["run_id"], {"gpu_uuids": []})
        assert fresh["attempt_id"] != attempt["attempt_id"]
        assert store.read_attempt(run["run_id"], attempt["attempt_id"])["execution"] == "exited"


def test_registration_and_evaluation_are_independent(setup):
    service, run, _, _ = setup
    registered = service.store.create_registration({"path": "unchanged"}, artifact_id="formal-sft")
    with pytest.raises(FileExistsError):
        service.store.create_registration({}, artifact_id="formal-sft")
    first = service.store.create_evaluation(run["run_id"], {"kind": "stage"})
    second = service.store.create_evaluation(run["run_id"], {"kind": "stage"})
    assert first["evaluation_id"] != second["evaluation_id"]
    assert registered["path"] == "unchanged"


def test_immutable_attempt_and_active_duplicate_rejected(setup):
    service, run, attempt, _ = setup
    with pytest.raises(ValueError, match="已有活动"):
        service.store.create_attempt(run["run_id"], {})
    with pytest.raises(ValueError, match="身份不可修改"):
        service.store.update_attempt(run["run_id"], attempt["attempt_id"], {"gpu_uuids": ["GPU-b"]})


def test_paths_reject_symlinks_escape_and_overlap(setup, tmp_path):
    service, _, _, inputs = setup
    file = inputs / "control-metadata.json"
    alias = inputs / "alias.json"
    alias.symlink_to(file)
    with pytest.raises(ValueError, match="符号链接"):
        input_path(alias, [inputs])
    with pytest.raises(ValueError, match="允许根"):
        input_path(inputs, [inputs])
    with pytest.raises(ValueError, match="重叠"):
        validate_layout(inputs / "outputs", [file.parent])
    with pytest.raises(ValueError, match="允许根"):
        input_path(tmp_path, [inputs])
    assert input_path(file, [inputs]) == file


def test_minimal_mounts_do_not_expose_history(setup):
    service, run, attempt, inputs = setup
    directory = service.store.attempt_dir(run["run_id"], attempt["attempt_id"])
    result = mounts([inputs], directory)
    assert all(mount["readonly"] for mount in result if mount["source"] == str(inputs))
    writable = {mount["source"] for mount in result if not mount["readonly"]}
    assert writable == {str(directory / name) for name in ("output", "control", "logs")}
    assert str(service.store.root) not in writable


def test_gpu_resolution_duplicate_alias_and_external_busy():
    calls = []
    def runner(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "0, GPU-a\n1, GPU-b\n" if "--query-gpu=index,uuid" in argv else "GPU-a, 123\n", "")
    runtime = DockerRuntime(runner)
    assert runtime.resolve_gpus("1,0", ["0", "1"]) == ["GPU-b", "GPU-a"]
    with pytest.raises(ValueError, match="别名重复"):
        runtime.resolve_gpus("0,GPU-a", ["0"])
    with pytest.raises(ValueError, match="计算进程"):
        runtime.check_idle(["GPU-a"])
    assert all(isinstance(argv, list) for argv in calls)


def test_persistent_claim_only_one_competitor(setup):
    service, run, attempt, _ = setup
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(service._claim, attempt) for _ in range(2)]
    assert sum(future.exception() is None for future in futures) == 1
    assert service._claims()["GPU-a"]["attempt_id"] == attempt["attempt_id"]


@pytest.mark.parametrize("field", ["name", "label", "uid", "id", "image", "gpu", "cuda_order"])
def test_container_identity_is_exact(setup, field):
    service, _, attempt, _ = setup
    record = container(service, attempt)
    if field == "name": record["Name"] = "/exact-container-copy"
    if field == "label": record["Config"]["Labels"]["org.agentic-arxiv.attempt"] = "other"
    if field == "uid": record["Config"]["User"] = "0:0"
    if field == "id": record["Id"] = "replacement"
    if field == "image": record["Config"]["Image"] = "latest"
    if field == "gpu": record["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] = ["GPU-b"]
    if field == "cuda_order": record["Config"]["Env"] = ["CUDA_VISIBLE_DEVICES=0"]
    with pytest.raises(IdentityMismatch):
        service.runtime.verify(record, attempt, service.profile)


def test_create_arguments_are_restricted_and_override_image_entrypoint(setup):
    service, _, attempt, _ = setup
    argv = service.runtime.create_arguments(service.profile, attempt, ["python", "-m", "agentic_arxiv", "train", "sft"], [])
    assert argv[argv.index("--entrypoint") + 1] == "python"
    assert argv[-5:] == [service.profile["image"], "-m", "agentic_arxiv", "train", "sft"]
    assert "--pull" in argv and "never" in argv and "--read-only" in argv
    assert "CUDA_VISIBLE_DEVICES=GPU-a" in argv
    assert not {"--privileged", "--rm", "--publish", "--ipc=host"} & set(argv)


def test_exit_zero_and_final_do_not_imply_training_success(setup, monkeypatch):
    service, run, attempt, _ = setup
    directory = service.store.attempt_dir(run["run_id"], attempt["attempt_id"])
    (directory / "output" / "final").mkdir()
    monkeypatch.setattr(service.runtime, "inspect", lambda name: container(service, attempt, running=False))
    status = service.status(run["run_id"])
    assert status["execution"] == "exited"
    assert status["training"] == "not_started"
    assert status["validation"] == "not_run"


@pytest.mark.parametrize("stage", ["sft", "dpo"])
@pytest.mark.parametrize("validation", ["passed", "failed", "skipped", "interrupted"])
def test_final_identity_waits_for_exit_and_remains_reusable(setup, monkeypatch, stage, validation):
    from agentic_arxiv.experiments.artifacts import resolve_model_reference
    from agentic_arxiv.runtime.react_contract import protocol_metadata

    service, _, _, inputs = setup
    run = service.store.create_run({"stage": stage, "total_budget": 10}, {}, {})
    attempt = service.store.create_attempt(run["run_id"], {"gpu_uuids": ["GPU-a"]})
    attempt = service.store.update_attempt(run["run_id"], attempt["attempt_id"], {
        "container_name": "exact-container", "container_id": "container-id"})
    directory = service.store.attempt_dir(run["run_id"], attempt["attempt_id"])
    final = directory / "output" / "final"
    base = inputs / "base"
    # 仅文件合同替身，不加载模型或执行训练。
    for model in (base, final):
        model.mkdir()
        atomic_json(model / "config.json", {})
        (model / "model.safetensors").touch()
    protocol = {**protocol_metadata(), "chat_template_hash": "declared-template"}
    manifest = {"stage": stage + "_full", "base_model": str(base), "protocol_metadata": protocol}
    if stage == "dpo":
        manifest.update(sft_source={"stage": "sft_full"}, reference_source={"stage": "sft_full"})
    atomic_json(final / "training_manifest.json", manifest)
    evidence = dict(run_id=run["run_id"], attempt_id=attempt["attempt_id"],
                    training="budget_completed", validation="in_progress", global_step=10)
    atomic_json(directory / "control" / "training-result.json", evidence)
    monkeypatch.setattr(service.runtime, "inspect", lambda _: container(service, attempt))
    assert not service.status(run["run_id"]).get("final_artifact")
    atomic_json(final / "verification_report.json", {
        "passed": validation == "passed", "stage": stage, "model_path": str(final),
        "protocol": {"training": protocol}})
    if stage == "dpo":
        atomic_json(final / "training_manifest.json", dict(manifest, verification={"passed": validation == "passed"}))
    atomic_json(directory / "control" / "training-result.json", dict(evidence, validation=validation))
    # 即便验证结果已写入，也等退出确认后再提交目录身份。
    assert not service.status(run["run_id"]).get("final_artifact")
    monkeypatch.setattr(service.runtime, "inspect", lambda _: container(service, attempt, running=False))
    frozen = service.status(run["run_id"])["final_artifact"]
    ref = {"kind": "run", "run_id": run["run_id"], "attempt_id": attempt["attempt_id"], "artifact": "final"}
    assert resolve_model_reference(ref, store_root=service.store.root, require_verified=False)["fingerprint"] == frozen["fingerprint"]
    assert service.status(run["run_id"])["final_artifact"] == frozen
    atomic_json(final / "config.json", {"changed_after_commit": True})
    with pytest.raises(ValueError, match="输入发生变化"):
        resolve_model_reference(ref, store_root=service.store.root, require_verified=False)


def test_docker_outage_preserves_claim_and_history(setup, monkeypatch):
    service, run, attempt, _ = setup
    service._claim(attempt)
    def outage(name): raise RuntimeUnavailable("offline")
    monkeypatch.setattr(service.runtime, "inspect", outage)
    assert service.status(run["run_id"])["execution"] == "unknown"
    assert "GPU-a" in service._claims()


def test_protection_evidence_not_overwritten_by_exit_zero(setup, monkeypatch):
    service, run, attempt, _ = setup
    directory = service.store.attempt_dir(run["run_id"], attempt["attempt_id"])
    atomic_json(directory / "control" / "training-result.json", dict(run_id=run["run_id"], attempt_id=attempt["attempt_id"], training="protection_stopped", validation="not_run", global_step=3))
    monkeypatch.setattr(service.runtime, "inspect", lambda name: container(service, attempt, running=False))
    result = service.status(run["run_id"])
    assert result["training"] == "protection_stopped" and result["global_step"] == 3
    with pytest.raises(ValueError, match="保护终止"):
        service.resume(run["run_id"], checkpoint="unused")


def test_safe_stop_is_idempotent_never_docker_stop_or_kill(setup, monkeypatch):
    service, run, attempt, _ = setup
    monkeypatch.setattr(service.runtime, "inspect", lambda name: container(service, attempt))
    calls = []
    monkeypatch.setattr(service.runtime, "call", lambda argv: calls.append(argv))
    first, second = service.stop(run["run_id"]), service.stop(run["run_id"])
    assert first["safe_stop_confirmed"] is False
    assert first["request"] == second["request"]
    assert calls == []


def test_uncertain_launch_does_not_repeat_create_or_release(setup, monkeypatch):
    service, run, attempt, inputs = setup
    service._claim(attempt)
    calls = []
    monkeypatch.setattr(service.runtime, "check_idle", lambda devices: None)
    def call(argv):
        calls.append(argv)
        raise RuntimeUnavailable("response lost")
    monkeypatch.setattr(service.runtime, "call", call)
    monkeypatch.setattr(service.runtime, "inspect", lambda name: container(service, attempt, running=False))
    result = service.launch_job(run, attempt, ["python", "-m", "agentic_arxiv"], [inputs / "control-metadata.json"])
    assert result["execution"] == "unknown" and result["claims_retained"]
    assert len(calls) == 1 and calls[0][:2] == ["docker", "create"]
    assert "GPU-a" in service._claims()


def test_response_lost_after_start_reconciles_same_container(setup, monkeypatch):
    service, run, attempt, inputs = setup
    service._claim(attempt)
    calls = []
    monkeypatch.setattr(service.runtime, "check_idle", lambda devices: None)
    def call(argv):
        calls.append(argv)
        if argv[:2] == ["docker", "create"]: return "container-id\n"
        raise RuntimeUnavailable("start response lost")
    monkeypatch.setattr(service.runtime, "call", call)
    monkeypatch.setattr(service.runtime, "inspect", lambda name: container(service, attempt))
    result = service.launch_job(run, attempt, ["python", "-m", "agentic_arxiv"], [inputs / "control-metadata.json"])
    assert result["started"] and not result["training_success"]
    assert len([argv for argv in calls if argv[:2] == ["docker", "create"]]) == 1


@pytest.mark.parametrize("started_at", ["2026-10-15T02:00:00.123456789Z", "0001-01-01T00:00:00Z", None])
@pytest.mark.parametrize("response_lost", [False, True])
def test_launch_records_only_post_start_container_time(setup, monkeypatch, started_at, response_lost):
    service, run, attempt, inputs = setup
    calls = []
    inspections = []
    monkeypatch.setattr(service.runtime, "check_idle", lambda _: None)

    def call(argv):
        calls.append(argv)
        if argv[:2] == ["docker", "create"]:
            return "container-id\n"
        if response_lost:
            raise RuntimeUnavailable("start response lost")
        return "container-id\n"

    def inspect(name):
        launched = any(argv[:2] == ["docker", "start"] for argv in calls)
        inspections.append(launched)
        record = container(service, attempt, running=launched)
        record["State"]["StartedAt"] = started_at if launched else "0001-01-01T00:00:00Z"
        return record

    monkeypatch.setattr(service.runtime, "call", call)
    monkeypatch.setattr(service.runtime, "inspect", inspect)
    assert service.launch_job(run, attempt, ["python", "-m", "agentic_arxiv"],
                              [inputs / "control-metadata.json"])["started"]
    saved = service.store.read_attempt(run["run_id"])
    assert saved["started_at"] == (started_at if started_at and started_at.startswith("2026") else None)
    assert inspections == [False, True]
    assert len([argv for argv in calls if argv[:2] == ["docker", "start"]]) == 1


@pytest.mark.parametrize("running", [False, True])
def test_status_corrects_old_zero_start_time_from_live_container(setup, monkeypatch, running):
    service, run, attempt, _ = setup
    service.store.update_attempt(run["run_id"], attempt["attempt_id"], {
        "started_at": "0001-01-01T00:00:00Z", "finished_at": "0001-01-01T00:00:00Z"})
    record = container(service, attempt, running=running)
    record["State"].update(StartedAt="2026-10-15T02:00:00Z",
                           FinishedAt="0001-01-01T00:00:00Z" if running else "2026-10-15T03:00:00Z")
    monkeypatch.setattr(service.runtime, "inspect", lambda _: record)
    result = service.status(run["run_id"])
    assert result["started_at"] == record["State"]["StartedAt"]
    assert result["costs"]["allocated_attempt_gpu_hours"] == (None if running else 1)
    saved = service.store.read_attempt(run["run_id"])
    assert saved["started_at"] == record["State"]["StartedAt"]
    assert saved["finished_at"] == (None if running else record["State"]["FinishedAt"])


def test_timeout_then_temporarily_absent_keeps_claim(setup, monkeypatch):
    service, run, attempt, inputs = setup
    service._claim(attempt)
    monkeypatch.setattr(service.runtime, "check_idle", lambda devices: None)
    monkeypatch.setattr(service.runtime, "inspect", lambda name: None)
    def lost(argv): raise RuntimeUnavailable("response lost")
    monkeypatch.setattr(service.runtime, "call", lost)
    result = service.launch_job(run, attempt, ["python", "-m", "agentic_arxiv"], [inputs / "control-metadata.json"])
    assert result["execution"] == "unknown" and result["management_exit_code"] == 5
    assert "GPU-a" in service._claims()


@pytest.mark.parametrize("uncertain", [False, True])
def test_start_failure_releases_only_with_definite_error_and_created_container(setup, monkeypatch, uncertain):
    service, run, attempt, inputs = setup
    service._claim(attempt)
    calls = []
    record = container(service, attempt, running=False)
    record["State"].update(Status="created", StartedAt="0001-01-01T00:00:00Z")
    monkeypatch.setattr(service.runtime, "inspect", lambda _: record)
    monkeypatch.setattr(service.runtime, "check_idle", lambda _: None)

    def call(argv):
        calls.append(argv)
        if argv[:2] == ["docker", "create"]:
            return "container-id\n"
        raise RuntimeUnavailable("start rejected" if not uncertain else "response lost", uncertain=uncertain)

    monkeypatch.setattr(service.runtime, "call", call)
    result = service.launch_job(run, attempt, ["python", "-m", "agentic_arxiv"], [inputs / "control-metadata.json"])
    assert result["management_exit_code"] == (5 if uncertain else 4)
    status = service.status(run["run_id"])
    assert status["execution"] == ("starting" if uncertain else "create_failed")
    assert ("GPU-a" in service._claims()) is uncertain
    assert len([argv for argv in calls if argv[:2] == ["docker", "start"]]) == 1
    assert not any(argv[1] in {"kill", "rm"} for argv in calls)
    if not uncertain:
        assert status["training"] == "failed"
        replacement = service.store.create_attempt(run["run_id"], {"gpu_uuids": ["GPU-a"]})
        service._claim(replacement)
        assert service._claims()["GPU-a"]["attempt_id"] == replacement["attempt_id"]


@pytest.mark.parametrize("reconcile", ["unavailable", "mismatch", "running"])
def test_definite_start_error_still_requires_exact_unstarted_container(setup, monkeypatch, reconcile):
    service, run, attempt, inputs = setup
    service._claim(attempt)
    calls = []
    record = container(service, attempt, running=False)
    record["State"]["Status"] = "created"
    monkeypatch.setattr(service.runtime, "check_idle", lambda _: None)

    def inspect(_):
        if len(calls) < 2:
            return record
        if reconcile == "unavailable":
            raise RuntimeUnavailable("inspect unavailable")
        if reconcile == "mismatch":
            return dict(record, Id="unrelated-container")
        return container(service, attempt)

    def call(argv):
        calls.append(argv)
        if argv[:2] == ["docker", "create"]:
            return "container-id\n"
        raise RuntimeUnavailable("start rejected", uncertain=False)

    monkeypatch.setattr(service.runtime, "inspect", inspect)
    monkeypatch.setattr(service.runtime, "call", call)
    result = service.launch_job(run, attempt, ["python", "-m", "agentic_arxiv"], [inputs / "control-metadata.json"])
    assert "GPU-a" in service._claims()
    assert result.get("management_exit_code") == (None if reconcile == "running" else 5)
    # 明确拒绝证据落盘；稍后只有核验相同容器确实未启动才能释放。
    if reconcile == "unavailable":
        monkeypatch.setattr(service.runtime, "inspect", lambda _: record)
        assert service.status(run["run_id"])["execution"] == "create_failed"
        assert "GPU-a" not in service._claims()


def test_post_start_inspect_error_is_not_evidence_that_start_failed(setup, monkeypatch):
    service, run, attempt, inputs = setup
    service._claim(attempt)
    inspections = []
    record = container(service, attempt, running=False)
    record["State"]["Status"] = "created"
    monkeypatch.setattr(service.runtime, "check_idle", lambda _: None)
    monkeypatch.setattr(service.runtime, "call", lambda _: "container-id\n")

    def inspect(_):
        inspections.append(True)
        if len(inspections) == 2:
            raise RuntimeUnavailable("inspect rejected", uncertain=False)
        return record

    monkeypatch.setattr(service.runtime, "inspect", inspect)
    result = service.launch_job(run, attempt, ["python", "-m", "agentic_arxiv"], [inputs / "control-metadata.json"])
    assert result["management_exit_code"] == 5
    assert not service.store.read_attempt(run["run_id"]).get("start_failure")
    assert service.status(run["run_id"])["execution"] == "starting"
    assert "GPU-a" in service._claims()


def test_force_stops_only_exact_container(setup, monkeypatch):
    service, run, attempt, _ = setup
    calls = []
    monkeypatch.setattr(service.runtime, "inspect", lambda name: container(service, attempt, running=not calls, exit_code=137 if calls else 0))
    monkeypatch.setattr(service.runtime, "call", lambda argv: calls.append(argv) or "")
    result = service.stop(run["run_id"], force=True)
    assert result["force_requested"]
    assert calls == [["docker", "kill", "--signal", "KILL", "container-id"]]
    assert service.store.read_attempt(run["run_id"])["training"] == "forced_interrupted"


def test_force_response_loss_keeps_evidence_and_later_reconciles(setup, monkeypatch):
    service, run, attempt, _ = setup
    service._claim(attempt)
    calls = []
    monkeypatch.setattr(service.runtime, "inspect", lambda name: container(service, attempt))
    def lost(argv):
        calls.append(argv)
        raise RuntimeUnavailable("kill response lost")
    monkeypatch.setattr(service.runtime, "call", lost)
    result = service.stop(run["run_id"], force=True)
    saved = service.store.read_attempt(run["run_id"])
    assert result["management_exit_code"] == 5 and saved["force_uncertain"]
    assert saved["force_target_id"] == "container-id" and saved["force_requested_at"]
    assert "GPU-a" in service._claims() and len(calls) == 1
    monkeypatch.setattr(service.runtime, "inspect", lambda name: container(service, attempt, running=False, exit_code=137))
    status = service.status(run["run_id"])
    assert status["training"] == "forced_interrupted" and status["force_uncertain"] is False
    assert "GPU-a" not in service._claims()


def test_completed_managed_final_can_be_reused_readonly(setup):
    service, run, attempt, inputs = setup
    final = service.store.attempt_dir(run["run_id"], attempt["attempt_id"]) / "output" / "final"
    final.mkdir()
    assert input_path(final, [inputs], managed_root=service.store.root) == final
    validate_layout(service.store.root, [final])
    with pytest.raises(ValueError, match="重叠"):
        validate_layout(service.store.root, [final], attempt_dir=final.parent.parent)


def test_sft_render_does_not_add_unsupported_snapshot(setup):
    from agentic_arxiv.training.mainline_config import stage_config
    service, _, _, inputs = setup
    model = str(inputs / "control-metadata.json")
    resolved = {"recipe": {"group": "M1", "stage": "sft"}, "training": stage_config("sft")}
    rendered = service._render_training(resolved, {"inputs": {"model": {"path": model}, "snapshot": {"path": model}}})
    assert rendered["defaults"]["model"] == model
    assert "snapshot" not in rendered["defaults"]


def test_grpo_split_is_explicit_and_bound_to_input(setup):
    from agentic_arxiv.training.mainline_config import stage_config
    service, _, _, inputs = setup
    path = str(inputs / "control-metadata.json")
    resolved = {"recipe": {"group": "M3", "stage": "grpo"}, "training": stage_config("grpo")}
    audit = {"inputs": {"model": {"path": path}, "split": {"path": path}}}
    with pytest.raises(ValueError, match="显式填写"):
        service._render_training(resolved, audit)
    resolved["training"]["defaults"]["split"] = path + ":rl_train"
    assert service._render_training(resolved, audit)["defaults"]["split"] == path + ":rl_train"
    resolved["training"]["defaults"]["split"] = path + ":iid_test"
    with pytest.raises(ValueError, match="train/rl_train"):
        service._render_training(resolved, audit)


def test_check_uses_readonly_nogpu_preflight_without_creating_run(setup, monkeypatch):
    from agentic_arxiv.training.mainline_config import stage_config
    import agentic_arxiv.experiments.artifacts as artifacts
    service, _, _, inputs = setup
    service.profile.update(uid=os.getuid(), gid=os.getgid())
    training = stage_config("sft")
    training["defaults"]["max_steps"] = 10
    recipe = {"group": "M1", "stage": "sft", "inputs": {"model": str(inputs / "control-metadata.json")},
              "budget": {"max_steps": 10}, "checkpoint": {"save_steps": 2}}
    base = inputs / "base-control.json"
    base.write_text("{}")
    policy = inputs / "policy-control.json"
    policy.write_text("{}")
    audit = {"inputs": {"model": {"path": recipe["inputs"]["model"],
        "base_model": {"path": str(base)},
        "metadata": {"source_baseline": {"configuration_baseline": {
            "path": "artifacts/mainline_baseline/source_configuration.json"}},
            "execution_source": {"path": "/missing/historical/source.py"}}}},
        "policies": {"task_pool": {"path": str(policy)}},
        "comparison": {"path": str(service.store.root / "runs/old/resolved-config.json")}}
    monkeypatch.setattr(artifacts, "audit_recipe_inputs", lambda *a, **k: audit)
    monkeypatch.setattr(service.runtime, "image", lambda ref: {"Id": "verified-image"})
    monkeypatch.setattr(service.runtime, "resolve_gpus", lambda *a: ["GPU-a"])
    monkeypatch.setattr(service.runtime, "check_idle", lambda *a: None)
    calls = []
    def call(argv, **kwargs):
        calls.append((argv, kwargs))
        return json.dumps(dict(schema_version=1, status="passed", stage="sft"))
    monkeypatch.setattr(service.runtime, "call", call)
    before = service.store.list_runs()
    result = service.check({"recipe": recipe, "training": training}, gpus="0")
    assert result["checked"] and service.store.list_runs() == before
    assert set(result["input_paths"]) == {recipe["inputs"]["model"], str(base), str(policy)}
    argv, kwargs = calls[0]
    assert "--read-only" in argv and "--gpus" not in argv
    assert "NVIDIA_VISIBLE_DEVICES=void" in argv
    assert all("readonly" in argv[index + 1] for index, token in enumerate(argv) if token == "--mount")
    assert json.loads(kwargs["input_text"])["workspace"] == "/tmp"
    def failed(*a, **k): return '{"schema_version":1,"status":"failed","stage":"sft"}'
    monkeypatch.setattr(service.runtime, "call", failed)
    with pytest.raises(ValueError, match="预检未通过"):
        service.check({"recipe": recipe, "training": training}, gpus="0")


def test_dpo_preflight_and_training_use_same_frozen_dependency_mounts(setup, monkeypatch):
    from agentic_arxiv.provenance import file_fingerprint
    from agentic_arxiv.training.mainline_config import stage_config
    import agentic_arxiv.experiments.artifacts as artifacts

    service, _, _, inputs = setup
    service.profile.update(uid=os.getuid(), gid=os.getgid())
    records = {}
    for name in ("model", "data", "data_manifest", "source_model_manifest", "split_file", "traces", "report"):
        path = inputs / (name + ".json")
        atomic_json(path, {"control_metadata": name})
        records[name] = {"path": str(path), "fingerprint": file_fingerprint(path)}
    records["data_manifest"]["runtime_dependencies"] = {
        name: records[name] for name in ("source_model_manifest", "split_file", "traces", "report")}
    records["data_manifest"]["metadata"] = {"historical_source": {"path": "/unmounted/reward.py"}}
    audit = {"inputs": {name: records[name] for name in ("model", "data", "data_manifest")}}
    training = stage_config("dpo")
    training["defaults"]["max_steps"] = 10
    recipe = {"group": "M2", "stage": "dpo", "inputs": {name: record["path"] for name, record in audit["inputs"].items()},
              "budget": {"max_steps": 10}, "checkpoint": {"save_steps": 2}}
    monkeypatch.setattr(artifacts, "audit_recipe_inputs", lambda *a, **k: audit)
    monkeypatch.setattr(service.runtime, "image", lambda ref: {"Id": "verified-image"})
    monkeypatch.setattr(service.runtime, "resolve_gpus", lambda *a: ["GPU-b"])
    monkeypatch.setattr(service.runtime, "check_idle", lambda *a: None)
    captured = {}

    def preflight(argv, **kwargs):
        captured["preflight_mounts"] = [argv[i + 1] for i, value in enumerate(argv) if value == "--mount"]
        captured["preflight_inputs"] = json.loads(kwargs["input_text"])["inputs"]
        return json.dumps({"schema_version": 1, "status": "passed", "stage": "dpo"})

    def launch(run, attempt, command, paths, **kwargs):
        captured["training_paths"] = paths
        context = service.store.attempt_dir(run["run_id"], attempt["attempt_id"]) / "control" / "managed-context.json"
        captured["context"] = json.loads(context.read_text())
        return {"started": True}

    monkeypatch.setattr(service.runtime, "call", preflight)
    monkeypatch.setattr(service, "launch_job", launch)
    assert service.start({"recipe": recipe, "training": training}, gpus="1")["started"]
    expected = {record["path"] for record in records.values()}
    assert set(captured["training_paths"]) == expected
    assert set(captured["preflight_mounts"]) == {f"type=bind,src={path},dst={path},readonly" for path in expected}
    assert captured["preflight_inputs"] == audit["inputs"]
    assert captured["context"]["identity"]["inputs"] == audit
    assert str(inputs) not in expected
    outside = inputs.parent / "outside.json"
    atomic_json(outside, {})
    records["report"].update(path=str(outside), fingerprint=file_fingerprint(outside))
    with pytest.raises(ValueError, match="允许根"):
        service.check({"recipe": recipe, "training": training}, gpus="1")


@pytest.mark.parametrize("stage,group", [("sft", "M1"), ("dpo", "M2"), ("grpo", "M3")])
@pytest.mark.parametrize("epoch_mode", [False, True])
def test_generated_training_config_roundtrips_real_cli_and_managed_identity(setup, monkeypatch, stage, group, epoch_mode):
    from agentic_arxiv.cli import parse_args
    from agentic_arxiv.training.checkpoint import ManagedCheckpointContext
    from agentic_arxiv.training.mainline_config import stage_config
    from agentic_arxiv.training.managed import validate_managed_config
    service, _, _, inputs = setup
    path = str(inputs / "control-metadata.json")
    training = stage_config(stage)
    training["defaults"]["max_steps"] = -1 if epoch_mode else 10
    training["defaults"]["epochs"] = 2
    policy = {"path": path, "fingerprint": {"control_metadata": "not-a-model"}}
    audit = {"inputs": {"model": policy}}
    for key in ("data", "data_manifest"):
        if key in training["defaults"]:
            audit["inputs"][key] = {"path": path}
    if stage == "grpo":
        audit["inputs"]["split"] = {"path": path}
        training["defaults"]["split"] = path + ":rl_train"
    if stage == "sft":
        audit["inputs"]["snapshot"] = {"path": path}
    budget = {"max_steps": None if epoch_mode else 10, "epochs": 2 if epoch_mode else None}
    resolved = {"recipe": {"group": group, "stage": stage, "budget": budget}, "training": training}
    checked = {"resolved": resolved, "audit": audit, "input_paths": [path], "gpu_uuids": ["GPU-b"], "image_id": "verified-image"}
    run = service.store.create_run({"stage": stage, "total_budget": budget["max_steps"], "epoch_budget": budget["epochs"]}, resolved, audit)
    captured = {}
    def launch(run, attempt, command, paths, **kwargs):
        captured.update(attempt=attempt, command=command)
        return {"started": True}
    monkeypatch.setattr(service, "launch_job", launch)
    service._launch_training(run, checked)
    command = captured["command"]
    _, args = parse_args(command[3:])
    context = ManagedCheckpointContext.from_file(args.managed_context)
    if stage == "sft":
        assert args.snapshot == path
    configured = stage_config(stage, args.config_path)
    validate_managed_config(context, vars(args), configured)
    assert configured["defaults"]["model"] == path
    assert configured["defaults"]["output_dir"] != training["defaults"]["output_dir"]
    if epoch_mode:
        assert configured["defaults"]["max_steps"] == -1 and configured["defaults"]["epochs"] == 2
        assert context.total_budget is None and context.epoch_budget == 2


def test_ddp_preserves_explicit_uuid_rank_order(setup, monkeypatch):
    from agentic_arxiv.training.mainline_config import stage_config
    from agentic_arxiv.training.launch import build_launch
    service, _, _, inputs = setup
    path = str(inputs / "control-metadata.json")
    training = stage_config("sft")
    training["defaults"]["max_steps"] = 10
    resolved = {"recipe": {"group": "M1", "stage": "sft", "budget": {"max_steps": 10}}, "training": training}
    audit = {"inputs": {"model": {"path": path}, "data": {"path": path}, "data_manifest": {"path": path}}}
    checked = {"resolved": resolved, "audit": audit, "input_paths": [path], "gpu_uuids": ["GPU-b", "GPU-a"], "image_id": "image"}
    run = service.store.create_run({"stage": "sft", "total_budget": 10}, resolved, audit)
    captured = {}
    def launch(run, attempt, command, paths, **kwargs):
        captured.update(attempt=attempt, command=command)
        return {}
    monkeypatch.setattr(service, "launch_job", launch)
    service._launch_training(run, checked)
    _, _, environment = build_launch(captured["command"][3:])
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-b,GPU-a"
    argv = service.runtime.create_arguments(service.profile, captured["attempt"], captured["command"], [])
    assert "CUDA_VISIBLE_DEVICES=GPU-b,GPU-a" in argv


def test_m0_check_validates_fixed_evaluation_without_creating_records(setup, monkeypatch):
    import agentic_arxiv.experiments.artifacts as artifacts
    import agentic_arxiv.experiments.evaluation as evaluation
    service, _, _, inputs = setup
    service.profile.update(uid=os.getuid(), gid=os.getgid())
    path = str(inputs / "control-metadata.json")
    audit = {"inputs": {"model": {"path": path}, "snapshot": {"path": path}, "split": {"path": path}}}
    recipe = {"stage": "evaluate", "inputs": {"model": path}, "evaluation": {"offline": True, "split": path + ":iid_test"}}
    monkeypatch.setattr(artifacts, "audit_recipe_inputs", lambda *a, **k: audit)
    monkeypatch.setattr(service.runtime, "image", lambda *a: {"Id": "verified-image"})
    monkeypatch.setattr(service.runtime, "resolve_gpus", lambda *a: ["GPU-a"])
    monkeypatch.setattr(service.runtime, "check_idle", lambda *a: None)
    calls = []
    def prepare(store, reference, kind, config, **kwargs):
        calls.append((reference, kind, config, kwargs))
        assert kwargs["persist"] is False
        return {"inputs": [path], "context": {"evaluation": config}}
    monkeypatch.setattr(evaluation, "prepare_evaluation", prepare)
    before = service.store.list_runs()
    service.check({"recipe": recipe}, gpus="0")
    assert service.store.list_runs() == before
    assert calls[0][2]["snapshot"] == path and calls[0][2]["split"] == path + ":iid_test"


def test_epoch_resume_can_reuse_older_attempt_without_rewriting_budget(setup, monkeypatch):
    from agentic_arxiv.training.mainline_config import stage_config
    import agentic_arxiv.training.checkpoint as checkpoint_module
    service, _, _, inputs = setup
    path = str(inputs / "control-metadata.json")
    training = stage_config("sft")
    training["defaults"].update(max_steps=-1, epochs=3)
    resolved = {"recipe": {"group": "M1", "stage": "sft", "budget": {"max_steps": None, "epochs": 3}}, "training": training}
    audit = {"inputs": {"model": {"path": path}, "data": {"path": path}, "data_manifest": {"path": path}}}
    checked = {"resolved": resolved, "audit": audit, "input_paths": [path], "gpu_uuids": ["GPU-b"], "image_id": "image"}
    run = service.store.create_run({"stage": "sft", "total_budget": None, "epoch_budget": 3}, resolved, audit)
    first = service.store.create_attempt(run["run_id"], {"gpu_uuids": ["GPU-b"]})
    source = service.store.attempt_dir(run["run_id"], first["attempt_id"]) / "output" / "checkpoint-4"
    source.mkdir()
    service.store.update_attempt(run["run_id"], first["attempt_id"], {"execution": "exited"})
    second = service.store.create_attempt(run["run_id"], {"gpu_uuids": ["GPU-b"]})
    service.store.update_attempt(run["run_id"], second["attempt_id"], {"execution": "create_failed", "training": "failed"})
    monkeypatch.setattr(service, "check", lambda *a, **k: checked)
    expected_records = []
    def validate(path, *, expected):
        expected_records.append(expected)
        return {"global_step": 4, "total_budget": 12, "attempt_id": first["attempt_id"]}
    monkeypatch.setattr(checkpoint_module, "validate_checkpoint", validate)
    captured = {}
    def launch(run, attempt, command, paths, **kwargs):
        captured.update(attempt=attempt, command=command)
        return {"started": True}
    monkeypatch.setattr(service, "launch_job", launch)
    assert service.resume(run["run_id"], checkpoint=str(source))["started"]
    new = captured["attempt"]
    context = json.loads((service.store.attempt_dir(run["run_id"], new["attempt_id"]) / "control" / "managed-context.json").read_text())
    config = json.loads((service.store.attempt_dir(run["run_id"], new["attempt_id"]) / "training-config.json").read_text())
    assert context["total_budget"] == 12 and context["epoch_budget"] == 3
    assert expected_records[0]["total_budget"] is None and expected_records[0]["epoch_budget"] == 3
    assert config["sft"]["defaults"]["max_steps"] == -1 and config["sft"]["defaults"]["epochs"] == 3
    assert new["attempt_id"] not in {first["attempt_id"], second["attempt_id"]}
    assert source.is_dir() and new["source_checkpoint_step"] == 4


def test_force_request_does_not_hide_naturally_completed_budget(setup, monkeypatch):
    service, run, attempt, _ = setup
    service.store.update_attempt(run["run_id"], attempt["attempt_id"], {"force_requested_at": "known-request", "force_target_id": "container-id", "final_artifact": {"control_identity": "not-a-model"}})
    directory = service.store.attempt_dir(run["run_id"], attempt["attempt_id"])
    atomic_json(directory / "control" / "training-result.json", dict(run_id=run["run_id"], attempt_id=attempt["attempt_id"], training="budget_completed", validation="in_progress", global_step=10))
    monkeypatch.setattr(service.runtime, "inspect", lambda name: container(service, attempt, running=False, exit_code=137))
    status = service.status(run["run_id"])
    assert status["training"] == "budget_completed" and status["validation"] == "interrupted"
