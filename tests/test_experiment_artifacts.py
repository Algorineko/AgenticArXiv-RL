"""只构造文件合同控制元数据；不生成模型参数、研究响应或训练轨迹。"""
import json
from importlib import resources
from pathlib import Path

import pytest

from agentic_arxiv.experiments.artifacts import (
    audit_recipe_inputs, cost_summary, register_artifact, resolve_file_reference,
    resolve_model_reference,
)
from agentic_arxiv.experiments.store import FileStore
from agentic_arxiv.provenance import file_fingerprint, read_protocol_metadata, sha256_file
from agentic_arxiv.runtime.react_contract import protocol_metadata


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def passed_report(model):
    metadata = json.loads((model / "training_manifest.json").read_text())
    write_json(model / "verification_report.json", {
        "passed": True, "stage": metadata["stage"].split("_", 1)[0], "model_path": str(model),
        "protocol": {"training": read_protocol_metadata(metadata)},
    })


@pytest.fixture
def model(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    write_json(base / "config.json", {})
    # 沿用 test_model_artifacts 的必要文件存在性替身，不能证明真实模型有效。
    (base / "model.safetensors").touch()
    final = tmp_path / "sft"
    final.mkdir()
    write_json(final / "config.json", {})
    (final / "model.safetensors").touch()
    write_json(final / "training_manifest.json", {
        "stage": "sft_full", "base_model": str(base),
        "protocol_metadata": {**protocol_metadata(), "chat_template_hash": "declared-template"},
    })
    passed_report(final)
    return final


def snapshot(path):
    return {str(item.relative_to(path)): sha256_file(item) for item in path.rglob("*") if item.is_file()}


def test_base_input_weight_error_is_a_management_validation_error(model, tmp_path):
    base = tmp_path / "base"
    (base / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="权重或分片索引缺失"):
        audit_recipe_inputs({"group": "M1", "inputs": {"base_model": str(base)}},
                            store_root=tmp_path / "managed")


@pytest.mark.parametrize("exit_code", [None, 0, 1])
def test_artifact_adapter_does_not_swallow_process_exits(model, tmp_path, monkeypatch, exit_code):
    def exit_process(*args, **kwargs):
        raise SystemExit(exit_code)

    monkeypatch.setattr("agentic_arxiv.experiments.artifacts.artifact_metadata", exit_process)
    with pytest.raises(SystemExit) as exc:
        resolve_model_reference(str(model), store_root=tmp_path / "managed")
    assert exc.value.code == exit_code


def test_registration_is_read_only_and_rechecks_identity(model, tmp_path):
    before = snapshot(model)
    root = tmp_path / "managed"
    registered = register_artifact(root, "M1", str(model))
    assert snapshot(model) == before
    assert registered["registration_only"] and not registered["resume_eligible"]
    ref = {"kind": "registration", "registration_id": registered["artifact_id"]}
    result = resolve_model_reference(ref, store_root=root, expected_stage="sft")
    assert result["path"] == str(model)
    assert registered["costs"]["total_gpu_hours"] is None
    record_path = root / "registrations" / registered["artifact_id"] / "reference.json"
    record_before = record_path.read_bytes()
    write_json(model / "config.json", {"changed": True})
    with pytest.raises(ValueError, match="输入发生变化"):
        resolve_model_reference(ref, store_root=root)
    assert record_path.read_bytes() == record_before


def test_registration_does_not_promote_unknown_or_failed_verification(model, tmp_path):
    (model / "verification_report.json").unlink()
    registered = register_artifact(tmp_path / "managed", "M1", str(model))
    assert registered["verification"]["status"] == "unknown"
    with pytest.raises(ValueError, match="阶段验证"):
        resolve_model_reference(str(model), store_root=tmp_path)


def test_explicit_registration_id_never_overwrites(model, tmp_path):
    first = register_artifact(tmp_path, "M1", str(model), artifact_id="shared-sft")
    assert first["artifact_id"] == "shared-sft"
    with pytest.raises(FileExistsError):
        register_artifact(tmp_path, "M1", str(model), artifact_id="shared-sft")
    with pytest.raises(ValueError, match="非法记录"):
        register_artifact(tmp_path, "M1", str(model), artifact_id="../outside")


def test_model_report_binding_and_base_fingerprint_are_rechecked(model, tmp_path):
    registered = register_artifact(tmp_path, "M1", str(model))
    ref = {"kind": "registration", "registration_id": registered["artifact_id"]}
    base = registered["base_model"]["path"]
    write_json(Path(base) / "config.json", {"changed-base": True})
    with pytest.raises(ValueError, match="base_model"):
        resolve_model_reference(ref, store_root=tmp_path)
    report_path = model / "verification_report.json"
    report = json.loads(report_path.read_text())
    report["model_path"] = str(tmp_path / "unrelated")
    write_json(report_path, report)
    with pytest.raises(ValueError, match="阶段验证"):
        resolve_model_reference(str(model), store_root=tmp_path)


def test_managed_reference_freezes_exact_attempt_identity(model, tmp_path):
    store = FileStore(tmp_path / "managed")
    run = store.create_run({}, {}, {})
    attempt = store.create_attempt(run["run_id"], {})
    final = store.attempt_dir(run["run_id"], attempt["attempt_id"]) / "output" / "final"
    # 路径别名只用于元数据合同回归，不启动训练或复制研究模型。
    final.symlink_to(model, target_is_directory=True)
    identity = resolve_model_reference(str(final), store_root=store.root, require_verified=False)
    store.update_attempt(run["run_id"], attempt["attempt_id"], {
        "training": "budget_completed", "validation": "passed", "final_artifact": identity,
    })
    ref = {"kind": "run", "run_id": run["run_id"], "attempt_id": attempt["attempt_id"], "artifact": "final"}
    assert resolve_model_reference(ref, store_root=store.root)["path"] == str(model)
    write_json(model / "config.json", {"changed": True})
    with pytest.raises(ValueError, match="输入发生变化"):
        resolve_model_reference(ref, store_root=store.root)


def test_explicit_stage_retry_proof_is_exact_immutable_and_read_only(model, tmp_path):
    passed_report(model)
    report = json.loads((model / "verification_report.json").read_text())
    write_json(model / "verification_report.json", dict(report, passed=False))
    before = snapshot(model)
    store = FileStore(tmp_path / "managed")
    origin = resolve_model_reference(str(model), store_root=store.root, require_verified=False)
    config = {"num_samples": 5}
    run = store.create_run({"stage": "evaluate", "evaluation": {
        "kind": "stage", "config": config, "source": origin,
    }}, {}, {})
    attempt = store.create_attempt(run["run_id"], {})
    evidence_path = store.attempt_dir(run["run_id"], attempt["attempt_id"]) / "output" / "evaluation-result.json"
    evidence = {"schema_version": 1, "kind": "stage", "run_id": run["run_id"],
                "attempt_id": attempt["attempt_id"], "execution": "complete", "passed": True,
                "training_executed": False, "qualification_eligible": True,
                "source": origin, "configuration": config, "report": report}
    write_json(evidence_path, evidence)
    verification_ref = {"run_id": run["run_id"], "attempt_id": attempt["attempt_id"]}
    ref = {"kind": "path", "path": str(model), "verification_ref": verification_ref}
    assert resolve_model_reference(ref, store_root=store.root)["verification"]["status"] == "passed"
    registered = register_artifact(store.root, "M1", str(model), metadata={"verification_ref": verification_ref})
    registered_ref = {"kind": "registration", "registration_id": registered["artifact_id"]}
    assert resolve_model_reference(registered_ref, store_root=store.root)["verification"]["kind"] == "stage"
    assert snapshot(model) == before
    with pytest.raises(ValueError, match="阶段验证"):
        resolve_model_reference(str(model), store_root=store.root)
    evidence["diagnostic"] = True
    write_json(evidence_path, evidence)
    with pytest.raises(ValueError, match="重验证据"):
        resolve_model_reference(ref, store_root=store.root)
    evidence.pop("diagnostic")
    evidence["configuration"] = {"num_samples": 1}
    write_json(evidence_path, evidence)
    with pytest.raises(ValueError, match="重验证据"):
        resolve_model_reference(registered_ref, store_root=store.root)
    write_json(model / "verification_report.json", {"passed": False})
    with pytest.raises(ValueError, match="阶段验证"):
        resolve_model_reference(str(model), store_root=tmp_path)


def test_m4_requires_dpo_and_never_changes_stage(model, tmp_path):
    before = snapshot(model)
    with pytest.raises(ValueError, match="stage"):
        resolve_model_reference(str(model), store_root=tmp_path, expected_stage="dpo")
    assert snapshot(model) == before


def test_protocol_and_base_declaration_are_not_guessed(model, tmp_path):
    manifest = model / "training_manifest.json"
    value = json.loads(manifest.read_text())
    value["protocol_metadata"].pop("chat_template_hash")
    write_json(manifest, value)
    with pytest.raises(ValueError, match="chat_template_hash"):
        register_artifact(tmp_path, "M1", str(model))
    value["protocol_metadata"]["chat_template_hash"] = "declared-template"
    value.pop("base_model")
    write_json(manifest, value)
    with pytest.raises(ValueError, match="base_model"):
        register_artifact(tmp_path, "M1", str(model))


def test_file_paths_and_exact_run_reference(model, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    document = workspace / "policy.json"
    write_json(document, {})
    assert resolve_file_reference("policy.json", workspace=workspace)["path"] == str(document)
    with pytest.raises(ValueError, match="明确路径"):
        resolve_file_reference({"kind": "registration", "registration_id": "x"})
    with pytest.raises(ValueError, match="attempt_id"):
        resolve_model_reference({"kind": "run", "run_id": "run-x", "artifact": "final"}, store_root=tmp_path)
    with pytest.raises(ValueError, match="非法记录"):
        resolve_model_reference({"kind": "registration", "registration_id": "../outside"}, store_root=tmp_path)


def test_common_sft_and_policy_fingerprint_audit(model, tmp_path):
    task_pool = tmp_path / "task-pool.json"
    policy = tmp_path / "selection-policy.json"
    write_json(task_pool, {"input_contract": "existing-task-pool-reference"})
    write_json(policy, {"input_contract": "existing-selection-policy-reference"})
    recipe = {"group": "M3", "inputs": {"model": str(model), "source_sft": str(model)},
              "task_pool": {"path": str(task_pool), "fingerprint": sha256_file(task_pool)},
              "filter_policy": {"path": str(policy), "fingerprint": sha256_file(policy)},
              "budget": {}}
    result = audit_recipe_inputs(recipe, store_root=tmp_path / "managed")
    assert result["common_sft"]["stage"] == "sft_full"
    assert result["costs"]["strict_budget_match"] is False
    recipe["filter_policy"]["fingerprint"] = "changed"
    with pytest.raises(ValueError, match="filter_policy"):
        audit_recipe_inputs(recipe, store_root=tmp_path)


@pytest.fixture
def dpo_recipe(model, tmp_path):
    package = tmp_path / "preference-package"
    package.mkdir()
    data, traces, split, report = [package / name for name in (
        "pairs.jsonl", "traces.jsonl", "split.json", "report.json",
    )]
    for path in (data, traces, split, report):
        write_json(path, {"control_metadata": path.name})
    manifest = package / "pairs.manifest.json"
    entries = {name: {"path": path.name, "sha256": sha256_file(path)} for name, path in (
        ("data", data), ("traces", traces), ("split_file", split), ("report", report),
    )}
    write_json(manifest, {
        "files": {name: entries[name] for name in ("data", "traces")},
        "split_file": entries["split_file"], "report": entries["report"],
        "source_model_manifest": file_fingerprint(model / "training_manifest.json"),
        # 元数据合同替身仅验证挂载解析，不声称是可供训练的偏好数据。
        "scoring": {"source": {"path": "/unmounted/old-reward.py", "sha256": "history"}},
        "source_baseline": {"path": "artifacts/mainline_baseline/source_configuration.json"},
    })
    return {"group": "M2", "inputs": {
        "model": str(model), "source_sft": str(model), "data": str(data), "data_manifest": str(manifest),
    }}


def test_dpo_audit_freezes_only_explicit_runtime_dependencies(dpo_recipe, tmp_path):
    manifest = Path(dpo_recipe["inputs"]["data_manifest"])
    before = snapshot(tmp_path)
    audited = audit_recipe_inputs(dpo_recipe, store_root=tmp_path / "managed")
    dependencies = audited["inputs"]["data_manifest"]["runtime_dependencies"]
    assert set(dependencies) == {"source_model_manifest", "split_file", "traces", "report"}
    assert dependencies["split_file"]["path"] == str(manifest.parent / "split.json")
    assert dependencies["source_model_manifest"]["path"] == str(
        Path(dpo_recipe["inputs"]["model"]) / "training_manifest.json")
    assert all(record["fingerprint"] == file_fingerprint(record["path"]) for record in dependencies.values())
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("name", ["source_model_manifest", "split_file", "data", "traces", "report"])
def test_dpo_audit_rejects_dependency_hash_mismatch(dpo_recipe, tmp_path, name):
    manifest = Path(dpo_recipe["inputs"]["data_manifest"])
    payload = json.loads(manifest.read_text())
    entry = payload["files"][name] if name in {"data", "traces"} else payload[name]
    entry["sha256"] = "changed"
    write_json(manifest, payload)
    with pytest.raises(ValueError, match=f"DPO {name} 文件缺失或哈希不一致"):
        audit_recipe_inputs(dpo_recipe, store_root=tmp_path / "managed")


@pytest.mark.parametrize("name", ["source_model_manifest", "split_file", "data", "traces", "report"])
def test_dpo_audit_rejects_missing_dependency_declaration(dpo_recipe, tmp_path, name):
    manifest = Path(dpo_recipe["inputs"]["data_manifest"])
    payload = json.loads(manifest.read_text())
    (payload["files"] if name in {"data", "traces"} else payload).pop(name)
    write_json(manifest, payload)
    with pytest.raises(ValueError, match=f"缺少 {name} 文件声明"):
        audit_recipe_inputs(dpo_recipe, store_root=tmp_path / "managed")


def test_dpo_audit_rejects_wrong_data_and_dependency_symlink(dpo_recipe, tmp_path):
    manifest = Path(dpo_recipe["inputs"]["data_manifest"])
    payload = json.loads(manifest.read_text())
    original_data = payload["files"]["data"]
    payload["files"]["data"] = payload["files"]["traces"]
    write_json(manifest, payload)
    with pytest.raises(ValueError, match="数据文件路径或指纹"):
        audit_recipe_inputs(dpo_recipe, store_root=tmp_path / "managed")
    payload["files"]["data"] = original_data
    alias = manifest.parent / "trace-alias.jsonl"
    alias.symlink_to(manifest.parent / "traces.jsonl")
    payload["files"]["traces"]["path"] = alias.name
    write_json(manifest, payload)
    with pytest.raises(ValueError, match="符号链接"):
        audit_recipe_inputs(dpo_recipe, store_root=tmp_path / "managed")


def test_dpo_lineage_must_match_explicit_common_sft(model, tmp_path):
    origin = resolve_model_reference(str(model), store_root=tmp_path)
    dpo = tmp_path / "dpo"
    dpo.mkdir()
    write_json(dpo / "config.json", {})
    (dpo / "model.safetensors").touch()
    metadata = dict(origin["metadata"], stage="dpo_full")
    source = {"model": origin["fingerprint"], "manifest": file_fingerprint(model / "training_manifest.json"),
              "stage": "sft_full"}
    metadata.update(sft_source=source, reference_source=source)
    write_json(dpo / "training_manifest.json", metadata)
    passed_report(dpo)
    recipe = {"group": "M4", "inputs": {"model": str(dpo), "source_sft": str(model)}}
    # 来源链通过后才诊断尚未填写的 task_pool，不自动准备缺失输入。
    with pytest.raises(ValueError, match="真实路径"):
        audit_recipe_inputs(recipe, store_root=tmp_path)
    metadata["sft_source"]["model"] = {"source": "different"}
    write_json(dpo / "training_manifest.json", metadata)
    with pytest.raises(ValueError, match="共同 SFT 来源不一致"):
        audit_recipe_inputs(recipe, store_root=tmp_path)


def test_costs_preserve_unknown_and_never_assert_equal_budget():
    report = cost_summary({"comparison_group": "M4", "costs": {"dpo_gpu_hours": 2}})
    assert report["total_gpu_hours"] is None
    assert report["known_total_gpu_hours"] == 2
    assert "preference_generation_gpu_hours" in report["unknown_components"]
    assert not report["strict_budget_match"]
    with pytest.raises(ValueError, match="成本"):
        cost_summary({"costs": {"dpo_gpu_hours": float("nan")}})


@pytest.mark.parametrize("declared_budget", [
    {"max_steps": 100, "epochs": None},
    {"max_steps": None, "epochs": 3},
])
def test_c1_requires_exact_m4_comparison_not_only_group_name(model, tmp_path, declared_budget):
    policy = tmp_path / "existing-policy-reference.json"
    write_json(policy, {"input_contract": "existing-policy-reference"})
    declared = {"path": str(policy), "fingerprint": sha256_file(policy)}
    recipe = {"group": "C1", "inputs": {"model": str(model), "source_sft": str(model)},
              "task_pool": declared, "filter_policy": declared,
              "budget": {"comparison_group": "M4", **declared_budget}}
    with pytest.raises(ValueError, match="comparison_reference"):
        audit_recipe_inputs(recipe, store_root=tmp_path)
    report = tmp_path / "m4-cost-reference.json"
    write_json(report, {"group": "M4", "costs": {"dpo_gpu_hours": None}})
    recipe["budget"]["comparison_reference"] = str(report)
    result = audit_recipe_inputs(recipe, store_root=tmp_path)
    assert result["comparison"]["fingerprint"] == file_fingerprint(report)
    assert result["costs"]["total_gpu_hours"] is None
    write_json(report, {"group": "M4", "filter_policy": {"fingerprint": "different"}})
    with pytest.raises(ValueError, match="filter_policy"):
        audit_recipe_inputs(recipe, store_root=tmp_path)


def test_six_packaged_templates_are_single_jobs_and_not_ready_inputs():
    root = resources.files("agentic_arxiv").joinpath("resources", "experiments", "recipes")
    expected = {"M0": "evaluate", "M1": "sft", "M2": "dpo", "M3": "grpo", "M4": "grpo", "C1": "grpo"}
    templates = {name: json.loads(root.joinpath(name + ".json").read_text()) for name in expected}
    for name, recipe in templates.items():
        assert recipe["stage"] == expected[name]
        assert not {"jobs", "queue", "pipeline", "depends_on"} & recipe.keys()
        assert "smoke" not in json.dumps(recipe).lower()
        assert recipe["budget"]["max_steps"] is None
        assert recipe["budget"]["epochs"] is None
        if name == "M0":
            assert recipe["checkpoint"] is None
        else:
            assert recipe["checkpoint"]["save_steps"] == (10 if expected[name] == "grpo" else 50)
    common = templates["M2"]["inputs"]["source_sft"]
    for name in ("M3", "M4", "C1"):
        assert templates[name]["inputs"]["source_sft"] == common
    assert templates["M4"]["inputs"]["model"] != common
