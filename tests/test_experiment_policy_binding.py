"""校验加载接线；权重仅用既有存在性替身，不启动模型、训练或容器。"""

import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agentic_arxiv.experiments.artifacts import resolve_model_reference
from agentic_arxiv.experiments.paths import mounts
from agentic_arxiv.experiments.preflight import _configuration
from agentic_arxiv.experiments.service import ExperimentService
from agentic_arxiv.experiments.store import atomic_json
from agentic_arxiv.provenance import file_fingerprint, read_protocol_metadata, sha256_file
from agentic_arxiv.runtime.react_contract import protocol_metadata
from agentic_arxiv.training.mainline_config import stage_config


@pytest.fixture(autouse=True)
def preserve_workspace():
    from agentic_arxiv.config import get_workspace, initialize_operation

    previous = get_workspace()
    yield
    initialize_operation(previous, load_env=False, logging=False)


def _full_files(path, *, config=None):
    path.mkdir()
    atomic_json(path / "config.json", config or {"control_metadata": "not-a-model"})
    (path / "model.safetensors").touch()
    return path


def _artifact(path, stage, base, *, sft_source=None):
    if stage.endswith("_qlora"):
        path.mkdir()
        atomic_json(path / "adapter_config.json", {"base_model_name_or_path": str(base)})
        (path / "adapter_model.safetensors").touch()
    else:
        _full_files(path)
    metadata = {"stage": stage, "base_model": str(base),
                "protocol_metadata": {**protocol_metadata(), "chat_template_hash": "declared-template"}}
    if sft_source:
        metadata.update(sft_source=sft_source, reference_source=copy.deepcopy(sft_source))
    atomic_json(path / "training_manifest.json", metadata)
    atomic_json(path / "verification_report.json", {
        "passed": True, "stage": stage.split("_", 1)[0], "model_path": str(path),
        "protocol": {"training": read_protocol_metadata(metadata)},
    })
    return path


@pytest.fixture
def binding_runtime(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    profile = dict(workspace=str(tmp_path), run_root=str(tmp_path / "runs"), input_roots=[str(inputs)],
                   image="local/train@sha256:" + "a" * 64, uid=os.getuid(), gid=os.getgid(),
                   gpus=["0"], shm_size="1g")
    service = ExperimentService(profile)
    observed = {"preflights": [], "launches": []}
    monkeypatch.setattr(service.runtime, "image", lambda *args: {"Id": "verified-image"})
    monkeypatch.setattr(service.runtime, "resolve_gpus", lambda *args: ["GPU-control"])
    monkeypatch.setattr(service.runtime, "check_idle", lambda *args: None)

    def preflight(argv, **kwargs):
        payload = json.loads(kwargs["input_text"])
        stage, training = _configuration(payload)
        observed["preflights"].append({
            "payload": payload, "training": training,
            "mounts": [argv[i + 1] for i, value in enumerate(argv) if value == "--mount"],
        })
        return json.dumps({"schema_version": 1, "status": "passed", "stage": stage})

    def launch(run, attempt, command, paths, *, extra_readonly=()):
        directory = service.store.attempt_dir(run["run_id"], attempt["attempt_id"])
        observed["launches"].append({
            "attempt": attempt, "command": command, "paths": paths,
            "mounts": mounts(paths, directory, extra_readonly=extra_readonly),
            "context": json.loads((directory / "control" / "managed-context.json").read_text()),
        })
        return {"started": True, "run_id": run["run_id"], "attempt_id": attempt["attempt_id"]}

    monkeypatch.setattr(service.runtime, "call", preflight)
    monkeypatch.setattr(service, "launch_job", launch)
    return service, inputs, observed


def _case(service, inputs, group, source_kind):
    stage = "dpo" if group == "M2" else "grpo"
    base = _full_files(inputs / "base")
    sft = _artifact(inputs / "sft", f"sft_{source_kind}", base)
    model = sft
    if group == "M4":
        source = resolve_model_reference(str(sft), store_root=service.store.root, expected_stage="sft")
        identity = {"model": source["fingerprint"], "stage": source["metadata"]["stage"],
                    "manifest": file_fingerprint(sft / "training_manifest.json")}
        model = _artifact(inputs / "dpo", f"dpo_{source_kind}", base, sft_source=identity)
    control = inputs / "control-metadata.json"
    atomic_json(control, {"control_metadata": "not-research-data"})
    recipe = {"group": group, "stage": stage, "budget": {"max_steps": 10},
              "checkpoint": {"save_steps": 2},
              "inputs": {"model": str(model), "source_sft": str(sft), "base_model": str(base)}}
    training = stage_config(stage)
    training["defaults"].update(max_steps=10, base_model="unused-old-config-base")
    if group == "M2":
        manifest = inputs / "control-package.json"
        record = {"path": str(control), "sha256": sha256_file(control)}
        atomic_json(manifest, {"source_model_manifest": record, "split_file": record,
                              "files": {"data": record, "traces": record}, "report": record})
        recipe["inputs"].update(data=str(control), data_manifest=str(manifest), snapshot=str(control))
    else:
        recipe["inputs"].update(split=str(control), snapshot=str(control))
        training["defaults"]["split"] = str(control) + ":rl_train"
        policy = {"path": str(control), "fingerprint": sha256_file(control)}
        recipe.update(task_pool=policy, filter_policy=copy.deepcopy(policy))
    if group == "C1":
        comparison = inputs / "control-comparison.json"
        atomic_json(comparison, {"group": "M4", "inputs": {"source_sft": str(sft)},
                                 "task_pool": recipe["task_pool"], "filter_policy": recipe["filter_policy"]})
        recipe["budget"].update(comparison_group="M4", comparison_reference=str(comparison))
    return {"recipe": recipe, "training": training}


def _parsed_launch(launch):
    from agentic_arxiv.cli import parse_args
    from agentic_arxiv.training.checkpoint import ManagedCheckpointContext
    from agentic_arxiv.training.managed import validate_managed_config

    _, args = parse_args(launch["command"][3:])
    context = ManagedCheckpointContext.from_file(args.managed_context)
    configured = stage_config(context.stage, args.config_path)
    validate_managed_config(context, vars(args), configured)
    return args, context, configured


def _assert_loader_source(defaults, declared_base, monkeypatch):
    from agentic_arxiv.training import model_setup

    class LoadingBoundary(Exception):
        pass

    calls = []

    def load(source, **kwargs):
        calls.append((source, kwargs))
        raise LoadingBoundary

    # 执行真实产物校验和全参 guard，只在设备/权重加载边界使用替身。
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "torch", SimpleNamespace(
            bfloat16=object(), cuda=SimpleNamespace(empty_cache=lambda: None, reset_peak_memory_stats=lambda: None)))
        patch.setitem(sys.modules, "transformers", SimpleNamespace(
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=load)))
        patch.setitem(sys.modules, "peft", SimpleNamespace(
            PeftModel=object(), get_peft_model=object(), prepare_model_for_kbit_training=object()))
        patch.setattr(model_setup.TrainingContext, "prepare_device", lambda self, **kwargs: None)
        patch.setattr(model_setup, "_qlora_runtime_guard", lambda: None)
        patch.setattr(model_setup, "_build_qlora_configs", lambda *args: (object(), object()))
        patch.setattr(model_setup, "quantized_model_device_map", lambda: {"": 0})
        arguments = {key: defaults[key] for key in ("qlora", "base_model", "lora_r", "lora_alpha", "lora_dropout")}
        for loader in (model_setup.load_training_policy, model_setup.load_reference_policy):
            if defaults["base_model"] is None:
                with pytest.raises(SystemExit, match="全参 SFT 产物不能"):
                    loader(defaults["model"], **{**arguments, "base_model": declared_base}, local_base_only=True)
            with pytest.raises(LoadingBoundary):
                loader(defaults["model"], **arguments, local_base_only=True, expected_protocol=protocol_metadata())
    expected = defaults["base_model"] or defaults["model"]
    assert len(calls) == 2 and all(source == expected and kwargs["local_files_only"] for source, kwargs in calls)


@pytest.mark.parametrize("group,model_reference", [
    ("M2", "model"), ("M3", "model"), ("M4", "model"),
    ("M4", "source_dpo"), ("M4", "both"), ("C1", "model"),
])
@pytest.mark.parametrize("source_kind", ["full", "qlora"])
@pytest.mark.parametrize("requested_qlora", [False, True])
def test_policy_binding_matches_preflight_launch_and_resume(binding_runtime, monkeypatch, group, model_reference, source_kind, requested_qlora):
    import agentic_arxiv.training.checkpoint as checkpoint_module

    service, inputs, observed = binding_runtime
    resolved = _case(service, inputs, group, source_kind)
    expected_model = resolved["recipe"]["inputs"]["model"]
    if model_reference != "model":
        resolved["recipe"]["inputs"]["source_dpo"] = {
            "kind": "path", "path": str(Path(expected_model).relative_to(service.workspace)),
        }
        if model_reference == "source_dpo":
            del resolved["recipe"]["inputs"]["model"]
    resolved["training"]["defaults"]["qlora"] = requested_qlora
    original = copy.deepcopy(resolved)
    input_bytes = {path: path.read_bytes() for path in inputs.rglob("*") if path.is_file()}
    started = service.start(resolved)
    assert started["started"] and resolved == original
    first = observed["launches"][0]
    args, context, configured = _parsed_launch(first)
    policy = context.identity["models"]["policy"]
    declared_base = policy["base_model"]["path"]
    expected_base = declared_base if source_kind == "qlora" else None
    assert args.base_model == configured["defaults"]["base_model"] == expected_base
    assert args.qlora is configured["defaults"]["qlora"] is requested_qlora
    assert args.model == configured["defaults"]["model"] == policy["path"] == expected_model
    assert context.identity["models"]["base"] == policy["base_model"]
    assert context.identity["models"]["reference"] == policy
    assert context.identity["inputs"]["inputs"]["base_model"]["path"] == declared_base
    preflight = observed["preflights"][0]
    assert "base_model" not in preflight["payload"]["inputs"]
    assert preflight["training"]["defaults"]["base_model"] == expected_base
    assert preflight["training"]["defaults"]["model"] == expected_model
    assert preflight["training"]["defaults"]["qlora"] is requested_qlora
    assert f"type=bind,src={declared_base},dst={declared_base},readonly" in preflight["mounts"]
    assert {"source": declared_base, "target": declared_base, "readonly": True} in first["mounts"]
    assert {"source": policy["path"], "target": policy["path"], "readonly": True} in first["mounts"]
    _assert_loader_source(configured["defaults"], declared_base, monkeypatch)

    directory = service.store.attempt_dir(started["run_id"], started["attempt_id"])
    old_config = (directory / "training-config.json").read_bytes()
    old_context = (directory / "control" / "managed-context.json").read_bytes()
    checkpoint = directory / "output" / "checkpoint-2"
    checkpoint.mkdir()
    service.store.update_attempt(started["run_id"], started["attempt_id"], {"execution": "create_failed", "training": "failed"})
    service._release(first["attempt"])
    validated = []

    def validate(path, *, expected):
        assert path == checkpoint and expected["identity"] == context.identity
        validated.append(expected)
        return {"global_step": 2, "total_budget": 10, "attempt_id": started["attempt_id"]}

    monkeypatch.setattr(checkpoint_module, "validate_checkpoint", validate)
    resumed = service.resume(started["run_id"], checkpoint=str(checkpoint))
    assert resumed["started"] and resumed["attempt_id"] != started["attempt_id"]
    second = observed["launches"][1]
    resumed_args, resumed_context, resumed_config = _parsed_launch(second)
    assert len(validated) == 1 and resumed_context.identity == context.identity
    assert resumed_args.resume_from_checkpoint == str(checkpoint)
    assert resumed_config["defaults"]["base_model"] == expected_base
    assert resumed_config["defaults"]["model"] == policy["path"]
    assert observed["preflights"][1] == preflight
    assert {"source": str(checkpoint), "target": str(checkpoint), "readonly": True} in second["mounts"]
    assert (directory / "training-config.json").read_bytes() == old_config
    assert (directory / "control" / "managed-context.json").read_bytes() == old_context
    assert {path: path.read_bytes() for path in input_bytes} == input_bytes


@pytest.mark.parametrize("source_kind", ["full", "qlora"])
@pytest.mark.parametrize("operation", ["check", "start"])
def test_m4_conflicting_model_references_fail_before_preflight_or_attempt(binding_runtime, monkeypatch, source_kind, operation):
    service, inputs, observed = binding_runtime
    resolved = _case(service, inputs, "M4", source_kind)
    metadata = json.loads((inputs / "dpo/training_manifest.json").read_text())
    other = _artifact(inputs / "other-dpo", f"dpo_{source_kind}", inputs / "base",
                      sft_source=metadata["sft_source"])
    resolved["recipe"]["inputs"]["source_dpo"] = str(other)
    before = {path: path.read_bytes() for path in inputs.rglob("*") if path.is_file()}
    image = Mock(return_value={"Id": "verified-image"})
    monkeypatch.setattr(service.runtime, "image", image)
    with pytest.raises(ValueError, match="M4.*model.*source_dpo.*同一"):
        getattr(service, operation)(resolved)
    image.assert_not_called()
    assert not observed["preflights"] and not observed["launches"]
    assert not service.store.root.exists()
    assert {path: path.read_bytes() for path in inputs.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("group", ["M2", "M3", "M4", "C1"])
def test_full_input_discards_stale_configured_base_without_explicit_input(binding_runtime, group):
    service, inputs, observed = binding_runtime
    resolved = _case(service, inputs, group, "full")
    del resolved["recipe"]["inputs"]["base_model"]
    assert service.start(resolved)["started"]
    args, context, configured = _parsed_launch(observed["launches"][0])
    assert args.base_model is configured["defaults"]["base_model"] is None
    assert observed["preflights"][0]["training"]["defaults"]["base_model"] is None
    assert context.identity["models"]["base"]["path"] == str(inputs / "base")


@pytest.mark.parametrize("origin", ["training", "inputs"])
def test_conflicting_adapter_base_is_not_hidden_by_normalization(binding_runtime, origin):
    service, inputs, observed = binding_runtime
    resolved = _case(service, inputs, "M2", "qlora")
    other = _full_files(inputs / "other-base", config={"different": "base"})
    if origin == "training":
        del resolved["recipe"]["inputs"]["base_model"]
        resolved["training"]["defaults"]["base_model"] = str(other.relative_to(service.workspace))
    else:
        resolved["recipe"]["inputs"]["base_model"] = str(other)
    with pytest.raises(ValueError, match="显式基座.*本地内容不一致"):
        service.start(resolved)
    assert not observed["preflights"] and not observed["launches"]
    assert not service.store.list_runs()


@pytest.mark.parametrize("origin", ["training", "inputs"])
def test_same_content_adapter_base_uses_existing_override_contract(binding_runtime, origin):
    service, inputs, observed = binding_runtime
    resolved = _case(service, inputs, "M2", "qlora")
    alternate = _full_files(inputs / "same-base")
    if origin == "training":
        del resolved["recipe"]["inputs"]["base_model"]
        resolved["training"]["defaults"]["base_model"] = str(alternate.relative_to(service.workspace))
    else:
        resolved["recipe"]["inputs"]["base_model"] = str(alternate)
    assert service.start(resolved)["started"]
    args, context, _ = _parsed_launch(observed["launches"][0])
    assert args.base_model == str(inputs / "base")
    assert observed["preflights"][0]["training"]["defaults"]["base_model"] == args.base_model
    if origin == "inputs":
        assert context.identity["inputs"]["inputs"]["base_model"]["path"] == str(alternate)
        assert {"source": str(alternate), "target": str(alternate), "readonly": True} in observed["launches"][0]["mounts"]


@pytest.mark.parametrize("changed_input", ["model", "base_model"])
def test_m1_explicit_base_remains_audited_and_readonly(binding_runtime, changed_input):
    from agentic_arxiv.training.managed import prepare_managed_training

    service, inputs, observed = binding_runtime
    model = _full_files(inputs / "model")
    base = _full_files(inputs / "explicit-base")
    control = inputs / "control-metadata.json"
    atomic_json(control, {"control_metadata": "not-research-data"})
    recipe = {"group": "M1", "stage": "sft", "budget": {"max_steps": 10}, "checkpoint": {"save_steps": 2},
              "inputs": {"model": str(model), "base_model": str(base), "data": str(control),
                         "data_manifest": str(control), "snapshot": str(control)}}
    training = stage_config("sft")
    training["defaults"]["max_steps"] = 10
    assert service.start({"recipe": recipe, "training": training})["started"]
    launched = observed["launches"][0]
    args, context, configured = _parsed_launch(launched)
    assert "base_model" not in configured["defaults"]
    assert "base_model" not in observed["preflights"][0]["training"]["defaults"]
    for path in (model, base):
        assert {"source": str(path), "target": str(path), "readonly": True} in launched["mounts"]
        assert f"type=bind,src={path},dst={path},readonly" in observed["preflights"][0]["mounts"]
    assert context.identity["models"]["policy"]["path"] == str(model)
    assert context.identity["inputs"]["inputs"]["base_model"]["path"] == str(base)
    prepare_managed_training(args.managed_context, stage="sft", output_dir=args.output_dir)
    changed = Path(recipe["inputs"][changed_input])
    atomic_json(changed / "config.json", {"changed": True})
    with pytest.raises(ValueError, match="受管输入指纹已改变"):
        prepare_managed_training(args.managed_context, stage="sft", output_dir=args.output_dir)
