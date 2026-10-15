"""纯控制元数据和接线回归；不构造训练数据或证明真实模型恢复。"""

import ast
from contextlib import nullcontext
from dataclasses import replace
import inspect
import json
import math
import shlex
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agentic_arxiv.cli import build_parser, parse_args
from agentic_arxiv.provenance import file_fingerprint, model_input_fingerprint
from agentic_arxiv.training.checkpoint import ManagedCheckpointContext
from agentic_arxiv.training.managed import (
    attach_managed_training, prepare_managed_training, record_training_result,
    validate_managed_config,
)


def _context(tmp_path, *, stage="sft", attempt="attempt-2"):
    source = tmp_path / "input-identity.json"
    source.write_text('{"schema_version": 1}\n')
    output = tmp_path / attempt / "output"
    data = {"schema_version": 1, "stage": stage, "run_id": "run-1", "attempt_id": attempt,
            "total_budget": 10, "output_dir": str(output),
            "control_dir": str(output.parent / "control"), "log_dir": str(output.parent / "logs"),
            "identity": {"inputs": {"data": {"path": str(source), "fingerprint": file_fingerprint(source)}},
                         "training_config": {"defaults": {"max_steps": 10, "lr": 0.01, "seed": 42},
                                             "trainer_parameters": {"save_steps": 5}},
                         "runtime": {"image": "test-control-identity"},
                         "models": {"policy": {"path": str(source), "fingerprint": file_fingerprint(source)}},
                         "devices": ["GPU-control-metadata"]}}
    path = tmp_path / "managed-context.json"
    path.write_text(json.dumps(data))
    return path, data


@pytest.mark.parametrize("stage", ["sft", "dpo", "grpo"])
def test_training_cli_explicit_context_and_resume_paths_are_workspace_scoped(stage, tmp_path):
    _, args = parse_args(["train", stage, "--workspace", str(tmp_path),
                          "--managed-context", "context.json", "--resume-from-checkpoint", "old/checkpoint-5"])
    assert args.managed_context == str(tmp_path / "context.json")
    assert args.resume_from_checkpoint == str(tmp_path / "old/checkpoint-5")


def test_documented_training_examples_use_only_registered_parameters():
    text = (Path(__file__).resolve().parents[1] / "docs/sft-dpo-managed-resume.md").read_text()
    examples = [line for line in text.splitlines() if line.startswith("agentic-arxiv train ")]
    assert len(examples) == 2
    for line in examples:
        args = build_parser().parse_args(shlex.split(line)[1:])
        assert args.managed_context and args.resume_from_checkpoint


@pytest.mark.parametrize("stage", ["sft", "dpo"])
def test_old_stage_defaults_need_no_managed_fields_and_new_resume_is_not_weight_reload(stage):
    _, args = parse_args(["train", stage])
    assert args.managed_context is None and args.resume_from_checkpoint is None
    with pytest.raises(ValueError, match="仅重载模型不是续训"):
        prepare_managed_training(None, stage=stage, output_dir="unused", resume_from_checkpoint="old")


def test_context_is_stage_bound_and_new_attempt_cannot_overwrite_old_output(tmp_path):
    path, data = _context(tmp_path)
    with pytest.raises(ValueError, match="阶段"):
        prepare_managed_training(path, stage="dpo", output_dir=data["output_dir"])
    output = Path(data["output_dir"])
    output.mkdir(parents=True)
    (output / "run_manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="拒绝覆盖"):
        prepare_managed_training(path, stage="sft", output_dir=output)


def test_resume_output_must_be_new_attempt_and_source_stays_read_only(tmp_path):
    path, data = _context(tmp_path)
    old = tmp_path / "attempt-1" / "output" / "checkpoint-5"
    old.mkdir(parents=True)
    marker = old / "diagnostic.json"
    marker.write_text("{}")
    before = marker.read_bytes()
    context = prepare_managed_training(path, stage="sft", output_dir=data["output_dir"],
                                       resume_from_checkpoint=old)
    assert context.attempt_id == "attempt-2" and marker.read_bytes() == before
    data["output_dir"] = str(old.parent / "new-nested-output")
    data["control_dir"] = str(old.parent / "new-control")
    data["log_dir"] = str(old.parent / "new-logs")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="原尝试"):
        prepare_managed_training(path, stage="sft", output_dir=data["output_dir"], resume_from_checkpoint=old)


def test_changed_disk_inputs_fail_without_updating_old_identity(tmp_path):
    path, data = _context(tmp_path)
    before = path.read_bytes()
    source = Path(data["identity"]["inputs"]["data"]["path"])
    source.write_text('{"schema_version": 2}\n')
    with pytest.raises(ValueError, match="指纹已改变"):
        prepare_managed_training(path, stage="sft", output_dir=data["output_dir"])
    assert path.read_bytes() == before


@pytest.mark.parametrize("changed", [None, "model", "base", "policy"])
def test_managed_audit_checks_runtime_inputs_not_historical_provenance(tmp_path, changed):
    from agentic_arxiv.training.managed import _validate_input_fingerprints

    records = {}
    for name in ("model", "base", "policy"):
        path = tmp_path / (name + ".json")
        path.write_text("{}")
        records[name] = {"path": str(path), "fingerprint": file_fingerprint(path)}
    historical = {"path": "artifacts/mainline_baseline/source_configuration.json",
                  "status": "known", "sha256": "old-source-hash"}
    records["model"].update(base_model=records["base"], metadata={
        "source_baseline": {"configuration_baseline": historical}},
        lineage={"source_manifest": historical})
    audit = {"inputs": {"model": records["model"]}, "policies": {"task_pool": records["policy"]},
             "comparison": {"path": "/unmounted/old-run.json", "fingerprint": historical},
             "common_sft": {"manifest": historical}}
    if changed:
        Path(records[changed]["path"]).write_text('{"changed": true}')
        with pytest.raises(ValueError, match="指纹已改变"):
            _validate_input_fingerprints(audit)
    else:
        _validate_input_fingerprints(audit)
        _validate_input_fingerprints({"policy": records["model"]})


@pytest.mark.parametrize("changed", [None, "source_model_manifest", "split_file", "traces", "report"])
def test_managed_dpo_rechecks_package_dependency_fingerprints(tmp_path, changed):
    path, data = _context(tmp_path, stage="dpo")
    dependencies = {}
    for name in ("source_model_manifest", "split_file", "traces", "report"):
        target = tmp_path / (name + ".json")
        target.write_text(json.dumps({"control_metadata": name}))
        dependencies[name] = {"path": str(target), "fingerprint": file_fingerprint(target)}
    data["identity"]["inputs"]["data"]["runtime_dependencies"] = dependencies
    path.write_text(json.dumps(data))
    before = path.read_bytes()
    if changed:
        Path(dependencies[changed]["path"]).write_text('{"changed": true}')
        with pytest.raises(ValueError, match="指纹已改变"):
            prepare_managed_training(path, stage="dpo", output_dir=data["output_dir"])
    else:
        assert prepare_managed_training(path, stage="dpo", output_dir=data["output_dir"]).stage == "dpo"
    assert path.read_bytes() == before


def test_managed_dependency_cannot_overlap_writable_output(tmp_path):
    from agentic_arxiv.training.managed import _validate_input_fingerprints

    path, data = _context(tmp_path, stage="dpo")
    output = Path(data["output_dir"])
    output.mkdir(parents=True)
    dependency = output / "traces.jsonl"
    dependency.write_text("{}")
    identities = data["identity"]["inputs"]
    identities["data"]["runtime_dependencies"] = {
        "traces": {"path": str(dependency), "fingerprint": file_fingerprint(dependency)}}
    with pytest.raises(ValueError, match="输出不能与只读输入重叠"):
        _validate_input_fingerprints(identities, output_dir=output)


@pytest.mark.parametrize("field,value", [("max_steps", 20), ("lr", 0.02), ("seed", 100)])
def test_actual_training_configuration_cannot_reset_original_budget_or_schedule(tmp_path, field, value):
    path, data = _context(tmp_path)
    context = ManagedCheckpointContext.from_file(path)
    requested = {"max_steps": 10, "lr": 0.01, "seed": 42, "output_dir": "new-attempt"}
    configured = {"trainer_parameters": {"save_steps": 5}}
    validate_managed_config(context, requested, configured)
    requested[field] = value
    with pytest.raises(ValueError, match=field):
        validate_managed_config(context, requested, configured)


@pytest.mark.parametrize("actual_budget", [None, 12])
def test_epoch_budget_preserves_original_epoch_mode_on_start_and_resume(tmp_path, actual_budget):
    path, data = _context(tmp_path)
    data.update(total_budget=actual_budget, epoch_budget=3)
    expected = data["identity"]["training_config"]
    expected["defaults"].update(max_steps=-1, epochs=3)
    path.write_text(json.dumps(data))
    context = ManagedCheckpointContext.from_file(path)
    requested = {**expected["defaults"], "output_dir": "new-attempt"}
    configured = {"trainer_parameters": expected["trainer_parameters"]}
    before = dict(requested)
    validate_managed_config(context, requested, configured)
    assert requested == before and requested["max_steps"] == -1
    requested["max_steps"] = 12
    with pytest.raises(ValueError, match="max_steps"):
        validate_managed_config(context, requested, configured)
    requested.update(max_steps=-1, epochs=4)
    with pytest.raises(ValueError, match="epochs"):
        validate_managed_config(context, requested, configured)


@pytest.mark.parametrize("planned", [None, 12])
def test_epoch_result_budget_is_unknown_until_trainer_computes_it(tmp_path, planned):
    path, data = _context(tmp_path)
    data.update(total_budget=None, epoch_budget=3)
    data["identity"]["training_config"]["defaults"].update(max_steps=-1, epochs=3)
    path.write_text(json.dumps(data))
    context = ManagedCheckpointContext.from_file(path)
    record_training_result(context, SimpleNamespace(is_main=True), training="failed",
                           validation="not_run", global_step=0, planned_steps=planned)
    saved = json.loads((Path(context.control_dir) / "training-result.json").read_text())
    assert saved["planned_steps"] == planned and saved["total_budget"] == planned
    assert saved["epoch_budget"] == 3


@pytest.mark.train
def test_epoch_budget_uses_real_pinned_trainer_sized_loader_interface_without_model(tmp_path):
    transformers = pytest.importorskip("transformers")

    class SizedOnly:
        def __init__(self, length):
            self.length = length

        def __len__(self):
            return self.length

    loader = SizedOnly(7)
    loader.dataset = SizedOnly(13)
    args = transformers.TrainingArguments(
        output_dir=str(tmp_path / "output"), max_steps=-1, num_train_epochs=3,
        gradient_accumulation_steps=2, per_device_train_batch_size=1,
        use_cpu=True, report_to=[],
    )
    # 绕过模型初始化，只调用固定 Trainer 的真实预算计算接口；不遍历样本。
    trainer = transformers.Trainer.__new__(transformers.Trainer)
    before = (args.max_steps, args.num_train_epochs, args.lr_scheduler_type, args.ignore_data_skip)
    values = trainer.set_initial_training_values(args, loader, total_train_batch_size=2)
    assert values[-1] == 12 and values[1] == 4 and values[4] is True
    assert (args.max_steps, args.num_train_epochs, args.lr_scheduler_type, args.ignore_data_skip) == before

    from agentic_arxiv.training.checkpoint import ManagedCheckpointAdapter
    from agentic_arxiv.training.distributed import TrainingContext
    path, data = _context(tmp_path)
    data.update(total_budget=None, epoch_budget=3)
    data["identity"]["training_config"]["defaults"].update(max_steps=-1, epochs=3)
    path.write_text(json.dumps(data))
    context = ManagedCheckpointContext.from_file(path)
    state = transformers.TrainerState(max_steps=values[-1], global_step=5, epoch=1.25)
    adapter = ManagedCheckpointAdapter(trainer, context=context, training_context=TrainingContext(1))
    adapter.bind_budget(args=args, state=state)
    assert context.total_budget is None and adapter.context.total_budget == 12
    assert state.global_step == 5 and state.epoch == 1.25
    resolved = json.loads((Path(context.control_dir) / "resolved-budget.json").read_text())
    assert resolved["total_budget"] == 12 and resolved["epoch_budget"] == 3

    resumed = replace(adapter.context, attempt_id="attempt-3", control_dir=str(tmp_path / "resumed-control"))
    resumed_adapter = ManagedCheckpointAdapter(trainer, context=resumed, training_context=TrainingContext(1))
    resumed_adapter.bind_budget(args=args, state=state)
    loader.length = 9
    changed = trainer.set_initial_training_values(args, loader, total_train_batch_size=2)
    with pytest.raises(ValueError, match="核定总步数与原预算不一致"):
        resumed_adapter.bind_budget(args=args, state=transformers.TrainerState(max_steps=changed[-1]))
    assert (args.max_steps, args.num_train_epochs, args.lr_scheduler_type, args.ignore_data_skip) == before


def test_reference_identity_checks_original_source_fingerprint_not_checkpoint_policy(tmp_path):
    path, data = _context(tmp_path, stage="dpo")
    source = tmp_path / "original-source-identity"
    source.mkdir()
    # 空目录指纹仅测试API的数据形状；不伪装成合格模型或恢复证据。
    fingerprint = model_input_fingerprint(source)
    data["identity"]["models"] = {"reference": {"path": str(source), "fingerprint": fingerprint}}
    path.write_text(json.dumps(data))
    context = ManagedCheckpointContext.from_file(path)
    context.validate_reference_identity(str(source))
    with pytest.raises(ValueError, match="参考"):
        context.validate_reference_identity(str(tmp_path / "checkpoint-policy"))


def test_managed_attach_restores_before_train_and_uses_committed_checkpoint(tmp_path, monkeypatch):
    path, _ = _context(tmp_path)
    context = ManagedCheckpointContext.from_file(path)
    from agentic_arxiv.training import checkpoint, callbacks
    adapter = SimpleNamespace(last_checkpoint=str(tmp_path / "checkpoint-5"))
    attach = Mock(return_value=adapter)
    callback_factory = Mock(return_value=object())
    monkeypatch.setattr(checkpoint, "attach_managed_checkpoint", attach)
    monkeypatch.setattr(callbacks, "ManagedStopCallback", callback_factory)
    trainer = SimpleNamespace(add_callback=Mock())
    state = {"schema_version": 1, "logical_step": 5}
    restore = Mock()
    callback, result = attach_managed_training(
        trainer, context, object(), resume_from_checkpoint="old/checkpoint-5",
        state_provider=lambda: state, state_restorer=restore,
        protection_reason=lambda: "canary_stopped",
    )
    assert result is adapter and callback is callback_factory.return_value
    assert attach.call_args.kwargs["resume_from_checkpoint"] == "old/checkpoint-5"
    assert attach.call_args.kwargs["state_restorer"] is restore
    assert attach.call_args.kwargs["state_provider"]() == state
    assert trainer._managed_termination_reason == "canary_stopped"
    assert callback_factory.call_args.kwargs["checkpoint_committed"](5) == adapter.last_checkpoint
    trainer.add_callback.assert_called_once_with(callback)


def test_training_result_distinguishes_safe_stop_from_budget_and_rank_writes(tmp_path):
    path, _ = _context(tmp_path)
    context = ManagedCheckpointContext.from_file(path)
    record_training_result(context, SimpleNamespace(is_main=False), training="user_safe_stop",
                           validation="not_run", global_step=5)
    result = Path(context.control_dir) / "training-result.json"
    assert not result.exists()
    record_training_result(context, SimpleNamespace(is_main=True), training="user_safe_stop",
                           validation="not_run", global_step=5, reason="user_safe_stop",
                           checkpoint=tmp_path / "checkpoint-5")
    saved = json.loads(result.read_text())
    assert saved["training"] == "user_safe_stop" and saved["validation"] == "not_run"
    assert saved["run_id"] == "run-1" and saved["attempt_id"] == "attempt-2"


@pytest.mark.parametrize("stage", ["sft", "dpo"])
def test_resume_is_trainer_state_resume_and_not_replacement_model_or_reset_budget(stage):
    from agentic_arxiv.training import dpo, sft
    main = (sft if stage == "sft" else dpo).main
    tree = ast.parse(inspect.getsource(main))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    restored_train = [node for node in calls if isinstance(node.func, ast.Attribute)
                      and node.func.attr == "train" and node.keywords]
    assert len(restored_train) == 1
    assert restored_train[0].keywords[0].arg == "resume_from_checkpoint"
    if stage == "dpo":
        identity_check = next(node for node in calls if isinstance(node.func, ast.Attribute)
                              and node.func.attr == "validate_reference_identity")
        assert isinstance(identity_check.args[0], ast.Name) and identity_check.args[0].id == "resolved"
        reference = next(node for node in calls if isinstance(node.func, ast.Name)
                         and node.func.id == "load_reference_policy")
        assert isinstance(reference.args[0], ast.Name) and reference.args[0].id == "resolved"
        audit = next(node for node in calls if isinstance(node.func, ast.Name)
                     and node.func.id == "load_dpo_package")
        assert any(item.arg == "expected_source_model" and isinstance(item.value, ast.Name)
                   and item.value.id == "source_model" for item in audit.keywords)


@pytest.mark.train
def test_reference_remains_frozen_independent_and_excluded_from_policy_optimizer():
    torch = pytest.importorskip("torch")
    from agentic_arxiv.training.dpo import _assert_independent_reference

    policy, reference = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
    reference.requires_grad_(False).eval()
    optimizer = SimpleNamespace(param_groups=[{"params": list(policy.parameters())}])
    _assert_independent_reference(policy, reference, optimizer)
    optimizer.param_groups[0]["params"].append(next(reference.parameters()))
    with pytest.raises(ValueError, match="优化器"):
        _assert_independent_reference(policy, reference, optimizer)
    reference.requires_grad_(True)
    with pytest.raises(ValueError, match="冻结"):
        _assert_independent_reference(policy, reference)
    reference.requires_grad_(False)
    reference.weight = policy.weight
    with pytest.raises(ValueError, match="冻结|共享"):
        _assert_independent_reference(policy, reference)


@pytest.mark.parametrize("actual_steps,user_stopped,expected", [
    (5, True, "user_safe_stop"), (5, False, "trainer_stopped_early"),
    (10, True, "max_steps_or_epochs_completed"),
])
@pytest.mark.parametrize("epoch_mode", [False, True])
def test_dpo_existing_training_completion_block_preserves_budget_and_stop_priority(
        actual_steps, user_stopped, expected, epoch_mode, tmp_path):
    """执行真实入口的控制流片段；不加载模型、不构造偏好样本。"""
    from agentic_arxiv.training import dpo
    from agentic_arxiv.config import workspace_path

    main = ast.parse(inspect.getsource(dpo.main)).body[0]
    start = next(index for index, node in enumerate(main.body) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "training_started" for target in node.targets))
    end = next(index for index, node in enumerate(main.body) if isinstance(node, ast.Expr)
               and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
               and node.value.func.attr == "save_model")
    projection = ast.FunctionDef(name="completion", args=ast.arguments(
        posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=main.body[start:end], decorator_list=[])
    program = ast.fix_missing_locations(ast.Module(body=[projection], type_ignores=[]))
    trainer = SimpleNamespace(state=SimpleNamespace(global_step=actual_steps, max_steps=10),
                              model=object(), ref_model=object(), optimizer=None,
                              train=Mock(return_value=SimpleNamespace(training_loss=0.25)))
    reasons, results = [], []
    execution = SimpleNamespace(is_main=True, memory_summary=lambda: {}, run_on_main=lambda operation: operation())
    namespace = {"time": time, "math": math, "workspace_path": workspace_path,
                 "trainer": trainer, "resume_from_checkpoint": str(tmp_path / "old/checkpoint-5"),
                 "max_steps": -1 if epoch_mode else 10, "stop_control": SimpleNamespace(
                     user_stopped=user_stopped, checkpoint="old/checkpoint-5", mark_failed=Mock()),
                 "managed": object(), "execution": execution, "verification": {"status": "pending"},
                 "preserve_managed_dataloader_epoch": lambda *args, **kwargs: nullcontext(),
                 "_assert_independent_reference": Mock(), "save_run": lambda reason: reasons.append(reason),
                 "record_training_result": lambda *args, **kwargs: results.append(kwargs)}
    exec(compile(program, "<DPO-entrypoint-control-flow>", "exec"), namespace)
    namespace["completion"]()
    trainer.train.assert_called_once_with(resume_from_checkpoint=str(tmp_path / "old/checkpoint-5"))
    assert reasons == [expected] and namespace["max_steps"] == (-1 if epoch_mode else 10)
    if actual_steps < 10:
        assert results[-1]["validation"] == "not_run"
        assert results[-1]["planned_steps"] == 10
        assert results[-1]["training"] == ("user_safe_stop" if user_stopped else "protection_stopped")
