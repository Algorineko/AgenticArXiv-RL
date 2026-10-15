"""用薄训练器验证 SFT/GRPO 入口的 writer 边界，不加载模型或执行训练。"""

from collections import defaultdict
from contextlib import nullcontext
import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agentic_arxiv.runtime.react_contract import protocol_metadata
from agentic_arxiv.training.distributed import TrainingContext


def _run_stage(stage, rank, tmp_path, monkeypatch, *, fail_training=False,
               actual_steps=2, user_stopped=False, resume=None, observed=None, epoch_mode=False):
    torch = pytest.importorskip("torch")
    trl = pytest.importorskip("trl")
    transformers = pytest.importorskip("transformers")
    from agentic_arxiv.training import grpo, sft

    module = sft if stage == "sft" else grpo
    if stage == "sft":
        # Snapshot parsing/binding has its own tests; this fixture isolates writers.
        monkeypatch.setattr(module, "_verification_environment", Mock(return_value=None))
        monkeypatch.setattr(module, "preserve_managed_dataloader_epoch", lambda *args, **kwargs: nullcontext())
    output = tmp_path / "output"
    if stage == "sft" and (resume or user_stopped or epoch_mode):
        managed = SimpleNamespace(log_dir=str(output / "logs"))
        monkeypatch.setattr(module, "prepare_managed_training", lambda *args, **kwargs: managed)
        monkeypatch.setattr(module, "validate_managed_config", lambda *args: None)
        def record_result(*args, **kwargs):
            if observed is not None:
                observed.setdefault("results", []).append(kwargs)

        monkeypatch.setattr(module, "record_training_result", record_result)
        stop_control = SimpleNamespace(user_stopped=user_stopped, checkpoint="old/checkpoint-1",
                                       mark_failed=Mock())
        monkeypatch.setattr(module, "attach_managed_training", lambda *args, **kwargs: (stop_control, None))
    responses = iter([{}, None, None] if stage == "sft" else [{}, output / "reward_probe_summary.json", (None, None), None, None])

    class Context(TrainingContext):
        def prepare_device(self, **kwargs):
            pass

        def memory_summary(self):
            return {"per_rank": [{"rank": i, "peak_allocated_gib": float(i + 1)} for i in range(4)],
                    "peak_allocated_gib": 4.0}

        def run_on_main(self, operation):
            return operation() if self.is_main else next(responses)

    execution = Context(4, rank, rank)
    monkeypatch.setattr(TrainingContext, "from_environment", classmethod(lambda cls: execution))
    model_path = tmp_path / "model"
    model_path.mkdir()
    parameter = torch.nn.Parameter(torch.ones(1))
    policy = SimpleNamespace(config=SimpleNamespace(use_cache=True),
                             parameters=lambda: iter([parameter]),
                             named_parameters=lambda: iter([("weight", parameter)]))
    tokenizer = SimpleNamespace(chat_template="synthetic", pad_token="<pad>", eos_token="<eos>")

    def save_tokenizer(path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer.json").write_text("{}")

    tokenizer.save_pretrained = Mock(side_effect=save_tokenizer)
    protocol = protocol_metadata(tokenizer)
    source = {"stage": "sft_full", "base_model": str(model_path), "protocol_metadata": protocol}
    (model_path / "training_manifest.json").write_text(json.dumps(source))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", Mock(return_value=tokenizer))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", Mock(return_value=policy))
    monkeypatch.setattr(module, "require_supported_config", lambda cls, values: values)

    class Config:
        def __init__(self, **kwargs):
            if observed is not None:
                observed["config"] = kwargs
            self.__dict__.update(kwargs)
            self.generation_batch_size = kwargs["per_device_train_batch_size"] * kwargs["gradient_accumulation_steps"] * 4

        def to_dict(self):
            return dict(self.__dict__)

    saved_models = []

    class Trainer:
        def __init__(self, *, model, args, **kwargs):
            self.model = model
            self.args = args
            self.state = SimpleNamespace(global_step=actual_steps, max_steps=2)
            self.accelerator = SimpleNamespace(num_processes=4, process_index=rank, distributed_type="MULTI_CPU")
            self.model_accepts_loss_kwargs = True
            self.compute_loss_func = None
            self._metrics = {"train": defaultdict(list)}

        def add_callback(self, callback):
            pass

        def train(self, **kwargs):
            if observed is not None:
                observed["train"] = kwargs
            if fail_training:
                raise ValueError("intentional trainer failure")
            return SimpleNamespace(training_loss=0.25)

        def save_model(self, path):
            saved_models.append(path)
            if execution.is_main:
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path) / "model.safetensors").write_bytes(b"synthetic-unit-test-placeholder")

    monkeypatch.setattr(trl, "SFTConfig" if stage == "sft" else "GRPOConfig", Config)
    monkeypatch.setattr(trl, "SFTTrainer" if stage == "sft" else "GRPOTrainer", Trainer)
    report = SimpleNamespace(passed=True, coverage={}, summary=lambda: "unit-test report")
    verifier = Mock()
    verifier.verify_sft.return_value = report
    verifier.verify_grpo.return_value = report
    monkeypatch.setattr(module, "StageVerifier", Mock(return_value=verifier))

    arguments = {"model": str(model_path), "output_dir": str(output), "qlora": False,
                 "max_steps": -1 if epoch_mode else 2,
                 "batch_size": 1, "grad_accum": 2 if stage == "sft" else 1,
                 "verify": True}
    if stage == "sft" and resume:
        arguments.update(managed_context="current-context.json", resume_from_checkpoint=str(resume))
    if stage == "sft":
        data = tmp_path / "data.jsonl"
        data.write_text('{}\n')
        manifest = {"kind": "qlora_sft_train_mix", "protocol_metadata": protocol}
        data.with_suffix(".jsonl.manifest.json").write_text(json.dumps(manifest))
        monkeypatch.setattr(module, "_verify_data_manifest", lambda *args: manifest)
        monkeypatch.setattr(module, "_load_sft_rows", lambda *args, **kwargs: [
            {"prompt": [{"role": "user", "content": "task"}],
             "completion": [{"role": "assistant", "content": "Action: FINISH"}]},
        ])
        monkeypatch.setattr(module, "_check_lengths", lambda *args: None)
        arguments["data"] = str(data)
    else:
        task = {"id": "synthetic", "task": "task", "expected_tools": []}
        snapshot = tmp_path / "snapshot.json"
        snapshot.write_text("{}")
        monkeypatch.setattr(module, "_load_tasks", lambda *args, **kwargs: [task])
        monkeypatch.setattr(module, "_artifact_metadata", lambda *args, **kwargs: source)
        monkeypatch.setattr(module, "attach_expected_paper_ids", lambda tasks, snapshot: tasks)
        monkeypatch.setattr(module, "_load_policy", lambda *args, **kwargs: policy)
        monkeypatch.setattr(module, "_gold_completion_tokens", lambda *args: 1)
        monkeypatch.setattr(module, "load_mock_env", lambda *args: object())
        arguments.update({"snapshot": str(snapshot), "allow_zero_variance": True,
                          "canary_steps": 0, "save_rollout_traces": True})

    if fail_training:
        with pytest.raises(ValueError, match="intentional trainer failure"):
            module.main(**arguments)
    else:
        module.main(**arguments)
    return output, tokenizer, verifier, saved_models


@pytest.mark.train
@pytest.mark.parametrize("stage", ["sft", "grpo"])
def test_main_rank_writes_manifest_and_runs_one_final_verification(stage, tmp_path, monkeypatch):
    output, tokenizer, verifier, models = _run_stage(stage, 0, tmp_path, monkeypatch)
    assert models == [str(output / "final")]
    tokenizer.save_pretrained.assert_called_once_with(str(output / "final"))
    report_method = verifier.verify_sft if stage == "sft" else verifier.verify_grpo
    report_method.assert_called_once()
    record = json.loads((output / "run_manifest.json").read_text())
    manifest = json.loads((output / "final/training_manifest.json").read_text())
    assert record["derived_values"]["distributed_training"]["world_size"] == 4
    assert record["derived_values"]["effective_batch_size"] == (8 if stage == "sft" else 4)
    assert record["resource_overrides"]["world_size"] == {"baseline": 1, "requested": 4}
    assert manifest["peak_vram_gib"] == 4.0 and manifest["actual_steps"] == 2
    if stage == "grpo":
        assert manifest["batch_size"] == 1 and manifest["generation_batch_size"] == 4
        assert (output / "rollout_traces.jsonl").exists()


@pytest.mark.train
@pytest.mark.parametrize("stage", ["sft", "grpo"])
def test_worker_participates_in_trainer_save_but_never_writes_shared_files(stage, tmp_path, monkeypatch):
    output, tokenizer, verifier, models = _run_stage(stage, 1, tmp_path, monkeypatch)
    assert models == [str(output / "final")]
    tokenizer.save_pretrained.assert_not_called()
    verifier.verify_sft.assert_not_called()
    verifier.verify_grpo.assert_not_called()
    assert not output.exists()


@pytest.mark.train
@pytest.mark.parametrize("stage", ["sft", "grpo"])
@pytest.mark.parametrize("rank", [0, 1])
def test_training_failure_has_only_main_diagnostic_and_no_success_final(stage, rank, tmp_path, monkeypatch):
    output, tokenizer, verifier, models = _run_stage(stage, rank, tmp_path, monkeypatch, fail_training=True)
    assert not models and not (output / "final").exists()
    tokenizer.save_pretrained.assert_not_called()
    if rank == 0:
        record = json.loads((output / "run_manifest.json").read_text())
        assert record["stop_reason"] == "training_exception:ValueError"
    else:
        assert not output.exists()


@pytest.mark.train
def test_sft_resume_passes_full_state_path_without_reissuing_budget(tmp_path, monkeypatch):
    observed = {}
    checkpoint = tmp_path / "previous-attempt" / "checkpoint-1"
    output, _, verifier, models = _run_stage(
        "sft", 0, tmp_path, monkeypatch, resume=checkpoint, observed=observed,
    )
    assert observed["train"] == {"resume_from_checkpoint": str(checkpoint)}
    assert observed["config"]["max_steps"] == 2
    assert observed["config"]["learning_rate"] == 1e-4
    assert observed["config"]["output_dir"] == str(output)
    assert models == [str(output / "final")]
    verifier.verify_sft.assert_called_once()


@pytest.mark.train
@pytest.mark.parametrize("user_stopped", [True, False])
def test_sft_user_or_protection_stop_never_exports_success_final_or_verifies(user_stopped, tmp_path, monkeypatch):
    output, tokenizer, verifier, models = _run_stage(
        "sft", 0, tmp_path, monkeypatch, actual_steps=1, user_stopped=user_stopped,
    )
    assert not models and not (output / "final").exists()
    tokenizer.save_pretrained.assert_not_called()
    verifier.verify_sft.assert_not_called()
    record = json.loads((output / "run_manifest.json").read_text())
    assert record["stop_reason"] == ("user_safe_stop" if user_stopped else "trainer_stopped_early")


@pytest.mark.train
def test_sft_completed_budget_wins_over_late_stop_request(tmp_path, monkeypatch):
    output, _, verifier, models = _run_stage(
        "sft", 0, tmp_path, monkeypatch, user_stopped=True,
    )
    assert models == [str(output / "final")]
    verifier.verify_sft.assert_called_once()
    assert json.loads((output / "run_manifest.json").read_text())["stop_reason"] == "max_steps_or_epochs_completed"


@pytest.mark.train
@pytest.mark.parametrize("actual_steps,user_stopped", [(1, True), (2, False)])
def test_sft_epoch_resume_keeps_epoch_scheduler_and_records_trainer_actual_plan(
        actual_steps, user_stopped, tmp_path, monkeypatch):
    observed = {}
    checkpoint = tmp_path / "previous-attempt/checkpoint-1"
    output, _, _, models = _run_stage(
        "sft", 0, tmp_path, monkeypatch, resume=checkpoint, observed=observed,
        epoch_mode=True, actual_steps=actual_steps, user_stopped=user_stopped,
    )
    assert observed["config"]["max_steps"] == -1
    assert observed["config"]["num_train_epochs"] == 3
    assert observed["config"]["lr_scheduler_type"] == "cosine"
    assert observed["train"] == {"resume_from_checkpoint": str(checkpoint)}
    assert observed["results"][-1]["planned_steps"] == 2
    assert observed["results"][-1]["training"] == ("user_safe_stop" if user_stopped else "budget_completed")
    assert models == ([] if user_stopped else [str(output / "final")])


def _dpo_main_tree():
    """只检查接线结构，不添加模型响应、token 或轨迹输入。"""
    from agentic_arxiv.training import dpo
    return ast.parse(inspect.getsource(dpo.main)).body[0]


def test_dpo_inspection_precedes_both_independent_weight_loads():
    tree = _dpo_main_tree()
    calls = {node.func.id: node.lineno for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    audit = calls["_audit_token_boundaries"]
    inspect_branch = next(node for node in tree.body if isinstance(node, ast.If)
                          and isinstance(node.test, ast.Name) and node.test.id == "inspect_only")
    assert audit < inspect_branch.lineno < calls["load_training_policy"] < calls["load_reference_policy"]
    assert any(isinstance(node, ast.Return) for node in ast.walk(inspect_branch))


def test_dpo_final_writers_and_verification_have_main_rank_broadcast_boundary():
    tree = _dpo_main_tree()
    broadcast_operations = {node.args[0].id for node in ast.walk(tree)
                            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and node.func.attr == "run_on_main" and node.args
                            and isinstance(node.args[0], ast.Name)}
    broadcast_operations.update(node.args[0].args[0].id for node in ast.walk(tree)
                                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                                and node.func.attr == "run_on_main" and node.args
                                and isinstance(node.args[0], ast.Call)
                                and isinstance(node.args[0].func, ast.Name) and node.args[0].func.id == "partial")
    assert {"save_final", "verify_final"} <= broadcast_operations
    assert any(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
               and isinstance(node.value.func, ast.Attribute)
               and node.value.func.attr == "save_model" for node in tree.body)
    verify = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "verify_final")
    assert any(isinstance(node, ast.Raise) for node in ast.walk(verify))
    release = next(node.lineno for node in ast.walk(tree)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr == "free_memory")
    assert release < verify.lineno


@pytest.mark.parametrize("managed", [False, True])
def test_dpo_cleanup_releases_managed_trainer_and_models_before_verification(managed):
    import gc
    import weakref
    from agentic_arxiv.training.checkpoint import ManagedCheckpointAdapter

    class Resource:
        pass

    class Accelerator:
        def free_memory(self):
            self.models.clear()

    def training_namespace():
        trainer, policy, reference, optimizer = (Resource() for _ in range(4))
        trainer.model, trainer.ref_model, trainer.optimizer = policy, reference, optimizer
        trainer.accelerator = Accelerator()
        trainer.accelerator.models = [policy, reference]
        adapter = (ManagedCheckpointAdapter(trainer, context=None, training_context=None)
                   if managed else None)
        callback = Resource() if managed else None
        if managed:
            callback.checkpoint_committed = lambda step: adapter.last_checkpoint
            trainer.callbacks = [callback]
            trainer._managed_checkpoint = adapter
        refs = [weakref.ref(item) for item in (trainer, policy, reference, optimizer)]
        return {"trainer": trainer, "policy": policy, "reference": reference,
                "stop_control": callback, "_checkpoint": adapter,
                "dataset": Resource(), "tokenizer": Resource(), "gc": gc}, refs

    tree = _dpo_main_tree()
    start = next(index for index, node in enumerate(tree.body)
                 if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                 and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "free_memory")
    end = next(index for index, node in enumerate(tree.body[start:], start)
               if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
               and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "collect")
    # 执行生产代码的实际清理片段，只提供内存资源及真实受管适配对象，不运行训练。
    cleanup = ast.Module(body=tree.body[start:end + 1], type_ignores=[])
    namespace, references = training_namespace()
    assert all(reference() is not None for reference in references)
    exec(compile(cleanup, "dpo-cleanup", "exec"), namespace)
    assert all(reference() is None for reference in references)


def test_dpo_uses_explicit_reference_without_second_peft_configuration():
    tree = _dpo_main_tree()
    trainer = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Name) and node.func.id == "DPOTrainer")
    keywords = {item.arg: item.value for item in trainer.keywords}
    assert isinstance(keywords["ref_model"], ast.Name) and keywords["ref_model"].id == "reference"
    assert "peft_config" not in keywords
    from agentic_arxiv.training.distributed import TrainingContext
    assert "average_tokens_across_devices" not in TrainingContext(4).trainer_kwargs(stage="dpo")


@pytest.mark.train
def test_installed_dpo_config_supports_every_explicit_training_field():
    trl = pytest.importorskip("trl")
    from agentic_arxiv.training.mainline_config import require_supported_config, stage_config
    tree = _dpo_main_tree()
    config = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "config_kwargs" for target in node.targets))
    keys = {key.value: None for key in config.keys if isinstance(key, ast.Constant)}
    keys.update(stage_config("dpo")["trainer_parameters"])
    keys.update(TrainingContext(4).trainer_kwargs(stage="dpo"))
    require_supported_config(trl.DPOConfig, keys)
    literal = {key.value: ast.literal_eval(value) for key, value in zip(config.keys, config.values)
               if isinstance(key, ast.Constant) and key.value in {
                   "max_length", "loss_type", "sync_ref_model", "precompute_ref_log_probs",
                   "f_divergence_type", "label_smoothing", "loss_weights", "padding_free", "use_liger_kernel"}}
    assert literal == {"max_length": None, "loss_type": ["sigmoid"], "sync_ref_model": False,
                       "precompute_ref_log_probs": False, "f_divergence_type": "reverse_kl",
                       "label_smoothing": 0.0, "loss_weights": [1.0], "padding_free": False,
                       "use_liger_kernel": False}
