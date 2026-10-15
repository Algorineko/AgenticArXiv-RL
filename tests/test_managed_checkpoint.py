"""只验证控制元数据、标准保存接线和文件事务，不是模型恢复证据。"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_arxiv.training.checkpoint import (
    COMPLETE_NAME, MANIFEST_NAME, CheckpointError, ManagedCheckpointContext,
    attach_managed_checkpoint, check_trainer_interfaces, validate_checkpoint,
)
from agentic_arxiv.training.distributed import TrainingContext


def context(attempt="attempt-1", **changes):
    values = dict(stage="sft", run_id="run-1", attempt_id=attempt, total_budget=10,
                  identity={"inputs": {"data": "existing-data-fingerprint"},
                            "training_config": {"batch": 1, "seed": 42},
                            "runtime": {"image": "immutable-image-identity"},
                            "models": {"policy": "existing-model-fingerprint"},
                            "devices": ["GPU-existing-uuid"]})
    values.update(changes)
    return ManagedCheckpointContext(**values)


class MetadataTrainer:
    """Explicit metadata-only writer; sentinel files are not usable model states."""

    def __init__(self, output):
        self.args = SimpleNamespace(output_dir=str(output), save_total_limit=1,
                                    save_strategy="steps", save_steps=2)
        self.state = SimpleNamespace(global_step=2, max_steps=10, best_global_step=None)
        self.accelerator = SimpleNamespace(scaler=None)
        self.fail = False
        self.callbacks = []

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def _get_output_dir(self, trial):
        return self.args.output_dir

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None):
        raise AssertionError("standard rotation must be suppressed until commit")

    def _load_rng_state(self, checkpoint):
        pass

    def _save_optimizer_and_scheduler(self, output_dir):
        pass

    def _save_rng_state(self, output_dir):
        pass

    def _save_scaler(self, output_dir):
        pass

    def _save_checkpoint(self, model, trial):
        target = Path(self._get_output_dir(trial)) / f"checkpoint-{self.state.global_step}"
        target.mkdir(parents=True, exist_ok=True)
        for name in ("optimizer.pt", "scheduler.pt", "training_args.bin", "rng_state.pth",
                     "adapter_model.safetensors"):
            (target / name).write_text("metadata-only test sentinel, not a model", encoding="utf-8")
        if self.fail:
            raise OSError("simulated disk full")
        (target / "trainer_state.json").write_text(json.dumps({
            "global_step": self.state.global_step, "max_steps": self.state.max_steps, "epoch": 0.2,
        }), encoding="utf-8")
        self._rotate_checkpoints(output_dir=self._get_output_dir(trial))


@pytest.fixture
def trainer(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr("agentic_arxiv.training.checkpoint.version",
                        lambda name: {"transformers": "4.57.6", "trl": "0.29.1"}[name])
    return MetadataTrainer(tmp_path / "attempt" / "output")


def attach(trainer, **kwargs):
    return attach_managed_checkpoint(trainer, context=kwargs.pop("context", context()),
                                     training_context=TrainingContext(), **kwargs)


def save(trainer, step=2):
    trainer.state.global_step = step
    trainer._save_checkpoint(None, None)
    return Path(trainer.args.output_dir) / f"checkpoint-{step}"


def test_context_unknown_stage_version_and_fields():
    for changes in ({"stage": "weights"}, {"schema_version": 2}, {"total_budget": True},
                    {"identity": {}}, {"total_budget": 0}, {"total_budget": None},
                    {"epoch_budget": True}, {"epoch_budget": 0}, {"epoch_budget": 1.5}):
        with pytest.raises(CheckpointError):
            context(**changes)
    with pytest.raises(CheckpointError, match="未知"):
        ManagedCheckpointContext.from_dict({**context().as_dict(), "surprise": True})


def test_context_operational_paths_are_not_research_identity(tmp_path):
    value = {**context().as_dict(), "output_dir": str(tmp_path), "control_dir": str(tmp_path / "control")}
    path = tmp_path / "context.json"
    path.write_text(json.dumps(value))
    loaded = ManagedCheckpointContext.from_file(path, expected_stage="sft")
    assert loaded.output_dir == str(tmp_path)
    assert loaded.as_dict() == context().as_dict()
    with pytest.raises(CheckpointError, match="阶段"):
        ManagedCheckpointContext.from_file(path, expected_stage="dpo")


def test_atomic_commit_and_extra_state_roundtrip(trainer):
    seen = []
    adapter = attach(trainer, state_provider=lambda: {"cursor": 2})
    path = save(trainer)
    manifest = validate_checkpoint(path, expected=context("attempt-2"))
    assert manifest["global_step"] == 2
    assert manifest["precision"] == {"scaler": "not_applicable"}
    assert adapter.last_checkpoint == path == adapter.latest_checkpoint
    other = MetadataTrainer(path.parent.parent.parent / "attempt-2" / "output")
    resumed = attach(other, context=context("attempt-2"), resume_from_checkpoint=path,
                     state_restorer=seen.append)
    resumed.restore_additional_state()
    assert seen == [{"cursor": 2}]
    assert not Path(other.args.output_dir).exists()


def test_failed_save_does_not_rotate_or_mark_complete(trainer):
    attach(trainer)
    previous = save(trainer)
    trainer.fail = True
    with pytest.raises(CheckpointError, match="disk full"):
        save(trainer, 4)
    assert validate_checkpoint(previous)["global_step"] == 2
    incomplete = previous.parent / ".checkpoint-4.incomplete" / "checkpoint-4"
    assert incomplete.is_dir()
    assert not (incomplete / COMPLETE_NAME).exists()
    assert not (previous.parent / "checkpoint-4").exists()


def test_commit_disk_failure_preserves_previous(trainer, monkeypatch):
    attach(trainer)
    previous = save(trainer)
    original = Path.open

    def failing_open(path, *args, **kwargs):
        if path.name == MANIFEST_NAME + ".tmp":
            raise OSError("manifest disk full")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(OSError, match="disk full"):
        save(trainer, 4)
    assert validate_checkpoint(previous)["global_step"] == 2


def test_rotation_only_after_commit_and_only_current_attempt(trainer):
    adapter = attach(trainer)
    first = save(trainer)
    second = save(trainer, 4)
    assert not first.exists()
    assert second == adapter.last_checkpoint
    assert validate_checkpoint(second)["global_step"] == 4


def test_rotation_protects_resume_source_even_inside_output(trainer):
    attach(trainer)
    first = save(trainer)
    # Direct adapter test of defence-in-depth; public managed entry forbids this layout.
    trainer._managed_checkpoint.resume_from_checkpoint = first.resolve()
    save(trainer, 4)
    assert first.exists()


@pytest.mark.parametrize("key,value", [("training_config", {"batch": 2, "seed": 42}),
                                      ("training_config", {"batch": 1, "seed": 41}),
                                      ("devices", ["GPU-other"]),
                                      ("inputs", {"data": "changed"}),
                                      ("runtime", {"image": "changed"})])
def test_strict_resume_identity(trainer, key, value):
    attach(trainer)
    path = save(trainer)
    expected = context().as_dict()
    expected["identity"][key] = value
    with pytest.raises(CheckpointError, match=key):
        validate_checkpoint(path, expected=expected)


def test_budget_stage_and_weights_only_rejected(trainer, tmp_path):
    attach(trainer)
    path = save(trainer)
    for expected in ({"total_budget": 11}, {"stage": "dpo"}, {"run_id": "other"}):
        with pytest.raises(CheckpointError, match="不兼容"):
            validate_checkpoint(path, expected=expected)
    weights_only = tmp_path / "old-checkpoint"
    weights_only.mkdir()
    with pytest.raises(CheckpointError):
        validate_checkpoint(weights_only)


@pytest.mark.parametrize("file", ["optimizer.pt", "rng_state.pth", "managed_state_0.pt",
                                  "rank-0.complete.json", "trainer_state.json", COMPLETE_NAME])
def test_corrupt_or_missing_files_rejected(trainer, file):
    attach(trainer)
    path = save(trainer)
    (path / file).write_text("changed")
    with pytest.raises(CheckpointError):
        validate_checkpoint(path)


def test_completed_budget_and_protection_cannot_resume(trainer):
    attach(trainer)
    trainer._managed_termination_reason = "canary"
    protected = save(trainer)
    with pytest.raises(CheckpointError, match="保护"):
        validate_checkpoint(protected)
    trainer._managed_termination_reason = None
    exhausted = save(trainer, 10)
    with pytest.raises(CheckpointError, match="耗尽"):
        validate_checkpoint(exhausted)
    assert validate_checkpoint(exhausted, require_remaining_budget=False)["global_step"] == 10


def test_rank_missing_does_not_publish(trainer):
    class MissingRank(TrainingContext):
        def run_on_main(self, operation):
            return operation()

        def gather(self, value):
            return [value, {"rank": 1, "error": None, "global_step": value["global_step"]}]

    ctx = context()
    ctx.identity["devices"].append("GPU-existing-uuid-2")
    attach_managed_checkpoint(trainer, context=ctx, training_context=MissingRank(world_size=2))
    with pytest.raises(CheckpointError, match="rank"):
        save(trainer)
    assert not list(Path(trainer.args.output_dir).glob("checkpoint-*"))
    assert not list(Path(trainer.args.output_dir).rglob(COMPLETE_NAME))


def test_unsupported_version_and_unsafe_modes_fail_before_save(trainer):
    with pytest.raises(CheckpointError, match="固定依赖"):
        check_trainer_interfaces(trainer, installed_versions={"transformers": "5", "trl": "0.29.1"})
    trainer.args.ignore_data_skip = True
    with pytest.raises(CheckpointError, match="数据"):
        attach(trainer)
    assert not Path(trainer.args.output_dir).exists()


def test_cuda_failure_does_not_start_reporting_collective(trainer, monkeypatch):
    topology = TrainingContext()
    attach_managed_checkpoint(trainer, context=context(), training_context=topology)

    def broken_state():
        raise RuntimeError("CUDA out of memory")

    trainer._managed_checkpoint.state_provider = broken_state
    monkeypatch.setattr(TrainingContext, "gather", lambda *args: pytest.fail("unsafe collective"))
    with pytest.raises(RuntimeError, match="CUDA"):
        save(trainer)
    assert not list(Path(trainer.args.output_dir).rglob(COMPLETE_NAME))


def test_fixed_installed_standard_trainer_interfaces(trainer):
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "4.57.6":
        pytest.skip("固定版本接口检查仅在受支持工程环境执行")
    from types import MethodType
    from transformers import Trainer

    # Exercise the actual pinned standard save orchestration, replacing only I/O
    # producers with metadata writers. This does not construct a model or train.
    trainer.hp_search_backend = None
    trainer.store_flos = lambda: None
    trainer.args.should_save = True
    trainer.args.save_only_model = False
    trainer.args.push_to_hub = False
    trainer.callback_handler = SimpleNamespace(callbacks=[])
    trainer.control = SimpleNamespace()

    def write_state(path):
        Path(path).write_text(json.dumps({"global_step": trainer.state.global_step,
                                         "max_steps": 10, "epoch": 0.2}))

    trainer.state.save_to_json = write_state
    trainer.save_model = lambda path, _internal_call: MetadataTrainer._save_checkpoint(trainer, None, None)
    trainer._save_checkpoint = MethodType(Trainer._save_checkpoint, trainer)
    attach(trainer)
    path = save(trainer)
    assert validate_checkpoint(path)["global_step"] == 2


def test_epoch_budget_binds_at_train_begin_and_retains_original_schedule(trainer, tmp_path):
    original = context(total_budget=None, epoch_budget=3, control_dir=str(tmp_path / "control"))
    trainer.args.max_steps = -1
    trainer.args.num_train_epochs = 3.0
    adapter = attach(trainer, context=original)
    assert adapter.context.total_budget is None
    with pytest.raises(CheckpointError, match="尚未"):
        save(trainer)
    control = SimpleNamespace()
    assert adapter.budget_callback in trainer.callbacks
    assert adapter.budget_callback.on_train_begin(trainer.args, trainer.state, control) is control
    assert original.total_budget is None
    assert adapter.context.total_budget == 10
    assert adapter.context.epoch_budget == 3
    assert (trainer.args.max_steps, trainer.args.num_train_epochs) == (-1, 3.0)
    evidence = json.loads((tmp_path / "control" / "resolved-budget.json").read_text())
    assert evidence == {"schema_version": 1, "run_id": "run-1", "attempt_id": "attempt-1",
                        "total_budget": 10, "epoch_budget": 3}
    path = save(trainer)
    manifest = validate_checkpoint(path, expected=original)
    assert (manifest["total_budget"], manifest["epoch_budget"]) == (10, 3)


def test_epoch_checkpoint_resume_keeps_actual_budget_and_rejects_changed_data_length(trainer, tmp_path):
    original = context(total_budget=None, epoch_budget=3, control_dir=str(tmp_path / "first-control"))
    trainer.args.max_steps = -1
    trainer.args.num_train_epochs = 3
    adapter = attach(trainer, context=original)
    adapter.bind_budget(args=trainer.args, state=trainer.state)
    checkpoint = save(trainer)
    resumed_trainer = MetadataTrainer(tmp_path / "resumed-output")
    resumed_trainer.args.max_steps = -1
    resumed_trainer.args.num_train_epochs = 3
    # Management normally supplies the checkpoint's known actual total_budget.
    # An initial unbound context is also pinned to the validated checkpoint here.
    resumed_context = context("attempt-2", total_budget=None, epoch_budget=3,
                              control_dir=str(tmp_path / "second-control"))
    resumed = attach(resumed_trainer, context=resumed_context, resume_from_checkpoint=checkpoint)
    assert resumed.context.total_budget == 10
    resumed_trainer.state.max_steps = 11
    with pytest.raises(CheckpointError, match="核定总步数"):
        resumed.bind_budget(args=resumed_trainer.args, state=resumed_trainer.state)
    resumed_trainer.state.max_steps = 10
    resumed.bind_budget(args=resumed_trainer.args, state=resumed_trainer.state)
    assert resumed.context.total_budget == 10
    assert resumed_trainer.args.max_steps == -1
    assert resumed_trainer.args.num_train_epochs == 3
    for desired in ({"total_budget": None, "epoch_budget": 4},
                    {"total_budget": None}, {"epoch_budget": 4},
                    {"total_budget": 20, "epoch_budget": 3}):
        with pytest.raises(CheckpointError):
            validate_checkpoint(checkpoint, expected=desired)


@pytest.mark.parametrize("max_steps,epochs", [(10, 3), (-1, 4), (-1, True)])
def test_epoch_budget_cannot_replace_original_scheduler(trainer, tmp_path, max_steps, epochs):
    adapter = attach(trainer, context=context(total_budget=None, epoch_budget=3,
                                               control_dir=str(tmp_path / "control")))
    trainer.args.max_steps = max_steps
    trainer.args.num_train_epochs = epochs
    with pytest.raises(CheckpointError, match="原 max_steps"):
        adapter.bind_budget(args=trainer.args, state=trainer.state)
    assert not (tmp_path / "control" / "resolved-budget.json").exists()


def test_epoch_actual_budget_evidence_cannot_be_overwritten(trainer, tmp_path):
    trainer.args.max_steps = -1
    trainer.args.num_train_epochs = 3
    adapter = attach(trainer, context=context(total_budget=None, epoch_budget=3,
                                               control_dir=str(tmp_path / "control")))
    adapter.bind_budget(args=trainer.args, state=trainer.state)
    adapter.bind_budget(args=trainer.args, state=trainer.state)
    evidence = tmp_path / "control" / "resolved-budget.json"
    original = evidence.read_bytes()
    payload = json.loads(original)
    payload["total_budget"] = 20
    evidence.write_text(json.dumps(payload))
    with pytest.raises(CheckpointError, match="已有不同"):
        adapter.bind_budget(args=trainer.args, state=trainer.state)
    assert json.loads(evidence.read_text())["total_budget"] == 20
