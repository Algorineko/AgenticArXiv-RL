"""No-model GRPO control-state checks, not evidence of real training recovery."""

from collections import defaultdict, deque
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")

from agentic_arxiv.training import trl_compat  # noqa: E402
from trl import GRPOTrainer  # noqa: E402
from agentic_arxiv.training.canary import CanaryCallback, CanaryResult  # noqa: E402
from agentic_arxiv.training.distributed import TrainingContext  # noqa: E402
from agentic_arxiv.training.grpo import RewardVarianceGuard  # noqa: E402
from agentic_arxiv.training.observability import RewardComponentTracker  # noqa: E402
from agentic_arxiv.training.rollout_audit import RolloutAuditWriter  # noqa: E402
from agentic_arxiv.training.trl_compat import (  # noqa: E402
    GRPOResumeState,
    reference_weight_fingerprint,
    require_managed_grpo_interfaces,
)

pytestmark = pytest.mark.train


def control_trainer(*, microstep=5, global_step=2):
    # Use the installed Trainer's actual methods without constructing a model,
    # dataset, accelerator process group, or synthetic rollout.
    trainer = GRPOTrainer.__new__(GRPOTrainer)
    trainer.args = SimpleNamespace(
        steps_per_generation=3, gradient_accumulation_steps=2,
        per_device_train_batch_size=1, generation_batch_size=3,
        sync_ref_model=False, seed=42, num_train_epochs=3, max_steps=-1,
    )
    trainer.state = SimpleNamespace(global_step=global_step, epoch=0.5, max_steps=18)
    trainer.train_dataset = range(16)
    trainer.shuffle_dataset = True
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer._step = microstep
    trainer._buffered_inputs = [
        {"control_tensor": torch.tensor([index]),
         "old_logps": torch.tensor([[float(index)]])}
        for index in range(3)
    ]
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer._metrics["train"]["reward_std"].append(0.25)
    trainer._logs = {"rewards": defaultdict(lambda: deque(maxlen=3)),
                     "completion": deque(maxlen=3)}
    trainer._logs["rewards"]["control"].append(0.5)
    trainer._total_train_tokens = 123
    trainer._current_train_step_time = 1.25
    trainer.num_iterations = 2
    trainer.num_generations = 3
    trainer.beta = 0
    trainer.ref_model = None
    trainer.optimizer = None
    trainer.use_vllm = False
    return trainer


def adapter_trainer(*, policy_value, dtype=torch.float32):
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(torch.nn.Sequential(torch.nn.Linear(2, 2)).to(dtype=dtype),
                           LoraConfig(r=1, target_modules=["0"]), autocast_adapter_dtype=False)
    model.add_adapter("ref", LoraConfig(r=1, target_modules=["0"]))
    model.to(dtype=dtype)
    model.set_adapter("default")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".default." in name:
                parameter.fill_(policy_value)
            elif ".ref." in name:
                parameter.fill_(2)
                parameter.requires_grad_(False)
    trainer = control_trainer()
    trainer.model = model
    trainer.beta = 0.1
    trainer.is_fsdp_enabled = False
    trainer.accelerator.unwrap_model = lambda model: model
    trainer.optimizer = torch.optim.AdamW(parameter for parameter in model.parameters()
                                         if parameter.requires_grad)
    return trainer


@pytest.mark.parametrize("safe_serialization", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("managed_loader", [True, False])
def test_real_peft_default_and_ref_restore_preserves_policy_optimizer_and_reference(
        tmp_path, safe_serialization, dtype, managed_loader):
    from agentic_arxiv.training.checkpoint import install_managed_adapter_restore

    source = adapter_trainer(policy_value=4, dtype=dtype)
    state = resume_state(source).state_dict()
    source.model.save_pretrained(tmp_path, safe_serialization=safe_serialization)
    assert (tmp_path / "ref" / "adapter_config.json").is_file()

    target = adapter_trainer(policy_value=1, dtype=dtype)
    restored = resume_state(target)
    restored.load_state_dict(state)
    parameter_ids = {name: id(value) for name, value in target.model.named_parameters()}
    optimizer_ids = {id(value) for group in target.optimizer.param_groups for value in group["params"]}
    reference_before = reference_weight_fingerprint(target)
    original_adapter_loader = target.model.load_adapter
    if managed_loader:
        install_managed_adapter_restore(target)
    restored.install_policy_adapter_restore()
    # Invoke the installed real Transformers loader, not a replacement stub.
    target._load_from_checkpoint(str(tmp_path))
    restored.validate_loaded_state()

    for name, value in target.model.named_parameters():
        assert id(value) == parameter_ids[name]
        assert value.dtype == dtype
        if ".default." in name:
            assert torch.equal(value, dict(source.model.named_parameters())[name])
            assert value.requires_grad and id(value) in optimizer_ids
        elif ".ref." in name:
            assert not value.requires_grad and id(value) not in optimizer_ids
    assert target.model.active_adapters == ["default"]
    assert target.model.load_adapter == original_adapter_loader
    assert reference_weight_fingerprint(target) == reference_before
    assert target._step == source._step
    assert torch.equal(target._buffered_inputs[0]["old_logps"], source._buffered_inputs[0]["old_logps"])


def test_adapter_restore_rejects_missing_root_policy(tmp_path):
    target = adapter_trainer(policy_value=1)
    restored = resume_state(target)
    restored.install_policy_adapter_restore()
    with pytest.raises(ValueError, match="缺少根 default 策略权重"):
        target._load_from_checkpoint(str(tmp_path))


def test_adapter_restore_rejects_changed_frozen_reference(tmp_path):
    source = adapter_trainer(policy_value=4)
    with torch.no_grad():
        for name, parameter in source.model.named_parameters():
            if ".ref." in name:
                parameter.fill_(3)
    source.model.save_pretrained(tmp_path)
    target = adapter_trainer(policy_value=1)
    restored = resume_state(target)
    restored.install_policy_adapter_restore()
    with pytest.raises(ValueError, match="原始冻结参考权重不一致"):
        target._load_from_checkpoint(str(tmp_path))


@pytest.mark.parametrize("invalid_weights", ["missing", "unexpected"])
def test_adapter_restore_rejects_incomplete_or_unexpected_policy_weights(tmp_path, invalid_weights):
    from safetensors.torch import load_file, save_file

    source = adapter_trainer(policy_value=4)
    source.model.save_pretrained(tmp_path)
    path = tmp_path / "adapter_model.safetensors"
    weights = load_file(path)
    if invalid_weights == "missing":
        weights.pop(next(iter(weights)))
    else:
        weights["unexpected.weight"] = torch.ones(1)
    save_file(weights, path)
    target = adapter_trainer(policy_value=1)
    restored = resume_state(target)
    restored.install_policy_adapter_restore()
    with pytest.raises(ValueError, match="default 策略权重不完整或不匹配"):
        target._load_from_checkpoint(str(tmp_path))


@pytest.mark.parametrize("full_reference", [True, False])
def test_adapter_restore_does_not_wrap_non_adapter_paths(full_reference):
    trainer = frozen_reference_trainer() if full_reference else control_trainer()
    original = trainer._load_from_checkpoint
    resume_state(trainer).install_policy_adapter_restore()
    assert trainer._load_from_checkpoint == original


def resume_state(trainer=None, *, track_sampler=True, **kwargs):
    controller = GRPOResumeState(
        trainer or control_trainer(), TrainingContext(),
        {"source": "original-sft", "fingerprint": "control-only"},
        curriculum_steps=30, **kwargs,
    )
    if track_sampler:
        controller.install_sampler_tracking()
        # Consume a control index to establish the actual RepeatSampler's
        # current epoch metadata, never a model generation or research sample.
        next(iter(controller.trainer._get_train_sampler()))
    return controller


def test_installed_trl_private_interface_is_explicitly_checked():
    trainer = control_trainer()
    require_managed_grpo_interfaces(trainer)
    with pytest.raises(ValueError, match="trl="):
        require_managed_grpo_interfaces(trainer, installed_version="0.30.0")
    del trainer._buffered_inputs
    with pytest.raises(ValueError, match="接口缺失"):
        require_managed_grpo_interfaces(trainer)


def test_microstep_cache_metrics_and_tracker_round_trip_without_resampling(tmp_path):
    source = control_trainer()
    tracker = RewardComponentTracker()
    tracker.bind(source)
    tracker.record_group(["control", "control"], [0.0, 1.0], num_generations=2)
    state = resume_state(source, tracker=tracker).state_dict()
    state_path = tmp_path / "grpo-rank-state.pt"
    torch.save(state, state_path)
    state = torch.load(state_path, weights_only=False, map_location="cpu")
    source._buffered_inputs[0]["control_tensor"].fill_(99)
    assert state["buffered_inputs"][0]["control_tensor"].item() == 0

    target = control_trainer(microstep=0, global_step=0)
    target_tracker = RewardComponentTracker()
    target_tracker.bind(target)
    sink = target_tracker._sink
    layout_before = vars(target.args).copy()
    restored = resume_state(target, tracker=target_tracker)
    restored.load_state_dict(state)
    # Standard Trainer loads its own progress before the begin callback.
    assert target.state.global_step == 0
    target.state.global_step = 2
    restored.validate_loaded_state()

    assert target._step == 5
    assert torch.equal(target._buffered_inputs[2]["old_logps"], torch.tensor([[2.0]]))
    assert target._total_train_tokens == 123
    assert target._current_train_step_time == 1.25
    assert target_tracker._sink is sink is target._metrics["train"]
    assert target_tracker._task_rewards == tracker._task_rewards
    assert target_tracker._task_group_stds == tracker._task_group_stds
    assert vars(target.args) == layout_before
    assert target.num_iterations == 2 and target.num_generations == 3
    assert target._logs["rewards"]["new_key"].maxlen == 3


@pytest.mark.parametrize("key,value", [
    ("schema_version", 2), ("trl_version", "0.30.0"),
    ("rank", 1), ("world_size", 2), ("curriculum_steps", 0),
    ("reference_identity", {"source": "checkpoint-policy"}),
    ("reference_weights", {"sha256": "changed"}),
])
def test_incompatible_identity_or_version_is_rejected(key, value):
    state = resume_state().state_dict()
    state[key] = value
    with pytest.raises(ValueError, match=key):
        resume_state().load_state_dict(state)


@pytest.mark.parametrize("key,value", [
    ("buffered_inputs", None), ("buffered_inputs", []),
    ("generation_position", 0), ("slice_position", 0),
    ("microstep", -1), ("global_step", 6),
])
def test_missing_cache_and_invalid_consumption_positions_are_rejected(key, value):
    state = resume_state().state_dict()
    state[key] = value
    with pytest.raises(ValueError):
        resume_state().load_state_dict(state)


def test_cache_is_optional_only_at_a_fresh_generation_window():
    trainer = control_trainer(microstep=6, global_step=3)
    trainer._buffered_inputs = None
    state = resume_state(trainer).state_dict()
    target = resume_state()
    target.load_state_dict(state)
    assert target.trainer._step == 6
    assert target.trainer._buffered_inputs is None


def test_changed_generation_reuse_layout_is_not_silently_adjusted():
    state = resume_state().state_dict()
    target = control_trainer()
    target.args.steps_per_generation = 1
    with pytest.raises(ValueError, match="layout"):
        resume_state(target).load_state_dict(state)
    assert target.args.steps_per_generation == 1


def test_standard_progress_must_match_before_next_rollout():
    state = resume_state().state_dict()
    target = resume_state(control_trainer(global_step=0))
    target.load_state_dict(state)
    with pytest.raises(ValueError, match="逻辑步数"):
        target.validate_loaded_state()


@pytest.mark.parametrize("field,value", [
    ("num_train_epochs", 4), ("max_steps", 18),
])
def test_epoch_budget_resume_rejects_changed_requested_budget(field, value):
    state = resume_state().state_dict()
    assert state["requested_num_train_epochs"] == 3
    assert state["requested_max_steps"] == -1
    assert state["planned_steps"] == 18
    trainer = control_trainer()
    setattr(trainer.args, field, value)
    args_before = vars(trainer.args).copy()
    with pytest.raises(ValueError, match=f"requested_{field}"):
        resume_state(trainer).load_state_dict(state)
    assert vars(trainer.args) == args_before
    assert trainer.state.max_steps == 18


@pytest.mark.parametrize("planned_steps", [None, True, 18.0, -1, 1])
def test_epoch_budget_resume_rejects_missing_or_invalid_original_step_budget(planned_steps):
    state = resume_state().state_dict()
    state["planned_steps"] = planned_steps
    with pytest.raises(ValueError, match="原始核定优化步数预算"):
        resume_state().load_state_dict(state)


def test_epoch_budget_is_verified_against_standard_state_without_rewriting_it():
    state = resume_state().state_dict()
    trainer = control_trainer()
    trainer.state.max_steps = 20
    target = resume_state(trainer)
    target.load_state_dict(state)
    assert trainer.args.num_train_epochs == 3
    assert trainer.args.max_steps == -1
    assert trainer.state.max_steps == 20
    with pytest.raises(ValueError, match="原始核定预算"):
        target.validate_loaded_state()
    trainer.state.max_steps = 18
    target.validate_loaded_state()
    assert target.restored_planned_steps == 18


def test_sampler_tracking_preserves_real_trl_index_order_and_global_rng():
    baseline = control_trainer()._get_train_sampler()
    trainer = control_trainer()
    controller = resume_state(trainer, track_sampler=False)
    controller.install_sampler_tracking()
    wrapped = trainer._get_train_sampler()
    before_rng = torch.get_rng_state().clone()
    assert len(wrapped) == len(baseline)
    assert list(wrapped) == list(baseline)
    assert list(wrapped) == list(baseline)
    assert torch.equal(before_rng, torch.get_rng_state())


def test_tracked_sampler_exposes_original_generator_for_accelerate_sync():
    controller = resume_state(track_sampler=False)
    controller.install_sampler_tracking()
    wrapped = controller.trainer._get_train_sampler()
    assert isinstance(wrapped.generator, torch.Generator)
    assert wrapped.generator is controller.sampler.generator
    assert wrapped.shuffle is controller.sampler.shuffle
    assert wrapped.num_samples == controller.sampler.num_samples


@pytest.mark.parametrize("epoch", [0.5, 1.0])
def test_sampler_resume_rebuilds_current_or_next_epoch_without_reordering(epoch):
    source = resume_state(track_sampler=False)
    source.install_sampler_tracking()
    wrapped_source = source.trainer._get_train_sampler()
    first_epoch = list(wrapped_source)
    source.trainer.state.epoch = epoch
    state = source.state_dict()
    expected = first_epoch if epoch == 0.5 else list(wrapped_source)

    target = resume_state(track_sampler=False)
    target.install_sampler_tracking()
    target.load_state_dict(state)
    restored_indices = list(target.trainer._get_train_sampler())
    assert target.pending_sampler_state is None
    assert restored_indices == expected
    # Mid-epoch replay is consumed by Trainer's existing batch skipping, not
    # by changing RepeatSampler's group/reuse contract.
    if epoch == 0.5:
        assert restored_indices[12:] == first_epoch[12:]


def test_nonzero_progress_without_sampler_tracking_cannot_be_saved():
    with pytest.raises(ValueError, match="采样器尚未接线"):
        resume_state(track_sampler=False).state_dict()


@pytest.mark.parametrize("completed_epochs,consumed_batches", [(1, 4), (2, 0), (2, 4)])
def test_multiple_epoch_resume_preserves_dataloader_position_with_accelerate_skip(
    completed_epochs, consumed_batches,
):
    from accelerate import skip_first_batches
    from torch.utils.data import DataLoader

    def controller_and_loader():
        trainer = control_trainer(microstep=0, global_step=0)
        trainer.train_dataset = range(8)
        trainer.num_generations = 2
        trainer.num_iterations = 1
        trainer.args.steps_per_generation = 2
        trainer.args.generation_batch_size = 4
        trainer.args.per_device_train_batch_size = 2
        trainer.state.max_steps = 24
        trainer.state.epoch = 0.0
        trainer._buffered_inputs = trainer._buffered_inputs[:2]
        controller = resume_state(trainer, track_sampler=False)
        controller.install_sampler_tracking()
        loader = DataLoader(
            trainer.train_dataset, batch_size=2,
            sampler=trainer._get_train_sampler(),
            generator=torch.Generator().manual_seed(13),
        )
        return controller, loader

    source, original_loader = controller_and_loader()
    batches_per_epoch = len(original_loader)
    assert batches_per_epoch == 16
    for _ in range(completed_epochs):
        list(original_loader)
    if consumed_batches:
        original_iterator = iter(original_loader)
        for _ in range(consumed_batches):
            next(original_iterator)
    else:
        original_iterator = None
    source.trainer._step = completed_epochs * batches_per_epoch + consumed_batches
    source.trainer.state.global_step = source.trainer._step // 2
    source.trainer.state.epoch = completed_epochs + consumed_batches / batches_per_epoch
    state = source.state_dict()
    expected = [batch.tolist() for batch in (
        original_iterator if original_iterator is not None else iter(original_loader))]

    restored, restored_loader = controller_and_loader()
    restored.load_state_dict(state)
    restored.trainer.state.global_step = source.trainer.state.global_step
    restored.validate_loaded_state()
    actual = [batch.tolist() for batch in skip_first_batches(
        restored_loader, num_batches=consumed_batches)]
    assert actual == expected
    assert restored.trainer.args.num_train_epochs == 3
    assert restored.trainer.args.max_steps == -1
    assert restored.trainer.state.max_steps == 24


@pytest.mark.parametrize("missing", ["epoch_start", "generator_current"])
def test_missing_local_sampler_rng_is_not_replaced_by_a_new_seed(missing):
    state = resume_state().state_dict()
    state["sampler"][missing] = None
    with pytest.raises(ValueError, match="采样器随机状态"):
        resume_state().load_state_dict(state)


def test_zero_variance_patience_is_consumed_across_resume():
    guard = RewardVarianceGuard(patience=3)
    control = SimpleNamespace(should_training_stop=False)
    for step in (1, 2):
        guard.on_log(None, SimpleNamespace(global_step=step), control,
                     logs={"frac_reward_zero_std": 1.0})
    state = resume_state(guard=guard).state_dict()
    restored_guard = RewardVarianceGuard(patience=3)
    resume_state(guard=restored_guard).load_state_dict(state)
    assert restored_guard.streak == 2
    restored_guard.on_log(None, SimpleNamespace(global_step=3), control,
                          logs={"frac_reward_zero_std": 1.0})
    assert control.should_training_stop and restored_guard.tripped
    assert restored_guard.stop_step == 3


@pytest.mark.parametrize("flag", ["tripped", "converged"])
def test_protection_stopped_guard_cannot_be_bypassed_by_resume(flag):
    guard = RewardVarianceGuard(patience=3)
    setattr(guard, flag, True)
    state = resume_state(guard=guard).state_dict()
    with pytest.raises(ValueError):
        resume_state(guard=RewardVarianceGuard(patience=3)).load_state_dict(state)


@pytest.mark.parametrize("key,value", [
    ("schema_version", True), ("schema_version", 1.0),
    ("patience", True), ("streak", 3), ("streak", 4),
])
def test_guard_rejects_noninteger_contract_and_exhausted_untripped_state(key, value):
    # patience=1 makes True numerically equal to the expected configuration,
    # so rejection must check types rather than relying on equality alone.
    patience = 1 if key == "patience" else 3
    guard = RewardVarianceGuard(patience=patience)
    state = guard.state_dict()
    state[key] = value
    assert state["tripped"] is False and state["converged"] is False
    with pytest.raises(ValueError):
        guard.load_state_dict(state)
    assert guard.streak == 0


def test_canary_history_round_trip_is_not_new_evaluation():
    # Already-computed monitoring metadata only; no evaluator or model is run.
    def callback():
        return CanaryCallback(SimpleNamespace(canary_task_ids=["control"]),
                              steps=10, min_reward=0.0, patience=3)

    canary = callback()
    canary._below_streak = 2
    canary._last_evaluated_step = 20
    canary._results = [CanaryResult(step, -0.5, 0.0, {"control": -0.5})
                       for step in (10, 20)]
    state = resume_state(canary=canary).state_dict()
    target = callback()
    resume_state(canary=target).load_state_dict(state)
    assert target._below_streak == 2
    assert target.state_dict() == canary.state_dict()
    state["canary"]["tripped"] = True
    with pytest.raises(ValueError, match="Canary"):
        resume_state(canary=callback()).load_state_dict(state)


def test_audit_inheritance_does_not_open_or_clear_the_old_attempt(tmp_path):
    old_path = tmp_path / "old-attempt" / "rollouts.jsonl"
    old_auditor = RolloutAuditWriter(old_path, num_generations=3)
    # Not a rollout: sentinel tests that restore never truncates the old file.
    old_path.write_text("old-attempt-preservation-sentinel\n", encoding="utf-8")
    old_bytes = old_path.read_bytes()
    state = resume_state(auditor=old_auditor).state_dict()
    new_auditor = RolloutAuditWriter(tmp_path / "new-attempt" / "rollouts.jsonl",
                                    num_generations=3)
    restored = resume_state(auditor=new_auditor)
    restored.load_state_dict(state)
    assert old_path.read_bytes() == old_bytes
    assert new_auditor.path.read_bytes() == b""
    assert restored.inherited_audit["committed_global_step"] == 2
    assert restored.inherited_audit["source_summary"] == old_auditor.summary()


def frozen_reference_trainer():
    trainer = control_trainer()
    trainer.beta = 0.04
    trainer.ref_model = torch.nn.Linear(2, 2, bias=False).requires_grad_(False)
    return trainer


def test_frozen_reference_hash_does_not_change_rng_or_accept_policy_weights():
    trainer = frozen_reference_trainer()
    before_rng = torch.get_rng_state().clone()
    original = reference_weight_fingerprint(trainer)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert original["kind"] == "frozen_reference"
    state = resume_state(trainer)
    with torch.no_grad():
        trainer.ref_model.weight.add_(1)
    assert reference_weight_fingerprint(trainer) != original
    with pytest.raises(ValueError, match="参考"):
        state.validate_loaded_state()


@pytest.mark.parametrize("violation", ["trainable", "in_optimizer"])
def test_reference_must_be_frozen_and_excluded_from_optimizer(violation):
    trainer = frozen_reference_trainer()
    if violation == "trainable":
        trainer.ref_model.requires_grad_(True)
    else:
        trainer.optimizer = SimpleNamespace(param_groups=[
            {"params": list(trainer.ref_model.parameters())}])
    with pytest.raises(ValueError, match="冻结且排除优化器"):
        reference_weight_fingerprint(trainer)
