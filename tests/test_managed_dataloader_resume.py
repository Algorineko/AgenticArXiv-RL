"""真实固定版本的配置、采样器和加载器索引回归，不加载模型或执行训练。"""

import ast
import inspect
from pathlib import Path
from types import MethodType

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("accelerate")
pytest.importorskip("trl")

from agentic_arxiv.training import dpo, sft, trl_compat  # noqa: E402,F401
from agentic_arxiv.training import dataloader_resume  # noqa: E402
from agentic_arxiv.training.checkpoint import CheckpointError  # noqa: E402
from agentic_arxiv.training.dataloader_resume import preserve_managed_dataloader_epoch  # noqa: E402
from agentic_arxiv.training.distributed import TrainingContext  # noqa: E402
from agentic_arxiv.training.mainline_config import stage_config  # noqa: E402
from accelerate import skip_first_batches  # noqa: E402
from accelerate.data_loader import prepare_data_loader  # noqa: E402
from transformers import Trainer  # noqa: E402
from trl import DPOConfig, DPOTrainer, GRPOTrainer, SFTConfig, SFTTrainer  # noqa: E402

pytestmark = pytest.mark.train
STAGES = {"sft": (sft, SFTConfig, SFTTrainer), "dpo": (dpo, DPOConfig, DPOTrainer)}
INDEX_COUNT = 128


def _trainer(stage, shards, budget_kind="epochs"):
    module, config_class, trainer_class = STAGES[stage]
    configured = stage_config(stage)
    namespace = {
        **configured["defaults"], "configured": configured,
        "out_path": Path("unused-index-only-output"), "backends": [], "managed": None,
        "actual_optim": configured["defaults"]["optim"], "execution": TrainingContext(shards),
        "_precision_flags": lambda: {"bf16": False, "fp16": False, "use_cpu": True},
        "precision_flags": lambda: {"bf16": False, "fp16": False, "use_cpu": True},
    }
    if budget_kind == "steps":
        namespace["max_steps"] = 3 * (INDEX_COUNT // shards // namespace["grad_accum"]) - 1
    tree = ast.parse(inspect.getsource(module.main))
    assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "config_kwargs"
                              for target in node.targets))
    # 执行生产入口的实际配置表达式，避免测试另配一个不同的 sampler/seed。
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "stage-config", "exec"), namespace)
    kwargs = namespace["config_kwargs"]
    if stage == "sft":
        kwargs.update(namespace["execution"].trainer_kwargs(stage="sft"))
    trainer = trainer_class.__new__(trainer_class)
    trainer.args = config_class(**kwargs)
    trainer.train_dataset = range(INDEX_COUNT)
    assert trainer.args.group_by_length is False
    assert trainer.args.accelerator_config.use_seedable_sampler is True
    return trainer


def _loader(trainer, shards, rank):
    args = trainer.args
    loader = torch.utils.data.DataLoader(
        trainer.train_dataset, sampler=trainer._get_train_sampler(),
        batch_size=args.per_device_train_batch_size, num_workers=args.dataloader_num_workers,
        drop_last=args.dataloader_drop_last,
    )
    return prepare_data_loader(
        loader, device=torch.device("cpu"), num_processes=shards, process_index=rank,
        split_batches=args.accelerator_config.split_batches,
        even_batches=args.accelerator_config.even_batches,
        use_seedable_sampler=args.accelerator_config.use_seedable_sampler,
        data_seed=args.data_seed, rng_types=[],
    )


def _indices(loader):
    return [int(index) for batch in loader for index in batch]


def _remaining_updates(trainer, loader, *, completed_steps=0, skip=skip_first_batches):
    args = trainer.args
    epochs, updates_per_epoch, _, _, _, _, total_steps = trainer.set_initial_training_values(
        args, loader, total_train_batch_size=(args.train_batch_size * args.gradient_accumulation_steps),
    )
    first_epoch = completed_steps // updates_per_epoch
    skipped_batches = (completed_steps % updates_per_epoch) * args.gradient_accumulation_steps
    updates = []
    for epoch in range(first_epoch, epochs):
        loader.set_epoch(epoch)
        epoch_loader = skip(loader, skipped_batches) if skipped_batches else loader
        batches = [tuple(int(index) for index in batch) for batch in epoch_loader]
        for offset in range(0, len(batches), args.gradient_accumulation_steps):
            updates.append(tuple(batches[offset:offset + args.gradient_accumulation_steps]))
            if completed_steps + len(updates) == total_steps:
                return updates
        skipped_batches = 0
    return updates


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("shards", [1, 4])
@pytest.mark.parametrize("budget_kind", ["steps", "epochs"])
@pytest.mark.parametrize("position", ["first_middle", "second_boundary", "second_middle", "third_middle"])
def test_resume_matches_remaining_indices_through_later_epochs(stage, shards, budget_kind, position):
    trainer = _trainer(stage, shards, budget_kind)
    original_args = trainer.args.to_dict()
    for rank in range(shards):
        continuous_loader = _loader(trainer, shards, rank)
        _, updates_per_epoch, _, _, _, _, total = trainer.set_initial_training_values(
            trainer.args, continuous_loader,
            total_train_batch_size=trainer.args.train_batch_size * trainer.args.gradient_accumulation_steps * shards,
        )
        completed = {"first_middle": 1, "second_boundary": updates_per_epoch,
                     "second_middle": updates_per_epoch + 1,
                     "third_middle": 2 * updates_per_epoch + 1}[position]
        continuous = _remaining_updates(trainer, continuous_loader)
        with preserve_managed_dataloader_epoch(trainer, enabled=True):
            # 取绑定到真实生产训练循环上的引用，不直接调用内部包装函数。
            skip = trainer._inner_training_loop.__func__.__globals__["skip_first_batches"]
            resumed = _remaining_updates(
                trainer, _loader(trainer, shards, rank), completed_steps=completed, skip=skip,
            )
        assert len(continuous) == total
        assert len(resumed) == total - completed
        assert resumed == continuous[completed:]
    assert trainer.args.to_dict() == original_args


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("shards", [1, 4])
def test_fixed_upstream_skip_loses_nonzero_epoch_without_managed_scope(stage, shards):
    trainer = _trainer(stage, shards)
    for rank in range(shards):
        loader = _loader(trainer, shards, rank)
        loader.set_epoch(1)
        expected = _indices(loader)[trainer.args.gradient_accumulation_steps:]
        source = _loader(trainer, shards, rank)
        source.set_epoch(1)
        skipped = skip_first_batches(source, trainer.args.gradient_accumulation_steps)
        assert skipped.iteration == 0
        assert _indices(skipped) != expected


@pytest.mark.parametrize("existing_instance_method", [False, True])
@pytest.mark.parametrize("raise_inside", [False, True])
def test_scope_preserves_code_metadata_other_instances_and_restores_exact_method(
        existing_instance_method, raise_inside):
    trainer = SFTTrainer.__new__(SFTTrainer)
    original = trainer._inner_training_loop
    if existing_instance_method:
        trainer._inner_training_loop = original
    other_trainers = [cls.__new__(cls) for cls in (SFTTrainer, DPOTrainer, GRPOTrainer)]
    globals_before = dict(original.__func__.__globals__)
    try:
        with preserve_managed_dataloader_epoch(trainer, enabled=True):
            scoped = trainer._inner_training_loop.__func__
            function = original.__func__
            assert scoped.__code__ is function.__code__
            assert scoped.__defaults__ is function.__defaults__
            assert scoped.__kwdefaults__ is function.__kwdefaults__
            assert scoped.__closure__ is function.__closure__
            assert scoped.__annotations__ == function.__annotations__
            assert inspect.signature(scoped) == inspect.signature(function)
            assert scoped.__module__ == function.__module__
            assert scoped.__qualname__ == function.__qualname__
            assert scoped.__doc__ == function.__doc__
            assert scoped.__wrapped__ is function
            assert scoped.__globals__ is not function.__globals__
            assert {key for key in globals_before if scoped.__globals__[key] is not globals_before[key]} == {
                "skip_first_batches",
            }
            assert all(other._inner_training_loop.__func__ is function for other in other_trainers)
            assert function.__globals__ == globals_before
            if raise_inside:
                raise ValueError("scope failure")
    except ValueError as exc:
        assert raise_inside and str(exc) == "scope failure"
    assert trainer._inner_training_loop == original
    if existing_instance_method:
        assert trainer.__dict__["_inner_training_loop"] is original
    else:
        assert "_inner_training_loop" not in trainer.__dict__
    assert original.__func__.__globals__ == globals_before


@pytest.mark.parametrize("package", dataloader_resume.PINNED_DATALOADER_VERSIONS)
def test_unknown_dependency_version_is_rejected_before_binding(package, monkeypatch):
    versions = {**dataloader_resume.PINNED_DATALOADER_VERSIONS, package: "unsupported"}
    monkeypatch.setattr(dataloader_resume, "version", versions.__getitem__)
    trainer = SFTTrainer.__new__(SFTTrainer)
    with pytest.raises(CheckpointError, match="固定依赖"):
        with preserve_managed_dataloader_epoch(trainer, enabled=True):
            pytest.fail("unsupported runtime entered training scope")
    assert "_inner_training_loop" not in trainer.__dict__


@pytest.mark.parametrize("trainer_class", [SFTTrainer, DPOTrainer, GRPOTrainer])
def test_disabled_scope_neither_checks_versions_nor_changes_any_trainer(trainer_class, monkeypatch):
    monkeypatch.setattr(dataloader_resume, "version", lambda name: pytest.fail("disabled scope imported training stack"))
    trainer = trainer_class.__new__(trainer_class)
    before = dict(trainer.__dict__)
    with preserve_managed_dataloader_epoch(trainer, enabled=False):
        assert trainer.__dict__ == before
    assert trainer.__dict__ == before


@pytest.mark.parametrize("change", ["override", "grpo", "skip_binding"])
def test_unknown_training_interface_is_rejected_before_binding(change, monkeypatch):
    trainer_class = GRPOTrainer if change == "grpo" else SFTTrainer
    trainer = trainer_class.__new__(trainer_class)
    if change == "override":
        trainer._inner_training_loop = MethodType(lambda self, **kwargs: None, trainer)
    elif change == "skip_binding":
        monkeypatch.setitem(Trainer._inner_training_loop.__globals__, "skip_first_batches", lambda *args: None)
    before = dict(trainer.__dict__)
    with pytest.raises(CheckpointError, match="不支持的 SFT/DPO"):
        with preserve_managed_dataloader_epoch(trainer, enabled=True):
            pytest.fail("unsupported interface entered training scope")
    assert trainer.__dict__ == before


@pytest.mark.parametrize("epoch", [None, -1, True, "1"])
def test_invalid_loader_epoch_is_not_silently_reset(epoch):
    trainer = _trainer("sft", 1)
    loader = _loader(trainer, 1, 0)
    loader.iteration = epoch
    with preserve_managed_dataloader_epoch(trainer, enabled=True):
        skip = trainer._inner_training_loop.__func__.__globals__["skip_first_batches"]
        with pytest.raises(CheckpointError, match="有效 epoch"):
            skip(loader, 8)


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("managed,resume", [(True, True), (True, False), (False, False)])
@pytest.mark.parametrize("raise_inside", [False, True])
def test_production_train_boundary_uses_and_releases_instance_scope(stage, managed, resume, raise_inside):
    module, _, _ = STAGES[stage]
    trainer = _trainer(stage, 1)
    original = trainer._inner_training_loop
    called = []

    def train_boundary(**kwargs):
        called.append(kwargs)
        loop = trainer._inner_training_loop.__func__
        assert (loop is not original.__func__) == (managed and resume)
        if managed and resume:
            expected_loader = _loader(trainer, 1, 0)
            expected_loader.set_epoch(1)
            expected = _indices(expected_loader)[8:]
            resumed_loader = _loader(trainer, 1, 0)
            resumed_loader.set_epoch(1)
            assert _indices(loop.__globals__["skip_first_batches"](resumed_loader, 8)) == expected
        if raise_inside:
            raise ValueError("train boundary failure")
        return object()

    trainer.train = train_boundary
    tree = ast.parse(inspect.getsource(module.main))
    boundaries = [node for node in ast.walk(tree) if isinstance(node, ast.With)
                  and any(isinstance(item.context_expr, ast.Call)
                          and isinstance(item.context_expr.func, ast.Name)
                          and item.context_expr.func.id == "preserve_managed_dataloader_epoch"
                          for item in node.items)]
    assert len(boundaries) == 1
    namespace = {**module.__dict__, "trainer": trainer, "managed": object() if managed else None,
                 "resume_from_checkpoint": "/previous/checkpoint-17" if resume else None}
    code = compile(ast.Module(body=boundaries, type_ignores=[]), "production-train-boundary", "exec")
    if raise_inside:
        with pytest.raises(ValueError, match="train boundary failure"):
            exec(code, namespace)
    else:
        exec(code, namespace)
    assert called == ([{"resume_from_checkpoint": "/previous/checkpoint-17"}] if resume else [{}])
    assert trainer._inner_training_loop == original
    assert "_inner_training_loop" not in trainer.__dict__
