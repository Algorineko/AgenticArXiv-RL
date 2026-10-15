"""真实 PEFT 微型层保存/恢复接口回归，不运行训练、量化模型或生成。"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
pytest.importorskip("trl")

from agentic_arxiv.training import trl_compat  # noqa: E402,F401
from agentic_arxiv.training.checkpoint import (  # noqa: E402
    CheckpointError, install_managed_adapter_restore, preserve_adapter_restore_contract,
)
from trl import DPOTrainer, SFTTrainer  # noqa: E402

pytestmark = pytest.mark.train


def adapter_trainer(trainer_class, dtype, value):
    model = peft.get_peft_model(torch.nn.Sequential(torch.nn.Linear(2, 2)).to(dtype),
                                peft.LoraConfig(r=1, target_modules=["0"]),
                                autocast_adapter_dtype=False)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".default." in name:
                parameter.fill_(value)
    trainer = trainer_class.__new__(trainer_class)
    trainer.model = model
    trainer.args = SimpleNamespace(world_size=1)
    trainer.is_fsdp_enabled = False
    trainer.is_deepspeed_enabled = False
    trainer.is_fsdp_xla_v1_enabled = False
    trainer.optimizer = torch.optim.AdamW(parameter for parameter in model.parameters()
                                         if parameter.requires_grad)
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda step: 1.0)
    return trainer


@pytest.mark.parametrize("trainer_class", [SFTTrainer, DPOTrainer])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("safe_serialization", [True, False])
def test_real_single_adapter_restore_preserves_dtype_and_optimizer_state(
        tmp_path, trainer_class, dtype, safe_serialization):
    source = adapter_trainer(trainer_class, dtype, 4)
    source.model.save_pretrained(tmp_path, safe_serialization=safe_serialization)
    for parameter in source.optimizer.param_groups[0]["params"]:
        source.optimizer.state[parameter] = {
            "step": torch.tensor(2.0), "exp_avg": torch.full_like(parameter, 0.25),
            "exp_avg_sq": torch.full_like(parameter, 0.5),
        }
    torch.save(source.optimizer.state_dict(), tmp_path / "optimizer.pt")
    torch.save(source.lr_scheduler.state_dict(), tmp_path / "scheduler.pt")

    target = adapter_trainer(trainer_class, dtype, 1)
    identities = {name: id(parameter) for name, parameter in target.model.named_parameters()}
    original_loader = target.model.load_adapter
    install_managed_adapter_restore(target)
    # Both calls are the installed fixed-version Trainer interfaces. There is
    # no Trainer.train, forward pass, dataset, optimizer step, or CUDA session.
    target._load_from_checkpoint(str(tmp_path))
    assert all(parameter.dtype == dtype for parameter in target.model.parameters())
    target._load_optimizer_and_scheduler(str(tmp_path))

    source_parameters = dict(source.model.named_parameters())
    for name, parameter in target.model.named_parameters():
        assert id(parameter) == identities[name]
        assert parameter.dtype == dtype
        if ".default." in name:
            assert parameter.requires_grad
            assert torch.equal(parameter, source_parameters[name])
            assert parameter in target.optimizer.state
            restored = target.optimizer.state[parameter]
            assert restored["exp_avg"].dtype == dtype
            assert torch.equal(restored["exp_avg"], source.optimizer.state[source_parameters[name]]["exp_avg"])
            assert restored["step"].item() == 2
        else:
            assert not parameter.requires_grad
    assert target.model.active_adapters == ["default"]
    assert target.model.load_adapter == original_loader


@pytest.mark.parametrize("mutation", ["dtype", "parameter", "frozen", "optimizer", "active"])
def test_restore_guard_rejects_parameter_or_optimizer_contract_changes(mutation):
    target = adapter_trainer(SFTTrainer, torch.bfloat16, 1)
    model = target.model
    original_loader = model.load_adapter
    adapter = model.base_model.model[0].lora_A["default"]
    with pytest.raises(CheckpointError, match="受管 adapter 恢复改变"):
        with preserve_adapter_restore_contract(target, model):
            if mutation == "dtype":
                adapter.weight.data = adapter.weight.data.float()
            elif mutation == "parameter":
                adapter.weight = torch.nn.Parameter(adapter.weight.detach().clone())
            elif mutation == "frozen":
                adapter.weight.requires_grad_(False)
            elif mutation == "optimizer":
                target.optimizer.param_groups[0]["params"] = []
            else:
                model.base_model.active_adapter = "ref"
    assert model.load_adapter == original_loader


def test_restore_guard_restores_instance_loader_when_loading_raises():
    target = adapter_trainer(SFTTrainer, torch.bfloat16, 1)
    original = target.model.load_adapter
    with pytest.raises(ValueError, match="invalid checkpoint"):
        with preserve_adapter_restore_contract(target, target.model):
            raise ValueError("invalid checkpoint")
    assert target.model.load_adapter == original
    assert "load_adapter" not in vars(target.model)


def test_non_peft_loader_is_not_changed():
    original = lambda path, model=None: None
    trainer = SimpleNamespace(model=torch.nn.Linear(2, 2), _load_from_checkpoint=original)
    install_managed_adapter_restore(trainer)
    assert trainer._load_from_checkpoint is original
