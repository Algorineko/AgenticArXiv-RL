"""CPU-only reference initialization/ownership checks; no rollout or training."""

import ast
import inspect
from types import SimpleNamespace
import weakref

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")

from accelerate import Accelerator  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import set_seed  # noqa: E402

from agentic_arxiv.training import grpo  # noqa: E402
from agentic_arxiv.training.trl_compat import reference_weight_fingerprint  # noqa: E402

pytestmark = pytest.mark.train


def fresh_adapter(*args, **kwargs):
    # The base represents a fixed, already trained full-model input. Only the
    # freshly attached LoRA weights consume the current initialization RNG.
    base = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        base[0].weight.fill_(1)
    return get_peft_model(base, LoraConfig(r=1, target_modules=["0"]))


def freeze_trl_reference(policy):
    # This is the fixed TRL 0.29.1 construction boundary: copy default into a
    # ref adapter before Trainer installs its seed. No model forward is called.
    policy.add_adapter("ref", policy.peft_config["default"])
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if ".default." in name:
                reference = policy.get_parameter(name.replace(".default.", ".ref."))
                reference.copy_(parameter)
                reference.requires_grad_(False)
    return SimpleNamespace(
        beta=0.1, ref_model=None, model=policy, optimizer=None,
        accelerator=SimpleNamespace(unwrap_model=lambda model: model),
    )


def test_managed_fresh_lora_reference_ignores_prior_process_rng(monkeypatch):
    monkeypatch.setattr(grpo, "_load_policy", fresh_adapter)
    references = []
    for prior_seed in (13, 72, 901):
        set_seed(prior_seed)
        policy = grpo._load_policy_for_run("immutable-full-model", managed=object(), seed=42)
        references.append(reference_weight_fingerprint(freeze_trl_reference(policy)))
    assert references == [references[0]] * 3


def test_managed_lora_reference_still_depends_on_requested_seed(monkeypatch):
    monkeypatch.setattr(grpo, "_load_policy", fresh_adapter)
    references = []
    for seed in (42, 43):
        policy = grpo._load_policy_for_run("immutable-full-model", managed=object(), seed=seed)
        references.append(reference_weight_fingerprint(freeze_trl_reference(policy)))
    assert references[0] != references[1]


def test_unmanaged_policy_keeps_existing_rng_behavior(monkeypatch):
    monkeypatch.setattr(grpo, "_load_policy", fresh_adapter)
    set_seed(123)
    expected = fresh_adapter().state_dict()
    expected_rng = torch.get_rng_state().clone()
    set_seed(123)
    actual = grpo._load_policy_for_run("immutable-full-model", managed=None, seed=999)
    assert all(torch.equal(value, expected[name]) for name, value in actual.state_dict().items())
    assert torch.equal(torch.get_rng_state(), expected_rng)


@pytest.mark.parametrize("different_rank", [False, True])
def test_reference_fingerprints_are_gathered_before_any_rank_rejects(different_rank):
    fingerprint = {"sha256": "initial-reference", "parameter_count": 2}
    seen = []

    def gather(value):
        seen.append(value)
        return [fingerprint, {**fingerprint, "sha256": "other"} if different_rank else fingerprint]

    state = SimpleNamespace(reference_weights=fingerprint)
    execution = SimpleNamespace(gather=gather)
    if different_rank:
        with pytest.raises(ValueError, match="各 rank 的初始冻结参考权重不一致"):
            grpo._validate_reference_initialization(state, execution)
    else:
        grpo._validate_reference_initialization(state, execution)
    assert seen == [fingerprint]


def test_main_binds_reference_before_checkpoint_state_restore_and_checks_after_ddp():
    tree = ast.parse(inspect.getsource(grpo.main))
    calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id: node.lineno
             for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, (ast.Attribute, ast.Name))}
    assert calls["_load_policy_for_run"] < calls["GRPOTrainer"]
    assert calls["GRPOResumeState"] < calls["_validate_reference_initialization"]
    assert calls["_validate_reference_initialization"] < calls["attach_managed_training"]
    callback = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.ClassDef) and node.name == "ResumeBoundaryCallback")
    begin = next(node for node in callback.body if isinstance(node, ast.FunctionDef))
    assert begin.name == "on_train_begin"
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == "validate_loaded_state" for node in ast.walk(begin))


def test_replacing_reference_releases_accelerator_ownership_without_other_state():
    accelerator = Accelerator(cpu=True)
    policy = accelerator.prepare_model(torch.nn.Linear(2, 2))
    optimizer = accelerator.prepare_optimizer(torch.optim.SGD(policy.parameters(), lr=0.1))
    reference = accelerator.prepare_model(torch.nn.Linear(2, 2).requires_grad_(False),
                                          evaluation_mode=True)
    # A cycle forces the public memory cleanup to collect rather than relying
    # on refcount alone, as happens with prepared mixed-precision forwards.
    reference.collection_probe = [reference]
    dead_reference = weakref.ref(reference)
    dead_weight = weakref.ref(reference.weight)
    trainer = SimpleNamespace(model=policy, ref_model=reference, accelerator=accelerator)
    del reference
    registered_models = accelerator._models
    optimizers = accelerator._optimizers
    schedulers = accelerator._schedulers
    dataloaders = accelerator._dataloaders
    accelerator.step = 7

    grpo._release_managed_reference(trainer)

    assert trainer.ref_model is None
    assert dead_reference() is None and dead_weight() is None
    assert accelerator._models is registered_models
    assert any(model is policy for model in registered_models)
    assert accelerator._optimizers is optimizers and any(value is optimizer for value in optimizers)
    assert accelerator._schedulers is schedulers and accelerator._dataloaders is dataloaders
    assert accelerator.step == 7
    # The old object must be gone before the replacement allocates its weights.
    trainer.ref_model = accelerator.prepare_model(torch.nn.Linear(2, 2).requires_grad_(False),
                                                  evaluation_mode=True)
    assert sum(model is trainer.ref_model for model in registered_models) == 1


@pytest.mark.parametrize("registration", ["missing", "unsupported"])
def test_reference_release_refuses_unknown_accelerator_ownership(registration):
    policy, reference = object(), object()
    models = [policy] if registration == "missing" else ()
    trainer = SimpleNamespace(model=policy, ref_model=reference, accelerator=SimpleNamespace(
        _models=models, unwrap_model=lambda model, **kwargs: model,
    ))
    with pytest.raises(ValueError, match="Accelerator 模型注册"):
        grpo._release_managed_reference(trainer)
    assert trainer.ref_model is reference
    assert trainer.accelerator._models is models


def test_reference_release_cannot_remove_policy():
    policy = object()
    trainer = SimpleNamespace(model=policy, ref_model=policy)
    with pytest.raises(ValueError, match="独立释放"):
        grpo._release_managed_reference(trainer)
    assert trainer.ref_model is policy
