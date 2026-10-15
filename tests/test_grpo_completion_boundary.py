"""真实回调及生产完成边界检查，不训练、不生成或写入模型文件。"""

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("transformers")

from transformers.trainer_callback import (  # noqa: E402
    CallbackHandler, DefaultFlowCallback, TrainerControl, TrainerState,
)
from agentic_arxiv.training import grpo  # noqa: E402

pytestmark = pytest.mark.train


def completion_boundary(*, guard=None, canary=None, user_stopped=False,
                        actual_steps=25, planned_steps=25, managed=True, epoch_budget=False):
    # Execute the production post-train classification through its first final
    # write, without constructing a Trainer, dataset or research-model input.
    tree = ast.parse(inspect.getsource(grpo.main)).body[0]
    start = next(index for index, node in enumerate(tree.body)
                 if isinstance(node, ast.Assign) and any(
                     isinstance(target, ast.Name) and target.id == "actual_steps"
                     for target in node.targets))
    end = next(index for index, node in enumerate(tree.body)
               if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
               and isinstance(node.value.func, ast.Attribute)
               and isinstance(node.value.func.value, ast.Name)
               and node.value.func.value.id == "trainer"
               and node.value.func.attr == "save_model")
    verification = next(index for index, node in enumerate(tree.body)
                        if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                        and node.test.id == "verify")
    assert start < end < verification
    function = ast.parse("def finish():\n    pass\n").body[0]
    function.body = tree.body[start:end + 1]
    namespace = {
        "trainer": SimpleNamespace(
            state=SimpleNamespace(global_step=actual_steps, max_steps=planned_steps),
            save_model=Mock(),
        ),
        "managed": object() if managed else None,
        "max_steps": -1 if epoch_budget else planned_steps,
        "guard": guard, "canary_cb": canary,
        "stop_callback": SimpleNamespace(user_stopped=user_stopped),
        "checkpoint_adapter": SimpleNamespace(last_checkpoint="checkpoint-control"),
        "execution": SimpleNamespace(memory_summary=lambda: {}, run_on_main=lambda operation: operation()),
        "save_run_diagnostic": Mock(return_value={}),
        "record_training_result": Mock(), "training_elapsed_seconds": 1.0,
        "tracker": SimpleNamespace(save_task_summary=Mock(return_value="reward-summary-control")),
        "workspace_path": Path, "output_dir": "/unused-control-output",
        "current_protocol": {}, "source_metadata": {}, "auditor": None,
        "print": lambda *args, **kwargs: None,
    }
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, inspect.getfile(grpo), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("epoch_budget", [False, True])
def test_final_step_zero_variance_protection_never_reaches_success_export(epoch_budget):
    guard = grpo.RewardVarianceGuard()
    args = SimpleNamespace(logging_first_step=False, logging_strategy="steps",
                           eval_strategy="no", save_strategy="steps")
    state = TrainerState(max_steps=25, logging_steps=1, save_steps=100)
    control = TrainerControl()
    handler = CallbackHandler([DefaultFlowCallback(), guard], None, None, None, None)
    for step in range(1, 26):
        state.global_step = step
        control = handler.on_step_begin(args, state, control)
        control = handler.on_step_end(args, state, control)
        control = handler.on_log(args, state, control,
                                 {"frac_reward_zero_std": 1.0 if step >= 21 else 0.0})
    assert guard.converged and not guard.tripped and guard.stop_step == 25
    assert control.should_save and control.should_training_stop

    boundary = completion_boundary(guard=guard, epoch_budget=epoch_budget)
    boundary["finish"]()
    boundary["trainer"].save_model.assert_not_called()
    boundary["record_training_result"].assert_called_once()
    result = boundary["record_training_result"].call_args.kwargs
    assert result["training"] == "protection_stopped"
    assert result["validation"] == "not_run"
    assert result["reason"] == "reward_variance_window_saturated"
    assert result["global_step"] == result["planned_steps"] == 25
    assert result["checkpoint"] == "checkpoint-control"


@pytest.mark.parametrize("cause,expected,raises", [
    ("converged", "protection_stopped", False),
    ("tripped", "protection_stopped", True),
    ("canary", "protection_stopped", True),
    ("user", "user_safe_stop", False),
    ("unknown", "failed", False),
])
def test_other_managed_stops_keep_their_recorded_state(cause, expected, raises):
    guard = grpo.RewardVarianceGuard()
    if cause in {"converged", "tripped"}:
        setattr(guard, cause, True)
    boundary = completion_boundary(
        guard=guard, canary=SimpleNamespace(tripped=cause == "canary"),
        user_stopped=cause in {"user", "converged"}, actual_steps=24,
    )
    if raises:
        with pytest.raises(SystemExit, match="1"):
            boundary["finish"]()
    else:
        boundary["finish"]()
    boundary["trainer"].save_model.assert_not_called()
    boundary["record_training_result"].assert_called_once()
    assert boundary["record_training_result"].call_args.kwargs["training"] == expected


@pytest.mark.parametrize("managed,converged", [(True, False), (False, False), (False, True)])
def test_budget_completion_and_unmanaged_export_behavior_are_unchanged(managed, converged):
    guard = grpo.RewardVarianceGuard()
    guard.converged = converged
    boundary = completion_boundary(guard=guard, managed=managed)
    boundary["finish"]()
    boundary["trainer"].save_model.assert_called_once_with("/unused-control-output/final")
    if managed:
        assert boundary["record_training_result"].call_args.kwargs["training"] == "budget_completed"
