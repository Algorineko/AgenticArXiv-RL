"""DDP 拓扑、全局 batch、主进程副作用与轻量启动器回归。"""

import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from agentic_arxiv.training.distributed import TrainingContext, grpo_generation_batch_size
from agentic_arxiv.training.launch import build_launch, main as launch
from agentic_arxiv.training.observability import RewardComponentTracker


@pytest.fixture(autouse=True)
def clean_launch_environment(monkeypatch):
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "CUDA_VISIBLE_DEVICES",
                 "ACCELERATE_USE_FSDP", "ACCELERATE_USE_DEEPSPEED"):
        monkeypatch.delenv(name, raising=False)


def test_default_context_keeps_single_process():
    context = TrainingContext.from_environment()
    assert context == TrainingContext()
    assert context.is_main and not context.is_distributed
    assert context.effective_batch_size(1, 8) == 8
    assert context.trainer_kwargs(stage="sft") == {}
    assert context.resource_overrides() == {}


def test_global_rank_is_not_local_device_index(monkeypatch):
    for name, value in {"WORLD_SIZE": "8", "RANK": "5", "LOCAL_RANK": "1"}.items():
        monkeypatch.setenv(name, value)
    context = TrainingContext.from_environment()
    assert context == TrainingContext(8, 5, 1)
    assert not context.is_main and context.is_distributed


@pytest.mark.parametrize("environment", [
    {"LOCAL_RANK": "0"}, {"WORLD_SIZE": "4", "RANK": "0"},
    {"WORLD_SIZE": "x", "RANK": "0", "LOCAL_RANK": "0"},
    {"WORLD_SIZE": "0", "RANK": "0", "LOCAL_RANK": "0"},
    {"WORLD_SIZE": "4", "RANK": "4", "LOCAL_RANK": "0"},
    {"WORLD_SIZE": "4", "RANK": "0", "LOCAL_RANK": "-1"},
])
def test_invalid_launcher_environment_fails_before_model_loading(monkeypatch, environment):
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(SystemExit):
        TrainingContext.from_environment()


def test_ddp_trainer_flags_and_effective_batch_do_not_change_algorithm_defaults():
    context = TrainingContext(4, 0, 0)
    assert context.effective_batch_size(1, 2) == 8
    assert context.resource_overrides() == {"world_size": {"baseline": 1, "requested": 4}}
    assert context.trainer_kwargs(stage="sft") == {
        "ddp_find_unused_parameters": False, "ddp_broadcast_buffers": False,
        "average_tokens_across_devices": True,
    }
    assert "average_tokens_across_devices" not in context.trainer_kwargs(stage="grpo")


@pytest.mark.parametrize("world,batch,accum,generations,expected", [
    (1, 4, 1, 4, 4), (4, 1, 1, 4, 4), (2, 1, 2, 4, 4), (4, 4, 1, 4, 16),
])
def test_grpo_groups_may_span_ranks_and_accumulation(world, batch, accum, generations, expected):
    assert grpo_generation_batch_size(batch, accum, generations, TrainingContext(world)) == expected


@pytest.mark.parametrize("batch,accum,generations", [(1, 1, 4), (0, 2, 4), (1, 0, 4), (1, 2, 1)])
def test_invalid_grpo_global_batch_is_not_silently_increased(batch, accum, generations):
    with pytest.raises(SystemExit):
        grpo_generation_batch_size(batch, accum, generations, TrainingContext(2))


@pytest.mark.local
def test_device_is_selected_before_quantized_load():
    torch = pytest.importorskip("torch")
    with patch.object(torch.cuda, "is_available", return_value=True), \
            patch.object(torch.cuda, "device_count", return_value=4), \
            patch.object(torch.cuda, "set_device") as select:
        TrainingContext(8, 5, 1).prepare_device(qlora=True)
    select.assert_called_once_with(1)


@pytest.mark.local
def test_missing_local_device_is_a_clear_error():
    torch = pytest.importorskip("torch")
    with patch.object(torch.cuda, "is_available", return_value=True), \
            patch.object(torch.cuda, "device_count", return_value=1), \
            pytest.raises(SystemExit, match="LOCAL_RANK"):
        TrainingContext(4, 1, 1).prepare_device(qlora=True)


@pytest.mark.parametrize("variable", ["ACCELERATE_USE_FSDP", "ACCELERATE_USE_DEEPSPEED"])
def test_sharded_qlora_fails_before_cuda_import(monkeypatch, variable):
    monkeypatch.setenv(variable, "true")
    with pytest.raises(SystemExit, match="DDP"):
        TrainingContext(4).prepare_device(qlora=True)


def test_validate_trainer_requires_global_token_loss_for_distributed_sft():
    trainer = SimpleNamespace(
        accelerator=SimpleNamespace(num_processes=4, process_index=0, distributed_type="MULTI_GPU"),
        args=SimpleNamespace(average_tokens_across_devices=True),
        model_accepts_loss_kwargs=True, compute_loss_func=None,
    )
    context = TrainingContext(4)
    context.validate_trainer(trainer, stage="sft")
    trainer.model_accepts_loss_kwargs = False
    with pytest.raises(SystemExit, match="num_items_in_batch"):
        context.validate_trainer(trainer, stage="sft")
    trainer.accelerator.num_processes = 1
    with pytest.raises(SystemExit, match="拓扑"):
        context.validate_trainer(trainer, stage="grpo")


@pytest.mark.parametrize("backend", ["FSDP", "DEEPSPEED"])
def test_multiturn_grpo_rejects_collective_generation_backends(backend):
    trainer = SimpleNamespace(accelerator=SimpleNamespace(
        num_processes=4, process_index=0, distributed_type=backend,
    ))
    with pytest.raises(SystemExit, match="DDP"):
        TrainingContext(4).validate_trainer(trainer, stage="grpo")


def test_only_main_executes_side_effect_and_broadcasts_success():
    group = Mock()
    operation = Mock(return_value={"saved": True})
    with patch.object(TrainingContext, "_process_group", return_value=group):
        assert TrainingContext(4).run_on_main(operation) == {"saved": True}
    operation.assert_called_once_with()
    assert group.broadcast_object_list.call_args.args[0][0] == {"ok": True, "result": {"saved": True}}


def test_worker_receives_result_without_writing():
    group = Mock()
    group.broadcast_object_list.side_effect = lambda payload, src: payload.__setitem__(0, {"ok": True, "result": 42})
    operation = Mock(side_effect=AssertionError("worker must not write"))
    with patch.object(TrainingContext, "_process_group", return_value=group):
        assert TrainingContext(4, 2, 2).run_on_main(operation) == 42
    operation.assert_not_called()


def test_main_failure_is_broadcast_before_reraise():
    group = Mock()
    with patch.object(TrainingContext, "_process_group", return_value=group):
        with pytest.raises(ValueError, match="save failed"):
            TrainingContext(4).run_on_main(Mock(side_effect=ValueError("save failed")))
    assert group.broadcast_object_list.call_args.args[0][0] == {"ok": False, "error": "ValueError: save failed"}


def test_worker_does_not_hang_when_main_verification_fails():
    group = Mock()
    group.broadcast_object_list.side_effect = lambda payload, src: payload.__setitem__(0, {"ok": False, "error": "verification failed"})
    with patch.object(TrainingContext, "_process_group", return_value=group):
        with pytest.raises(RuntimeError, match="verification failed"):
            TrainingContext(4, 1, 1).run_on_main(Mock())


def test_tracker_does_not_merge_distinct_groups_of_the_same_task():
    tracker = RewardComponentTracker()
    tracker.record_group(["t"] * 8, [0.0] * 4 + [1.0] * 4, num_generations=4)
    summary = tracker.task_summary()["tasks"]["t"]
    assert summary["sample_count"] == 8 and summary["group_count"] == 2
    assert summary["zero_std_fraction"] == 1.0
    assert summary["informative_group_fraction"] == 0.0


@pytest.mark.parametrize("ids,rewards,g", [(["a"], [1.0], 4), (["a", "b"], [1, 2], 2), (["a"], [], 1)])
def test_tracker_rejects_partial_or_mixed_generation_groups(ids, rewards, g):
    with pytest.raises(ValueError):
        RewardComponentTracker().record_group(ids, rewards, num_generations=g)


def test_launcher_keeps_training_arguments_and_scopes_gpu_selection():
    args, command, env = build_launch([
        "--gpus", "0,1,2,3", "sft", "--max-length", "9216", "--grad-accum", "2",
    ])
    assert args.num_processes == 4 and env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert "--nproc-per-node=4" in command
    assert command[-8:] == ["--module", "agentic_arxiv", "train", "sft", "--max-length", "9216", "--grad-accum", "2"]


@pytest.mark.parametrize("stage", ["sft", "dpo", "grpo"])
def test_launcher_dry_run_has_no_process_or_file_side_effect(tmp_path, capsys, stage):
    with patch("agentic_arxiv.training.launch.subprocess.call") as call:
        assert launch(["--gpus", "0,1,2,3", "--dry-run", "--log-dir", str(tmp_path / "logs"),
                       stage, "--batch-size", "1", "--grad-accum", "8"]) == 0
    call.assert_not_called()
    result = json.loads(capsys.readouterr().out)
    assert "--tee=3" in result["command"]
    assert result["command"][-5:] == [stage, "--batch-size", "1", "--grad-accum", "8"]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("arguments", [
    ["--num-processes", "1", "sft"], ["--gpus", "0,0,1,2", "sft"],
    ["--gpus", "0,1", "grpo"], ["--gpus", "-1", "sft"],
])
def test_launcher_rejects_invalid_device_or_process_counts(arguments):
    with pytest.raises(SystemExit) as error:
        build_launch(arguments)
    assert error.value.code == 2


def test_launcher_rejects_nested_torchrun(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "0")
    with pytest.raises(SystemExit) as error:
        build_launch(["sft"])
    assert error.value.code == 2


def test_launcher_propagates_worker_failure():
    with patch("agentic_arxiv.training.launch.subprocess.call", return_value=17) as call:
        assert launch(["--num-processes", "2", "--gpus", "1,2", "grpo", "--batch-size", "2"]) == 17
    assert call.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "1,2"


@pytest.mark.train
def test_installed_trl_configs_support_the_explicit_ddp_fields():
    trl = pytest.importorskip("trl")
    from agentic_arxiv.training.mainline_config import require_supported_config

    context = TrainingContext(4)
    for stage, config in (("sft", trl.SFTConfig), ("grpo", trl.GRPOConfig)):
        require_supported_config(config, context.trainer_kwargs(stage=stage))


def test_launcher_help_and_dry_run_are_lightweight_outside_repository(tmp_path):
    probe = """
import sys
from pathlib import Path
def audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        assert Path(args[0]).name != '.env'
sys.addaudithook(audit)
from agentic_arxiv.training.launch import main
try:
    assert main(sys.argv[1:]) == 0
except SystemExit as exc:
    assert exc.code == 0
assert not {'torch', 'transformers', 'trl', 'accelerate', 'datasets', 'peft'} & set(sys.modules)
"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    for arguments in (["--help"], ["--dry-run", "--gpus", "0,1,2,3", "sft", "--max-length", "9216"]):
        result = subprocess.run([sys.executable, "-I", "-c", probe, *arguments], cwd=tmp_path,
                                env=environment, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())
