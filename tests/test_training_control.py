"""Pure file protocol and no-model optimizer-boundary control regression."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agentic_arxiv.training.callbacks import ManagedStopCallback
from agentic_arxiv.training.control import read_stop_request, request_stop


def state(step=2, budget=10):
    return SimpleNamespace(global_step=step, max_steps=budget)


def control():
    return SimpleNamespace(should_training_stop=False, should_save=False)


def callback(tmp_path, **kwargs):
    return ManagedStopCallback(tmp_path, "run", "attempt", checkpoint_committed=lambda step:
                               tmp_path / f"checkpoint-{step}", **kwargs)


def test_stop_request_is_concurrently_idempotent_and_attempt_bound(tmp_path):
    with ThreadPoolExecutor(max_workers=8) as pool:
        requests = list(pool.map(lambda _: request_stop(tmp_path, "run", "attempt"), range(20)))
    assert len({row["request_id"] for row in requests}) == 1
    assert read_stop_request(tmp_path, "run", "other") is None
    newer = request_stop(tmp_path, "run", "other")
    assert newer["request_id"] != requests[0]["request_id"]
    assert read_stop_request(tmp_path, "run", "attempt") is None


def test_microbatches_do_not_read_or_communicate_and_save_precedes_stop(tmp_path):
    distributed = SimpleNamespace(is_main=True, run_on_main=Mock(side_effect=lambda op: op()))
    cb = callback(tmp_path, distributed=distributed)
    request_stop(tmp_path, "run", "attempt")
    flags = control()
    # Trainer's accumulation hook is inherited and deliberately has no control logic.
    substep = getattr(cb, "on_substep_end", lambda *args, **kwargs: None)
    substep(None, state(), flags)
    assert not flags.should_training_stop and not flags.should_save
    distributed.run_on_main.assert_not_called()
    cb.on_step_end(None, state(), flags)
    assert flags.should_save and not flags.should_training_stop
    assert json.loads((tmp_path / "stop-status.json").read_text())["status"] == "checkpoint_pending"
    count = distributed.run_on_main.call_count
    cb.on_step_end(None, state(), flags)
    assert distributed.run_on_main.call_count == count
    cb.on_save(None, state(), flags)
    assert flags.should_training_stop and cb.user_stopped
    assert cb.reason == "user_safe_stop"
    snapshot = json.loads((tmp_path / "stop-status.json").read_text())
    assert snapshot["checkpoint"].endswith("checkpoint-2")
    assert snapshot["run_id"] == "run" and snapshot["attempt_id"] == "attempt"


def test_old_request_does_not_stop_new_attempt(tmp_path):
    request_stop(tmp_path, "run", "old")
    cb = callback(tmp_path)
    flags = control()
    cb.on_step_end(None, state(), flags)
    assert not flags.should_save and not cb.user_stopped


@pytest.mark.parametrize("reason,budget,expected", [
    ("canary_threshold", 10, "canary_threshold"),
    (None, 2, "max_steps_or_epochs_completed"),
])
def test_protection_and_completed_budget_win_over_request(tmp_path, reason, budget, expected):
    request_stop(tmp_path, "run", "attempt")
    cb = callback(tmp_path, protection_reason=lambda: reason)
    flags = control()
    cb.on_step_end(None, state(budget=budget), flags)
    assert not flags.should_save and not cb.user_stopped and cb.reason == expected


def test_protection_detected_during_logging_wins_before_save_confirmation(tmp_path):
    request_stop(tmp_path, "run", "attempt")
    reason = [None]
    cb = callback(tmp_path, protection_reason=lambda: reason[0])
    flags = control()
    cb.on_step_end(None, state(), flags)
    reason[0] = "reward_variance_initial_signal_missing"
    flags.should_training_stop = True
    cb.on_save(None, state(), flags)
    assert not cb.user_stopped and cb.reason == reason[0]


def test_missing_commit_fails_and_failure_path_never_communicates_again(tmp_path):
    request_stop(tmp_path, "run", "attempt")
    distributed = SimpleNamespace(is_main=True, run_on_main=Mock(side_effect=lambda op: op()))
    cb = ManagedStopCallback(tmp_path, "run", "attempt", distributed=distributed,
                             checkpoint_committed=lambda step: None)
    flags = control()
    cb.on_step_end(None, state(), flags)
    with pytest.raises(RuntimeError, match="提交确认"):
        cb.on_save(None, state(), flags)
    assert not cb.user_stopped and not flags.should_training_stop
    count = distributed.run_on_main.call_count
    cb.on_step_end(None, state(), flags)
    cb.on_save(None, state(), flags)
    cb.mark_failed(RuntimeError("NCCL failure"), 2)
    assert distributed.run_on_main.call_count == count
    assert json.loads((tmp_path / "stop-status.json").read_text())["status"] == "failed"


def test_collective_failure_does_not_attempt_recovery_collective(tmp_path):
    distributed = SimpleNamespace(is_main=True, run_on_main=Mock(side_effect=RuntimeError("NCCL")))
    cb = callback(tmp_path, distributed=distributed)
    with pytest.raises(RuntimeError, match="NCCL"):
        cb.on_step_end(None, state(), control())
    distributed.run_on_main.assert_called_once()
    cb.on_step_end(None, state(), control())
    distributed.run_on_main.assert_called_once()


def test_later_training_failure_revokes_stop_confirmation(tmp_path):
    request_stop(tmp_path, "run", "attempt")
    cb = callback(tmp_path)
    flags = control()
    cb.on_step_end(None, state(), flags)
    cb.on_save(None, state(), flags)
    assert cb.user_stopped
    cb.mark_failed(ValueError("invalid loss"), 2)
    assert not cb.user_stopped
    assert json.loads((tmp_path / "stop-status.json").read_text())["status"] == "failed"


def test_worker_failure_never_overwrites_main_status(tmp_path):
    cb = callback(tmp_path, distributed=SimpleNamespace(is_main=False, run_on_main=Mock()))
    cb.mark_failed(RuntimeError("worker"), 3)
    assert not (tmp_path / "stop-status.json").exists()


def test_corrupted_current_request_fails_closed(tmp_path):
    request_stop(tmp_path, "run", "attempt")
    path = tmp_path / "stop-request.json"
    request = json.loads(path.read_text())
    request["schema_version"] = True
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="停止请求"):
        read_stop_request(tmp_path, "run", "attempt")


def _control_gloo_worker(rank, directory):
    """Only control metadata and CPU collectives; no model or training workload."""
    import datetime
    import os
    import sys

    import torch.distributed as dist
    from agentic_arxiv.training.distributed import TrainingContext

    root = Path(directory)
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    dist.init_process_group("gloo", init_method=(root / "rendezvous").as_uri(),
                            rank=rank, world_size=4, timeout=datetime.timedelta(seconds=20))
    try:
        topology = TrainingContext(4, rank, rank)
        if rank == 0:
            request_stop(root / "control", "run", "attempt")
        topology.gather("ready")
        cb = callback(root / "control", distributed=topology)
        flags = control()
        # Different local tool runtimes must not add different collectives.
        time.sleep(rank * 0.03)
        for _ in range(rank + 1):
            cb.on_substep_end(None, state(), flags)
        assert not flags.should_training_stop
        cb.on_step_end(None, state(), flags)
        assert flags.should_save and not flags.should_training_stop
        cb.on_save(None, state(), flags)
        assert cb.user_stopped and flags.should_training_stop
        # All ranks receive a main-side commit acknowledgement error.
        bad = ManagedStopCallback(root / "control", "run", "attempt", distributed=topology,
                                  checkpoint_committed=lambda step: None)
        flags = control()
        bad.on_step_end(None, state(), flags)
        try:
            bad.on_save(None, state(), flags)
        except RuntimeError:
            pass
        else:
            raise AssertionError("missing commit was accepted")
        assert not bad.user_stopped
        # No further communication may be attempted after the failure.
        bad._main = lambda operation: (_ for _ in ()).throw(AssertionError("unsafe collective"))
        bad.mark_failed(RuntimeError("NCCL error"), 2)
        bad.on_step_end(None, state(), flags)
        bad.on_save(None, state(), flags)
        (root / f"control-rank-{rank}.json").write_text(json.dumps({"passed": True}))
    finally:
        dist.destroy_process_group()


@pytest.mark.local
def test_four_rank_gloo_stop_decision_and_error_propagation(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("需要 CPU/Gloo 分布式后端")
    processes = torch.multiprocessing.spawn(_control_gloo_worker, args=(str(tmp_path),),
                                            nprocs=4, join=False)
    deadline = time.monotonic() + 90
    try:
        while not processes.join(timeout=5):
            assert time.monotonic() < deadline, "控制通信回归超时"
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
    assert all(json.loads((tmp_path / f"control-rank-{rank}.json").read_text())["passed"]
               for rank in range(4))
