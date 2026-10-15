"""SFT verification input binding, without loading weights or training."""

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agentic_arxiv.cli import parse_args
from agentic_arxiv.training import sft


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data/mock_arxiv_snapshot.json"


def context(snapshot=SNAPSHOT):
    inputs = {"snapshot": {"path": str(snapshot)}} if snapshot else {}
    return SimpleNamespace(identity={"inputs": {"inputs": inputs}})


def test_sft_snapshot_argument_is_workspace_scoped(tmp_path):
    _, args = parse_args(["train", "sft", "--workspace", str(tmp_path), "--snapshot", "snapshot.json"])
    assert args.snapshot == str(tmp_path / "snapshot.json")


def test_managed_verification_uses_frozen_snapshot_outside_current_workspace(tmp_path, monkeypatch):
    from agentic_arxiv.config import get_workspace, initialize_operation

    previous = get_workspace()
    factory = Mock()
    monkeypatch.setattr("agentic_arxiv.runtime.env.MockArxivEnv", factory)
    try:
        initialize_operation(tmp_path, load_env=False, logging=False)
        environment = sft._verification_environment(str(SNAPSHOT), context())
        assert environment is factory.return_value
        factory.assert_called_once_with(snapshot_path=SNAPSHOT, mode="replay")
        assert not list(tmp_path.iterdir())
    finally:
        initialize_operation(previous, load_env=False, logging=False)


def test_managed_verification_without_cli_snapshot_still_uses_frozen_input(monkeypatch):
    factory = Mock()
    monkeypatch.setattr("agentic_arxiv.runtime.env.MockArxivEnv", factory)
    assert sft._verification_environment(None, context()) is factory.return_value
    factory.assert_called_once_with(snapshot_path=SNAPSHOT, mode="replay")


def test_managed_verification_cannot_fall_back_or_substitute_snapshot(tmp_path):
    with pytest.raises(ValueError, match="明确 snapshot"):
        sft._verification_environment(None, context(None))
    with pytest.raises(ValueError, match="输入身份不一致"):
        sft._verification_environment(str(tmp_path / "other.json"), context())


def test_unmanaged_implicit_snapshot_keeps_legacy_verifier_fallback():
    assert sft._verification_environment(None, None) is None


def test_sft_checks_snapshot_before_trainer_and_passes_environment_to_verifier():
    calls = [node for node in ast.walk(ast.parse(inspect.getsource(sft.main)))
             if isinstance(node, ast.Call)]
    binding = next(node for node in calls if isinstance(node.func, ast.Name)
                   and node.func.id == "_verification_environment")
    trainer = next(node for node in calls if isinstance(node.func, ast.Name)
                   and node.func.id == "SFTTrainer")
    assert binding.lineno < trainer.lineno
    verify = next(node for node in calls if isinstance(node.func, ast.Attribute)
                  and node.func.attr == "verify_sft")
    assert any(keyword.arg == "env" and isinstance(keyword.value, ast.Name)
               and keyword.value.id == "verification_env" for keyword in verify.keywords)
