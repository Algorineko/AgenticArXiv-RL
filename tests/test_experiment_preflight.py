"""Read-only delegation tests, without new research data, real tokenizer or model."""

import copy
import io
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from agentic_arxiv.experiments import preflight
from agentic_arxiv.training.mainline_config import stage_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def preserve_workspace():
    from agentic_arxiv.config import get_workspace, initialize_operation

    previous = get_workspace()
    try:
        yield
    finally:
        initialize_operation(previous, load_env=False, logging=False)


def payload(stage, workspace=ROOT):
    return {"stage": stage, "training": copy.deepcopy(stage_config(stage)),
            "workspace": str(workspace)}


def test_preflight_import_never_imports_training_stack_or_stage_modules(tmp_path):
    probe = """
import sys
import agentic_arxiv.experiments.preflight
assert not {'torch', 'transformers', 'trl', 'accelerate', 'datasets', 'peft'} & set(sys.modules)
assert 'agentic_arxiv.runtime.env' not in sys.modules
assert not any(name in sys.modules for name in ('agentic_arxiv.training.sft', 'agentic_arxiv.training.dpo', 'agentic_arxiv.training.grpo'))
"""
    result = subprocess.run([sys.executable, "-c", probe], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("stage", ["sft", "dpo", "grpo"])
def test_dispatch_uses_resolved_inputs_without_creating_output(tmp_path, monkeypatch, stage):
    data = payload(stage, tmp_path)
    data["training"]["defaults"]["verify"] = False
    model = tmp_path / "existing-model-reference"
    data["inputs"] = {"model": {"path": str(model)}}
    audit = Mock(return_value={"existing_audit": "passed"})
    monkeypatch.setattr(preflight, f"_{stage}", audit)
    result = preflight.run_preflight(data)
    assert result["status"] == "passed"
    assert result["model_weights_loaded"] is False and result["training_started"] is False
    assert audit.call_args.args[0]["model"] == str(model)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("change", [
    lambda data: data.update(stage="evaluate"),
    lambda data: data.update(unknown=True),
    lambda data: data["training"]["defaults"].update(unknown=1),
    lambda data: data["training"]["defaults"].update(skip_data_manifest_check=True),
    lambda data: data["training"].update(defaults=[]),
])
def test_invalid_configuration_fails_before_stage_audit(monkeypatch, change):
    data = payload("sft")
    change(data)
    audit = Mock()
    monkeypatch.setattr(preflight, "_sft", audit)
    with pytest.raises(ValueError):
        preflight.run_preflight(data)
    audit.assert_not_called()


def test_grpo_refuses_split_substitution_and_smoke_before_tokenizer(monkeypatch):
    data = payload("grpo")
    data["inputs"] = {"split": {"path": str(ROOT / "data/splits/v7_86.json")}}
    data["training"]["defaults"]["split"] = "data/splits/v2_62.json:train"
    tokenizer = Mock()
    monkeypatch.setattr(preflight, "_tokenizer", tokenizer)
    with pytest.raises(ValueError, match="切分文件"):
        preflight.run_preflight(data)
    data["inputs"] = {}
    data["training"]["defaults"]["smoke"] = True
    with pytest.raises(ValueError, match="冒烟"):
        preflight.run_preflight(data)
    tokenizer.assert_not_called()


def test_grpo_existing_heldout_split_is_rejected_without_tokenizer(monkeypatch):
    data = payload("grpo")
    data["training"]["defaults"].update(task_set="expanded", split="data/splits/v7_86.json:dev")
    tokenizer = Mock()
    monkeypatch.setattr(preflight, "_tokenizer", tokenizer)
    with pytest.raises(SystemExit, match="训练切分"):
        preflight.run_preflight(data)
    tokenizer.assert_not_called()


@pytest.mark.parametrize("verify_stage", [True, False])
def test_sft_reuses_manifest_messages_and_length_audits(monkeypatch, tmp_path, verify_stage):
    from agentic_arxiv.config import initialize_operation
    from agentic_arxiv.training import sft

    initialize_operation(tmp_path, load_env=False)
    data = payload("sft", tmp_path)["training"]
    model = tmp_path / "existing-model-reference"
    model.mkdir()
    data["defaults"]["model"] = str(model)
    data["defaults"]["data"] = str(ROOT / "data/sft/sft_v3_train_mix.jsonl")
    data["defaults"]["verify"] = verify_stage
    if verify_stage:
        data["defaults"]["snapshot"] = str(ROOT / "data/mock_arxiv_snapshot.json")
    # Sentinels stand for already audited inputs, not synthetic training rows.
    rows = []
    manifest = {"control_metadata": "manifest-result"}
    verify = Mock(return_value=manifest)
    load = Mock(return_value=rows)
    lengths = Mock()
    metadata = Mock()
    check_protocol = Mock()
    tokenizer = object()
    snapshot_check = Mock()
    monkeypatch.setattr(sft, "_verification_environment", snapshot_check)
    monkeypatch.setattr(sft, "_verify_data_manifest", verify)
    monkeypatch.setattr(sft, "_load_sft_rows", load)
    monkeypatch.setattr(sft, "_check_lengths", lengths)
    monkeypatch.setattr(sft, "_training_model_metadata", metadata)
    monkeypatch.setattr("agentic_arxiv.provenance.require_training_protocol", check_protocol)
    monkeypatch.setattr(preflight, "_tokenizer", Mock(return_value=(tokenizer, {"control": "protocol"})))
    result = preflight._sft(data["defaults"], data["trainer_parameters"])
    verify.assert_called_once()
    load.assert_called_once()
    lengths.assert_called_once_with(tokenizer, rows, data["defaults"]["max_length"])
    check_protocol.assert_called_once_with(manifest, {"control": "protocol"}, source_label="SFT 数据 manifest")
    metadata.assert_called_once_with(str(model), {"control": "protocol"})
    assert result["data_manifest"] == "passed"
    assert result["verification_snapshot"] == ("passed" if verify_stage else "not_required")
    if verify_stage:
        snapshot_check.assert_called_once_with(data["defaults"]["snapshot"], None)
    else:
        snapshot_check.assert_not_called()
    assert list(tmp_path.iterdir()) == [model]


def test_sft_resolved_snapshot_is_forwarded_without_changing_default_schema(tmp_path, monkeypatch):
    data = payload("sft", tmp_path)
    snapshot = ROOT / "data/mock_arxiv_snapshot.json"
    data["inputs"] = {"snapshot": {"path": str(snapshot)}}
    audit = Mock(return_value={})
    monkeypatch.setattr(preflight, "_sft", audit)
    preflight.run_preflight(data)
    assert audit.call_args.args[0]["snapshot"] == str(snapshot)
    assert "snapshot" not in stage_config("sft")["defaults"]


@pytest.mark.parametrize("snapshot", [None, "missing-snapshot.json", "malformed-snapshot.json"])
def test_sft_missing_or_invalid_verification_snapshot_fails_before_tokenizer(tmp_path, monkeypatch, snapshot):
    data = payload("sft", tmp_path)
    if snapshot:
        path = tmp_path / snapshot
        if snapshot == "malformed-snapshot.json":
            path.write_text("not json")
        data["inputs"] = {"snapshot": {"path": str(path)}}
    tokenizer = Mock()
    monkeypatch.setattr(preflight, "_tokenizer", tokenizer)
    with pytest.raises((ValueError, FileNotFoundError)):
        preflight.run_preflight(data)
    tokenizer.assert_not_called()


@pytest.mark.parametrize("verify", [True, False])
def test_dpo_reuses_existing_inspect_only_entry_without_managed_attempt(tmp_path, monkeypatch, verify):
    from agentic_arxiv.provenance import sha256_file
    from agentic_arxiv.runtime import env
    from agentic_arxiv.training import dpo

    audit = Mock()
    monkeypatch.setattr(dpo, "main", audit)
    replay_environment = env.MockArxivEnv
    replay = Mock(wraps=replay_environment)
    monkeypatch.setattr(env, "MockArxivEnv", replay)
    execute = Mock(side_effect=AssertionError("preflight must not execute tools"))
    save = Mock(side_effect=AssertionError("preflight must not write snapshots"))
    monkeypatch.setattr(replay_environment, "execute_tool", execute)
    monkeypatch.setattr(replay_environment, "save_snapshot", save)
    snapshot = ROOT / "tests/fixtures/t4_snapshot_for_t5.json"
    before = sha256_file(snapshot)
    data = payload("dpo", tmp_path)
    data["training"]["defaults"]["verify"] = verify
    # A disabled verifier must not even open an unusable snapshot.
    data["inputs"] = {"snapshot": {"path": str(snapshot if verify else ROOT / "README.md")}}
    result = preflight.run_preflight(data)
    assert audit.call_args.kwargs["inspect_only"] is True
    assert "managed_context" not in audit.call_args.kwargs
    assert result["checks"]["verification_snapshot"] == ("passed" if verify else "not_required")
    if verify:
        replay.assert_called_once_with(snapshot_path=snapshot, mode="replay")
    else:
        replay.assert_not_called()
    execute.assert_not_called()
    save.assert_not_called()
    assert sha256_file(snapshot) == before
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("snapshot", [None, "missing-snapshot.json", "not-json", "invalid-structure"])
def test_dpo_requires_loadable_verification_snapshot_before_package_or_tokenizer(tmp_path, monkeypatch, snapshot):
    from agentic_arxiv.training import dpo

    data = payload("dpo", tmp_path)
    if snapshot is None:
        # Even a valid configured fallback is not a frozen, mounted input.
        data["training"]["defaults"]["snapshot"] = str(ROOT / "tests/fixtures/t4_snapshot_for_t5.json")
    else:
        path = tmp_path / snapshot
        if snapshot in {"not-json", "invalid-structure"}:
            path.write_text("not json" if snapshot == "not-json" else "[]")
        data["inputs"] = {"snapshot": {"path": str(path)}}
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    audit = Mock()
    monkeypatch.setattr(dpo, "main", audit)
    with pytest.raises((ValueError, FileNotFoundError, AttributeError)):
        preflight.run_preflight(data)
    audit.assert_not_called()
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_dpo_without_verification_does_not_require_snapshot_input(tmp_path, monkeypatch):
    from agentic_arxiv.runtime import env
    from agentic_arxiv.training import dpo

    data = payload("dpo", tmp_path)
    data["training"]["defaults"]["verify"] = False
    replay, audit = Mock(), Mock()
    monkeypatch.setattr(env, "MockArxivEnv", replay)
    monkeypatch.setattr(dpo, "main", audit)
    assert preflight.run_preflight(data)["checks"]["verification_snapshot"] == "not_required"
    audit.assert_called_once()
    replay.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_dpo_corrupt_verification_snapshot_reports_machine_failure(tmp_path, monkeypatch, capsys):
    from agentic_arxiv.training import dpo

    data = payload("dpo", tmp_path)
    data["inputs"] = {"snapshot": {"path": str(ROOT / "README.md")}}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(data)))
    audit = Mock()
    monkeypatch.setattr(dpo, "main", audit)
    assert preflight.main() == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "failed" and result["stage"] == "dpo"
    assert "JSONDecodeError" in result["error"]
    audit.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_dpo_existing_parameter_guard_runs_before_loading_tokenizer(monkeypatch):
    data = payload("dpo")
    data["inputs"] = {"snapshot": {"path": str(ROOT / "tests/fixtures/t4_snapshot_for_t5.json")}}
    data["training"]["defaults"]["beta"] = -1.0
    tokenizer = Mock()
    monkeypatch.setattr(preflight, "_tokenizer", tokenizer)
    with pytest.raises(ValueError, match="beta"):
        preflight.run_preflight(data)
    tokenizer.assert_not_called()


def test_machine_result_does_not_mix_diagnostics_into_stdout(monkeypatch, capsys):
    data = payload("sft")
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(data)))

    def audit(value):
        print("existing audit diagnostic")
        return {"status": "passed", "stage": "sft"}

    monkeypatch.setattr(preflight, "run_preflight", audit)
    assert preflight.main() == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["status"] == "passed"
    assert "existing audit diagnostic" in output.err


def test_machine_failure_reports_json_and_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"stage":"sft","workspace":null}'))
    assert preflight.main() == 2
    output = capsys.readouterr()
    result = json.loads(output.out)
    assert result["status"] == "failed" and result["stage"] == "sft"
