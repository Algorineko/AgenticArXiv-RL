"""Container-side, read-only training input audit. Importing is stdlib-only.

The host controller delegates this explicit entry point to a no-GPU, offline,
read-only container. No model, Trainer, optimizer, or research output is created.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import copy
import json
from pathlib import Path
import sys


def _configuration(payload):
    if not isinstance(payload, dict) or set(payload) - {"stage", "training", "inputs", "workspace"}:
        raise ValueError("预检只接受 stage/training/inputs/workspace")
    if payload.get("stage") not in {"sft", "dpo", "grpo"}:
        raise ValueError("只读训练预检阶段必须为 sft/dpo/grpo")
    if not isinstance(payload.get("workspace"), str) or not Path(payload["workspace"]).is_dir():
        raise ValueError("预检需要已存在的明确 workspace")
    from agentic_arxiv.config import initialize_operation, workspace_path
    from agentic_arxiv.training.mainline_config import _validate_section, stage_config

    initialize_operation(payload["workspace"], load_env=False, logging=False)
    stage = payload["stage"]
    training = copy.deepcopy(payload.get("training"))
    if not isinstance(training, dict) or not {"defaults", "trainer_parameters"} <= set(training):
        raise ValueError("预检 training 需要 defaults/trainer_parameters")
    baseline = stage_config(stage)
    # DPO generation_defaults describe a separate explicit data-generation job.
    # They are not executed here, but the common strict validator still sees them.
    if stage == "dpo" and "generation_defaults" not in training:
        training["generation_defaults"] = baseline["generation_defaults"]
    if set(training) != set(baseline):
        raise ValueError("预检 training 包含未知配置组")
    if any(not isinstance(value, dict) for value in training.values()):
        raise ValueError("预检配置组必须为对象")
    _validate_section(stage, training, baseline)
    defaults = training["defaults"]
    inputs = payload.get("inputs", {})
    if not isinstance(inputs, dict):
        raise ValueError("预检 inputs 必须为对象")
    if stage == "dpo" and defaults.get("verify") and "snapshot" not in inputs:
        raise ValueError("受管 DPO 阶段验证需要明确 inputs.snapshot 输入")
    for name in ("model", "base_model", "data", "data_manifest", "snapshot"):
        if name in inputs and (name in defaults or (stage == "sft" and name == "snapshot")):
            reference = inputs[name]
            path = reference.get("path") if isinstance(reference, dict) else reference
            if not isinstance(path, str) or not path:
                raise ValueError(f"预检输入 {name} 缺少解析后的实际路径")
            defaults[name] = str(workspace_path(path))
    if stage == "grpo" and "split" in inputs:
        reference = inputs["split"]
        path = reference.get("path") if isinstance(reference, dict) else reference
        split = defaults.get("split")
        if not isinstance(path, str) or not isinstance(split, str) or ":" not in split:
            raise ValueError("GRPO 预检需要明确 FILE:train/rl_train 切分")
        filename, subset = split.rsplit(":", 1)
        if workspace_path(filename).resolve() != workspace_path(path).resolve():
            raise ValueError("GRPO 预检切分文件与已解析输入不一致")
        defaults["split"] = f"{workspace_path(path)}:{subset}"
    if defaults.get("skip_data_manifest_check") or defaults.get("smoke"):
        raise ValueError("正式受管预检不允许跳过数据审计或使用冒烟模式")
    return stage, training


def _tokenizer(source):
    from transformers import AutoTokenizer
    from agentic_arxiv.provenance import require_training_protocol
    from agentic_arxiv.runtime.react_contract import protocol_metadata

    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    protocol = protocol_metadata(tokenizer)
    require_training_protocol(protocol, protocol, source_label="预检实际 tokenizer",
                              require_chat_template_hash=True)
    return tokenizer, protocol


def _sft(defaults, trainer_parameters):
    from agentic_arxiv.config import resolve_model_source, workspace_path
    from agentic_arxiv.provenance import require_training_protocol
    from agentic_arxiv.training.sft import (
        _check_lengths, _load_sft_rows, _training_model_metadata, _verify_data_manifest,
    )

    if trainer_parameters.get("completion_only_loss") is not True:
        raise ValueError("SFT 主线必须仅监督当前 assistant completion")
    if defaults.get("verify"):
        if not defaults.get("snapshot"):
            raise ValueError("受管 SFT 阶段验证需要明确 snapshot 输入")
        from agentic_arxiv.training.sft import _verification_environment

        _verification_environment(defaults["snapshot"], None)
    if not defaults.get("data"):
        raise ValueError("受管 SFT 预检需要明确 data 输入")
    data = workspace_path(defaults["data"])
    manifest_path = (workspace_path(defaults["data_manifest"]) if defaults.get("data_manifest")
                     else data.with_suffix(data.suffix + ".manifest.json"))
    manifest = _verify_data_manifest(data, manifest_path)
    source = resolve_model_source(defaults["model"], local_only=True)
    tokenizer, protocol = _tokenizer(source)
    require_training_protocol(manifest, protocol, source_label="SFT 数据 manifest")
    _training_model_metadata(source, protocol)
    rows = _load_sft_rows(data, expected_protocol=protocol)
    _check_lengths(tokenizer, rows, defaults["max_length"])
    return {"samples": len(rows), "data_manifest": "passed", "message_protocol": "passed",
            "tokenizer_protocol": "passed", "token_lengths": "passed",
            "verification_snapshot": "passed" if defaults.get("verify") else "not_required"}


def _dpo(defaults, trainer_parameters):
    if defaults.get("verify"):
        if not defaults.get("snapshot"):
            raise ValueError("受管 DPO 阶段验证需要明确 snapshot 输入")
        from agentic_arxiv.config import workspace_path
        from agentic_arxiv.runtime.env import MockArxivEnv

        # DPO 的 inspect_only 在阶段验证前返回，必须单独检查回放环境。
        MockArxivEnv(snapshot_path=workspace_path(defaults["snapshot"]), mode="replay")
    from agentic_arxiv.training.dpo import main as audit_dpo

    # This existing explicit branch returns before importing TRL/datasets,
    # constructing policies or creating output. Reuse its full parameter,
    # package, frozen-source and exact token-prefix checks rather than copying
    # a smaller approximation here. No managed context is supplied during check.
    audit_dpo(**{**defaults, "inspect_only": True})
    return {"preference_package": "passed", "sft_source": "passed",
            "tokenizer_protocol": "passed", "token_boundaries": "passed",
            "verification_snapshot": "passed" if defaults.get("verify") else "not_required"}


def _grpo(defaults, trainer_parameters):
    from agentic_arxiv.config import resolve_model_source, workspace_path
    from agentic_arxiv.llm.artifacts import artifact_metadata
    from agentic_arxiv.provenance import require_training_protocol
    from agentic_arxiv.runtime.react_contract import protocol_metadata
    from agentic_arxiv.tasks.reference import attach_expected_paper_ids
    from agentic_arxiv.tools.bootstrap import require_all_tools
    from agentic_arxiv.training.grpo import _load_tasks, _gold_completion_tokens
    from agentic_arxiv.training.grpo_reward import build_prompt_dataset

    require_all_tools("GRPO 只读预检")
    tasks = _load_tasks(defaults["task_set"], defaults["split"], smoke=False)
    source = resolve_model_source(defaults["model"], local_only=True)
    metadata = artifact_metadata(source, base_model=defaults.get("base_model"),
                                 allowed_stages={"sft_full", "sft_qlora", "dpo_full", "dpo_qlora"},
                                 expected_protocol=protocol_metadata(), local_base_only=True)
    if not defaults.get("snapshot"):
        raise ValueError("受管 GRPO 预检需要明确 snapshot 输入")
    snapshot = workspace_path(defaults["snapshot"])
    tasks = attach_expected_paper_ids(tasks, snapshot)
    required_turns = max((len(task.get("expected_tools") or []) + 1 for task in tasks), default=1)
    if defaults["max_turns"] < required_turns:
        raise ValueError(f"GRPO max_turns 小于标准链所需的 {required_turns}")
    rows = build_prompt_dataset(tasks)
    if not rows or any(row["prompt"][0].get("_task_id") != row["task_id"] for row in rows):
        raise ValueError("GRPO 任务 prompt 缺少可信隐藏 task_id")
    from datasets import Dataset

    dataset = Dataset.from_list(rows)
    if [row["prompt"][0].get("_task_id") for row in dataset] != [row["task_id"] for row in rows]:
        raise ValueError("datasets 转换后隐藏 task_id 元数据丢失")
    tokenizer, protocol = _tokenizer(source)
    require_training_protocol(metadata, protocol, source_label="GRPO 输入模型",
                              require_chat_template_hash=True)
    required_tokens = _gold_completion_tokens(tokenizer, tasks)
    if defaults["max_completion_length"] < required_tokens:
        raise ValueError(f"GRPO max_completion_length 小于标准动作所需的 {required_tokens}")
    return {"tasks": len(tasks), "task_ids": [task["id"] for task in tasks],
            "training_split": "passed", "snapshot_semantics": "passed", "prompt_identity": "passed",
            "tokenizer_protocol": "passed", "completion_length": "passed"}


def run_preflight(payload):
    """Use existing stage input audits without calling any stage's train entry."""
    stage, training = _configuration(payload)
    checks = {"sft": _sft, "dpo": _dpo, "grpo": _grpo}[stage](
        training["defaults"], training["trainer_parameters"])
    return {"schema_version": 1, "status": "passed", "stage": stage, "checks": checks,
            "model_weights_loaded": False, "training_started": False}


def main():
    stage = None
    try:
        payload = json.load(sys.stdin)
        if isinstance(payload, dict):
            stage = payload.get("stage")
        # Existing audit functions print diagnostics; stdout stays one machine result.
        with redirect_stdout(sys.stderr):
            result = run_preflight(payload)
    except (Exception, SystemExit) as exc:
        result = {"schema_version": 1, "status": "failed", "stage": stage,
                  "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
