"""审计并语言扩增 parametric v1 专家 seed。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from agentic_arxiv.data_pipeline.augment_sft_data import (
    WRAPPER_SETS,
    augment_validated_rows,
    split_assistant_action,
    split_prompt_task,
)
from agentic_arxiv.data_pipeline.common import canonical_hash, validate_data_protocol
from agentic_arxiv.provenance import (
    read_json, read_jsonl, require_training_protocol, sha256_file, write_json, write_jsonl,
)
from agentic_arxiv.runtime.react_contract import protocol_metadata, validate_react_messages


def validate_parametric_rows(
    rows: Iterable[Dict[str, Any]],
    split_payload: Dict[str, Any],
    manifest: Dict[str, Any],
    *,
    split_name: str,
    input_path: Path | None = None,
) -> List[Dict[str, Any]]:
    rows = list(rows)
    if not rows:
        raise ValueError("parametric seed 为空")
    if manifest.get("kind") != "train_only_parametric_expert_seed":
        raise ValueError(f"manifest kind 错误: {manifest.get('kind')!r}")
    require_training_protocol(manifest, protocol_metadata(), source_label="parametric manifest")
    validate_data_protocol(rows)
    if manifest.get("sample_rows") != len(rows):
        raise ValueError("manifest sample_rows 与实际行数不一致")
    if input_path is not None and manifest.get("output_sha256") != sha256_file(input_path):
        raise ValueError("parametric seed 的 SHA256 与 manifest 不一致")

    groups = split_payload["split"]
    train = set(groups["train"])
    heldout = set(groups["dev"] + groups["iid_test"] + groups["ood_test"])
    task_entries = manifest.get("tasks") or []
    task_manifest = {item["derived_task_id"]: item for item in task_entries}
    if len(task_manifest) != manifest.get("derived_tasks"):
        raise ValueError("manifest 派生任务数量或 id 唯一性错误")

    # The seed records the split file it was generated from.  Rebuilding the
    # name from the version number used to assume the `*_62.json` filename and
    # so rejected any newer split; take the actual filename instead.
    # ``--include-t5`` 的种子在生成端带 `_t5` 后缀（见 generate_parametric_sft_data），
    # 这里按 manifest 的标志还原，否则 T5 种子会被误判为来源不符。
    expected_source = f"{split_name}:train:parametric_v1"
    if manifest.get("include_t5"):
        expected_source += "_t5"
    source_ids = {row.get("source_task_id") for row in rows}
    if source_ids != set(task_manifest):
        raise ValueError("seed 的 source_task_id 与 manifest 派生任务不一致")
    if {row.get("source_split") for row in rows} != {expected_source}:
        raise ValueError("parametric seed 的 source_split 不匹配")

    counts = Counter(row["source_task_id"] for row in rows)
    for task_id, task_info in task_manifest.items():
        parent = task_info.get("parent_task_id")
        if parent not in train or parent in heldout:
            raise ValueError(f"派生任务父级不属于纯 train: {task_id} -> {parent}")
        expected_rows = max(1, len(task_info.get("steps") or []) + 1)
        if counts[task_id] != expected_rows:
            raise ValueError(
                f"派生任务决策行数错误: {task_id}, "
                f"expected={expected_rows}, actual={counts[task_id]}"
            )

    fingerprints = set()
    steps_seen: Dict[str, set] = {task_id: set() for task_id in task_manifest}
    for index, row in enumerate(rows):
        require_training_protocol(row, manifest, source_label=f"parametric row {index}")
        task_id = row["source_task_id"]
        task_info = task_manifest[task_id]
        if row.get("derived_task_id") != task_id:
            raise ValueError(f"row {index} 的 derived_task_id 不一致")
        if row.get("parent_task_id") != task_info["parent_task_id"]:
            raise ValueError(f"row {index} 的 parent_task_id 不一致")
        if row.get("generation_parameters") != task_info["parameters"]:
            raise ValueError(f"row {index} 的 generation_parameters 不一致")
        if row.get("dataset_stage") != "parametric_v1_expert_seed":
            raise ValueError(f"row {index} 的 dataset_stage 错误")

        messages = row.get("messages")
        validate_react_messages(messages, require_completion=True)
        _, task_text, _ = split_prompt_task(messages[0].get("content", ""))
        if task_text.strip() != task_info["task"].strip():
            raise ValueError(f"row {index} 的任务正文与 manifest 不一致")
        split_assistant_action(messages[-1].get("content", ""))

        fingerprint = row.get("sample_sha256")
        if fingerprint != canonical_hash(messages):
            raise ValueError(f"row {index} 的 sample_sha256 与 messages 不匹配")
        if fingerprint in fingerprints:
            raise ValueError(f"row {index} 指纹重复")
        fingerprints.add(fingerprint)
        steps_seen[task_id].add(row.get("trajectory_step"))

    for task_id, task_info in task_manifest.items():
        expected_rows = max(1, len(task_info.get("steps") or []) + 1)
        if steps_seen[task_id] != set(range(expected_rows)):
            raise ValueError(f"派生任务 trajectory_step 不连续: {task_id}")
    return rows


def main(
    *,
    input: str | Path = "data/sft/sft_react_messages_v3_parametric_seed.jsonl",
    manifest: str | Path | None = None,
    split_file: str | Path = "data/splits/v3_81.json",
    output: str | Path = "data/sft/sft_react_messages_v3_parametric_linguistic.jsonl",
    wrapper_set: str = "base",
    workspace: str | Path | None = None,
) -> None:
    """审计并语言扩增参数化专家数据；不覆盖来源 seed。"""
    from agentic_arxiv.config import initialize_operation, workspace_path

    initialize_operation(workspace)
    input_path = workspace_path(input, workspace=workspace)
    manifest_path = (
        workspace_path(manifest, workspace=workspace) if manifest
        else input_path.with_suffix(input_path.suffix + ".manifest.json")
    )
    split_path = workspace_path(split_file, workspace=workspace)
    output_path = workspace_path(output, workspace=workspace)
    for label, path in (
        ("input", input_path), ("manifest", manifest_path), ("split", split_path)
    ):
        if not path.exists():
            raise SystemExit(f"{label} 不存在: {path}")
    if input_path.resolve() == output_path.resolve():
        raise SystemExit("禁止覆盖 parametric seed 原件")
    out_manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if output_path.exists() or out_manifest_path.exists():
        raise SystemExit(f"拒绝覆盖已有数据或 manifest: {output_path}")

    rows = read_jsonl(input_path)
    manifest = read_json(manifest_path)
    split_payload = read_json(split_path)
    if manifest.get("split_sha256") != sha256_file(split_path):
        raise SystemExit("参数化 seed 声明的切分 SHA256 与实际切分不一致")
    validated = validate_parametric_rows(
        rows, split_payload, manifest, split_name=split_path.name,
        input_path=input_path,
    )
    augmented = augment_validated_rows(validated, wrapper_set=wrapper_set)
    for row in augmented:
        row["dataset_stage"] = "parametric_v1_linguistic"
    write_jsonl(output_path, augmented)

    derived_tasks = len({row["source_task_id"] for row in augmented})
    parent_tasks = len({row["parent_task_id"] for row in augmented})
    out_manifest = {
        "protocol_metadata": validate_data_protocol(augmented),
        "version": 1,
        "kind": "parametric_v1_linguistic",
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": sha256_file(manifest_path),
        "split_file": str(split_path),
        "split_sha256": sha256_file(split_path),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "seed_rows": len(rows),
        "output_rows": len(augmented),
        "derived_tasks": derived_tasks,
        "parent_train_tasks": parent_tasks,
        "wrapper_set": wrapper_set,
        "task_variants": len(WRAPPER_SETS[wrapper_set]),
        "thought_variants": 2,
        "unique_sample_fingerprints": len({row["sample_sha256"] for row in augmented}),
        "heldout_parent_overlap": 0,
        "note": (
            f"{len(augmented)} rows are {len(WRAPPER_SETS[wrapper_set]) * 2} linguistic views "
            f"of {len(rows)} decisions from {derived_tasks} derived tasks."
        ),
    }
    if manifest.get("include_t5"):
        out_manifest["include_t5"] = True
    write_json(out_manifest_path, out_manifest)

    print("PARAMETRIC_AUGMENTATION_OK")
    print(f"seed rows:       {len(rows)}")
    print(f"output rows:     {len(augmented)}")
    print(f"derived tasks:   {derived_tasks}")
    print(f"parent tasks:    {parent_tasks}")
    print("heldout overlap: 0")
    print(f"output:          {output_path}")
    print(f"manifest:        {out_manifest_path}")
