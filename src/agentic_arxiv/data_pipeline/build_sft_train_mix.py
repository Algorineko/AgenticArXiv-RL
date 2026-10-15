"""确定性混合原始 train 语言扩增与参数化语言扩增数据。"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from agentic_arxiv.config import initialize_operation, workspace_path
from agentic_arxiv.data_pipeline.common import canonical_hash, validate_data_protocol
from agentic_arxiv.provenance import (
    read_json,
    read_jsonl,
    require_training_protocol,
    sha256_file,
    write_json,
    write_jsonl,
)
from agentic_arxiv.runtime.react_contract import protocol_metadata, validate_react_messages


def build_mix(
    original_rows: Iterable[Dict[str, Any]],
    parametric_rows: Iterable[Dict[str, Any]],
    *,
    seed: int,
) -> List[Dict[str, Any]]:
    original_rows = list(original_rows)
    parametric_rows = list(parametric_rows)
    if not original_rows or not parametric_rows:
        raise ValueError("两个混合来源都不能为空")
    validate_data_protocol(original_rows + parametric_rows)

    mixed: List[Dict[str, Any]] = []
    seen = set()
    for source, rows in (
        ("original_train_linguistic", original_rows),
        ("parametric_v1_linguistic", parametric_rows),
    ):
        for index, original in enumerate(rows):
            row = dict(original)
            validate_react_messages(row.get("messages"), require_completion=True)
            fingerprint = canonical_hash(row.get("messages"))
            if row.get("sample_sha256") != fingerprint:
                raise ValueError(f"{source} row {index} 的 sample_sha256 不匹配")
            if fingerprint in seen:
                raise ValueError(f"两个来源存在重复 messages: {fingerprint}")
            seen.add(fingerprint)
            row["mixture_source"] = source
            mixed.append(row)

    random.Random(seed).shuffle(mixed)
    return mixed


def validate_source_manifest(
    path: Path, manifest_path: Path, *, expected_kind: str, expected_rows: int,
) -> Dict[str, Any]:
    manifest = read_json(manifest_path)
    require_training_protocol(manifest, protocol_metadata(), source_label="混合来源 manifest")
    if manifest.get("augmentation_kind") == "linguistic_semantics_preserving":
        actual_kind = "original_train_linguistic"
    else:
        actual_kind = manifest.get("kind")
    if actual_kind != expected_kind:
        raise ValueError(f"来源 kind 错误: expected={expected_kind}, actual={actual_kind}")
    if manifest.get("output_sha256") != sha256_file(path):
        raise ValueError(f"来源文件 SHA256 与 manifest 不一致: {path}")
    row_count = manifest.get("output_rows")
    if row_count != expected_rows:
        raise ValueError(f"来源行数错误: {path}, expected={expected_rows}, actual={row_count}")
    return manifest


def main(
    *,
    original: Path | str = "data/sft/sft_react_messages_v3_linguistic.jsonl",
    parametric: Path | str = "data/sft/sft_react_messages_v3_parametric_linguistic.jsonl",
    output: Path | str = "data/sft/sft_react_messages_v3_train_mix.jsonl",
    seed: int = 42,
    workspace: Path | str | None = None,
) -> Dict[str, Any]:
    workspace = Path.cwd() if workspace is None else Path(workspace).expanduser().resolve()
    initialize_operation(workspace)
    original_path = workspace_path(original, workspace=workspace)
    parametric_path = workspace_path(parametric, workspace=workspace)
    output_path = workspace_path(output, workspace=workspace)
    original_manifest_path = original_path.with_suffix(original_path.suffix + ".manifest.json")
    parametric_manifest_path = parametric_path.with_suffix(parametric_path.suffix + ".manifest.json")
    for label, path in (
        ("original", original_path), ("original manifest", original_manifest_path),
        ("parametric", parametric_path), ("parametric manifest", parametric_manifest_path),
    ):
        if not path.exists():
            raise SystemExit(f"{label} 不存在: {path}")
    if output_path.resolve() in {original_path.resolve(), parametric_path.resolve()}:
        raise SystemExit("output 不能覆盖任何输入")
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise SystemExit(f"拒绝覆盖已有数据或 manifest: {output_path}")

    # 各来源行数以已冻结的 manifest 为准，不写死某一代数据的数量。
    original_rows = read_jsonl(original_path)
    parametric_rows = read_jsonl(parametric_path)
    original_manifest = validate_source_manifest(
        original_path, original_manifest_path,
        expected_kind="original_train_linguistic", expected_rows=len(original_rows),
    )
    parametric_manifest = validate_source_manifest(
        parametric_path, parametric_manifest_path,
        expected_kind="parametric_v1_linguistic", expected_rows=len(parametric_rows),
    )
    for rows, source_manifest in ((original_rows, original_manifest),
                                  (parametric_rows, parametric_manifest)):
        for index, row in enumerate(rows):
            require_training_protocol(row, source_manifest, source_label=f"混合来源 row {index}")

    # 同名或包含关系不能把跨版本 train 混称为同一个声明切分。
    split_files = [workspace_path(str(source.get("split_file") or ""), workspace=workspace)
                   for source in (original_manifest, parametric_manifest)]
    for path, source in zip(split_files, (original_manifest, parametric_manifest)):
        if not path.is_file() or source.get("split_sha256") != sha256_file(path):
            raise ValueError(f"混合来源缺少有效的切分文件或其 SHA256: {path}")
    if (split_files[0].name != split_files[1].name
            or original_manifest["split_sha256"] != parametric_manifest["split_sha256"]):
        raise ValueError("两个混合来源必须声明同一个版本化 train 切分")
    source_split = f"{split_files[0].name}:train"
    payload = read_json(split_files[0])
    groups = payload["split"]
    train = set(groups["train"])
    heldout = set(groups["dev"] + groups["iid_test"] + groups["ood_test"])
    if train & heldout:
        raise ValueError("train 与留出切分存在交叉污染")
    for index, row in enumerate(original_rows):
        if row.get("source_split") != source_split or row.get("source_task_id") not in train:
            raise ValueError(f"原始来源 row {index} 不属于声明的 train")
    for index, row in enumerate(parametric_rows):
        declared = row.get("source_split")
        if (not isinstance(declared, str) or not declared.startswith(source_split + ":")
                or row.get("parent_task_id") not in train):
            raise ValueError(f"参数化来源 row {index} 的父任务不属于声明的 train")

    mixed = build_mix(original_rows, parametric_rows, seed=seed)
    source_counts = Counter(row["mixture_source"] for row in mixed)
    original_tasks = {row["source_task_id"] for row in original_rows}
    parametric_tasks = {row["source_task_id"] for row in parametric_rows}
    if original_tasks & parametric_tasks:
        raise ValueError("原始任务 id 与派生任务 id 冲突")
    write_jsonl(output_path, mixed)

    manifest = {
        "version": 3,
        "kind": "qlora_sft_train_mix",
        "protocol_metadata": validate_data_protocol(mixed),
        "source_split": source_split,
        "split_file": str(split_files[0]),
        "split_sha256": original_manifest["split_sha256"],
        "shuffle_seed": seed,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "output_rows": len(mixed),
        "unique_sample_fingerprints": len({row["sample_sha256"] for row in mixed}),
        "semantic_task_instances": len(original_tasks | parametric_tasks),
        "original_semantic_tasks": len(original_tasks),
        "parametric_semantic_tasks": len(parametric_tasks),
        "source_counts": dict(sorted(source_counts.items())),
        "rows_per_semantic_task": {
            "original": len(original_rows) / len(original_tasks),
            "parametric": len(parametric_rows) / len(parametric_tasks),
        },
        "sources": [
            {
                "path": str(original_path), "sha256": sha256_file(original_path),
                "manifest_sha256": sha256_file(original_manifest_path),
                "rows": len(original_rows), "manifest_kind": original_manifest.get("augmentation_kind"),
            },
            {
                "path": str(parametric_path), "sha256": sha256_file(parametric_path),
                "manifest_sha256": sha256_file(parametric_manifest_path),
                "rows": len(parametric_rows), "manifest_kind": parametric_manifest.get("kind"),
            },
        ],
        "policy": (
            "All rows are kept; per-semantic-task density of the two sources is "
            f"{len(original_rows) / len(original_tasks):.2f} vs "
            f"{len(parametric_rows) / len(parametric_tasks):.2f} rows/task; "
            "file order is deterministically shuffled."
        ),
    }
    write_json(manifest_path, manifest)

    print("SFT_MIX_OK")
    print(f"output rows:      {len(mixed)}")
    print(f"semantic tasks:   {manifest['semantic_task_instances']}")
    print(f"source counts:    {dict(source_counts)}")
    print(f"rows/task:        {manifest['rows_per_semantic_task']}")
    print(f"shuffle seed:     {seed}")
    print(f"output:           {output_path}")
    print(f"manifest:         {manifest_path}")
    return manifest
