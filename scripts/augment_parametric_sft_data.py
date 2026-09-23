#!/usr/bin/env python3
"""审计并语言扩增 parametric v1 专家 seed。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "AgenticArxiv"
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from augment_sft_data import (  # noqa: E402
    TASK_WRAPPERS,
    augment_validated_rows,
    canonical_hash,
    read_jsonl,
    sha256_file,
    split_assistant_action,
    split_prompt_task,
    write_jsonl,
)


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
    expected_source = f"{split_name}:train:parametric_v1"
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
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError(f"row {index} 不是两轮 messages")
        _, task_text, _ = split_prompt_task(messages[0].get("content", ""))
        if task_text.strip() != task_info["task"].strip():
            raise ValueError(f"row {index} 的任务正文与 manifest 不一致")
        split_assistant_action(messages[1].get("content", ""))

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


def main() -> None:
    parser = argparse.ArgumentParser(description="语言扩增参数化 SFT 专家 seed")
    parser.add_argument("--input", default="data/sft/sft_v2_parametric_seed.jsonl")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split-file", default="data/splits/v2_62.json")
    parser.add_argument("--output", default="data/sft/sft_v2_parametric_linguistic.jsonl")
    args = parser.parse_args()

    input_path = Path(args.input)
    manifest_path = (
        Path(args.manifest) if args.manifest
        else input_path.with_suffix(input_path.suffix + ".manifest.json")
    )
    split_path = Path(args.split_file)
    output_path = Path(args.output)
    for label, path in (
        ("input", input_path), ("manifest", manifest_path), ("split", split_path)
    ):
        if not path.exists():
            raise SystemExit(f"{label} 不存在: {path}")
    if input_path.resolve() == output_path.resolve():
        raise SystemExit("禁止覆盖 parametric seed 原件")

    rows = read_jsonl(input_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    validated = validate_parametric_rows(
        rows, split_payload, manifest, split_name=split_path.name,
        input_path=input_path,
    )
    augmented = augment_validated_rows(validated)
    for row in augmented:
        row["dataset_stage"] = "parametric_v1_linguistic"
    write_jsonl(output_path, augmented)

    derived_tasks = len({row["source_task_id"] for row in augmented})
    parent_tasks = len({row["parent_task_id"] for row in augmented})
    out_manifest = {
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
        "task_variants": len(TASK_WRAPPERS),
        "thought_variants": 2,
        "unique_sample_fingerprints": len({row["sample_sha256"] for row in augmented}),
        "heldout_parent_overlap": 0,
        "note": "1908 rows are 12 linguistic views of 159 decisions from 65 derived tasks.",
    }
    out_manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    out_manifest_path.write_text(
        json.dumps(out_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("PARAMETRIC_AUGMENTATION_OK")
    print(f"seed rows:       {len(rows)}")
    print(f"output rows:     {len(augmented)}")
    print(f"derived tasks:   {derived_tasks}")
    print(f"parent tasks:    {parent_tasks}")
    print("heldout overlap: 0")
    print(f"output:          {output_path}")
    print(f"manifest:        {out_manifest_path}")


if __name__ == "__main__":
    main()
