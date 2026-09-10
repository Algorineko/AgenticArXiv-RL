#!/usr/bin/env python3
"""确定性混合原始 train 语言扩增与参数化语言扩增数据。"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from augment_sft_data import canonical_hash, read_jsonl, sha256_file, write_jsonl  # noqa: E402


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

    mixed: List[Dict[str, Any]] = []
    seen = set()
    for source, rows in (
        ("original_train_linguistic", original_rows),
        ("parametric_v1_linguistic", parametric_rows),
    ):
        for index, original in enumerate(rows):
            row = dict(original)
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
    path: Path, manifest_path: Path, *, expected_kind: str, expected_rows: int
) -> Dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="构建最终 QLoRA SFT 训练混合数据")
    parser.add_argument("--original", default="data/sft/sft_v1_linguistic.jsonl")
    parser.add_argument(
        "--parametric", default="data/sft/sft_v2_parametric_linguistic.jsonl"
    )
    parser.add_argument("--output", default="data/sft/sft_v3_train_mix.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    original_path = Path(args.original)
    parametric_path = Path(args.parametric)
    output_path = Path(args.output)
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

    original_manifest = validate_source_manifest(
        original_path, original_manifest_path,
        expected_kind="original_train_linguistic", expected_rows=1020,
    )
    parametric_manifest = validate_source_manifest(
        parametric_path, parametric_manifest_path,
        expected_kind="parametric_v1_linguistic", expected_rows=1908,
    )
    original_rows = read_jsonl(original_path)
    parametric_rows = read_jsonl(parametric_path)
    if len(original_rows) != 1020 or len(parametric_rows) != 1908:
        raise ValueError("JSONL 实际行数与冻结方案不一致")

    mixed = build_mix(original_rows, parametric_rows, seed=args.seed)
    write_jsonl(output_path, mixed)
    source_counts = Counter(row["mixture_source"] for row in mixed)
    original_tasks = {row["source_task_id"] for row in original_rows}
    parametric_tasks = {row["source_task_id"] for row in parametric_rows}
    if original_tasks & parametric_tasks:
        raise ValueError("原始任务 id 与派生任务 id 冲突")

    manifest = {
        "version": 3,
        "kind": "qlora_sft_train_mix",
        "shuffle_seed": args.seed,
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
            "All rows are kept because the two sources are balanced per semantic task "
            "(28.33 vs 29.35 rows/task); file order is deterministically shuffled."
        ),
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("SFT_MIX_OK")
    print(f"output rows:      {len(mixed)}")
    print(f"semantic tasks:   {manifest['semantic_task_instances']}")
    print(f"source counts:    {dict(source_counts)}")
    print(f"rows/task:        {manifest['rows_per_semantic_task']}")
    print(f"shuffle seed:     {args.seed}")
    print(f"output:           {output_path}")
    print(f"manifest:         {manifest_path}")


if __name__ == "__main__":
    main()
