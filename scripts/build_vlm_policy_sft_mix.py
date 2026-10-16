#!/usr/bin/env python3
"""构建策略侧多模态 SFT 混合集（plan.md S1 第三步）。

三个来源统一成一个 TRL 视觉路径可消费的 messages+images 混合集：

1. 工具调用轨迹（``data/sft/sft_v3_t5_parametric_linguistic.jsonl``，纯文本，
   训练"何时调工具"的决策能力）——messages 原样透传，``images=[]``；
2. figureqa v2（图文，训练识图）——只取 ``split=="train"`` 行，content 转
   分段格式（image 部件 + text 部件），``images=[image_relpath]``；
3. 论文总结自蒸馏（纯文本生成，训练 abstractive 总结）——只取 train 行。

val/val2 行一律不进混合集（评测走各自的留出集）。输出按 seed 洗牌，
manifest 冻结全部输入输出哈希与配比。

用法::

    python scripts/build_vlm_policy_sft_mix.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="构建策略侧多模态 SFT 混合集")
    parser.add_argument("--tool-trajectories", default="data/sft/sft_v3_t5_parametric_linguistic.jsonl")
    parser.add_argument("--figureqa-v2", default="data/vlm/figureqa_v2.jsonl")
    parser.add_argument("--summary-sft", default="data/vlm/paper_summary_sft.jsonl")
    parser.add_argument("--output", default="data/vlm/vlm_policy_sft_v1.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tool", type=int, default=0, help="消融：限制工具轨迹条数（0=全量）")
    parser.add_argument("--max-figureqa", type=int, default=0)
    parser.add_argument("--max-summary", type=int, default=0)
    args = parser.parse_args()

    mixed: List[Dict[str, Any]] = []
    counts: Dict[str, Dict[str, int]] = {}
    inputs: Dict[str, str] = {}

    def add(source: str, row: Dict[str, Any], *, max_n: int = 0) -> bool:
        if max_n and counts.get(source, {}).get("rows", 0) >= max_n:
            return False
        entry = {
            "messages": row["messages"],
            "images": row.get("images") or [],
            "source": source,
        }
        for lineage in ("sample_sha256", "paper_id", "question"):
            if row.get(lineage) is not None:
                entry[lineage] = row[lineage]
        mixed.append(entry)
        counts.setdefault(source, {})
        counts[source]["rows"] = counts[source].get("rows", 0) + 1
        multimodal = 1 if entry["images"] else 0
        counts[source]["multimodal"] = counts[source].get("multimodal", 0) + multimodal
        return True

    tool_path = Path(args.tool_trajectories)
    if tool_path.exists():
        inputs["tool_trajectories"] = sha256_file(tool_path)
        for row in load_jsonl(tool_path):
            add("tool", {"messages": row["messages"], "sample_sha256": row.get("sample_sha256")},
                max_n=args.max_tool)
    else:
        print(f"⚠️ 工具轨迹缺失，跳过: {tool_path}")

    v2_path = Path(args.figureqa_v2)
    if v2_path.exists():
        inputs["figureqa_v2"] = sha256_file(v2_path)
        for row in load_jsonl(v2_path):
            if row.get("split") != "train":
                continue
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": row["prompt"]},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": row["target"]},
                ]},
            ]
            add("figureqa_v2", {**row, "messages": messages, "images": [row["image_relpath"]]},
                max_n=args.max_figureqa)
    else:
        print(f"⚠️ figureqa v2 缺失，跳过: {v2_path}")

    summary_path = Path(args.summary_sft)
    if summary_path.exists():
        inputs["summary_sft"] = sha256_file(summary_path)
        for row in load_jsonl(summary_path):
            if row.get("split") != "train":
                continue
            messages = [
                {"role": "user", "content": [{"type": "text", "text": row["prompt"]}]},
                {"role": "assistant", "content": [{"type": "text", "text": row["target"]}]},
            ]
            add("summary_sft", {**row, "messages": messages, "images": []},
                max_n=args.max_summary)
    else:
        print(f"⚠️ 总结 SFT 缺失，跳过: {summary_path}")

    random.Random(args.seed).shuffle(mixed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in mixed:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "version": 1,
        "kind": "vlm_policy_sft_mix",
        "seed": args.seed,
        "inputs_sha256": inputs,
        "output": output_path.as_posix(),
        "output_sha256": sha256_file(output_path),
        "counts": counts,
        "rows_total": len(mixed),
        "val_excluded": "figureqa_v2 的 val/val2 与 summary 的 val2 行一律不进混合集",
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("VLM_POLICY_SFT_MIX_OK")
    print(f"rows total: {len(mixed)} | per source: { {k: v['rows'] for k, v in counts.items()} }")
    print(f"multimodal rows: {sum(v.get('multimodal', 0) for v in counts.values())}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
