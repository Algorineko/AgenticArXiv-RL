#!/usr/bin/env python3
"""对通过严格校验的 SFT seed 做确定性、可审计的语言扩增。

本脚本只改变两部分：
1. ReAct prompt 中的“当前任务”措辞；
2. assistant 当前步的 Thought 措辞。

Action、历史 Observation、工具参数和步骤顺序保持逐字不变。因此这一步增加的是
语言鲁棒性，不冒充新的任务语义。参数组合扩增应使用另一版本的数据生成器完成。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "AgenticArxiv"
sys.path.insert(0, str(PACKAGE_ROOT))

from benchmark.metrics import classify_blocked_terminal_semantics  # noqa: E402
from benchmark.task_spec import reference_terminal_thought  # noqa: E402
from benchmark.tasks_expanded import get_expanded_tasks  # noqa: E402


TASK_START = "当前任务："
TASK_END = "\n请按照ReAct框架的格式思考和行动:"

TASK_WRAPPERS = (
    "{task}",
    "请完成下面这项任务：\n{task}",
    "用户提出了以下需求，请准确处理：\n{task}",
    "请使用可用工具处理这个请求：\n{task}",
    "需要你完成以下事项：\n{task}",
    "请按任务中的参数和约束执行：\n{task}",
)


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def split_prompt_task(prompt: str) -> Tuple[str, str, str]:
    """返回任务前缀、任务正文、任务后缀；标记不唯一时拒绝猜测。"""
    if prompt.count(TASK_START) != 1 or prompt.count(TASK_END) != 1:
        raise ValueError("ReAct prompt 中找不到唯一的当前任务边界")
    before, rest = prompt.split(TASK_START, 1)
    task, after = rest.split(TASK_END, 1)
    if not task.strip():
        raise ValueError("ReAct prompt 的当前任务为空")
    return before + TASK_START, task, TASK_END + after


def replace_prompt_task(prompt: str, new_task: str) -> str:
    before, _, after = split_prompt_task(prompt)
    return before + new_task + after


def split_assistant_action(content: str) -> Tuple[str, str]:
    marker = "\nAction:"
    if content.count(marker) != 1:
        raise ValueError("assistant 内容中找不到唯一的 Action 边界")
    thought, action = content.split(marker, 1)
    return thought, action


def decision_thought(action: str, task_def: Dict[str, Any] | None = None) -> str:
    raw = action.strip()
    if raw == "FINISH":
        if task_def and task_def.get("expected_terminal_mode") == "blocked":
            return f"Thought: {reference_terminal_thought(task_def, variant=1)}"
        return "Thought: 任务要求的操作已经完成，应立即结束，避免产生额外工具调用"
    try:
        parsed = json.loads(raw)
        tool_name = parsed["name"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"无法从专家 Action 中解析工具名: {raw}") from exc
    return f"Thought: 结合任务要求和当前会话状态，下一步应调用 {tool_name}"


def validate_seed_rows(
    rows: Iterable[Dict[str, Any]], split_payload: Dict[str, Any]
) -> List[Dict[str, Any]]:
    rows = list(rows)
    groups = split_payload["split"]
    train_ids = set(groups["train"])
    held_out = set(groups["dev"] + groups["iid_test"] + groups["ood_test"])
    source_ids = {row.get("source_task_id") for row in rows}

    if not rows:
        raise ValueError("seed 数据为空")
    if None in source_ids:
        raise ValueError("seed 样本缺少 source_task_id，无法做泄漏审计")
    if source_ids != train_ids:
        missing = sorted(train_ids - source_ids)
        extra = sorted(source_ids - train_ids)
        raise ValueError(f"seed 来源不等于完整 train: missing={missing}, extra={extra}")
    if source_ids & held_out:
        raise ValueError(f"seed 混入留出任务: {sorted(source_ids & held_out)}")

    expected_source = f"v{split_payload['version']}_62.json:train"
    actual_sources = {row.get("source_split") for row in rows}
    if actual_sources != {expected_source}:
        raise ValueError(
            f"seed source_split 不匹配: expected={expected_source}, actual={actual_sources}"
        )

    known_tasks = {task["id"]: task["task"] for task in get_expanded_tasks()}
    held_out_texts = {known_tasks[task_id].strip() for task_id in held_out}
    for index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError(f"seed row {index} 不是两轮 messages 格式")
        if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
            raise ValueError(f"seed row {index} 的 role 顺序错误")
        _, task_text, _ = split_prompt_task(messages[0].get("content", ""))
        if task_text.strip() in held_out_texts:
            raise ValueError(f"seed row {index} 包含留出任务原文")
        split_assistant_action(messages[1].get("content", ""))
    return rows


def augment_rows(
    seed_rows: Iterable[Dict[str, Any]],
    split_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    seed_rows = validate_seed_rows(seed_rows, split_payload)
    return augment_validated_rows(seed_rows)


def augment_validated_rows(
    seed_rows: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """扩增已由调用方完成来源审计的数据；不在此放宽任何来源规则。"""
    seed_rows = list(seed_rows)
    if not seed_rows:
        raise ValueError("待扩增数据为空")
    augmented: List[Dict[str, Any]] = []
    seen = set()
    task_catalog = {task["id"]: task for task in get_expanded_tasks()}

    for parent_index, parent in enumerate(seed_rows):
        messages = parent["messages"]
        prompt = messages[0]["content"]
        assistant = messages[1]["content"]
        _, original_task, _ = split_prompt_task(prompt)
        original_thought, action = split_assistant_action(assistant)
        parent_hash = canonical_hash(messages)
        contract_id = parent.get("parent_task_id") or parent.get("source_task_id")
        task_def = task_catalog.get(str(contract_id))
        thoughts = (original_thought, decision_thought(action, task_def))

        if (
            action.strip() == "FINISH"
            and task_def
            and task_def.get("expected_terminal_mode") == "blocked"
        ):
            for thought in thoughts:
                semantic = classify_blocked_terminal_semantics(
                    task_def,
                    [{"thought": thought, "action": "FINISH"}],
                )
                if semantic != "explained_block":
                    raise ValueError(
                        "blocked FINISH seed 没有说明正确阻塞原因: "
                        f"task={contract_id}, semantic={semantic}, thought={thought!r}"
                    )

        for task_variant, wrapper in enumerate(TASK_WRAPPERS):
            varied_task = wrapper.format(task=original_task)
            varied_prompt = replace_prompt_task(prompt, varied_task)
            for thought_variant, thought in enumerate(thoughts):
                varied_assistant = f"{thought}\nAction:{action}"
                varied_messages = [
                    {"role": "user", "content": varied_prompt},
                    {"role": "assistant", "content": varied_assistant},
                ]
                fingerprint = canonical_hash(varied_messages)
                if fingerprint in seen:
                    raise ValueError(
                        f"扩增产生重复样本: parent={parent_index}, "
                        f"task_variant={task_variant}, thought_variant={thought_variant}"
                    )
                seen.add(fingerprint)

                row = {key: value for key, value in parent.items() if key != "messages"}
                row.update({
                    "messages": varied_messages,
                    "parent_sample_sha256": parent_hash,
                    "sample_sha256": fingerprint,
                    "augmentation": {
                        "kind": "linguistic_semantics_preserving",
                        "task_variant": task_variant,
                        "thought_variant": thought_variant,
                    },
                })
                augmented.append(row)

    expected = len(seed_rows) * len(TASK_WRAPPERS) * 2
    if len(augmented) != expected:
        raise AssertionError(f"扩增数量错误: expected={expected}, actual={len(augmented)}")
    return augmented


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="确定性扩增已校验的 SFT seed 数据")
    parser.add_argument("--input", required=True, help="85 条 seed JSONL")
    parser.add_argument("--output", required=True, help="扩增后的 JSONL")
    parser.add_argument(
        "--split-file", default="data/splits/v2_62.json",
        help="用于泄漏审计的版本化切分文件",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    split_path = Path(args.split_file)
    if not input_path.exists():
        raise SystemExit(f"seed 数据不存在: {input_path}")
    if not split_path.exists():
        raise SystemExit(f"切分文件不存在: {split_path}")
    if input_path.resolve() == output_path.resolve():
        raise SystemExit("input 与 output 不能是同一文件，必须保留 seed 原件")

    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    seed_rows = read_jsonl(input_path)
    augmented = augment_rows(seed_rows, split_payload)
    write_jsonl(output_path, augmented)

    manifest = {
        "version": 1,
        "augmentation_kind": "linguistic_semantics_preserving",
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "seed_rows": len(seed_rows),
        "output_rows": len(augmented),
        "source_tasks": len({row["source_task_id"] for row in augmented}),
        "source_split": sorted({row["source_split"] for row in augmented}),
        "task_variants": len(TASK_WRAPPERS),
        "thought_variants": 2,
        "unique_sample_fingerprints": len({row["sample_sha256"] for row in augmented}),
        "heldout_overlap": 0,
        "semantic_task_count": len({row["source_task_id"] for row in augmented}),
        "note": "1020 rows are linguistic variants of 36 task semantics; they are not 1020 independent tasks.",
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("SFT_AUGMENTATION_OK")
    print(f"seed rows:       {len(seed_rows)}")
    print(f"output rows:     {len(augmented)}")
    print(f"source tasks:    {manifest['source_tasks']}")
    print(f"unique samples:  {manifest['unique_sample_fingerprints']}")
    print(f"heldout overlap: {manifest['heldout_overlap']}")
    print(f"output:           {output_path}")
    print(f"manifest:         {manifest_path}")


if __name__ == "__main__":
    main()
