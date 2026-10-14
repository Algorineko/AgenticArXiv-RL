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
from agents.react_contract import protocol_metadata, validate_react_messages  # noqa: E402
from rl.run_manifest import require_training_protocol  # noqa: E402
# 共享实现；augment_parametric_sft_data / build_sft_train_mix 仍从本模块转导入这两个名字
from utils.io_utils import read_jsonl, write_jsonl  # noqa: E402


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

# 定向语言扩增使用与基础集不相交的措辞框架：同一批专家决策换一套说法，
# 产出的样本指纹与基础混合集不同，才能真正给弱项族加权而不是生成重复行。
TARGETED_TASK_WRAPPERS = (
    "以下是本次要处理的任务，请逐步完成：\n{task}",
    "请根据下面的描述决定下一步动作：\n{task}",
    "收到一条新的用户请求，请按要求执行：\n{task}",
    "请在当前会话状态下处理这项工作：\n{task}",
    "任务说明如下，请严格按参数操作：\n{task}",
    "请阅读任务并给出合适的工具调用：\n{task}",
)

WRAPPER_SETS = {"base": TASK_WRAPPERS, "targeted": TARGETED_TASK_WRAPPERS}


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_data_protocol(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """新训练数据不能把历史未知协议重标为当前协议。"""
    declarations = [require_training_protocol(
        row, protocol_metadata(), source_label=f"SFT row {index}",
    ) for index, row in enumerate(rows)]
    if not declarations:
        raise ValueError("SFT 数据为空")
    known_hashes = {item["chat_template_hash"] for item in declarations
                    if item["chat_template_hash"] is not None}
    if len(known_hashes) > 1:
        raise ValueError("SFT 来源使用不兼容的 chat template")
    declared = dict(declarations[0])
    # 部分来源未知时，混合 manifest 不把已知的一部分伪装成全部已知。
    if any(item["chat_template_hash"] is None for item in declarations):
        declared["chat_template_hash"] = None
        declared["metadata_status"] = "partial"
    return declared


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
    rows: Iterable[Dict[str, Any]], split_payload: Dict[str, Any], *,
    split_name: str = "v2_62.json",
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

    expected_source = f"{split_name}:train"
    actual_sources = {row.get("source_split") for row in rows}
    if actual_sources != {expected_source}:
        raise ValueError(
            f"seed source_split 不匹配: expected={expected_source}, actual={actual_sources}"
        )

    known_tasks = {task["id"]: task["task"] for task in get_expanded_tasks()}
    held_out_texts = {known_tasks[task_id].strip() for task_id in held_out}
    validate_data_protocol(rows)
    for index, row in enumerate(rows):
        messages = row.get("messages")
        validate_react_messages(messages, require_completion=True)
        _, task_text, _ = split_prompt_task(messages[0].get("content", ""))
        if task_text.strip() in held_out_texts:
            raise ValueError(f"seed row {index} 包含留出任务原文")
        split_assistant_action(messages[-1].get("content", ""))
    return rows


def augment_rows(
    seed_rows: Iterable[Dict[str, Any]],
    split_payload: Dict[str, Any],
    *, split_name: str = "v2_62.json",
) -> List[Dict[str, Any]]:
    seed_rows = validate_seed_rows(seed_rows, split_payload, split_name=split_name)
    return augment_validated_rows(seed_rows)


def augment_validated_rows(
    seed_rows: Iterable[Dict[str, Any]],
    *,
    wrapper_set: str = "base",
) -> List[Dict[str, Any]]:
    """扩增已由调用方完成来源审计的数据；不在此放宽任何来源规则。

    ``wrapper_set`` 选择措辞框架：``base`` 是默认数据集用的六种；``targeted`` 是
    定向扩增的另外六种，与前者不相交，因而不会和基础混合集撞指纹。
    """
    if wrapper_set not in WRAPPER_SETS:
        raise ValueError(f"未知的 wrapper_set: {wrapper_set!r}，可选 {sorted(WRAPPER_SETS)}")
    wrappers = WRAPPER_SETS[wrapper_set]
    seed_rows = list(seed_rows)
    if not seed_rows:
        raise ValueError("待扩增数据为空")
    validate_data_protocol(seed_rows)
    augmented: List[Dict[str, Any]] = []
    seen = set()
    task_catalog = {task["id"]: task for task in get_expanded_tasks()}

    for parent_index, parent in enumerate(seed_rows):
        messages = parent["messages"]
        validate_react_messages(messages, require_completion=True)
        prompt = messages[0]["content"]
        assistant = messages[-1]["content"]
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

        for task_variant, wrapper in enumerate(wrappers):
            varied_task = wrapper.format(task=original_task)
            varied_prompt = replace_prompt_task(prompt, varied_task)
            for thought_variant, thought in enumerate(thoughts):
                varied_assistant = f"{thought}\nAction:{action}"
                # 只改初始任务与本步 Thought，全部历史消息保持原字节。
                varied_messages = [dict(message) for message in messages]
                varied_messages[0]["content"] = varied_prompt
                varied_messages[-1]["content"] = varied_assistant
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
                        "wrapper_set": wrapper_set,
                        "task_variant": task_variant,
                        "thought_variant": thought_variant,
                    },
                })
                augmented.append(row)

    expected = len(seed_rows) * len(wrappers) * 2
    if len(augmented) != expected:
        raise AssertionError(f"扩增数量错误: expected={expected}, actual={len(augmented)}")
    return augmented


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="确定性扩增已校验的 SFT seed 数据")
    parser.add_argument("--input", required=True, help="新协议 seed JSONL")
    parser.add_argument("--output", required=True, help="扩增后的 JSONL")
    parser.add_argument("--manifest", default=None, help="默认读取 seed 旁的 manifest")
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
    out_manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if output_path.exists() or out_manifest_path.exists():
        raise SystemExit(f"拒绝覆盖已有数据或 manifest: {output_path}")

    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    seed_rows = read_jsonl(input_path)
    seed_manifest_path = (Path(args.manifest) if args.manifest else
                          input_path.with_suffix(input_path.suffix + ".manifest.json"))
    seed_manifest = json.loads(seed_manifest_path.read_text(encoding="utf-8"))
    require_training_protocol(seed_manifest, protocol_metadata(), source_label="seed manifest")
    if (seed_manifest.get("kind") != "train_only_expert_seed"
            or seed_manifest.get("output_sha256") != sha256_file(input_path)
            or seed_manifest.get("sample_rows") != len(seed_rows)
            or seed_manifest.get("unique_sample_fingerprints") != len(seed_rows)
            or (seed_manifest.get("split") or {}).get("sha256") != sha256_file(split_path)):
        raise SystemExit("seed manifest 类型、哈希或行数不匹配")
    fingerprints = set()
    for index, row in enumerate(seed_rows):
        require_training_protocol(row, seed_manifest, source_label=f"seed row {index}")
        fingerprint = canonical_hash(row.get("messages"))
        if row.get("sample_sha256") != fingerprint or fingerprint in fingerprints:
            raise SystemExit(f"seed row {index} 指纹不匹配或重复")
        fingerprints.add(fingerprint)
    augmented = augment_rows(seed_rows, split_payload, split_name=split_path.name)
    write_jsonl(output_path, augmented)

    manifest = {
        "version": 1,
        "augmentation_kind": "linguistic_semantics_preserving",
        "protocol_metadata": validate_data_protocol(augmented),
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "input_manifest": str(seed_manifest_path),
        "input_manifest_sha256": sha256_file(seed_manifest_path),
        "split_file": str(split_path),
        "split_sha256": sha256_file(split_path),
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
        "note": "语言扩增行数不是独立任务数量，来源任务及切分保持不变。",
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
