#!/usr/bin/env python3
"""模块 C1：步级策略熵诊断（只做前向，不训练；需要 torch + transformers）。

回答一个问题：首次出错的那一步，模型是「犹豫」还是「笃定」？

    犹豫：该步动作 token 的熵高，同组 4 条轨迹在这一状态往往会分叉，
          前缀树上有对照，模块 A/B 能给出信号；
    笃定：熵低，4 条轨迹犯同一个错，前缀树没有分叉，属于探索问题，
          跨步信用分配帮不上忙。

输入是 ``train_grpo --rollout_trace_path`` 写出的审计 JSONL。对每条记录，按
``make_multiturn_rollout_func`` 的拼接方式重建「prompt + 各轮输出 + 观察」，
做一次前向，取每个 token 的预测分布熵，再按轮次和「思考 / 动作」两段求均值。
逐段重新分词与原始采样的 token 不完全一致，结果是近似值。

用法（仓库根目录）::

    python scripts/step_entropy_probe.py --traces outputs/sc_probe/rollouts.jsonl \\
        --model models/sft_t5 --out outputs/sc_probe/step_entropy.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "AgenticArxiv"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from rl.step_credit import WRONG, compute_group_step_credit  # noqa: E402

MODEL_OBSERVATION_CHARS = 1000
Segment = Tuple[int, str, str]          # (轮次, "thought" / "action" / "observation", 文本)


# --------------------------------------------------------------------- 纯函数（可在 CPU 上测试）
def split_assistant_turn(raw_text: str) -> Tuple[str, str]:
    """把一轮输出切成「思考」与「动作」两段，与 rollout 一样先截掉 Observation: 之后的部分。"""
    text = raw_text
    stop = text.find("Observation:")
    if stop >= 0:
        text = text[:stop]
    marker = text.find("Action:")
    if marker < 0:
        return text, ""
    cut = marker + len("Action:")
    return text[:cut], text[cut:]


def rebuild_segments(trajectory: Mapping[str, Any]) -> List[Segment]:
    """按 rollout 的拼接顺序重建 completion：每轮思考、动作，工具步之后接观察。"""
    turns = list(trajectory.get("raw_assistant_turns") or [])
    history = list(trajectory.get("history") or [])
    segments: List[Segment] = []
    for index, raw in enumerate(turns[: len(history)]):
        thought, action = split_assistant_turn(str(raw))
        segments.append((index, "thought", thought))
        segments.append((index, "action", action))
        step = history[index]
        name = str(step.get("action") or "")
        if name in ("FINISH", "PARSE_ERROR") or step.get("parse_failed"):
            continue
        observation = step.get("model_observation")
        if observation is None:
            observation = str(step.get("observation") or "")[:MODEL_OBSERVATION_CHARS]
        segments.append((index, "observation", f"\nObservation: {observation}\nThought:"))
    return segments


def attach_credit(
    records: Sequence[Mapping[str, Any]],
    entropies: Mapping[Tuple[int, int, int], Dict[int, Dict[str, float]]],
    tasks_by_id: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """把每步熵与前缀树信息（标签、该状态下的备选动作数）对齐，输出逐步行。"""
    groups: Dict[Tuple[int, int], List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(int(record.get("batch_index", 0)), int(record["group_index"]))].append(record)
    rows: List[Dict[str, Any]] = []
    for (batch_index, group_index), members in sorted(groups.items()):
        members = sorted(members, key=lambda r: int(r.get("generation_index", 0)))
        task = tasks_by_id.get(members[0]["task_id"])
        if task is None or len(members) < 2:
            continue
        credit = compute_group_step_credit(
            task,
            [m["trajectory"].get("history") or [] for m in members],
            [float(m["reward"]) for m in members],
        )
        for member, steps, (first_index, first_kind) in zip(members, credit.steps, credit.first_errors):
            key = (batch_index, group_index, int(member.get("generation_index", 0)))
            per_step = entropies.get(key, {})
            for index, info in enumerate(steps):
                values = per_step.get(index, {})
                rows.append({
                    "task_id": member["task_id"],
                    "batch_index": batch_index,
                    "group_index": group_index,
                    "generation_index": key[2],
                    "step": index,
                    "label": info.label,
                    "is_first_error": index == first_index,
                    "first_error_kind": first_kind if index == first_index else "",
                    "n_alternatives": info.n_alternatives,
                    "node_size": info.node_size,
                    "advantage": info.advantage,
                    "thought_entropy": values.get("thought"),
                    "action_entropy": values.get("action"),
                })
    return rows


def summarize_entropy(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """首错步按「同状态有无备选动作」分成犹豫型与笃定型，并比较动作熵。"""
    rows = list(rows)
    all_action = [r["action_entropy"] for r in rows if r.get("action_entropy") is not None]
    first = [r for r in rows if r.get("is_first_error") and r.get("label") == WRONG]
    hesitant = [r for r in first if r["n_alternatives"] > 1]
    confident = [r for r in first if r["n_alternatives"] <= 1]

    def mean_entropy(items: Sequence[Mapping[str, Any]]) -> Optional[float]:
        values = [r["action_entropy"] for r in items if r.get("action_entropy") is not None]
        return statistics.fmean(values) if values else None

    median = statistics.median(all_action) if all_action else None
    high = [r for r in first if median is not None and r.get("action_entropy") is not None
            and r["action_entropy"] > median]
    return {
        "step_rows": len(rows),
        "first_error_steps": len(first),
        "first_error_with_alternatives": len(hesitant),
        "first_error_without_alternatives": len(confident),
        "fraction_with_alternatives": len(hesitant) / len(first) if first else None,
        "mean_action_entropy_all_steps": statistics.fmean(all_action) if all_action else None,
        "mean_action_entropy_first_error_with_alternatives": mean_entropy(hesitant),
        "mean_action_entropy_first_error_without_alternatives": mean_entropy(confident),
        "first_error_above_median_entropy": len(high),
        "first_error_kinds": {
            kind: sum(1 for r in first if r["first_error_kind"] == kind)
            for kind in sorted({r["first_error_kind"] for r in first})
        },
    }


# --------------------------------------------------------------------- 需要 torch 的部分
def segment_entropies(model: Any, tokenizer: Any, prompt_messages: Any,
                      segments: Sequence[Segment], chunk: int = 256) -> Dict[int, Dict[str, float]]:
    """一次前向，返回 {轮次: {"thought": 均值熵, "action": 均值熵}}。"""
    import torch

    encoded = tokenizer.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True)
    ids = list(encoded["input_ids"] if "input_ids" in encoded else encoded)
    bounds = []
    for index, role, text in segments:
        piece = list(tokenizer(text, add_special_tokens=False)["input_ids"]) if text else []
        bounds.append((index, role, len(ids), len(ids) + len(piece)))
        ids.extend(piece)
    if len(ids) < 2:
        return {}
    device = next(model.parameters()).device
    with torch.no_grad():
        hidden = model(torch.tensor([ids], device=device), output_hidden_states=False).logits[0]
    out: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for index, role, start, end in bounds:
        if role == "observation" or end <= start:
            continue
        # 位置 p 的 token 由 logits[p-1] 预测
        for left in range(start - 1, end - 1, chunk):
            right = min(left + chunk, end - 1)
            logp = torch.log_softmax(hidden[left:right].float(), dim=-1)
            ent = -(logp.exp() * logp).sum(dim=-1)
            out[index][role].extend(ent.tolist())
    return {index: {role: statistics.fmean(values) for role, values in roles.items() if values}
            for index, roles in out.items()}


def _load_records(path: str, limit: int) -> List[Dict[str, Any]]:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="步级策略熵诊断（模块 C1）")
    parser.add_argument("--traces", required=True, help="train_grpo --rollout_trace_path 的 JSONL")
    parser.add_argument("--model", required=True, help="产生这些 rollout 的策略模型目录")
    parser.add_argument("--task_set", default="expanded", choices=["default", "expanded"])
    parser.add_argument("--max_records", type=int, default=0, help="只处理前 N 条（0=全部；应为组大小的整数倍）")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--out", default=None, help="逐步明细 JSONL")
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from benchmark.task_io import load_tasks
    from rl.grpo_reward import _strip_prompt_task_metadata, build_prompt_dataset

    tasks_by_id = {task["id"]: task for task in load_tasks(args.task_set, None)}
    prompts = {row["task_id"]: _strip_prompt_task_metadata(row["prompt"])
               for row in build_prompt_dataset(list(tasks_by_id.values()))}
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), device_map="auto"
    )
    model.eval()

    records = _load_records(args.traces, args.max_records)
    entropies: Dict[Tuple[int, int, int], Dict[int, Dict[str, float]]] = {}
    for number, record in enumerate(records, 1):
        task_id = record["task_id"]
        if task_id not in prompts:
            continue
        key = (int(record.get("batch_index", 0)), int(record["group_index"]),
               int(record.get("generation_index", 0)))
        entropies[key] = segment_entropies(model, tokenizer, prompts[task_id],
                                           rebuild_segments(record["trajectory"]))
        if number % 20 == 0:
            print(f"  已处理 {number}/{len(records)}", flush=True)

    rows = attach_credit(records, entropies, tasks_by_id)
    print(json.dumps(summarize_entropy(rows), ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
