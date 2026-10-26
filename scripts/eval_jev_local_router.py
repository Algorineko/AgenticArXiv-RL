#!/usr/bin/env python3
"""本地 next-tool 判别头的冻结评测（4 suite + 双臂对比）。

suite 划分：

- **S1 first-step**：81 任务首步。``--context-mode v1_generic`` 严格复刻
  外部 Jev 86.42% pilot 的口径（通用 setup 上下文、无候选论文注入）；
  ``v2_snapshot`` 注入回放候选论文（与训练分布一致）。两口径分列，不混比。
- **S2 full-chain**：数据集中全部 gold_replay 决策点（含链尾 FINISH）。
- **S3 held-out steps**：dev/iid_test/ood_test 任务的 policy 轨迹决策点
  （训练不可见）。
- **S4 error-pattern**：带 error_pattern 标注的留出决策点切片，定向检验
  hard negative 是否起效。

输出 JSON 对齐既有 artifacts schema；``--compare A.json B.json`` 输出
markdown 对照表 + 同决策点 McNemar 精确检验 + bootstrap CI。

用法::

    HIP_VISIBLE_DEVICES=3 python scripts/eval_jev_local_router.py \
        --router-model outputs/jev_rlcd_armB/final \
        --output artifacts/jev_local_router_armB.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "AgenticArxiv"
sys.path.insert(0, str(PACKAGE_ROOT))

from agents.prompt_templates import build_visible_setup_context  # noqa: E402
from benchmark.tasks import get_dependency_chain, get_task_by_id  # noqa: E402
from benchmark.tasks_expanded import get_expanded_tasks  # noqa: E402
from rl.train_jev_rlcd import RouterDataset, load_registry_tools  # noqa: E402
from routing.local_prompt import ROUTER_PROMPT_VERSION  # noqa: E402

DEFAULT_SNAPSHOT = "data/mock_arxiv_snapshot.json"
DEFAULT_SPLIT_FILE = "data/splits/v3_81.json"
SEARCH_TOOLS = {"get_recently_submitted_cs_papers", "search_arxiv_papers"}
ACCEPT_THRESHOLD = 0.80


def effective_setup(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """与 scripts/build_jev_rlcd_dataset.py 同语义（setup + 依赖 gold 链）。"""
    setup = [dict(action) for action in (task.get("setup") or [])]
    task_id = str(task.get("id") or "")
    if task_id:
        for dependency_id in get_dependency_chain(task_id)[:-1]:
            dependency = get_task_by_id(dependency_id)
            if dependency is None:
                continue
            tools = list(dependency.get("expected_tools") or [])
            arguments = list(dependency.get("expected_tool_args") or [])
            setup.extend(
                {
                    "name": tool,
                    "args": dict(arguments[index] or {})
                    if index < len(arguments)
                    else {},
                }
                for index, tool in enumerate(tools)
            )
    return setup


def paper_lists_for(tasks: List[Dict[str, Any]], snapshot: Optional[Path]) -> Dict[str, List[Dict[str, Any]]]:
    if snapshot is None:
        return {}
    from rl.env import MockArxivEnv

    env = MockArxivEnv(snapshot_path=snapshot, mode="replay")
    out: Dict[str, List[Dict[str, Any]]] = {}
    for task in tasks:
        current: List[Dict[str, Any]] = []
        for action in effective_setup(task):
            name = str(action.get("name") or "")
            if name not in SEARCH_TOOLS:
                continue
            try:
                result = env.execute_tool(name, dict(action.get("args") or {}))
            except (KeyError, TypeError, ValueError):
                continue
            if isinstance(result, list):
                current = [dict(p) for p in result if isinstance(p, dict)]
        if current:
            out[str(task["id"])] = current
    return out


def first_step_rows(
    context_mode: str,
    snapshot: Optional[Path],
    split_file: Path,
    max_state_chars: int,
) -> List[Dict[str, Any]]:
    """S1：81 任务首步决策点（state 现渲染，prompt 不入库口径同训练）。"""
    split_payload = json.loads(split_file.read_text(encoding="utf-8"))
    task_split = {
        task_id: split
        for split, ids in split_payload["split"].items()
        for task_id in ids
    }
    tasks = [
        task
        for task in get_expanded_tasks()
        if task["id"] in task_split
    ]
    papers = paper_lists_for(tasks, snapshot if context_mode == "v2_snapshot" else None)
    rows: List[Dict[str, Any]] = []
    for task in sorted(tasks, key=lambda t: t["id"]):
        setup = effective_setup(task)
        context = build_visible_setup_context({"setup": setup})
        block = papers.get(str(task["id"]))
        if block:
            lines = [context, "[Current candidate papers in session]"]
            for index, paper in enumerate(block, 1):
                paper_id = str(paper.get("id") or paper.get("paper_id") or "unknown")
                title = str(paper.get("title") or "untitled")
                lines.append(f"{index}. {paper_id} — {title}")
            context = "\n".join(lines)
        expected = list(task.get("expected_tools") or [])
        gold_tool = expected[0] if expected else "FINISH"
        state_text = context
        if len(state_text) > max_state_chars:
            state_text = "[earlier context truncated]\n" + state_text[-max_state_chars:]
        rows.append(
            {
                "decision_id": f"s1::{task['id']}",
                "suite": "S1_first_step",
                "task_id": str(task["id"]),
                "split": task_split[str(task["id"])],
                "task_text": str(task.get("task") or ""),
                "state_text": state_text,
                "gold_tool": gold_tool,
                "negative_tools": [],
            }
        )
    return rows


def dataset_rows(data_path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in data_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("source") == "gold_replay":
            suite = "S2_full_chain"
        elif row.get("split") in {"dev", "iid_test", "ood_test"}:
            suite = (
                "S4_error_pattern"
                if row.get("error_pattern")
                else "S3_held_out_steps"
            )
        else:
            continue
        row = dict(row)
        row["suite"] = suite
        row.setdefault("decision_id", f"{row.get('task_id')}#{row.get('step_index')}#{row.get('source')}")
        rows.append(row)
    return rows


def run_suite(
    model: Any,
    dataset: RouterDataset,
    device: Any,
    *,
    tau: float,
    scoring: str,
    agg: str,
) -> List[Dict[str, Any]]:
    import torch

    results = []
    started = time.time()
    for index in range(len(dataset)):
        batch = dataset.sample(index, device)
        t0 = time.perf_counter()
        with torch.no_grad():
            z = jev_score(
                model, batch, scoring=scoring, agg=agg
            )
        selected, confidence, probabilities = jev_decision(z, dataset.names, tau)
        gold_name = dataset.names[batch["gold"]]
        results.append(
            {
                "decision_id": dataset.rows[index].get("decision_id"),
                "suite": dataset.rows[index].get("suite"),
                "task_id": dataset.rows[index].get("task_id"),
                "split": dataset.rows[index].get("split"),
                "error_pattern": dataset.rows[index].get("error_pattern"),
                "expected": gold_name,
                "selected": selected,
                "correct": int(selected == gold_name),
                "confidence": round(confidence, 6),
                "probabilities": probabilities,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            }
        )
    print(f"  scored {len(results)} decisions in {time.time()-started:.1f}s")
    return results


def jev_score(model: Any, batch: Dict[str, Any], *, scoring: str, agg: str):
    from rl import jev_scoring

    return jev_scoring.score_candidates(
        model,
        batch["prompt_ids"],
        batch["suffix_ids"],
        batch["suffix_lens"],
        agg=agg,
        scoring=scoring,
        name_starts=batch["name_starts"],
    )


def jev_decision(z, names, tau):
    from rl import jev_scoring

    return jev_scoring.decision_from_z(z, names, tau)


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_suite: Dict[str, List[Dict[str, Any]]] = {}
    for row in results:
        by_suite.setdefault(str(row["suite"]), []).append(row)
    accuracy_by_suite = {
        suite: round(sum(r["correct"] for r in rows) / len(rows), 4)
        for suite, rows in sorted(by_suite.items())
    }
    n = max(1, len(results))
    accept = sum(1 for r in results if r["confidence"] >= ACCEPT_THRESHOLD)
    confusion: Dict[str, Dict[str, int]] = {}
    for row in results:
        confusion.setdefault(row["expected"], {}).setdefault(row["selected"], 0)
        confusion[row["expected"]][row["selected"]] += 1
    return {
        "n": len(results),
        "accuracy": round(sum(r["correct"] for r in results) / n, 4),
        "accuracy_by_suite": accuracy_by_suite,
        "accept_rate_at_0.80": round(accept / n, 4),
        "mean_latency_ms": round(
            sum(r["elapsed_ms"] for r in results) / n, 2
        ),
        "error_patterns": dict(
            Counter(
                str(r["error_pattern"])
                for r in results
                if r.get("error_pattern")
            )
        ),
        "per_class_confusion": confusion,
    }


def mcnemar_exact(b: int, c: int) -> float:
    """双侧精确二项 McNemar p 值。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1))
    return min(1.0, 2 * tail / 2**n)


def compare(left_path: Path, right_path: Path) -> str:
    left = json.loads(left_path.read_text(encoding="utf-8"))
    right = json.loads(right_path.read_text(encoding="utf-8"))

    def by_id(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return {str(r["decision_id"]): r for r in payload["results"]}

    lmap, rmap = by_id(left), by_id(right)
    common = sorted(set(lmap) & set(rmap))
    b = sum(1 for k in common if lmap[k]["correct"] and not rmap[k]["correct"])
    c = sum(1 for k in common if not lmap[k]["correct"] and rmap[k]["correct"])
    p_value = mcnemar_exact(b, c)
    rng = random.Random(42)
    diffs = []
    for _ in range(1000):
        sample = [common[rng.randrange(len(common))] for _ in common]
        diff = sum(rmap[k]["correct"] for k in sample) - sum(
            lmap[k]["correct"] for k in sample
        )
        diffs.append(diff / max(1, len(common)))
    diffs.sort()
    lo, hi = diffs[24], diffs[974]

    lines = [
        f"| suite | {left_path.stem} acc | {right_path.stem} acc | Δ |",
        "|---|---|---|---|",
    ]
    ls, rs = left["summary"], right["summary"]
    suites = sorted(set(ls["accuracy_by_suite"]) | set(rs["accuracy_by_suite"]))
    for suite in suites:
        lv = ls["accuracy_by_suite"].get(suite)
        rv = rs["accuracy_by_suite"].get(suite)
        delta = None if lv is None or rv is None else round(rv - lv, 4)
        lines.append(f"| {suite} | {lv} | {rv} | {delta} |")
    lines.append(f"| overall | {ls['accuracy']} | {rs['accuracy']} | {round(rs['accuracy']-ls['accuracy'],4)} |")
    lines.append("")
    lines.append(
        f"McNemar (paired n={len(common)}): discordant {b}/{c}, "
        f"exact p={p_value:.4f}; bootstrap acc Δ (B−A) 95% CI "
        f"[{lo:+.4f}, {hi:+.4f}]"
    )
    return "\n".join(lines)


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description="本地 next-tool 判别头评测")
    parser.add_argument(
        "--router-model",
        default="",
        help="判别模型目录；--compare 模式下可省",
    )
    parser.add_argument("--data", default="data/rlcd/rlcd_v1.jsonl")
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--split-file", default=DEFAULT_SPLIT_FILE)
    parser.add_argument(
        "--context-mode",
        choices=["v1_generic", "v2_snapshot"],
        default="v1_generic",
        help="S1 的上下文口径：v1 对标外部 Jev 86.42%%，v2 与训练分布一致",
    )
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--agg", choices=["mean", "sum"], default="mean")
    parser.add_argument("--scoring", choices=["cache", "batch"], default="cache")
    parser.add_argument("--max-state-chars", type=int, default=12000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--s2-limit", type=int, default=0, help="0 = 全量")
    parser.add_argument(
        "--compare",
        default="",
        help="给定另一份评测 JSON 时，只输出两臂对比表（不加载模型）",
    )
    args = parser.parse_args()

    if args.compare:
        print(compare(Path(args.compare), Path(args.output)))
        return

    tools = load_registry_tools()
    model_path = Path(args.router_model)
    if not model_path.exists():
        raise SystemExit(f"判别模型不存在: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path.resolve()), local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path.resolve()),
        local_files_only=True,
        attn_implementation="sdpa",
        dtype="auto",
    )
    model.to("cuda")
    model.eval()

    s1_rows = first_step_rows(
        args.context_mode, Path(args.snapshot), Path(args.split_file),
        args.max_state_chars,
    )
    s2s3_rows = dataset_rows(REPO_ROOT / args.data)
    if args.s2_limit:
        s2s3_rows = [
            row
            for row in s2s3_rows
            if row["suite"] != "S2_full_chain"
        ] + [
            row
            for row in s2s3_rows
            if row["suite"] == "S2_full_chain"
        ][: args.s2_limit]
    all_rows = s1_rows + s2s3_rows

    dataset = RouterDataset(
        all_rows, tokenizer=tokenizer, tools=tools,
        max_state_chars=args.max_state_chars, desc="eval",
    )
    print(f"🔎 评测决策点：S1={len(s1_rows)} 数据集={len(s2s3_rows)} "
          f"context={args.context_mode}")
    results = run_suite(
        model, dataset, torch.device("cuda"),
        tau=args.tau, scoring=args.scoring, agg=args.agg,
    )
    summary = summarize(results)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "jev_local_router_eval",
        "router_model": str(model_path),
        "context_mode": args.context_mode,
        "prompt_version": ROUTER_PROMPT_VERSION,
        "tau": args.tau,
        "agg": args.agg,
        "scoring": args.scoring,
        "candidates": dataset.names,
        "summary": summary,
        "results": results,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print("JEV_LOCAL_ROUTER_EVAL_OK")
    print(json.dumps(summary["accuracy_by_suite"], ensure_ascii=False))
    print(f"overall acc={summary['accuracy']} "
          f"accept@0.80={summary['accept_rate_at_0.80']} "
          f"latency={summary['mean_latency_ms']}ms")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
