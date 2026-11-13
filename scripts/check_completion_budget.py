#!/usr/bin/env python3
"""估算「一条完全正确的轨迹」在多轮 GRPO rollout 中要占多少 completion token。

多轮 rollout 里，模型每轮生成的 token 和环境拼回来的观察 token
（``"\\nObservation: " + str(result)[:1000] + "\\nThought:"``）都计入
``max_completion_length``。预算小于标准轨迹所需时，rollout 会在中途被截断，
后面的步骤永远拿不到训练信号，跨步信用分配也就无从谈起。

本脚本对每条任务：在离线快照上执行 setup 与标准工具步，按 rollout 的同一
拼接方式构造文本，用策略模型的分词器数 token，与给定预算比较。

用法（仓库根目录）::

    python scripts/check_completion_budget.py --tokenizer models/sft_t5 \\
        --snapshot data/mock_arxiv_snapshot.json \\
        --split data/splits/v6_grpo_train.json:rl_train --budget 256

不装 transformers 时可加 ``--tokenizer char`` 用字符数做上界粗估。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "AgenticArxiv"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from rl.grpo_reward import (  # noqa: E402
    TOOL_ERROR_PREFIX,
    _dispatch_environment_tool,
    _prepare_rollout_environment,
)

DEFAULT_THOUGHT = "根据任务要求和上一步的观察，确定下一步要调用的工具与参数"
FINAL_THOUGHT = "所有步骤均已成功执行，任务已完成"
MODEL_OBSERVATION_CHARS = 1000      # 与 make_multiturn_rollout_func 一致
PER_TURN_OVERHEAD = 2               # 每轮结束符等特殊 token 的余量


def assistant_turn_text(name: Optional[str], args: Optional[Mapping[str, Any]], thought: str) -> str:
    if name is None:
        return f"Thought: {thought}\nAction: FINISH"
    action = json.dumps({"name": name, "args": dict(args or {})}, ensure_ascii=False)
    return f"Thought: {thought}\nAction: {action}"


def observation_suffix(result_text: str) -> str:
    return f"\nObservation: {result_text[:MODEL_OBSERVATION_CHARS]}\nThought:"


def estimate_task_budget(
    task: Mapping[str, Any],
    environment: Any,
    count_tokens: Callable[[str], int],
    tasks_by_id: Mapping[str, Mapping[str, Any]],
    thought: str = DEFAULT_THOUGHT,
) -> Dict[str, Any]:
    """在环境上回放标准答案，返回逐轮 token 数与合计。"""
    _prepare_rollout_environment(environment, task["id"], tasks_by_id)
    tools = list(task.get("expected_tools") or [])
    args_list = list(task.get("expected_tool_args") or [None] * len(tools))
    turns: List[Dict[str, Any]] = []
    total = 0
    errors = []
    for index, name in enumerate(tools):
        args = args_list[index] if index < len(args_list) and args_list[index] is not None else {}
        policy = count_tokens(assistant_turn_text(name, args, thought)) + PER_TURN_OVERHEAD
        try:
            result_text = str(_dispatch_environment_tool(environment, name, args))
        except Exception as exc:  # noqa: BLE001
            result_text = f"{TOOL_ERROR_PREFIX}{exc}"
            errors.append(f"step {index + 1} {name}: {exc}")
        observation = count_tokens(observation_suffix(result_text))
        turns.append({"tool": name, "policy_tokens": policy, "observation_tokens": observation})
        total += policy + observation
    final = count_tokens(assistant_turn_text(None, None, FINAL_THOUGHT)) + PER_TURN_OVERHEAD
    turns.append({"tool": "FINISH", "policy_tokens": final, "observation_tokens": 0})
    total += final
    # 第一轮之后预算就用完，等价于「只做第 1 步就被截断」。
    first_turn = turns[0]["policy_tokens"] + turns[0]["observation_tokens"] if tools else final
    return {
        "task_id": task["id"],
        "chain_length": len(tools),
        "total_tokens": total,
        "first_turn_tokens": first_turn,
        "turns": turns,
        "replay_errors": errors,
    }


def recommend_budget(totals: Sequence[int], margin: float = 1.25, round_to: int = 64) -> int:
    """最长标准轨迹乘以余量，向上取整到 round_to 的倍数。"""
    if not totals:
        return 0
    return int(math.ceil(max(totals) * margin / round_to) * round_to)


def _load_tokenizer(name: str) -> Callable[[str], int]:
    if name == "char":
        return len
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    return lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _load_split_tasks(split: Optional[str], task_set: str) -> List[Dict[str, Any]]:
    from benchmark.task_io import load_tasks

    tasks = load_tasks(task_set, None)
    if not split:
        return tasks
    path, _, name = split.rpartition(":")
    data = json.loads(Path(path if Path(path).is_absolute() else REPO_ROOT / path).read_text(encoding="utf-8"))
    wanted = set(data["split"][name])
    return [task for task in tasks if task["id"] in wanted]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tokenizer", required=True, help="策略模型目录（取其分词器），或 char")
    parser.add_argument("--snapshot", default="data/mock_arxiv_snapshot.json")
    parser.add_argument("--task_set", default="expanded", choices=["default", "expanded"])
    parser.add_argument("--split", default=None, help="FILE:NAME，例如 data/splits/v6_grpo_train.json:rl_train")
    parser.add_argument("--budget", type=int, default=256, help="要检查的 max_completion_length")
    parser.add_argument("--out", default=None, help="把逐任务结果写成 JSON")
    args = parser.parse_args(argv)

    from rl.multiturn_env import AgenticArxivMultiTurnEnv

    snapshot = Path(args.snapshot)
    if not snapshot.is_absolute():
        snapshot = REPO_ROOT / snapshot
    if not snapshot.exists():
        parser.error(f"快照不存在: {snapshot}")
    count_tokens = _load_tokenizer(args.tokenizer)
    tasks = _load_split_tasks(args.split, args.task_set)
    tasks_by_id = {task["id"]: task for task in tasks}
    environment = AgenticArxivMultiTurnEnv(snapshot_path=snapshot)

    rows = [estimate_task_budget(task, environment, count_tokens, tasks_by_id) for task in tasks]
    rows.sort(key=lambda row: (-row["chain_length"], -row["total_tokens"]))
    over = [row for row in rows if row["total_tokens"] > args.budget]
    first_over = [row for row in rows if row["first_turn_tokens"] > args.budget]

    print(f"{'task_id':<34} {'L':>2} {'整条':>6} {'第1轮':>6}  超出预算")
    for row in rows:
        flag = "整条超出" if row["total_tokens"] > args.budget else ""
        if row["first_turn_tokens"] > args.budget:
            flag = "第1轮就超出"
        print(f"{row['task_id']:<34} {row['chain_length']:>2} {row['total_tokens']:>6} "
              f"{row['first_turn_tokens']:>6}  {flag}")
        for error in row["replay_errors"]:
            print(f"    ⚠️ 回放失败 {error}")
    print()
    print(f"预算 {args.budget}：{len(over)}/{len(rows)} 条任务的标准轨迹放不下；"
          f"其中 {len(first_over)} 条第 1 轮就放不下。")
    print(f"建议 --max_completion_length 至少 {recommend_budget([r['total_tokens'] for r in rows])}"
          "（最长标准轨迹 × 1.25，取 64 的倍数；模型实际的思考文本可能更长）")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
