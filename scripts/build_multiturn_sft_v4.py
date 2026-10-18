#!/usr/bin/env python3
"""多轮 SFT 数据 v4：同任务决策链 + 快照 Observation，对齐 runner 推理分布。

根因 #2（plan.md S3 复盘）：v3 及之前的轨迹是「单轮决策」样本
（user=任务+手册，assistant=一次 Thought/Action），训练出的 27B 不会
「读 Observation 后继续/修正」，benchmark 一半任务迭代耗尽。

本脚本把同一派生任务的多条决策行串成链：

1. 按 (source_task_id, 语言视图) 分组、按 trajectory_step 排序；
2. 依次用 MockArxivEnv（离线快照回放）执行每步 Action，捕获确定性 Observation；
3. 每一步 k 生成一条训练样本：user = runner 同款 build_messages(task, 手册,
   历史(步骤 1..k-1 含 Observation))，assistant = 该步的 gold 决策——
   **训练前缀与推理前缀逐字节同分布**（含会话状态块）；
4. 快照缺失等 env 错误 → 链在该步截断（与 benchmark 行为一致）。

用法::

    python scripts/build_multiturn_sft_v4.py --output data/sft/sft_v4_multiturn.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "AgenticArxiv"
sys.path.insert(0, str(PACKAGE_ROOT))

import os as _os

_os.environ.setdefault("STORE_BACKEND", "memory")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="多轮 SFT v4 数据构建")
    parser.add_argument("--seed-data", default="data/sft/sft_v3_t5_parametric_linguistic.jsonl")
    parser.add_argument("--snapshot", default="data/mock_arxiv_snapshot.json")
    parser.add_argument("--output", default="data/sft/sft_v4_multiturn.jsonl")
    args = parser.parse_args()

    from tools.bootstrap import register_all_tools
    from tools.tool_registry import registry
    from agents.prompt_templates import format_tool_description

    register_all_tools()
    canonical_desc = format_tool_description(registry.list_tools())

    from agents.agent_engine import ReActAgent
    from agents.side_effects import LocalSideEffectManager
    from utils.llm_client import LLMClient
    from rl.env import MockArxivEnv
    from benchmark.task_spec import build
    from benchmark.tasks_expanded import EXPANDED_SPECS

    class _Fake(LLMClient):
        def __init__(self):
            pass

        def chat_completions(self, model, messages, **kw):
            return {"choices": [{"message": {"role": "assistant", "content": ""}}], "usage": {}}

    agent = ReActAgent(_Fake(), side_effect_mgr=LocalSideEffectManager(), env=None)

    rows = []
    with Path(args.seed_data).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # 分组键：派生血缘（同一任务同一语言视图 = 一条链）
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        aug = row.get("augmentation")
        aug_key = json.dumps(aug, ensure_ascii=False, sort_keys=True) if aug else ""
        key = (row.get("source_task_id") or row.get("parent_task_id") or row["sample_sha256"][:16], aug_key)
        groups[key].append(row)

    snapshot_path = Path(args.snapshot)
    chains_total = steps_total = 0
    truncated = 0
    out_rows: List[Dict[str, Any]] = []
    task_re = __import__("re").compile(r"当前任务：(.+)")

    for gi, (key, group) in enumerate(sorted(groups.items())):
        group.sort(key=lambda r: r.get("trajectory_step") or 0)
        # 任务文本就在行内（「当前任务：…」行），无需查任务集
        first_user = [m for m in group[0]["messages"] if m["role"] == "user"][0]["content"]
        m_task = task_re.search(first_user)
        if not m_task:
            continue
        task_text = m_task.group(1).strip()

        env = MockArxivEnv(snapshot_path=snapshot_path, mode="replay")
        session_id = f"v4_{gi}"
        agent.session_id = session_id
        agent.env = env
        # setup：铺会话状态（与 runner 的 _apply_setup 等价的可见历史）
        initial_history = ""
        try:
            for action in (spec.get("setup") if isinstance(spec, dict) else getattr(spec, "setup", None)) or []:
                obs = env.execute_tool(action.get("tool"), action.get("args") or {})
                initial_history = f"{initial_history}\n{obs}".strip()
        except Exception:  # noqa: BLE001 - setup 缺快照则按空会话
            initial_history = ""

        history_text = initial_history
        chain_broke = False
        for row in group:
            assistant = [m for m in row["messages"] if m["role"] == "assistant"]
            if not assistant:
                continue
            decision = assistant[-1]["content"]
            # 每步一条训练样本：历史(不含本步) + 本步 gold 决策
            msgs, _extra = agent.build_messages(task_text, canonical_desc, history_text)
            out_rows.append({
                "messages": [{"role": "user", "content": msgs[0]["content"]},
                             {"role": "assistant", "content": decision}],
                "source": "v4_multiturn",
                "derived_task_id": row.get("derived_task_id"),
                "trajectory_step": row.get("trajectory_step"),
            })
            steps_total += 1

            # 推进链：执行本步 Action 得到 Observation（快照缺失 → 链截断）
            try:
                action_json = json.loads(
                    __import__("re").search(r"\{.*\}", decision, __import__("re").DOTALL).group(0)
                )
                chain_args = dict(action_json.get("args") or {})
                # 链内 session 统一：搜索(不含 session_id，框架同步用默认)与
                # 下载/翻译(显式 session_id) 必须落在同一会话，否则断链
                chain_args["session_id"] = session_id
                obs = env.execute_tool(action_json.get("name"), chain_args)
            except Exception:  # noqa: BLE001 - 快照缺失/解析失败：链到此为止
                chain_broke = True
                break
            obs_text = obs if isinstance(obs, str) else json.dumps(obs, ensure_ascii=False, default=str)
            history_text = (
                f"{history_text}\n\n{decision}\nObservation: {obs_text}".strip()
                if history_text else f"{decision}\nObservation: {obs_text}"
            )
        if chain_broke:
            truncated += 1
        chains_total += 1
        if (gi + 1) % 25 == 0:
            print(f"  [{gi+1}/{len(groups)}] 链 {chains_total} | 样本 {steps_total} | 截断链 {truncated}", flush=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "kind": "sft_v4_multiturn",
        "version": 1,
        "seed_data": Path(args.seed_data).as_posix(),
        "seed_sha256": sha256_file(Path(args.seed_data)),
        "snapshot": Path(args.snapshot).as_posix(),
        "snapshot_sha256": sha256_file(snapshot_path),
        "output": output_path.as_posix(),
        "output_sha256": sha256_file(output_path),
        "summary": {
            "chains": chains_total,
            "rows": len(out_rows),
            "truncated_chains": truncated,
            "note": "user 内容 = runner 同款 build_messages（含会话状态块与 Observation 历史），与推理逐字节同分布",
        },
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("SFT_V4_MULTITURN_OK")
    print(f"chains: {chains_total} | rows: {len(out_rows)} | truncated: {truncated}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
