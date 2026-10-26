#!/usr/bin/env python3
"""构造 RLCD next-tool 判别头的决策点数据集（纯 CPU，离线）。

四个来源池（v1 实现前三个，sft_pool 留待后续版本）：

1. ``gold_replay``：v3_81 冻结任务的 gold 链逐前缀回放。每任务
   len(expected_tools) 个工具决策 + 1 个链尾 FINISH 决策；
   observation 用本脚本的确定性渲染器合成（与 policy 轨迹的真实
   observation 文本不完全同构，训练/评测两侧同构即可）。
2. ``policy_success``：58 个 ``artifacts/*/traces.jsonl`` 的轨迹步。
   轨迹级成功 = 工具序列严格匹配（``_check_tool_sequence``）且正常
   FINISH；失败轨迹的已匹配前缀同样产出正样本（``prefix_clean=True``，
   前缀与 gold 逐位一致是构造保证的）。
3. ``policy_failure_neg``：首分歧步的硬负样本——state 是策略真实走到
   的状态，gold 是 expected 里该步的工具，negative 是策略实际选错的
   工具（"策略真会选的错" 天然 hard）。error_pattern 规则标注：
   ``title_ref_wrong_search`` / ``infeasible_called_tool`` /
   ``recent_vs_keyword`` / ``other``。

标签全部由 ``expected_tools`` 确定性重算，无人工标注（RLVR 口径）。

防泄漏：任务级切分沿 v3_81（train 51 / dev 8 / iid_test 18 / ood_test 4），
决策点继承所属任务的 split；dev/iid/ood 的任务不得出现在 train 行；
manifest 记录各池计数与输出 SHA256。

用法::

    python scripts/build_jev_rlcd_dataset.py                     # 全量
    python scripts/build_jev_rlcd_dataset.py --max-traces 200    # 冒烟
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "AgenticArxiv"
sys.path.insert(0, str(PACKAGE_ROOT))

from agents.prompt_templates import build_visible_setup_context  # noqa: E402
from benchmark.metrics import _check_tool_sequence, _extract_tool_sequence  # noqa: E402
from benchmark.tasks import get_dependency_chain, get_task_by_id  # noqa: E402
from benchmark.tasks_expanded import get_expanded_tasks  # noqa: E402
from tools.bootstrap import require_all_tools  # noqa: E402
from tools.tool_registry import registry  # noqa: E402

FINISH = "FINISH"
DEFAULT_SNAPSHOT = "data/mock_arxiv_snapshot.json"
DEFAULT_SPLIT_FILE = "data/splits/v3_81.json"
DEFAULT_OUTPUT = "data/rlcd/rlcd_v1.jsonl"
MAX_NEGATIVES_PER_STATE = 2
SEARCH_TOOLS = {"get_recently_submitted_cs_papers", "search_arxiv_papers"}

HASH_SAMPLE_BYTES = 1024 * 1024


def sha256_head(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(HASH_SAMPLE_BYTES))
    return digest.hexdigest()


def decision_id(state_text: str, task_id: str, step_index: int, source: str) -> str:
    payload = f"{state_text}|{task_id}|{step_index}|{source}".encode("utf-8")
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def effective_setup(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """任务开始前已成立的动作（setup + 依赖任务的 gold 链）。

    与 ``scripts/jev_route_smoke.py::_effective_setup`` 同语义；本仓库
    已有多处各自持有该小段逻辑的先例，这里保持 builder 独立可跑。
    """
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


def render_paper_block(papers: List[Dict[str, Any]]) -> str:
    if not papers:
        return ""
    lines = ["[Current candidate papers in session]"]
    for index, paper in enumerate(papers, 1):
        paper_id = str(paper.get("id") or paper.get("paper_id") or "unknown")
        title = str(paper.get("title") or "untitled")
        lines.append(f"{index}. {paper_id} — {title}")
    return "\n".join(lines)


def visible_context(task: Dict[str, Any], papers: List[Dict[str, Any]]) -> str:
    """会话初始状态：setup 渲染 + 候选论文块（不含 gold 工具名）。"""
    context = build_visible_setup_context({"setup": effective_setup(task)})
    paper_block = render_paper_block(papers)
    if paper_block:
        context = f"{context}\n{paper_block}"
    return context


def render_observation(name: str, result: Any) -> str:
    """gold 链 observation 的确定性渲染（紧凑、可复现）。"""
    if isinstance(result, list):
        lines = [f"成功获取 {len(result)} 篇论文。候选列表:"]
        for index, paper in enumerate(result, 1):
            paper_id = str(paper.get("id") or paper.get("paper_id") or "unknown")
            title = str(paper.get("title") or "untitled")
            lines.append(f"  {index}. {paper_id} — {title}")
        return "\n".join(lines)
    if isinstance(result, dict):
        paper_id = result.get("paper_id")
        head = (
            f"已对论文 {paper_id} 完成 {name}。"
            if paper_id
            else f"{name} 执行成功。"
        )
        extra = {
            key: value
            for key, value in result.items()
            if key not in {"paper_id"}
        }
        text = json.dumps(extra, ensure_ascii=False, sort_keys=True)[:300]
        return f"{head}\n{text}" if extra else head
    return f"{name} 执行成功。{str(result)[:300]}"


def history_block(action_json: str, observation: str) -> str:
    return f"Action: {action_json}\nObservation: {observation}"


def build_state(
    initial_context: str, blocks: List[str], max_chars: int
) -> str:
    """state_text = 初始可见上下文 + 已发生步的 Action/Observation 块。"""
    parts = [initial_context] + list(blocks)
    state = "\n".join(parts)
    if len(state) <= max_chars:
        return state
    return "[earlier context truncated]\n" + state[-max_chars:]


def classify_error_pattern(
    task: Dict[str, Any], gold_tool: str, negative_tool: str
) -> str:
    expected = list(task.get("expected_tools") or [])
    task_text = str(task.get("task") or "")
    if not expected and negative_tool != FINISH:
        return "infeasible_called_tool"
    if negative_tool == "search_arxiv_papers" and gold_tool != "search_arxiv_papers":
        has_reference = any(
            marker in task_text
            for marker in ("第", "篇", "arXiv", "ID", "《", "标题")
        )
        if has_reference:
            return "title_ref_wrong_search"
    if {gold_tool, negative_tool} == {
        "get_recently_submitted_cs_papers",
        "search_arxiv_papers",
    }:
        return "recent_vs_keyword"
    return "other"


class DatasetBuilder:
    def __init__(
        self,
        *,
        snapshot_path: Path,
        split_file: Path,
        max_state_chars: int,
        replay_check: bool,
    ) -> None:
        # 回放要执行真实工具，registry 必须先注册齐（同 benchmark/RL 入口口径）
        require_all_tools("Jev RLCD dataset builder")
        split_payload = json.loads(split_file.read_text(encoding="utf-8"))
        self.task_split: Dict[str, str] = {
            task_id: split
            for split, ids in split_payload["split"].items()
            for task_id in ids
        }
        self.tasks_by_id = {task["id"]: task for task in get_expanded_tasks()}
        self.snapshot_path = snapshot_path
        self.max_state_chars = max_state_chars
        self.replay_check = replay_check
        self._papers_cache: Dict[str, Optional[List[Dict[str, Any]]]] = {}
        self.stats: Counter[str] = Counter()

    def _task_env(self):
        """任务级 MultiTurnEnv（会话感知；与 semantic_oracle 同一执行路径）。

        paper 工具的 ref 解析依赖会话记忆，MockArxivEnv 裸 execute_tool
        没有这层状态；每任务独立实例防串扰。
        """
        from rl.multiturn_env import AgenticArxivMultiTurnEnv

        env = AgenticArxivMultiTurnEnv(self.snapshot_path)
        env.reset(task_id="rlcd_builder")
        return env

    @staticmethod
    def _call(env, name: str, args: Dict[str, Any]) -> Any:
        method = getattr(env, name, None)
        if method is None:
            raise ValueError(f"环境没有工具方法: {name}")
        clean = {k: v for k, v in dict(args or {}).items() if k != "session_id"}
        return method(**clean)

    def _run_setup(self, env, task: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """执行 setup + 依赖链，返回最后一次搜索的候选论文（无搜索则空表）。

        返回 None 仅表示 setup 执行失败（快照缺 key 等），调用方按噪声处理；
        空列表是合法结果——很多任务的搜索本身就是 gold 第 0 步。
        """
        papers: List[Dict[str, Any]] = []
        for action in effective_setup(task):
            name = str(action.get("name") or "")
            try:
                result = self._call(env, name, dict(action.get("args") or {}))
            except Exception:  # noqa: BLE001
                return None
            if name in SEARCH_TOOLS and isinstance(result, list):
                papers = [dict(paper) for paper in result if isinstance(paper, dict)]
        return papers

    def _papers_for(self, task: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """任务级候选论文缓存（确定性回放，同任务只算一次）。"""
        task_id = str(task["id"])
        if task_id not in self._papers_cache:
            self._papers_cache[task_id] = self._run_setup(
                self._task_env(), task
            )
        return self._papers_cache[task_id]

    # ---------- 池 1：gold_replay ----------

    def gold_rows(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        task_id = str(task["id"])
        expected = list(task.get("expected_tools") or [])
        args_list = task.get("expected_tool_args") or []
        env = self._task_env()
        papers = self._run_setup(env, task)
        if papers is None:
            raise SystemExit(
                f"gold_replay: 任务 {task_id} 的 setup 回放失败（快照缺 key?）"
            )
        initial = visible_context(task, papers or [])
        rows: List[Dict[str, Any]] = []
        blocks: List[str] = []
        for step_index, tool in enumerate(expected):
            rows.append(
                self._row(
                    task=task,
                    state_text=build_state(initial, blocks, self.max_state_chars),
                    step_index=step_index,
                    source="gold_replay",
                    gold_tool=tool,
                    gold_args=dict(args_list[step_index] or {})
                    if step_index < len(args_list)
                    else {},
                )
            )
            args = dict(args_list[step_index] or {}) if step_index < len(args_list) else {}
            try:
                result = self._call(env, tool, dict(args))
                observation = render_observation(tool, result)
            except Exception as exc:  # noqa: BLE001 - gold 链断链响亮上报
                raise SystemExit(
                    f"gold_replay: 任务 {task_id} 第 {step_index + 1} 步 {tool} "
                    f"回放失败（快照缺 key?）: {exc}"
                )
            action_json = json.dumps({"name": tool, "args": args}, ensure_ascii=False)
            blocks.append(history_block(action_json, observation))
        # 链尾 FINISH 决策
        rows.append(
            self._row(
                task=task,
                state_text=build_state(initial, blocks, self.max_state_chars),
                step_index=len(expected),
                source="gold_replay",
                gold_tool=FINISH,
                gold_args={},
            )
        )
        self.stats["gold_rows"] += len(rows)
        return rows

    # ---------- 池 2/3：policy 轨迹 ----------

    def trace_rows(
        self, trace: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
        task = self.tasks_by_id.get(str(trace.get("task_id") or ""))
        if task is None:
            self.stats["skipped_unknown_task"] += 1
            return [], [], False
        history = list(trace.get("history") or [])
        expected = list(task.get("expected_tools") or [])
        actual = _extract_tool_sequence(history)
        exact = _check_tool_sequence(actual, expected)
        if not history:
            self.stats["skipped_empty_history"] += 1
            return [], [], False

        # 结构校验 + 任务级候选论文缓存（标签来自 TaskSpec 确定性重算，
        # 不做轨迹逐步回放——标签正确性不依赖回放，见 manifest notes）
        papers = self._papers_for(task)
        if papers is None:
            self.stats["skipped_setup_fail"] += 1
            return [], [], False
        self.stats["traces_accepted"] += 1

        initial = visible_context(task, papers or [])

        # 首分歧点（actual 的工具序列下标）：值不同处，或长度不足处
        divergence = None
        for index in range(min(len(actual), len(expected))):
            if actual[index] != expected[index]:
                divergence = index
                break
        if divergence is None and len(actual) != len(expected):
            divergence = min(len(actual), len(expected))

        positives: List[Dict[str, Any]] = []
        negative: Optional[Dict[str, Any]] = None
        blocks: List[str] = []
        tool_seen = 0
        # 终止决策看到的 state：最后一个工具步的 observation 之后、
        # 任何 FINISH/FORCE_STOP 步之前
        state_after_last_tool = build_state(initial, blocks, self.max_state_chars)

        for step in history:
            action_text = str(step.get("action") or "")
            name = self._action_tool_name(action_text)
            state_before = build_state(initial, blocks, self.max_state_chars)
            if name is not None:
                j = tool_seen
                if divergence is None or j < divergence:
                    positives.append(
                        self._row(
                            task=task,
                            state_text=state_before,
                            step_index=j,
                            source="policy_success",
                            gold_tool=expected[j],
                            gold_args={},
                            prefix_clean=True,
                        )
                    )
                elif j == divergence and negative is None:
                    gold_tool = (
                        expected[divergence]
                        if divergence < len(expected)
                        else FINISH
                    )
                    if name != gold_tool:
                        negative = self._row(
                            task=task,
                            state_text=state_before,
                            step_index=divergence,
                            source="policy_failure_neg",
                            gold_tool=gold_tool,
                            gold_args={},
                            negative_tools=[name],
                            error_pattern=classify_error_pattern(
                                task, gold_tool, name
                            ),
                        )
                tool_seen += 1
            blocks.append(
                history_block(action_text or "{}", str(step.get("observation") or ""))
            )
            if name is not None:
                state_after_last_tool = build_state(
                    initial, blocks, self.max_state_chars
                )

        # 提前停：divergence 落在全部工具步之后（假 FINISH / force stop），
        # 策略在此处的选择（FINISH）就是硬负样本
        if (
            divergence is not None
            and divergence >= tool_seen
            and negative is None
        ):
            gold_tool = (
                expected[divergence] if divergence < len(expected) else FINISH
            )
            if FINISH != gold_tool:
                negative = self._row(
                    task=task,
                    state_text=state_after_last_tool,
                    step_index=divergence,
                    source="policy_failure_neg",
                    gold_tool=gold_tool,
                    gold_args={},
                    negative_tools=[FINISH],
                    error_pattern=classify_error_pattern(task, gold_tool, FINISH),
                )

        # 严格成功轨迹补链尾 FINISH 决策（状态 = 全部工具步之后）
        if exact:
            positives.append(
                self._row(
                    task=task,
                    state_text=state_after_last_tool,
                    step_index=len(actual),
                    source="policy_success",
                    gold_tool=FINISH,
                    gold_args={},
                )
            )
        self.stats["policy_positives"] += len(positives)
        if negative is not None:
            self.stats["policy_negatives"] += 1
        return positives, ([negative] if negative is not None else []), True

    @staticmethod
    def _parse_action(action_text: str) -> Optional[Dict[str, Any]]:
        text = (action_text or "").strip()
        if not text or text.upper() in {"FINISH", "FORCE_STOP", "ERROR"}:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict) or not parsed.get("name"):
            return None
        return parsed

    @classmethod
    def _action_tool_name(cls, action_text: str) -> Optional[str]:
        parsed = cls._parse_action(action_text)
        return str(parsed["name"]) if parsed else None

    def _row(
        self,
        *,
        task: Dict[str, Any],
        state_text: str,
        step_index: int,
        source: str,
        gold_tool: str,
        gold_args: Dict[str, Any],
        negative_tools: Optional[List[str]] = None,
        error_pattern: Optional[str] = None,
        prefix_clean: bool = True,
    ) -> Dict[str, Any]:
        task_id = str(task["id"])
        self.stats["rows"] += 1
        return {
            "decision_id": decision_id(state_text, task_id, step_index, source),
            "source": source,
            "task_id": task_id,
            "template_key": list(
                (
                    str(task.get("template") or task.get("category") or "?"),
                    len(task.get("expected_tools") or []),
                )
            ),
            "split": self.task_split.get(task_id, "unassigned"),
            "step_index": step_index,
            "task_text": str(task.get("task") or ""),
            "state_text": state_text,
            "gold_tool": gold_tool,
            "gold_args": gold_args,
            "negative_tools": negative_tools or [],
            "negative_provenance": [],
            "prefix_clean": prefix_clean,
            "error_pattern": error_pattern,
            "prompt_version": "jev_router_prompt_v1",
        }


def merge_rows(
    positives: List[Dict[str, Any]], negatives: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """同 (task, state) 的正样本行合并，负样本挂行不占行。

    - 正样本行按 (task_id, state_text) 去重，gold 相同则并成一行；
    - 负样本 (negative, error_pattern) 对按同键合并进行行的 negative_tools
      （上限 MAX_NEGATIVES_PER_STATE，gold 本身不入列）；
    - 只出现在失败池的 state 保留其行：该行的 gold 来自 expected
      （仍是有效正标签），negative_tools 提供对比信号。
    """
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    negative_pairs: Dict[Tuple[str, str], List[Tuple[str, Optional[str]]]] = {}

    for row in negatives:
        key = (row["task_id"], row["state_text"])
        bucket = negative_pairs.setdefault(key, [])
        entry = (str(row["negative_tools"][0]), row.get("error_pattern"))
        if entry not in bucket:
            bucket.append(entry)

    for row in positives:
        by_key.setdefault((row["task_id"], row["state_text"]), row)

    for key, entries in negative_pairs.items():
        target = by_key.get(key)
        if target is None:
            first = next(
                row
                for row in negatives
                if (row["task_id"], row["state_text"]) == key
            )
            target = dict(first)
            target["negative_tools"] = []
            by_key[key] = target
        for tool, pattern in entries:
            if tool == target["gold_tool"] or tool in target["negative_tools"]:
                continue
            if len(target["negative_tools"]) < MAX_NEGATIVES_PER_STATE:
                target["negative_tools"].append(tool)
                if pattern and not target.get("error_pattern"):
                    target["error_pattern"] = pattern

    return list(by_key.values())


def leak_assertions(rows: List[Dict[str, Any]], task_split: Dict[str, str]) -> None:
    eval_tasks = {t for t, s in task_split.items() if s != "train"}
    for row in rows:
        if row["split"] == "train" and row["task_id"] in eval_tasks:
            raise SystemExit(
                f"泄漏断言失败：train 行含留出任务 {row['task_id']}"
            )
        if row["split"] not in {"train", "dev", "iid_test", "ood_test"}:
            raise SystemExit(
                f"泄漏断言失败：任务 {row['task_id']} 不在冻结切分内"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="RLCD 判别头数据集构造")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--split-file", default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--trace-glob", default="artifacts/*/traces.jsonl")
    parser.add_argument("--max-traces", type=int, default=0, help="0 = 全量")
    parser.add_argument("--max-state-chars", type=int, default=12000)
    parser.add_argument(
        "--no-replay-check",
        action="store_true",
        help="跳过轨迹回放校验（快照缺 key 时降级用，正常不要开）",
    )
    args = parser.parse_args()

    require_all_tools("RLCD dataset builder")
    snapshot_path = Path(args.snapshot)
    if not snapshot_path.exists():
        raise SystemExit(f"快照不存在: {snapshot_path}")
    split_file = Path(args.split_file)

    builder = DatasetBuilder(
        snapshot_path=snapshot_path,
        split_file=split_file,
        max_state_chars=args.max_state_chars,
        replay_check=not args.no_replay_check,
    )
    task_split = builder.task_split

    positives: List[Dict[str, Any]] = []
    negatives: List[Dict[str, Any]] = []
    for task_id in sorted(task_split):
        task = builder.tasks_by_id.get(task_id)
        if task is None:
            builder.stats["skipped_split_task_missing"] += 1
            continue
        positives.extend(builder.gold_rows(task))

    trace_files = sorted(REPO_ROOT.glob(args.trace_glob))
    if not trace_files:
        raise SystemExit(f"没有找到轨迹文件: {args.trace_glob}")
    trace_budget = args.max_traces
    for trace_file in trace_files:
        for line in trace_file.read_text(encoding="utf-8").splitlines():
            if trace_budget and builder.stats["traces_seen"] >= trace_budget:
                break
            if not line.strip():
                continue
            builder.stats["traces_seen"] += 1
            try:
                trace = json.loads(line)
            except json.JSONDecodeError:
                builder.stats["skipped_bad_json"] += 1
                continue
            row_pos, row_neg, _ = builder.trace_rows(trace)
            positives.extend(row_pos)
            negatives.extend(row_neg)
        if trace_budget and builder.stats["traces_seen"] >= trace_budget:
            break

    rows = merge_rows(positives, negatives)
    leak_assertions(rows, task_split)
    rows.sort(key=lambda r: (r["split"], r["task_id"], r["step_index"], r["source"]))

    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    )
    output_path.write_text(payload, encoding="utf-8")

    by_split = Counter(row["split"] for row in rows)
    by_source = Counter(row["source"] for row in rows)
    with_negative = sum(1 for row in rows if row["negative_tools"])
    by_error = Counter(
        row["error_pattern"] for row in rows if row.get("error_pattern")
    )
    manifest = {
        "kind": "jev_rlcd_decision_points",
        "version": 1,
        "output": str(output_path.relative_to(REPO_ROOT)),
        "output_sha256": hashlib.sha256(
            output_path.read_bytes()
        ).hexdigest(),
        "output_rows": len(rows),
        "by_split": dict(by_split),
        "by_source": dict(by_source),
        "rows_with_negatives": with_negative,
        "error_patterns": dict(by_error),
        "prompt_version": "jev_router_prompt_v1",
        "snapshot": str(snapshot_path),
        "snapshot_sha256_head": sha256_head(snapshot_path),
        "split_file": str(split_file),
        "stats": dict(builder.stats),
        "notes": [
            "sft_pool 未在 v1 实现（SFT 行的 state 格式与 router prompt 不同构，"
            "重建有损）；正样本由 gold_replay + policy_success 提供",
            "policy 轨迹标签由 expected_tools 确定性重算，未使用 raw_data.csv",
            "回放校验口径 = setup+每工具步在冻结快照上无异常执行（不做 "
            "observation 字符串比对，渲染器与 base_agent 不同构）",
        ],
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("RLCD_DATASET_OK")
    print(f"rows: {len(rows)}  by_split: {dict(by_split)}")
    print(f"by_source: {dict(by_source)}")
    print(f"rows_with_negatives: {with_negative}  error_patterns: {dict(by_error)}")
    print(f"stats: {dict(builder.stats)}")
    print(f"output: {output_path}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
