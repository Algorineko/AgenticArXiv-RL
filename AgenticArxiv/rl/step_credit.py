"""rl/step_credit.py — 确定性前缀树上的分层信用分配（纯函数，不 import torch）。

把 GRPO 给整条轨迹的一个优势值，拆到 ReAct 的每一步：

    模块 A（奖励分解）：同组 G 条轨迹按「规范化动作键」建前缀树，
        V_R(s) = 经过节点 s 的轨迹最终奖励（含封顶，原样）的均值，
        A_t    = [V_R(s_{t+1}) - V_R(s_t)] / (σ_R + eps)。
        各步之和恰好等于 GRPO 给这条轨迹的优势（守恒）。
    模块 B（进度门控）：仅当组内奖励零方差、而规则进度 Φ 有方差时启用，
        Φ = 按顺序严格匹配的标准步数 / (L+1)，FINISH 算最后一步，
        A_t = β · [V_Φ(s_{t+1}) - V_Φ(s_t)] / (σ_Φ + eps)。

确定性快照回放保证「规范化动作前缀相同 ⇒ 环境状态相同」，因此前缀树上的
同一节点就是同一状态，不需要像 GiGPO 那样对观察做哈希匹配。

本模块只做纯 Python 计算：PR 1 用它离线诊断 rollout 审计文件；PR 2 由
训练器子类调用它，把结果展开成逐 token 优势。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from benchmark.metrics import (  # noqa: E402
    NON_TOOL_ACTIONS,
    _count_tool_failures,
    _match_arg_value,
    _parse_tool_action,
    _same_paper,
    classify_blocked_terminal_semantics,
    observation_reports,
    resolved_paper_id,
)
from rl.reward import _has_figure_answer  # noqa: E402

PARSE_ERROR_KEY = "PARSE_ERROR"
_VERSION_SUFFIX = re.compile(r"v\d+$")
# 与 TRL 的 GRPO 归一化保持一致：样本标准差（torch.std 默认 correction=1）+ 1e-4。
DEFAULT_EPS = 1e-4
_ZERO = 1e-8

CORRECT, WRONG, OFF_PREFIX = "correct", "wrong", "off_prefix"


# --------------------------------------------------------------------- 动作键
def _terminal_of(action: Any) -> Optional[str]:
    """识别终止动作，兼容裸字符串和 JSON 外壳两种写法。"""
    if isinstance(action, str):
        text = action.strip()
        if text.upper() in NON_TOOL_ACTIONS:
            return text.upper()
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None
    elif isinstance(action, Mapping):
        parsed = action
    else:
        return None
    if isinstance(parsed, Mapping):
        name = parsed.get("name")
        if isinstance(name, str) and name.strip().upper() in NON_TOOL_ACTIONS:
            return name.strip().upper()
    return None


def _is_parse_failure(step: Mapping[str, Any]) -> bool:
    action = step.get("action", "")
    return bool(step.get("parse_failed")) or action == PARSE_ERROR_KEY or observation_reports(
        step.get("observation", ""), ("无法解析",), action
    )


def canonical_action_key(step: Mapping[str, Any]) -> str:
    """把一步动作规范化成键。键相同 ⇒ 在确定性回放下产生相同的环境状态。

    规范化刻意保守：只合并能证明等价的写法（aspect 大小写与 cs. 前缀、ref
    实际解析到同一篇论文）。漏合并只会让对照变少，错合并才会给出错误信号。
    思考文本不进键。
    """
    terminal = _terminal_of(step.get("action", ""))
    if terminal is not None:
        return terminal
    if _is_parse_failure(step):
        return PARSE_ERROR_KEY
    parsed = _parse_tool_action(step.get("action", ""))
    if parsed is None:
        return PARSE_ERROR_KEY
    name = str(parsed.get("name", ""))
    args = dict(parsed.get("parameters", parsed.get("args", {})) or {})
    args.pop("session_id", None)
    normalized: Dict[str, Any] = {}
    for key, value in args.items():
        if key == "aspect" and isinstance(value, str):
            value = value.strip().upper()
            value = value[3:] if value.startswith("CS.") else value
        normalized[key] = value
    if "ref" in normalized:
        paper = resolved_paper_id(step.get("observation"))
        if paper:
            normalized["ref"] = "paper:" + _VERSION_SUFFIX.sub("", paper.strip())
    return name + json.dumps(normalized, sort_keys=True, ensure_ascii=False, default=str)


# --------------------------------------------------------------------- 规则进度
@dataclass(frozen=True)
class GoldItem:
    tool: str                      # 工具名，或 "FINISH"
    args: Optional[Mapping[str, Any]] = None
    paper_id: Optional[str] = None


def gold_items(task_def: Mapping[str, Any]) -> List[GoldItem]:
    """标准步骤序列，末尾追加 FINISH 作为第 L+1 个标准步。"""
    tools = list(task_def.get("expected_tools") or [])
    args = list(task_def.get("expected_tool_args") or [None] * len(tools))
    papers = list(task_def.get("expected_paper_ids") or [None] * len(tools))
    items = [
        GoldItem(tool, args[i] if i < len(args) else None, papers[i] if i < len(papers) else None)
        for i, tool in enumerate(tools)
    ]
    items.append(GoldItem("FINISH"))
    return items


def _tool_step_matches(step: Mapping[str, Any], gold: GoldItem) -> Tuple[bool, str]:
    if _is_parse_failure(step):
        return False, "parse_error"
    parsed = _parse_tool_action(step.get("action", ""))
    if parsed is None:
        return False, "parse_error"
    if parsed.get("name") != gold.tool:
        return False, "wrong_tool"
    if _count_tool_failures([dict(step)]) > 0:
        return False, "tool_fail"
    if gold.tool == "analyze_figure" and not _has_figure_answer(str(step.get("observation", "") or "")):
        return False, "tool_fail"
    predicted = dict(parsed.get("parameters", parsed.get("args", {})) or {})
    for key, expected in (gold.args or {}).items():
        if key == "ref" and gold.paper_id:
            resolved = resolved_paper_id(step.get("observation"))
            ok = bool(resolved) and _same_paper(resolved, str(gold.paper_id))
        else:
            ok = _match_arg_value(predicted.get(key), expected, key)
        if not ok:
            return False, "wrong_args"
    return True, ""


@dataclass
class TrajectoryLabels:
    progress_count: int            # 按顺序匹配上的标准步数（含 FINISH）
    total: int                     # L + 1
    labels: List[str]              # 每步 correct / wrong / off_prefix
    first_error_index: Optional[int]
    first_error_kind: str

    @property
    def phi(self) -> float:
        return self.progress_count / self.total if self.total else 0.0


def label_trajectory(task_def: Mapping[str, Any], history: Sequence[Mapping[str, Any]]) -> TrajectoryLabels:
    """严格前缀匹配：在第一次偏离之前逐步核对标准答案，之后的步骤不再打标签。"""
    gold = gold_items(task_def)
    blocked = task_def.get("expected_terminal_mode") == "blocked"
    matched = 0
    labels: List[str] = []
    first_error: Optional[int] = None
    kind = ""
    for index, step in enumerate(history):
        if first_error is not None:
            labels.append(OFF_PREFIX)
            continue
        target = gold[matched]
        terminal = _terminal_of(step.get("action", ""))
        if target.tool == "FINISH":
            if terminal == "FINISH":
                ok = True
                if blocked:
                    ok = classify_blocked_terminal_semantics(dict(task_def), list(history[: index + 1])) == "explained_block"
                if ok:
                    matched += 1
                    labels.append(CORRECT)
                else:
                    first_error, kind = index, "bad_terminal"
                    labels.append(WRONG)
            else:
                first_error = index
                kind = "parse_error" if _is_parse_failure(step) else ("bad_terminal" if terminal else "extra_call")
                labels.append(WRONG)
            continue
        if terminal is not None:
            first_error, kind = index, ("early_finish" if terminal == "FINISH" else "bad_terminal")
            labels.append(WRONG)
            continue
        ok, why = _tool_step_matches(step, target)
        if ok:
            matched += 1
            labels.append(CORRECT)
        else:
            first_error, kind = index, why
            labels.append(WRONG)
    if first_error is None and matched < len(gold):
        kind = "unfinished"            # 被截断或撞 max_turns，没有出错但也没走完
    return TrajectoryLabels(matched, len(gold), labels, first_error, kind)


# --------------------------------------------------------------------- 前缀树分解
def prefix_tree_deltas(key_seqs: Sequence[Sequence[str]], values: Sequence[float]) -> List[List[float]]:
    """返回每条轨迹每一步的 V(s_{t+1}) - V(s_t)。

    中间节点取「经过该节点的所有轨迹」的均值；最后一步之后取轨迹自身的值
    （叶子约定）。于是各步之和 = values[i] - mean(values)，严格守恒；同一状态
    下做了同一动作的不同轨迹，在非末步拿到相同信用。
    """
    if len(key_seqs) != len(values):
        raise ValueError("key_seqs and values must have the same length")
    sums: Dict[Tuple[str, ...], float] = defaultdict(float)
    counts: Dict[Tuple[str, ...], int] = defaultdict(int)
    for keys, value in zip(key_seqs, values):
        for t in range(len(keys) + 1):
            prefix = tuple(keys[:t])
            sums[prefix] += float(value)
            counts[prefix] += 1
    node_value = {prefix: sums[prefix] / counts[prefix] for prefix in sums}
    deltas: List[List[float]] = []
    for keys, value in zip(key_seqs, values):
        row = []
        for t in range(len(keys)):
            before = node_value[tuple(keys[:t])]
            after = float(value) if t == len(keys) - 1 else node_value[tuple(keys[: t + 1])]
            row.append(after - before)
        deltas.append(row)
    return deltas


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


@dataclass
class StepInfo:
    key: str
    label: str
    advantage: float               # 已归一化的步优势 A_t
    delta_reward: float            # 未归一化的 ΔV_R
    delta_progress: float          # 未归一化的 ΔV_Φ
    node_size: int                 # 经过 s_t 的轨迹数
    n_alternatives: int            # s_t 下出现过几种不同动作


@dataclass
class GroupCredit:
    task_id: str
    mode: str                      # reward / progress / none
    category: str                  # informative / plateau_recoverable / identical / all_success / flat
    rewards: List[float]
    phis: List[float]
    trajectory_advantages: List[float]
    steps: List[List[StepInfo]] = field(default_factory=list)
    first_errors: List[Tuple[Optional[int], str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_group_step_credit(
    task_def: Mapping[str, Any],
    histories: Sequence[Sequence[Mapping[str, Any]]],
    rewards: Sequence[float],
    *,
    use_progress: bool = True,
    beta: float = 1.0,
    eps: float = DEFAULT_EPS,
    scale: str = "group",          # group：除以组内样本标准差（TRL 默认）；none：不缩放（Dr.GRPO）
) -> GroupCredit:
    if len(histories) != len(rewards):
        raise ValueError("histories and rewards must have the same length")
    rewards = [float(r) for r in rewards]
    key_seqs = [[canonical_action_key(step) for step in history] for history in histories]
    labels = [label_trajectory(task_def, history) for history in histories]
    phis = [lab.phi for lab in labels]

    sigma_r = _sample_std(rewards)
    sigma_p = _sample_std(phis)
    mean_r = sum(rewards) / len(rewards)
    denom_r = (sigma_r + eps) if scale == "group" else 1.0
    traj_adv = [(r - mean_r) / denom_r for r in rewards] if sigma_r > _ZERO else [0.0] * len(rewards)

    d_reward = prefix_tree_deltas(key_seqs, rewards)
    d_progress = prefix_tree_deltas(key_seqs, phis)
    if sigma_r > _ZERO:
        mode, source, denom, coef = "reward", d_reward, denom_r, 1.0
        category = "informative"
    elif use_progress and sigma_p > _ZERO:
        mode, source, denom, coef = "progress", d_progress, ((sigma_p + eps) if scale == "group" else 1.0), beta
        category = "plateau_recoverable"
    else:
        mode, source, denom, coef = "none", None, 1.0, 0.0
        if len({tuple(k) for k in key_seqs}) == 1:
            category = "identical"
        elif min(rewards) >= 0.999:
            category = "all_success"
        else:
            category = "flat"
    if mode == "none" and sigma_p > _ZERO and not use_progress:
        category = "plateau_recoverable"

    node_sizes: Dict[Tuple[str, ...], int] = Counter()
    children: Dict[Tuple[str, ...], set] = defaultdict(set)
    for keys in key_seqs:
        for t in range(len(keys)):
            node_sizes[tuple(keys[:t])] += 1
            children[tuple(keys[:t])].add(keys[t])

    steps: List[List[StepInfo]] = []
    for i, keys in enumerate(key_seqs):
        row = []
        for t, key in enumerate(keys):
            prefix = tuple(keys[:t])
            adv = coef * source[i][t] / denom if source is not None else 0.0
            row.append(StepInfo(
                key=key,
                label=labels[i].labels[t],
                advantage=adv,
                delta_reward=d_reward[i][t],
                delta_progress=d_progress[i][t],
                node_size=node_sizes[prefix],
                n_alternatives=len(children[prefix]),
            ))
        steps.append(row)
    return GroupCredit(
        task_id=str(task_def.get("id", "")),
        mode=mode,
        category=category,
        rewards=rewards,
        phis=phis,
        trajectory_advantages=traj_adv,
        steps=steps,
        first_errors=[(lab.first_error_index, lab.first_error_kind) for lab in labels],
    )


# --------------------------------------------------------------------- 批量与 token 展开
def compute_batch_step_credit(
    task_ids: Sequence[str],
    trajectories: Sequence[Mapping[str, Any]],
    rewards: Sequence[float],
    tasks_by_id: Mapping[str, Mapping[str, Any]],
    num_generations: int,
    **kwargs: Any,
) -> List[GroupCredit]:
    """按 TRL 的约定把连续 num_generations 条视为一组（prompt-major）。"""
    size = len(task_ids)
    if not (size == len(trajectories) == len(rewards)):
        raise ValueError("task_ids, trajectories and rewards must have the same length")
    if size % num_generations:
        raise ValueError(f"batch size {size} is not a multiple of num_generations={num_generations}")
    groups = []
    for start in range(0, size, num_generations):
        ids = {str(t) for t in task_ids[start:start + num_generations]}
        if len(ids) != 1:
            raise ValueError(f"mixed task ids inside one GRPO group: {sorted(ids)}")
        task_def = tasks_by_id[ids.pop()]
        histories = [list((traj or {}).get("history") or []) for traj in trajectories[start:start + num_generations]]
        groups.append(compute_group_step_credit(task_def, histories, rewards[start:start + num_generations], **kwargs))
    return groups


def token_advantages(
    trajectory_advantage: float,
    step_advantages: Sequence[float],
    turn_spans: Sequence[Tuple[int, int]],
    length: int,
    weight: float,
) -> List[float]:
    """把步优势铺到 completion 的 token 上。

    第 t 步模型生成的 token（turn_spans[t] = [start, end)）取
    (1-weight)·A_traj + weight·A_t；其余位置（观察 token、padding）保持 A_traj，
    它们本来就被 env_mask / completion_mask 排除在 loss 之外。weight=0 时与 GRPO 一致。
    """
    out = [float(trajectory_advantage)] * int(length)
    for t, (start, end) in enumerate(turn_spans):
        if t >= len(step_advantages):
            break
        value = (1.0 - weight) * trajectory_advantage + weight * step_advantages[t]
        for j in range(max(0, int(start)), min(int(end), int(length))):
            out[j] = value
    return out


def align_step_advantages(
    groups: Sequence[GroupCredit],
    trainer_advantages: Sequence[float],
    num_generations: int,
) -> Tuple[List[List[float]], float]:
    """把步优势对齐到训练器实际使用的轨迹优势尺度。

    reward 模式下，按组求一个缩放系数，使「各步之和 = 训练器给这条轨迹的优势」，
    这样无论 TRL 用 scale_rewards=group / batch / none，守恒都成立。
    返回 (按 batch 顺序排列的逐轨迹步优势, 最大守恒误差)。
    progress / none 模式的组在 TRL 里轨迹优势本来就是 0，原样返回。
    """
    aligned: List[List[float]] = []
    max_error = 0.0
    for g_index, group in enumerate(groups):
        base = g_index * num_generations
        actual = [float(trainer_advantages[base + j]) for j in range(len(group.steps))]
        factor = 1.0
        if group.mode == "reward":
            denominator = sum(abs(x) for x in group.trajectory_advantages)
            factor = sum(abs(x) for x in actual) / denominator if denominator > _ZERO else 0.0
        for j, row in enumerate(group.steps):
            values = [info.advantage * factor for info in row]
            if group.mode == "reward" and values:
                max_error = max(max_error, abs(sum(values) - actual[j]))
            aligned.append(values)
    return aligned, max_error


class StepCreditCollector:
    """reward 函数每算完一个 batch 就记录一次；训练器子类随后取走。"""

    def __init__(self, tasks_by_id: Mapping[str, Mapping[str, Any]], num_generations: int, **kwargs: Any):
        self.tasks_by_id = dict(tasks_by_id)
        self.num_generations = int(num_generations)
        self.kwargs = kwargs
        self._latest: Optional[Tuple[List[GroupCredit], List[List[Tuple[int, int]]]]] = None

    def record_batch(self, task_ids, trajectories, rewards) -> None:
        groups = compute_batch_step_credit(
            task_ids, trajectories, rewards, self.tasks_by_id, self.num_generations, **self.kwargs
        )
        spans = [[tuple(span) for span in ((t or {}).get("turn_token_spans") or [])] for t in trajectories]
        self._latest = (groups, spans)

    def pop(self) -> Tuple[List[GroupCredit], List[List[Tuple[int, int]]]]:
        if self._latest is None:
            raise RuntimeError("step credit was not recorded for this batch")
        latest, self._latest = self._latest, None
        return latest


# --------------------------------------------------------------------- 诊断汇总
def summarize(groups: Sequence[GroupCredit]) -> Dict[str, Any]:
    """PR 1 要报告的数字：信号覆盖、零方差成因、单样本节点、信用准确率。"""
    modes = Counter(g.mode for g in groups)
    categories = Counter(g.category for g in groups)
    plateau_levels = Counter(
        round(g.rewards[0], 3) for g in groups if g.category == "plateau_recoverable"
    )
    labeled = agree = nonzero = 0
    steps_total = singleton = 0
    trajectories = unfinished = 0
    first_errors = Counter()
    for g in groups:
        for (index, kind) in g.first_errors:
            trajectories += 1
            if index is not None:
                first_errors[kind] += 1
            elif kind == "unfinished":
                unfinished += 1
        for row in g.steps:
            for info in row:
                steps_total += 1
                singleton += int(info.node_size == 1)
                if info.label in (CORRECT, WRONG):
                    labeled += 1
                    if abs(info.advantage) > _ZERO:
                        nonzero += 1
                        agree += int((info.label == CORRECT) == (info.advantage > 0))
    n = len(groups) or 1
    return {
        "group_count": len(groups),
        "mode_fraction": {k: v / n for k, v in sorted(modes.items())},
        "category_count": dict(sorted(categories.items())),
        "informative_fraction_grpo": categories.get("informative", 0) / n,
        "informative_fraction_with_progress": (modes.get("reward", 0) + modes.get("progress", 0)) / n,
        "plateau_reward_levels": {str(k): v for k, v in sorted(plateau_levels.items())},
        "singleton_step_fraction": singleton / steps_total if steps_total else 0.0,
        "credit_coverage": nonzero / labeled if labeled else 0.0,
        "credit_accuracy": agree / nonzero if nonzero else None,
        "first_error_kinds": dict(first_errors.most_common()),
        # 没出错却没走完（被 token 预算截断或撞 max_turns）的轨迹占比；偏高说明先要处理截断
        "unfinished_fraction": unfinished / trajectories if trajectories else 0.0,
    }


def explain_group(group: GroupCredit) -> str:
    """把一组的逐步信用打印成可读表格，便于人工核对。"""
    lines = [f"task={group.task_id} mode={group.mode} category={group.category}"]
    for i, row in enumerate(group.steps):
        lines.append(
            f"  traj{i}: R={group.rewards[i]:+.3f} Φ={group.phis[i]:.2f} A_traj={group.trajectory_advantages[i]:+.3f}"
        )
        for t, info in enumerate(row):
            short = info.key if len(info.key) <= 70 else info.key[:67] + "..."
            lines.append(
                f"    s{t}: {short:<70} {info.label:<10} A={info.advantage:+.3f}"
                f" ΔV_R={info.delta_reward:+.3f} ΔV_Φ={info.delta_progress:+.3f}"
                f" node={info.node_size} alt={info.n_alternatives}"
            )
    return "\n".join(lines)


# --------------------------------------------------------------------- CLI
def _load_audit_groups(path: str) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                groups[(int(record.get("batch_index", 0)), int(record["group_index"]))].append(record)
    return groups


def _report(args: argparse.Namespace) -> int:
    from benchmark.task_io import load_tasks

    tasks_by_id = {t["id"]: t for t in load_tasks(args.task_set, None)}
    results: List[GroupCredit] = []
    skipped = 0
    for _, records in sorted(_load_audit_groups(args.traces).items()):
        records.sort(key=lambda r: int(r.get("generation_index", 0)))
        task_id = records[0]["task_id"]
        if task_id not in tasks_by_id or len(records) < 2:
            skipped += 1
            continue
        histories = [r["trajectory"].get("history") or [] for r in records]
        rewards = [float(r["reward"]) for r in records]
        results.append(compute_group_step_credit(
            tasks_by_id[task_id], histories, rewards,
            use_progress=not args.no_progress, beta=args.beta,
        ))
    summary = summarize(results)
    summary["skipped_groups"] = skipped
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for group in results[: args.explain]:
        print(explain_group(group))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"summary": summary, "groups": [g.to_dict() for g in results]},
                      handle, ensure_ascii=False, indent=2)
    return 0


def _demo(_: argparse.Namespace) -> int:
    """不依赖任何数据文件的演示：chain_ro5_dl_three 上的两种组。"""
    from benchmark.task_io import load_tasks
    from rl.reward import RewardCalculator

    task = {t["id"]: t for t in load_tasks("expanded", None)}["chain_ro5_dl_three"]
    calc = RewardCalculator(curriculum_steps=0)

    def call(name, args, obs):
        return {"thought": "t", "action": json.dumps({"name": name, "args": args}), "observation": obs}

    search = call("get_recently_submitted_cs_papers", {"aspect": "RO", "days": 7, "max_results": 5},
                  "[{'id': '2601.00001v1'}]")
    dl = [None] + [
        call("download_arxiv_pdf", {"ref": i}, f"{{'paper_id': '2601.0000{i}v1', 'status': 'READY'}}")
        for i in (1, 2, 3)
    ]
    fin = {"thought": "done", "action": "FINISH", "observation": "任务完成"}
    scenarios = {
        "mixed": [[search, fin], [search, dl[1], dl[2], fin], [search, dl[1], dl[2], dl[3], fin],
                  [search, dl[1], dl[2], dl[3], fin]],
        "all_early_stop": [[search, fin], [search, dl[1], fin], [search, dl[1], dl[2], fin], [search, fin]],
    }
    for name, histories in scenarios.items():
        rewards = [calc.compute_reward_breakdown(task, {"history": h})[0].total for h in histories]
        group = compute_group_step_credit(task, histories, rewards)
        print(f"== {name}")
        print(explain_group(group))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prefix-tree step credit assignment")
    sub = parser.add_subparsers(dest="command", required=True)
    report = sub.add_parser("report", help="对 GRPO rollout 审计 JSONL 做离线信用分解诊断")
    report.add_argument("--traces", required=True, help="train_grpo --rollout_trace_path 产出的 JSONL")
    report.add_argument("--task_set", default="expanded", choices=["default", "expanded"])
    report.add_argument("--beta", type=float, default=1.0)
    report.add_argument("--no_progress", action="store_true", help="关闭模块 B，只看奖励分解")
    report.add_argument("--explain", type=int, default=0, help="打印前 N 组的逐步明细")
    report.add_argument("--out", default=None, help="把汇总和逐组明细写成 JSON")
    report.set_defaults(func=_report)
    demo = sub.add_parser("demo", help="不需要数据文件的演示")
    demo.set_defaults(func=_demo)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
