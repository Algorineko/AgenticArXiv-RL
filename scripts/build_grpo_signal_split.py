"""从冻结策略探测结果构建 GRPO 训练切分。

先对当前策略跑一次冻结探测（README 阶段3；`--lr 0.0` 不更新权重）：

    python -m AgenticArxiv.rl.train_grpo --model outputs/sft_messages_v3/final \\
        --task_set expanded --split data/splits/v7_86.json:train \\
        --lr 0.0 --max_steps 408 --allow_zero_variance --no-verify \\
        --output_dir outputs/grpo_messages_v3_frozen_probe

再用本脚本按「每任务至少 2 个有组内奖励方差的 prompt 组」筛选训练任务：

    python -m scripts.build_grpo_signal_split \\
        --probe outputs/grpo_messages_v3_frozen_probe/reward_probe_summary.json \\
        --source-split data/splits/v7_86.json:train \\
        --model outputs/sft_messages_v3/final \\
        --output data/splits/grpo_messages_v3_signal_train.json

候选只允许来自声明的 train，且不能与 dev/iid_test/ood_test 重叠。
输出必须是新路径，历史切分不能被覆盖。以上命令仅供未来有研究输入时运行。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "AgenticArxiv"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from rl.run_manifest import evaluation_protocol_relation, read_protocol_metadata  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_split_ref(ref: str) -> Tuple[Path, str]:
    file_part, _, name = ref.rpartition(":")
    if not file_part or not name:
        raise SystemExit(f"切分引用要写成 FILE:NAME，收到 {ref!r}")
    return Path(file_part), name


def load_eligible_tasks(ref: str) -> List[str]:
    path, name = parse_split_ref(ref)
    if name != "train":
        raise SystemExit(f"信号选择候选必须来自 FILE:train，拒绝切分 {name!r}")
    document = json.loads(path.read_text(encoding="utf-8"))
    split = document.get("split", {})
    if not isinstance(split, Mapping):
        raise SystemExit(f"{path} 的 split 必须为对象")
    if name not in split:
        raise SystemExit(f"{path} 里没有切分 {name!r}（可选：{sorted(split)}）")
    eligible = split[name]
    if not isinstance(eligible, list) or not all(isinstance(task_id, str) for task_id in eligible):
        raise SystemExit(f"{path}:train 必须为任务 ID 列表")
    if len(set(eligible)) != len(eligible):
        raise SystemExit(f"{path}:train 包含重复任务 ID")
    for heldout in ("dev", "iid_test", "ood_test"):
        members = split.get(heldout, [])
        if not isinstance(members, list) or not all(isinstance(task_id, str) for task_id in members):
            raise SystemExit(f"{path}:{heldout} 必须为任务 ID 列表")
        overlap = sorted(set(eligible) & set(members))
        if overlap:
            raise SystemExit(f"{path}:train 与留出切分 {heldout} 交叉: {overlap}")
    return list(eligible)


def informative_group_count(entry: Mapping[str, Any]) -> int:
    groups = int(entry.get("group_count") or 0)
    fraction = float(entry.get("informative_group_fraction") or 0.0)
    return int(round(fraction * groups))


def select_tasks(
    probe_tasks: Mapping[str, Mapping[str, Any]],
    eligible: Sequence[str],
    min_informative_groups: int,
) -> Dict[str, List[str]]:
    buckets: Dict[str, List[str]] = {
        "rl_train": [],
        "dropped_low_signal": [],
        "ceiling_control": [],
        "missing": [],
    }
    for task_id in eligible:
        entry = probe_tasks.get(task_id)
        if entry is None or int(entry.get("group_count") or 0) <= 0:
            buckets["missing"].append(task_id)
            continue
        informative = informative_group_count(entry)
        if informative >= min_informative_groups:
            buckets["rl_train"].append(task_id)
        elif informative == 0:
            buckets["ceiling_control"].append(task_id)
        else:
            buckets["dropped_low_signal"].append(task_id)
    return buckets


def audit_entry(entry: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(entry)
    merged["informative_group_count"] = informative_group_count(entry)
    return merged


def build_document(
    *,
    probe_path: Path,
    probe: Mapping[str, Any],
    source_ref: str,
    model: str,
    num_generations: int,
    seed: int,
    min_informative_groups: int,
    buckets: Mapping[str, List[str]],
) -> Dict[str, Any]:
    probe_tasks = probe["tasks"]
    selected = buckets["rl_train"]
    controls = buckets["ceiling_control"]
    dropped = buckets["dropped_low_signal"]
    audited = selected + controls + dropped
    audit = {task_id: audit_entry(probe_tasks[task_id]) for task_id in audited}
    groups = sum(audit[task_id]["group_count"] for task_id in selected)
    informative = sum(
        audit[task_id]["informative_group_count"] for task_id in selected
    )
    groups_per_task = Counter(
        probe_tasks[task_id]["group_count"] for task_id in selected
    ).most_common(1)[0][0]
    model_protocol = read_protocol_metadata(Path(model))
    probe_protocol = read_protocol_metadata(probe)
    return {
        "version": 6,
        "protocol_metadata": probe_protocol,
        "training_protocol_metadata": model_protocol,
        "protocol_relation": evaluation_protocol_relation(model_protocol, probe_protocol),
        "status": "probe_selected_pending_train",
        "task_set": "expanded",
        "purpose": (
            "setup-aware GRPO train split selected from frozen-SFT "
            "within-prompt reward variance"
        ),
        "derived_from": [source_ref, str(probe_path)],
        "source": {
            "model": model,
            "policy_frozen": True,
            "learning_rate": 0.0,
            "num_generations": num_generations,
            "groups_per_task": groups_per_task,
            "rollouts_per_task": groups_per_task * num_generations,
            "seed": seed,
            "reward_curriculum_steps": 0,
            "reward_summary": str(probe_path),
            "reward_summary_sha256": sha256_file(probe_path),
            "model_protocol_metadata": model_protocol,
        },
        "selection_policy": {
            "eligibility": (
                f"Only tasks in {source_ref} are eligible; dev, iid_test and "
                "ood_test remain held out."
            ),
            "include": (
                f"Keep tasks with at least {min_informative_groups} informative "
                f"prompt groups out of {groups_per_task} in the frozen-policy probe."
            ),
            "exclude": (
                "Move tasks with zero informative prompt groups to "
                "ceiling_control; drop tasks with exactly one informative group "
                "as inconclusive."
            ),
            "rationale": (
                "GRPO needs within-prompt reward variance. A task whose "
                "generations always receive the same reward contributes no "
                "relative-advantage gradient."
            ),
        },
        "split": {
            "rl_train": selected,
            "ceiling_control": controls,
            "dropped_low_signal": dropped,
        },
        "audit": audit,
        "summary": {
            "probe_task_count": len(probe_tasks),
            "eligible_task_count": len(selected) + len(controls) + len(dropped),
            "selected_task_count": len(selected),
            "ceiling_control_task_count": len(controls),
            "dropped_task_count": len(dropped),
            "selected_prompt_group_count": groups,
            "selected_rollout_count": groups * num_generations,
            "informative_group_count": informative,
            "informative_group_fraction": (informative / groups) if groups else 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, help="reward_probe_summary.json 路径")
    parser.add_argument(
        "--source-split", required=True, help="候选任务来源，仅 FILE:train（如 data/splits/v7_86.json:train）"
    )
    parser.add_argument("--model", required=True, help="被探测的策略模型路径")
    parser.add_argument("--output", required=True, help="新输出切分路径，拒绝覆盖既存文件")
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--min-informative-groups", type=int, default=2)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise SystemExit(f"输出路径已存在，拒绝覆盖历史切分: {output}")
    eligible = load_eligible_tasks(args.source_split)
    probe_path = Path(args.probe)
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    if "tasks" not in probe:
        raise SystemExit(f"{probe_path} 里没有 tasks 字段，不是探测汇总文件")

    buckets = select_tasks(probe["tasks"], eligible, args.min_informative_groups)
    if buckets["missing"]:
        raise SystemExit(
            f"探测结果缺少 {len(buckets['missing'])} 个任务的统计："
            f"{buckets['missing'][:8]}（探测是否完全覆盖了 {args.source_split}？）"
        )
    if not buckets["rl_train"]:
        raise SystemExit(
            "探测没有选出任何有信号的任务；先确认探测的 lr 确实为 0、"
            "模型是当前策略，再考虑下调 --min-informative-groups"
        )

    document = build_document(
        probe_path=probe_path,
        probe=probe,
        source_ref=args.source_split,
        model=args.model,
        num_generations=args.num_generations,
        seed=args.seed,
        min_informative_groups=args.min_informative_groups,
        buckets=buckets,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    except FileExistsError as exc:
        raise SystemExit(f"输出路径已存在，拒绝覆盖历史切分: {output}") from exc
    summary = document["summary"]
    print(f"已写入 {output}")
    print(
        "  rl_train={selected_task_count} 条 / {selected_prompt_group_count} 组"
        "（有信息组 {informative_group_count}，占 {informative_group_fraction:.3f}）".format(
            **summary
        )
    )
    print(
        "  ceiling_control={ceiling_control_task_count} 条，"
        "低信号丢弃={dropped_task_count} 条".format(**summary)
    )


if __name__ == "__main__":
    main()
