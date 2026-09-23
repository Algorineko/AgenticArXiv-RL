#!/usr/bin/env python3
"""奖励课程的实际表现分析（README TODO P1 的证据工具）。

课程「前 30 步把 tool / argument / outcome 权重压到 1/3」是先验设定的档位。
要把它定档，需要回答一个可证伪的问题：**分开压这三个分量，真的让它们学得更
好吗**——而不是让曲线看起来更平滑。

这个脚本读一次 GRPO 运行写出的 TensorBoard 事件，按课程窗口切成两段
（压制期 / 全权重期），逐分量对比均值与斜率，并给出结论：

  - 压制期的 format / process 是否确实处于较低水平（课程的前提）；
  - 全权重期 tool / argument / outcome 是否高于压制期（课程的效果）；
  - 压制期与全权重期的总量不可直接比较——权重变了，total 会自己跳一档，
    所以只看分量自身的趋势。

用法：
    python scripts/analyze_curriculum.py outputs/grpo/logs/<run>
    python scripts/analyze_curriculum.py <run> --curriculum-steps 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


#: 与 rl/reward.py 的 RewardSchedule 字段一致。
WEIGHTED_COMPONENTS = ("format", "tool", "argument", "process", "outcome")
CURRICULUM_SCALED = ("tool", "argument", "outcome")
DIAGNOSTICS = ("result_quality", "efficiency")


def _load_event_dir(path: Path):
    """Accept either a run directory or a parent containing several runs."""
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    candidates = [path] if (path / "events.out.tfevents").exists() or list(path.glob("events.out.tfevents*")) else []
    if not candidates:
        candidates = sorted({p.parent for p in path.rglob("events.out.tfevents*")})
    if not candidates:
        raise SystemExit(f"❌ 在 {path} 下找不到 TensorBoard 事件文件")
    if len(candidates) > 1:
        names = ", ".join(str(c) for c in candidates[:5])
        raise SystemExit(
            f"❌ {path} 下有多个 run（{names}…），请指定其中一个"
        )
    accumulator = EventAccumulator(str(candidates[0]), size_guidance={"scalars": 0})
    accumulator.Reload()
    return accumulator


#: TRL 把指标写进 `_metrics["train"]`，Trainer 落盘时会加上 split 前缀，
#: 于是同一份曲线在不同版本里叫 `reward_components/x` 或 `train/reward_components/x`。
#: 两种都要认，否则工具会在数据明明存在时报「没有找到曲线」。
_TAG_PREFIXES = ("", "train/")


def _scalar_series(accumulator, tag: str) -> List[Tuple[int, float]]:
    for prefix in _TAG_PREFIXES:
        try:
            events = accumulator.Scalars(prefix + tag)
        except KeyError:
            continue
        return [(event.step, float(event.value)) for event in events]
    return []


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _split(series, boundary: int):
    early = [value for step, value in series if step < boundary]
    late = [value for step, value in series if step >= boundary]
    return early, late


def _fmt(value: Optional[float]) -> str:
    return "  n/a " if value is None else f"{value:+.3f}"


def _collect_rows(accumulator, curriculum_steps: int) -> list:
    rows = []
    for name in WEIGHTED_COMPONENTS + DIAGNOSTICS:
        series = _scalar_series(accumulator, f"reward_components/{name}")
        if not series:
            continue
        early, late = _split(series, curriculum_steps)
        rows.append((name, series, early, late))
    if not rows:
        raise SystemExit(
            "❌ 没有找到 reward_components/* 曲线。训练时请加 --report_to tensorboard。"
        )
    return rows


def _print_component_table(rows) -> None:
    print("| 分量 | 压制期均值 | 全权重期均值 | 变化 |")
    print("|---|---:|---:|---:|")
    for name, _series, early, late in rows:
        early_mean, late_mean = _mean(early), _mean(late)
        delta = (
            f"{late_mean - early_mean:+.3f}"
            if early_mean is not None and late_mean is not None
            else "n/a"
        )
        print(f"| `{name}` | {_fmt(early_mean)} | {_fmt(late_mean)} | {delta} |")


def _print_weight_curve(accumulator) -> None:
    """权重曲线是课程确实生效的直接证据。

    没有它就没法区分「策略变强」与「权重放开」——两者在总奖励上长得一样。
    """
    weight_series = _scalar_series(accumulator, "reward_weights/tool")
    if not weight_series:
        print(
            "\n⚠️  没有 `reward_weights/*` 曲线，无法区分「策略变强」与「课程放开权重」。"
        )
        return
    distinct = sorted({round(value, 6) for _step, value in weight_series})
    print(f"\n`reward_weights/tool` 出现过的取值: {distinct}")
    if len(distinct) == 1:
        print("⚠️  权重全程没有变化 —— 课程没生效，或者训练步数还没跨过窗口。")


def _verdicts(accumulator, rows, curriculum_steps: int) -> List[str]:
    by_name = {name: (early, late) for name, _s, early, late in rows}
    verdicts: List[str] = []

    for name in ("format", "process"):
        early = by_name.get(name, ([], []))[0]
        mean = _mean(early)
        if mean is None:
            continue
        if mean >= 0.9:
            verdicts.append(
                f"- ⚠️  `{name}` 在压制期就已经到 {mean:.3f}：这一档不需要课程保护，"
                "压低它对应的学习信号反而是浪费预算。"
            )
        else:
            verdicts.append(
                f"- ✅ `{name}` 在压制期均值 {mean:.3f}，确有提升空间，符合"
                "「先学 ReAct 结构」的前提。"
            )

    for name in CURRICULUM_SCALED:
        early, late = by_name.get(name, ([], []))
        early_mean, late_mean = _mean(early), _mean(late)
        if early_mean is None or late_mean is None:
            continue
        if late_mean > early_mean:
            verdicts.append(
                f"- ✅ `{name}` 全权重期 {late_mean:+.3f} 高于压制期 {early_mean:+.3f}："
                "放开权重后确实还在学。"
            )
        else:
            verdicts.append(
                f"- ⚠️  `{name}` 全权重期 {late_mean:+.3f} 未超过压制期 {early_mean:+.3f}："
                "要么已收敛，要么这一步的档位/步数需要重新考虑。"
            )

    total_early = _mean(
        [v for s, v in _scalar_series(accumulator, "reward") if s < curriculum_steps]
    )
    total_late = _mean(
        [v for s, v in _scalar_series(accumulator, "reward") if s >= curriculum_steps]
    )
    if total_early is not None and total_late is not None:
        verdicts.append(
            f"- ℹ️  `reward` 总量 {total_early:+.3f} → {total_late:+.3f}。"
            "跨窗口比较总量没有意义（权重被放开了），必须看上面的分量。"
        )

    return verdicts


def analyze(log_dir: Path, curriculum_steps: int) -> int:
    accumulator = _load_event_dir(log_dir)
    if not set(accumulator.Tags().get("scalars", [])):
        raise SystemExit(f"❌ {log_dir} 的事件里没有任何 scalar")

    print(f"# 奖励课程分析: {log_dir}")
    print(
        f"课程窗口: 前 {curriculum_steps} 步压制 "
        f"{'/'.join(CURRICULUM_SCALED)} 的权重\n"
    )

    rows = _collect_rows(accumulator, curriculum_steps)
    _print_component_table(rows)
    _print_weight_curve(accumulator)

    print("\n## 结论")
    verdicts = _verdicts(accumulator, rows, curriculum_steps)
    print("数据不足，无法得出结论。" if not verdicts else "\n".join(verdicts))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log_dir", type=Path, help="TensorBoard 事件目录")
    parser.add_argument(
        "--curriculum-steps",
        type=int,
        default=30,
        help="课程窗口长度，须与训练时的 --reward_curriculum_steps 一致（默认 30）",
    )
    args = parser.parse_args()
    if args.curriculum_steps < 1:
        raise SystemExit("--curriculum-steps 必须为正")
    return analyze(args.log_dir, args.curriculum_steps)


if __name__ == "__main__":
    sys.exit(main())
