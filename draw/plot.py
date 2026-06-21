#!/usr/bin/env python3
# draw/plot.py
"""
从 data/ 目录读取 benchmark 数据，生成对比图表到 draw/images/。

用法:
  cd AgenticArxiv 项目根目录
  python draw/plot.py                        # 使用默认路径
  python draw/plot.py --data data/raw_data.csv --output draw/images
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------- 中文字体 ----------

def _setup_chinese_font():
    """尝试设置中文字体，失败则回退英文"""
    import warnings
    from matplotlib.font_manager import FontManager
    fm = FontManager()
    available = {f.name for f in fm.ttflist}

    candidates = [
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
        "Noto Sans CJK SC", "SimHei", "Microsoft YaHei",
        "PingFang SC", "STHeiti", "AR PL UMing CN",
    ]
    for font_name in candidates:
        if font_name in available:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            # 验证渲染
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                fig, ax = plt.subplots()
                ax.set_title("平均耗时分解")
                fig.canvas.draw()
                plt.close(fig)
                if not any("missing from font" in str(x.message) for x in w):
                    return True
    # 回退英文
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    return False

_HAS_CN = _setup_chinese_font()

# 全局字号
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

def _label(cn: str, en: str) -> str:
    return cn if _HAS_CN else en


# ---------- 数据加载 ----------

def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _group_by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    return dict(groups)


# ---------- 颜色/样式 ----------

AGENT_COLORS = {
    "regex": "#4C78A8",
    "mcp": "#F58518",
    "skill_cli": "#54A24B",
}
AGENT_LABELS = {
    "regex": "Regex",
    "mcp": "MCP",
    "skill_cli": "Skill-CLI",
}
AGENT_ORDER = ["regex", "mcp", "skill_cli"]


def _ordered_agents(agents: list[str]) -> list[str]:
    return [a for a in AGENT_ORDER if a in agents]


def _per_task_agent_data(rows: list[dict], value_key: str, cast=float):
    """按 task × agent 聚合数据，返回 (tasks, agents, data_dict, by_agent_all)
    data_dict: (agent, task_id) -> [values]
    by_agent_all: agent -> [rows]
    """
    by_task = _group_by(rows, "task_id")
    tasks = sorted(by_task.keys())
    data: dict[tuple[str, str], list] = {}
    all_agents: set[str] = set()
    for task_id, items in by_task.items():
        for r in items:
            a = r["agent_type"]
            all_agents.add(a)
            data.setdefault((a, task_id), []).append(cast(r[value_key]))
    agents = _ordered_agents(list(all_agents))
    by_agent_all = _group_by(rows, "agent_type")
    return tasks, agents, data, by_agent_all


def _subtitle_line(by_agent_all, agents, value_key, fmt="{:.0f}", unit="", cast=float):
    """生成 per-agent 聚合统计的 subtitle 字符串"""
    parts = []
    for a in agents:
        v = _mean([cast(r[value_key]) for r in by_agent_all[a]])
        parts.append(f"{AGENT_LABELS[a]}: {fmt.format(v)}{unit}")
    return " | ".join(parts)


def _grouped_bar_positions(n_tasks, n_agents):
    bar_w = 0.8 / max(n_agents, 1)
    return bar_w


# ---------- 图表 ----------

def plot_time_breakdown(rows: list[dict], output_dir: str):
    """饼状图：每个 task × agent 的 LLM / Tool / Overhead 时间分解"""
    by_task = _group_by(rows, "task_id")
    tasks = sorted(by_task.keys())
    if not tasks:
        return

    # (agent, task) -> (avg_llm, avg_tool, avg_oh)
    atd: dict[tuple[str, str], tuple] = {}
    all_agents: set[str] = set()
    for tid, items in by_task.items():
        by_a = _group_by(items, "agent_type")
        for a, aitems in by_a.items():
            all_agents.add(a)
            atd[(a, tid)] = (
                _mean([float(r["llm_ms"]) for r in aitems]),
                _mean([float(r["tool_ms"]) for r in aitems]),
                _mean([float(r["overhead_ms"]) for r in aitems]),
            )

    agents = _ordered_agents(list(all_agents))
    n_tasks, n_agents = len(tasks), len(agents)

    # 组件颜色
    comp_colors = ["#E45756", "#F2A86B", "#88BEDC"]  # LLM, Tool, Overhead
    comp_names = ["LLM", _label("Tool", "Tool"), _label("Overhead", "Overhead")]

    # subtitle
    by_agent_all = _group_by(rows, "agent_type")
    sub = _subtitle_line(by_agent_all, agents, "total_ms", unit="ms")

    fig, axes = plt.subplots(n_agents, n_tasks,
                             figsize=(n_tasks * 2.5, n_agents * 2.8 + 0.8))
    # 确保 axes 始终是 2D
    if n_agents == 1:
        axes = [axes]
    if n_tasks == 1:
        axes = [[ax] for ax in axes]

    for i, a in enumerate(agents):
        for j, t in enumerate(tasks):
            ax = axes[i][j]
            llm, tool, oh = atd.get((a, t), (0, 0, 0))
            total = llm + tool + oh
            if total == 0:
                ax.set_visible(False)
                continue

            sizes = [llm, tool, oh]
            pcts = [v / total * 100 for v in sizes]

            def autopct_func(pct, vals=sizes):
                idx = [i for i, p in enumerate([v / total * 100 for v in vals])
                       if abs(p - pct) < 0.1]
                ms = vals[idx[0]] if idx else 0
                return f"{ms:.0f}ms\n({pct:.0f}%)" if pct >= 5 else ""

            wedges, texts, autotexts = ax.pie(
                sizes, colors=comp_colors, autopct=autopct_func,
                startangle=90, pctdistance=0.65,
                textprops={"fontsize": 9},
            )
            for at in autotexts:
                at.set_fontsize(8)

            # 行首显示 agent 名
            if j == 0:
                ax.set_ylabel(AGENT_LABELS.get(a, a), fontsize=14, fontweight="bold")
            # 首行显示 task 名
            if i == 0:
                ax.set_title(t, fontsize=14, fontweight="bold")
            # 右下角显示总耗时
            ax.text(0, -1.25, f"{total:.0f}ms", ha="center", fontsize=9, color="gray")

    # 图例
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=n) for c, n in zip(comp_colors, comp_names)]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=13,
               bbox_to_anchor=(0.5, 0.01))

    title = _label("各任务耗时分解", "Time Breakdown per Task")
    fig.suptitle(f"{title} (N={len(rows)})\n{sub}", fontsize=20)
    fig.tight_layout(rect=[0, 0.08, 1, 0.99])
    fig.savefig(os.path.join(output_dir, "time_breakdown.png"), dpi=300)
    plt.close(fig)
    print(f"  time_breakdown.png")


def plot_completion_rate(rows: list[dict], output_dir: str):
    """按任务分组：任务完成率 + 工具调用准确率（上下两个子图）"""
    by_task = _group_by(rows, "task_id")
    tasks = sorted(by_task.keys())
    if not tasks:
        return

    # (agent, task) -> (completion%, accuracy%)
    atd: dict[tuple[str, str], tuple] = {}
    all_agents: set[str] = set()
    for tid, items in by_task.items():
        by_a = _group_by(items, "agent_type")
        for a, aitems in by_a.items():
            all_agents.add(a)
            comp = _mean([1 if r["completed"] == "True" else 0 for r in aitems]) * 100
            acc = _mean([1 if r["tool_accurate"] == "True" else 0 for r in aitems]) * 100
            atd[(a, tid)] = (comp, acc)

    agents = _ordered_agents(list(all_agents))
    n_tasks, n_agents = len(tasks), len(agents)
    bar_w = 0.8 / max(n_agents, 1)

    # per-agent aggregates for subtitle
    by_agent_all = _group_by(rows, "agent_type")
    comp_parts, acc_parts = [], []
    for a in agents:
        lbl = AGENT_LABELS.get(a, a)
        cv = _mean([1 if r["completed"] == "True" else 0 for r in by_agent_all[a]]) * 100
        av = _mean([1 if r["tool_accurate"] == "True" else 0 for r in by_agent_all[a]]) * 100
        comp_parts.append(f"{lbl}: {cv:.0f}%")
        acc_parts.append(f"{lbl}: {av:.0f}%")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, n_tasks * 2), 9), sharex=True)

    def _draw_rate_bars(ax, metric_idx, ylabel, subtitle):
        for j, a in enumerate(agents):
            positions = [i + j * bar_w for i in range(n_tasks)]
            vals = [atd.get((a, t), (0, 0))[metric_idx] for t in tasks]
            bars = ax.bar(positions, vals, bar_w,
                          label=AGENT_LABELS.get(a, a), color=AGENT_COLORS.get(a, "#999"))
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                        f"{v:.0f}%", ha="center", fontsize=12)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 115)
        ax.set_title(subtitle, fontsize=18)
        ax.legend(fontsize=12)

    _draw_rate_bars(ax1, 0,
                    _label("完成率 (%)", "Completion (%)"),
                    _label("任务完成率", "Completion Rate") + f": {' | '.join(comp_parts)}")
    _draw_rate_bars(ax2, 1,
                    _label("准确率 (%)", "Accuracy (%)"),
                    _label("工具调用准确率", "Tool Accuracy") + f": {' | '.join(acc_parts)}")

    ax2.set_xticks([i + bar_w * (n_agents - 1) / 2 for i in range(n_tasks)])
    ax2.set_xticklabels(tasks, rotation=30, ha="right")

    title = _label("各任务准确性对比", "Accuracy per Task")
    fig.suptitle(f"{title} (N={len(rows)})", fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 1.005])
    fig.savefig(os.path.join(output_dir, "accuracy_comparison.png"), dpi=300)
    plt.close(fig)
    print(f"  accuracy_comparison.png")


def plot_iterations(rows: list[dict], output_dir: str):
    """抖动散点图：各任务在不同 Agent 模式下的迭代次数分布"""
    import numpy as np

    by_task = _group_by(rows, "task_id")
    tasks = sorted(by_task.keys())
    if not tasks:
        return

    by_agent_task: dict[str, dict[str, list[int]]] = {}
    for task_id, items in by_task.items():
        for r in items:
            at = r["agent_type"]
            by_agent_task.setdefault(at, {}).setdefault(task_id, []).append(int(r["iterations"]))

    agents = _ordered_agents(list(by_agent_task.keys()))
    n_tasks = len(tasks)
    n_agents = len(agents)
    group_width = 0.8
    point_w = group_width / max(n_agents, 1)

    fig, ax = plt.subplots(figsize=(max(8, n_tasks * 1.5), 5))

    for j, a in enumerate(agents):
        color = AGENT_COLORS.get(a, "#999")
        for i, t in enumerate(tasks):
            iterations = by_agent_task.get(a, {}).get(t, [0])
            if not iterations:
                continue

            x_offset = (j - (n_agents - 1) / 2) * point_w
            x_jitter = np.random.normal(0, 0.04, len(iterations))
            x_pos = i + x_offset + x_jitter

            ax.scatter(x_pos, iterations, alpha=0.6, s=50, color=color,
                      edgecolors="black", linewidth=0.5, label=AGENT_LABELS.get(a, a) if i == 0 else "")

            mean_val = _mean(iterations)
            ax.plot([i + x_offset - point_w * 0.35, i + x_offset + point_w * 0.35],
                   [mean_val, mean_val], color="black", linewidth=2.5, zorder=3)

    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    by_agent_all = _group_by(rows, "agent_type")
    sub = _subtitle_line(by_agent_all, agents, "iterations", fmt="{:.1f}")

    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels.append(l)
            unique_handles.append(h)
    ax.legend(unique_handles, unique_labels, fontsize=12)

    ax.set_ylabel(_label("迭代次数", "Iterations"))
    title = _label("各任务迭代次数分布", "Iteration Distribution per Task")
    ax.set_title(f"{title} (N={len(rows)})\n{sub}", fontsize=15)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "iteration_boxplot.png"), dpi=300)
    plt.close(fig)
    print(f"  iteration_boxplot.png")


def plot_iterations_heatmap(rows: list[dict], output_dir: str):
    """热力图：任务 × agent 的平均迭代次数"""
    import numpy as np

    by_task = _group_by(rows, "task_id")
    tasks = sorted(by_task.keys())
    if not tasks:
        return

    by_agent_task: dict[str, dict[str, list[int]]] = {}
    for task_id, items in by_task.items():
        for r in items:
            at = r["agent_type"]
            by_agent_task.setdefault(at, {}).setdefault(task_id, []).append(int(r["iterations"]))

    agents = _ordered_agents(list(by_agent_task.keys()))
    n_tasks = len(tasks)
    n_agents = len(agents)

    # 构建数据矩阵 (agent × task)
    data = np.zeros((n_agents, n_tasks))
    for i, a in enumerate(agents):
        for j, t in enumerate(tasks):
            iterations = by_agent_task.get(a, {}).get(t, [0])
            data[i, j] = _mean(iterations)

    fig, ax = plt.subplots(figsize=(max(8, n_tasks * 1.2), 4))

    im = ax.imshow(data, cmap="Blues", aspect="auto", vmin=2.0, vmax=4.0)

    # 添加数值标签
    for i in range(n_agents):
        for j in range(n_tasks):
            text = ax.text(j, i, f"{data[i, j]:.1f}",
                          ha="center", va="center", color="black", fontsize=12)

    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_yticks(range(n_agents))
    ax.set_yticklabels([AGENT_LABELS.get(a, a) for a in agents])

    # colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(_label("平均迭代次数", "Avg Iterations"), rotation=270, labelpad=15, fontsize=13)

    title = _label("各任务迭代次数热力图", "Iteration Heatmap per Task")
    ax.set_title(f"{title} (N={len(rows)})", fontsize=15)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "iteration_heatmap.png"), dpi=300)
    plt.close(fig)
    print(f"  iteration_heatmap.png")


def plot_per_task_time(rows: list[dict], output_dir: str):
    """分组条形图：每个任务在不同 Agent 下的总耗时"""
    by_task = _group_by(rows, "task_id")
    tasks = sorted(by_task.keys())
    if not tasks:
        return

    by_agent_task = {}  # agent -> {task_id: avg_ms}
    for task_id, items in by_task.items():
        for r in items:
            at = r["agent_type"]
            by_agent_task.setdefault(at, {}).setdefault(task_id, []).append(float(r["total_ms"]))

    agents = _ordered_agents(list(by_agent_task.keys()))
    n_tasks = len(tasks)
    n_agents = len(agents)
    bar_w = 0.8 / max(n_agents, 1)

    fig, ax = plt.subplots(figsize=(max(8, n_tasks * 1.5), 5))
    for j, a in enumerate(agents):
        avgs = [_mean(by_agent_task.get(a, {}).get(t, [0])) for t in tasks]
        positions = [i + j * bar_w for i in range(n_tasks)]
        ax.bar(positions, avgs, bar_w, label=AGENT_LABELS.get(a, a), color=AGENT_COLORS.get(a, "#999"))

    ax.set_xticks([i + bar_w * (n_agents - 1) / 2 for i in range(n_tasks)])
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    # subtitle: per-agent average
    by_agent_all = _group_by(rows, "agent_type")
    sub = _subtitle_line(by_agent_all, agents, "total_ms", unit="ms")

    ax.set_ylabel(_label("平均耗时 (ms)", "Avg Time (ms)"))
    title = _label("各任务平均耗时", "Avg Time per Task")
    ax.set_title(f"{title} (N={len(rows)})\n{sub}", fontsize=15)
    ax.legend(fontsize=12)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "per_task_time.png"), dpi=300)
    plt.close(fig)
    print(f"  per_task_time.png")


def plot_token_usage(rows: list[dict], output_dir: str):
    """按任务分组：各 Agent 平均 Token 使用量"""
    has_tokens = any(int(r.get("tokens", 0)) > 0 for r in rows)
    if not has_tokens:
        print(f"  token_usage.png (skipped: no token data)")
        return

    tasks, agents, data, by_agent_all = _per_task_agent_data(rows, "tokens", cast=int)
    if not tasks:
        return

    n_tasks, n_agents = len(tasks), len(agents)
    bar_w = 0.8 / max(n_agents, 1)
    sub = _subtitle_line(by_agent_all, agents, "tokens", cast=int)

    fig, ax = plt.subplots(figsize=(max(10, n_tasks * 2), 6))
    for j, a in enumerate(agents):
        positions = [i + j * bar_w for i in range(n_tasks)]
        avgs = [_mean(data.get((a, t), [0])) for t in tasks]
        bars = ax.bar(positions, avgs, bar_w,
                      label=AGENT_LABELS.get(a, a), color=AGENT_COLORS.get(a, "#999"))
        for bar, v in zip(bars, avgs):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 50,
                        f"{v:.0f}", ha="center", fontsize=11)

    ax.set_xticks([i + bar_w * (n_agents - 1) / 2 for i in range(n_tasks)])
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_ylabel(_label("词元用量 (Tokens)", "Tokens"))
    title = _label("各任务平均 Token 用量", "Avg Token Usage per Task")
    ax.set_title(f"{title} (N={len(rows)})\n{sub}", fontsize=15)
    ax.legend(fontsize=12)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "token_usage.png"), dpi=300)
    plt.close(fig)
    print(f"  token_usage.png")


# ---------- 工具函数 ----------

def _mean(vals: list) -> float:
    return sum(vals) / len(vals) if vals else 0.0


# ---------- 主入口 ----------

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Benchmark 绘图")
    parser.add_argument("--data", default=os.path.join(repo_root, "data", "raw_data.csv"),
                        help="CSV 数据路径")
    parser.add_argument("--output", default=os.path.join(repo_root, "draw", "images"),
                        help="图片输出目录")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"错误: 数据文件不存在: {args.data}")
        print("请先运行 benchmark:")
        print("  cd AgenticArxiv && python -m benchmark.run_benchmark")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    rows = load_csv(args.data)
    print(f"加载 {len(rows)} 条数据，开始绘图...")
    print(f"中文字体: {'OK' if _HAS_CN else 'fallback to English'}")

    plot_time_breakdown(rows, args.output)
    plot_completion_rate(rows, args.output)
    plot_iterations(rows, args.output)
    plot_iterations_heatmap(rows, args.output)
    plot_per_task_time(rows, args.output)
    plot_token_usage(rows, args.output)

    print(f"\n绘图完成，输出目录: {args.output}")


if __name__ == "__main__":
    main()
