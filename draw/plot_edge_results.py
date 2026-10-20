#!/usr/bin/env python3
"""端侧 VLM 后训练结果图：8B 全链路 strict 演进 + clean-val ROUGE 对比。

数据源（真实评测产物，不硬编码数字）：
- artifacts/bench_*_greedy|v3|…/summary.json → 81 任务 strict success
- artifacts/vlm_figureqa_eval/*.json → clean-val（真图 ≥300px）ROUGE-1/L

输出 imgs/edge_chain_benchmark.png 与 imgs/edge_rouge_evolution.png（dpi=300）。

用法::

    python3 draw/plot_edge_results.py [--artifacts artifacts] [--output imgs]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# CJK 字体探测（与 draw/plot.py 同款候选与回退）
from matplotlib.font_manager import FontManager  # noqa: E402
import warnings  # noqa: E402

fm = FontManager()
available = {f.name for f in fm.ttflist}
for _name in ["WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "Noto Sans CJK SC",
              "SimHei", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK"]:
    if _name in available:
        plt.rcParams["font.sans-serif"] = [_name, "DejaVu Sans"]
        break
else:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 参考调色板（dataviz skill 校验通过：ΔE 24.7 CVD）
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SURFACE = "#ffffff"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.alpha": 0.6, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
})


def strict_from_summary(path: Path) -> float:
    s = json.loads(path.read_text(encoding="utf-8"))["summary_by_agent"]["regex"]
    return float(s["strict_success_rate"]) * 100


def rouge_clean(path: Path, artifacts: Path, min_side: int = 300):
    seed = {}
    seed_file = artifacts.parent / "data" / "vlm" / "figureqa_seed.jsonl"
    for line in seed_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            seed[(r["paper_id"], r["figure_no"], r["question"])] = r["image_relpath"]
    d = json.loads(path.read_text(encoding="utf-8"))["results"]
    r1, r2, n = 0.0, 0.0, 0
    for r in d:
        rel = seed[(r["paper_id"], r["figure_no"], r["question"])]
        try:
            if max(Image.open(artifacts.parent / "AgenticArxiv" / rel).size) < min_side:
                continue
        except Exception:  # noqa: BLE001 - 坏图行排除（口径与评测报告一致）
            continue
        n += 1
        r1 += r["rouge1_f"]
        r2 += r["rougeL_f"]
    return n, r1 / n * 100, r2 / n * 100


def plot_chain_benchmark(artifacts: Path, output: Path) -> None:
    stages = [
        ("27B base\n(greedy)", "bench_base_q38_27b_greedy/summary.json"),
        ("8B base\n(参考)", "bench_base_q3vl8b_greedy/summary.json"),
        ("8B SFT v3", "bench_sft_q3vl8b_v3/summary.json"),
        ("8B DPO v3", "bench_dpo_q3vl8b_v3/summary.json"),
        ("8B GRPO v3", "bench_grpo_q3vl8b_v3/summary.json"),
    ]
    values, labels = [], []
    for label, rel in stages:
        p = artifacts / rel
        if not p.exists():
            continue
        values.append(strict_from_summary(p))
        labels.append(label)

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    colors = [GRID] + [BLUE] * (len(values) - 1)
    bars = ax.bar(range(len(values)), values, width=0.58, color=colors, zorder=3)
    for i, (bar, v) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.2, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=10.5,
                color=INK if i else INK_2,
                fontweight="bold" if i == len(values) - 1 else "normal")
    ax.axhline(values[0], color=INK_2, linestyle=(0, (4, 3)), linewidth=1.2, zorder=2)
    ax.text(len(values) - 0.45, values[0] + 1.2, "27B base",
            ha="right", va="bottom", fontsize=9, color=INK_2)
    ax.set_xticks(range(len(labels)), labels, fontsize=10)
    ax.set_ylabel("81-task strict success (%)")
    ax.set_ylim(0, max(values) * 1.22)
    ax.set_title("8B full-chain post-training · strict success (greedy protocol)",
                 fontsize=12, color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(output, dpi=300, facecolor=SURFACE)
    plt.close(fig)


def plot_rouge_evolution(artifacts: Path, output: Path) -> None:
    models = [
        ("8B base", "vlm_figureqa_eval/qwen3vl_8b_base.json"),
        ("8B SFT v3", "vlm_figureqa_eval/qwen3vl_8b_sft_v3.json"),
        ("8B DPO v3", "vlm_figureqa_eval/qwen3vl_8b_dpo_v3.json"),
        ("8B GRPO v3", "vlm_figureqa_eval/qwen3vl_8b_grpo_v3.json"),
        ("27B base", "vlm_figureqa_eval/qwen38_27b_base.json"),
        ("27B cap SFT", "vlm_figureqa_eval/qwen38_27b_cap.json"),
    ]
    r1s, r2s, labels = [], [], []
    for label, rel in models:
        p = artifacts / rel
        if not p.exists():
            continue
        n, r1, r2 = rouge_clean(p, artifacts)
        r1s.append(r1)
        r2s.append(r2)
        labels.append(label)

    import numpy as np

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    b1 = ax.bar(x - w / 2, r1s, w - 0.03, color=BLUE, label="ROUGE-1", zorder=3)
    b2 = ax.bar(x + w / 2, r2s, w - 0.03, color=ORANGE, label="ROUGE-L", zorder=3)
    for bar, v in zip(b1, r1s):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.25, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9, color=INK)
    base_r1 = r1s[0]
    for i, v in enumerate(r1s):
        gain = (v - base_r1) / base_r1 * 100
        if gain >= 20 and i > 0:
            ax.text(b1[i].get_x() + b1[i].get_width() / 2, r1s[i] + 1.1,
                    f"+{gain:.0f}%", ha="center", va="bottom",
                    fontsize=9.5, color=BLUE, fontweight="bold")
    ax.set_xticks(x, labels, fontsize=10)
    ax.set_ylabel("clean-val ROUGE (58 real figures, clipart rows excluded)")
    ax.set_ylim(0, max(max(r1s), max(r2s)) * 1.25)
    ax.set_title("Figure QA · clean-val ROUGE (labels = gain over 8B base when ≥20%)",
                 fontsize=12, color=INK, pad=12)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    fig.tight_layout()
    fig.savefig(output, dpi=300, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="端侧 VLM 后训练结果图")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--output", default="imgs")
    args = parser.parse_args()
    artifacts = Path(args.artifacts)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    plot_chain_benchmark(artifacts, out / "edge_chain_benchmark.png")
    plot_rouge_evolution(artifacts, out / "edge_rouge_evolution.png")
    print("EDGE_PLOTS_OK")
    for p in ("edge_chain_benchmark.png", "edge_rouge_evolution.png"):
        f = out / p
        print(f"  {f} ({f.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
