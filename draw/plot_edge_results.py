#!/usr/bin/env python3
"""端侧 VLM 后训练结果图（合并版）：单图双面板。

(a) 8B 全链路 strict success 演进（greedy 协议，81 任务，含 27B base 对照）
(b) 图表识别 clean-val ROUGE（58 真图，6 模型 × ROUGE-1/L）

数据源（真实评测产物）：artifacts/bench_*/summary.json 与
artifacts/vlm_figureqa_eval/*.json。输出 imgs/edge_results_combined.png（dpi=300）。

用法::

    python3 draw/plot_edge_results.py [--artifacts artifacts] [--output imgs]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# CJK 字体探测（有中文字体则自动启用；图表文本默认英文，不依赖 CJK）
from matplotlib.font_manager import FontManager  # noqa: E402

for _name in ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC"]:
    if _name in {f.name for f in FontManager().ttflist}:
        plt.rcParams["font.sans-serif"] = [_name, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

# 参考调色板（dataviz skill 校验通过：CVD ΔE 24.7，对比度全过）
BLUE, ORANGE = "#2a78d6", "#eb6834"
GRAY = "#c9c8c3"          # 参照条（非系列色，中性）
INK, INK_2 = "#0b0b0b", "#52514e"
GRID, SURFACE = "#e4e3df", "#ffffff"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "xtick.color": INK_2, "ytick.color": INK,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10,
    "axes.grid": True, "grid.color": GRID, "grid.alpha": 0.55, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
})


def strict(path: Path) -> float:
    s = json.loads(path.read_text(encoding="utf-8"))["summary_by_agent"]["regex"]
    return float(s["strict_success_rate"]) * 100


def rouge_clean(path: Path, repo: Path, min_side: int = 300):
    seed = {}
    seed_file = repo / "data" / "vlm" / "figureqa_seed.jsonl"
    for line in seed_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            seed[(r["paper_id"], r["figure_no"], r["question"])] = r["image_relpath"]
    d = json.loads(path.read_text(encoding="utf-8"))["results"]
    r1 = r2 = 0.0
    n = 0
    for r in d:
        rel = seed[(r["paper_id"], r["figure_no"], r["question"])]
        try:
            if max(Image.open(repo / "AgenticArxiv" / rel).size) < min_side:
                continue
        except Exception:  # noqa: BLE001 - 坏图行排除（口径与评测报告一致）
            continue
        n += 1
        r1 += r["rouge1_f"]
        r2 += r["rougeL_f"]
    return n, r1 / n * 100, r2 / n * 100


def style_axis(ax) -> None:
    ax.grid(axis="y", visible=True)
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser(description="端侧 VLM 后训练结果图（合并版）")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--output", default="imgs")
    args = parser.parse_args()
    artifacts = Path(args.artifacts)
    repo = artifacts.parent
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(12.6, 9.0), gridspec_kw={"height_ratios": [1, 1.12], "hspace": 0.42}
    )

    # ---------- (a) 8B 全链路 strict 演进 ----------
    chain = [
        ("27B base\n(greedy)", "bench_base_q38_27b_greedy/summary.json", GRAY),
        ("8B SFT v3", "bench_sft_q3vl8b_v3/summary.json", BLUE),
        ("8B DPO v3", "bench_dpo_q3vl8b_v3/summary.json", BLUE),
        ("8B GRPO v3", "bench_grpo_q3vl8b_v3/summary.json", BLUE),
    ]
    vals = [strict(artifacts / rel) for _, rel, _c in chain]
    labels = [c[0] for c in chain]
    x = np.arange(len(vals))
    bars = ax_a.bar(x, vals, width=0.52, color=[c for *_, c in chain], zorder=3)
    for bar, v in zip(bars, vals):
        ax_a.text(bar.get_x() + bar.get_width() / 2, v + 1.1, f"{v:.1f}%",
                  ha="center", va="bottom", fontsize=11.5,
                  color=INK, fontweight="bold" if v == max(vals) else "normal")
    ax_a.set_xticks(x, labels)
    ax_a.set_ylabel("strict success (%)", fontsize=10.5)
    ax_a.set_ylim(0, max(vals) * 1.22)
    style_axis(ax_a)
    ax_a.set_title("(a)  8B full-chain post-training · 81-task strict success (greedy protocol)",
                   loc="left", fontsize=13, color=INK, pad=12, fontweight="bold")

    # ---------- (b) clean-val ROUGE ----------
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
        n, r1, r2 = rouge_clean(artifacts / rel, repo)
        r1s.append(r1)
        r2s.append(r2)
        labels.append(label)
    x = np.arange(len(labels))
    w = 0.36
    b1 = ax_b.bar(x - w / 2, r1s, w - 0.035, color=BLUE, label="ROUGE-1", zorder=3)
    b2 = ax_b.bar(x + w / 2, r2s, w - 0.035, color=ORANGE, label="ROUGE-L", zorder=3)
    for bar, v in zip(b1, r1s):
        ax_b.text(bar.get_x() + bar.get_width() / 2, v + 0.22, f"{v:.3f}",
                  ha="center", va="bottom", fontsize=9, color=INK)
    base_r1 = r1s[0]
    for i, (bar, v) in enumerate(zip(b1, r1s)):
        gain = (v - base_r1) / base_r1 * 100
        if i > 0 and gain >= 20:
            ax_b.text(bar.get_x() + bar.get_width() / 2, v + 1.0,
                      f"+{gain:.0f}%", ha="center", va="bottom",
                      fontsize=10, color=BLUE, fontweight="bold")
    ax_b.set_xticks(x, labels)
    ax_b.set_ylabel("clean-val ROUGE (58 real figures)", fontsize=10.5)
    ax_b.set_ylim(0, max(max(r1s), max(r2s)) * 1.26)
    style_axis(ax_b)
    ax_b.legend(frameon=False, fontsize=10.5, loc="upper left", ncol=2)
    ax_b.set_title("(b)  Figure QA · clean-val ROUGE (labels = gain over 8B base when ≥20%)",
                   loc="left", fontsize=13, color=INK, pad=12, fontweight="bold")

    fig.savefig(out / "edge_results_combined.png", dpi=300, facecolor=SURFACE,
                bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("EDGE_COMBINED_OK")
    f = out / "edge_results_combined.png"
    print(f"  {f} ({f.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
