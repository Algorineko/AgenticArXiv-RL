#!/usr/bin/env python3
"""figureqa v2 数据集：seed 冻结行 + 扩采图 + 新题型。

纪律（对应 local_docs/plan.md S1 的 GO 判据）：

- **seed 行逐字节保留**：``figureqa_seed.jsonl`` 的 630 行原样进入输出，
  其中 94 行 val 引用的（论文, 图, 问题）组合永不进入任何训练划分；
- **按论文切分防泄漏**：新图行按论文归属划分——seed val 论文的新图一律
  ``val2``（不训练），其余 ``train``；
- **新题型 prompt 定义在本文件内**（``_V2_QUESTION_PROMPTS``），不触碰
  ``tools.figure_analysis_tool._QUESTION_PROMPTS`` 冻结面，快照回放确定性不受影响；
- 去重：新图与 seed 图按图片字节头部 sha256 去重，近重复不进池。

监督目标仍然全部来自作者 caption（确定性规则），``figure_type`` 用
caption 关键词映射到封闭枚举，未命中关键词的图跳过（宁缺毋滥）。

用法::

    python scripts/reextract_figures.py            # 先扩图池
    python scripts/build_figure_qa_v2_dataset.py   # 再构建数据集
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "AgenticArxiv"
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_figure_qa_dataset import FIGURE_PREFIX_RE, build_target, clean_text  # noqa: E402
from tools.figure_analysis_tool import (  # noqa: E402
    _AXES_HINTS,
    _TREND_HINTS,
    _matching_sentences,
    _sentences,
)
from utils.io_utils import write_jsonl  # noqa: E402

# 新题型：prompt 本地定义（不进 env 工具冻结面），target 仍是 caption 确定性规则。
_V2_QUESTION_PROMPTS = {
    "figure_type": (
        "What type of figure is this? Answer with only the figure type "
        "(for example: line plot, bar chart, scatter plot, heatmap, table, "
        "schematic diagram, photograph)."
    ),
    "reading": (
        "What are the key numbers or quantities reported for this figure? "
        "Answer in one sentence."
    ),
    "caption": (
        "Reproduce the caption of this figure as written in the paper."
    ),
}

# 封闭枚举 → caption 关键词（有序，特异在前，首触即停）。
_FIGURE_TYPE_KEYWORDS: List[Tuple[str, Tuple[str, ...]]] = [
    ("heatmap", ("heat map", "heatmap")),
    ("pie chart", ("pie chart", "pie graph")),
    ("bar chart", ("bar chart", "bar graph", "histogram", "bar plot")),
    ("scatter plot", ("scatter",)),
    ("line plot", ("line plot", "line chart", "line graph", "learning curve", "curves")),
    ("table", (" table", "tables")),  # 带边界防 "tablet"/"portable"
    ("photograph", ("photograph", "photo of", "microscop", "camera capture", "real-world image")),
    ("map", ("map of", "map showing", "map visualizing", "geographic")),
    ("schematic diagram", (
        "diagram", "schematic", "architecture", "pipeline", "framework",
        "workflow", "flowchart", "overview of", "illustration of",
    )),
]

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def sha256_head(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
    return digest.hexdigest()


def figure_type_target(caption: str) -> str:
    """caption 关键词 → 封闭枚举；未命中返回 ""（该行不生成）。"""
    lowered = " " + clean_text(caption).lower()
    for label, keywords in _FIGURE_TYPE_KEYWORDS:
        if any(kw in lowered for kw in keywords):
            return label
    return ""


def reading_target(caption: str) -> str:
    """取含数字最多的那一句（至少含 1 个数字），保证目标中的数字都来自 caption。"""
    best, best_hits = "", 0
    for sentence in _sentences(clean_text(caption)):
        hits = len(_NUMBER_RE.findall(sentence))
        if hits > best_hits:
            best, best_hits = sentence, hits
    if best_hits < 1:
        return ""
    return best.strip()


def caption_target(caption: str) -> str:
    """caption 全文（≤4 句，去 Figure N 前缀）——图→文的完整接地监督。"""
    from build_figure_qa_dataset import describe_target

    cleaned = FIGURE_PREFIX_RE.sub("", clean_text(caption))
    sentences = _sentences(cleaned)
    if not sentences:
        return cleaned
    return " ".join(sentences[:4])


def build_v2_target(caption: str, question: str) -> str:
    if question == "figure_type":
        return figure_type_target(caption)
    if question == "reading":
        return reading_target(caption)
    if question == "caption":
        return caption_target(caption)
    return build_target(caption, question)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 figureqa v2 数据集（seed 冻结 + 扩采 + 新题型）")
    parser.add_argument("--seed-data", default="data/vlm/figureqa_seed.jsonl")
    parser.add_argument("--extraction", default="data/vlm/figureqa_v2_extraction.json")
    parser.add_argument("--output", default="data/vlm/figureqa_v2.jsonl")
    parser.add_argument("--image-root", default="AgenticArxiv")
    parser.add_argument("--seed-questions", nargs="+", default=["describe", "axes", "trend"])
    parser.add_argument("--v2-questions", nargs="+", default=["figure_type", "reading", "caption"])
    args = parser.parse_args()

    seed_path = Path(args.seed_data)
    extraction_path = Path(args.extraction)
    for path in (seed_path, extraction_path):
        if not path.exists():
            raise SystemExit(f"输入不存在: {path}（先跑 scripts/reextract_figures.py）")
    image_root = Path(args.image_root).resolve()

    seed_rows = load_jsonl(seed_path)
    val_papers = sorted({row["paper_id"] for row in seed_rows if row.get("split") == "val"})
    val_paper_set = set(val_papers)

    # seed 图去重指纹（近重复新图不得入池）
    seed_hashes = set()
    for row in seed_rows:
        p = (image_root / row["image_relpath"]).resolve()
        if p.exists():
            seed_hashes.add(sha256_head(p))

    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    out_root = Path(extraction.get("out_root", "AgenticArxiv/output/pdf_figures_v2"))

    new_rows: List[Dict[str, Any]] = []
    stats = {
        "figures_v2_total": 0,
        "figures_v2_dup_of_seed": 0,
        "figures_v2_used": 0,
        "rows_skipped_no_target": 0,
    }
    counts: Dict[str, Dict[str, int]] = {"train": {}, "val2": {}}
    questions = list(args.seed_questions) + list(args.v2_questions)

    for paper_id, record in sorted(extraction.get("papers", {}).items()):
        split = "val2" if paper_id in val_paper_set else "train"
        for figure in record.get("figures", []):
            stats["figures_v2_total"] += 1
            fpath = Path(figure["path"])
            if not fpath.is_file() or fpath.stat().st_size == 0:
                continue
            digest = figure.get("sha256_head") or sha256_head(fpath)
            if digest in seed_hashes:
                stats["figures_v2_dup_of_seed"] += 1
                continue
            seed_hashes.add(digest)  # v2 内部也去重
            caption = clean_text(figure.get("caption"))
            if not caption:
                continue
            try:
                rel = fpath.resolve().relative_to(image_root)
            except ValueError as exc:
                raise ValueError(f"figure path outside image root: {fpath}") from exc
            used = False
            for question in questions:
                target = build_v2_target(caption, question)
                if not target:
                    stats["rows_skipped_no_target"] += 1
                    continue
                new_rows.append(
                    {
                        "paper_id": paper_id,
                        "figure_no": int(figure.get("figure_no") or 0),
                        "question": question,
                        "prompt": (
                            _V2_QUESTION_PROMPTS[question]
                            if question in _V2_QUESTION_PROMPTS
                            else None
                        )
                        or _legacy_prompt(question),
                        "target": target,
                        "image_relpath": rel.as_posix(),
                        "split": split,
                        "source": "v2",
                    }
                )
                counts[split][question] = counts[split].get(question, 0) + 1
                used = True
            if used:
                stats["figures_v2_used"] += 1

    all_rows = list(seed_rows) + new_rows
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, all_rows)

    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    train_rows = sum(1 for row in all_rows if row.get("split") == "train")
    manifest = {
        "version": 2,
        "kind": "figure_qa_v2",
        "seed_data": seed_path.as_posix(),
        "seed_sha256": sha256_file(seed_path),
        "extraction": extraction_path.as_posix(),
        "extraction_sha256": sha256_file(extraction_path),
        "output": output_path.as_posix(),
        "output_sha256": sha256_file(output_path),
        "discipline": {
            "seed_rows_verbatim": True,
            "seed_val_rows": sum(1 for row in seed_rows if row.get("split") == "val"),
            "val_papers_frozen": val_papers,
            "val2_never_trained": True,
        },
        "v2_question_prompts": _V2_QUESTION_PROMPTS,
        "figure_type_keywords": {label: list(kws) for label, kws in _FIGURE_TYPE_KEYWORDS},
        "summary": {
            "rows_total": len(all_rows),
            "rows_train": train_rows,
            "rows_val_frozen": sum(1 for row in all_rows if row.get("split") == "val"),
            "rows_val2": sum(1 for row in all_rows if row.get("split") == "val2"),
            "counts": counts,
            **stats,
        },
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("FIGURE_QA_V2_DATASET_OK")
    print(f"rows total:        {len(all_rows)} (train {train_rows})")
    print(f"val frozen:        {manifest['summary']['rows_val_frozen']} rows / {len(val_papers)} papers")
    print(f"val2 (no-train):   {manifest['summary']['rows_val2']}")
    print(f"v2 figures:        used {stats['figures_v2_used']} / {stats['figures_v2_total']} (dup-of-seed {stats['figures_v2_dup_of_seed']})")
    print(f"counts:            {counts}")
    print(f"output:            {output_path}")


def _legacy_prompt(question: str) -> str:
    """旧三型的 prompt 单一事实来源仍在 env 工具；这里仅为审计一致而引用。"""
    from tools.figure_analysis_tool import _QUESTION_PROMPTS

    return _QUESTION_PROMPTS[question]


if __name__ == "__main__":
    main()
