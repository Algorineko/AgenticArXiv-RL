#!/usr/bin/env python3
"""figureqa v2 的图池扩展：用放宽的阈值从本地缓存 PDF 重抽论文图。

与 T4（``tools.paper_figures_tool``）共用同一套抽取核心
（``_figure_candidates`` / ``_caption_for``），但有两点不同：

- 阈值放宽：``MAX_FIGURES`` 8→16、``MIN_FIGURE_PIXELS`` 96→64，扩出更多可用图；
- **只增不改**：全部写入独立根目录 ``output/pdf_figures_v2/<paper_id>/``，
  与 seed 冻结的 ``output/pdf_figures/`` 零交集，绝不覆盖既有图片
  （94 行冻结 val 引用的文件路径与字节都不动）。

输入是 ``output/pdf_raw/`` 的缓存 PDF（paper_id 即文件名去 ``.pdf``），
不联网。输出 sidecar JSON（``--output``，默认
``data/vlm/figureqa_v2_extraction.json``），由
``scripts/build_figure_qa_v2_dataset.py`` 合并进数据集。

用法::

    python scripts/reextract_figures.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "AgenticArxiv"
sys.path.insert(0, str(PACKAGE_ROOT))

from tools import paper_figures_tool as t4  # noqa: E402

SHA256_SAMPLE_BYTES = 1024 * 1024


def sha256_head(path: Path) -> str:
    """图片字节头部哈希：近重复去重用（全文哈希更严，头部已足够稳）。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(SHA256_SAMPLE_BYTES))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="figureqa v2 图池扩展（本地 PDF 重抽）")
    parser.add_argument("--pdf-dir", default="AgenticArxiv/output/pdf_raw")
    parser.add_argument("--output", default="data/vlm/figureqa_v2_extraction.json")
    parser.add_argument("--out-root", default="AgenticArxiv/output/pdf_figures_v2")
    parser.add_argument("--max-figures", type=int, default=16)
    parser.add_argument("--min-pixels", type=int, default=64)
    args = parser.parse_args()

    import pymupdf

    pdf_dir = Path(args.pdf_dir)
    out_root = Path(args.out_root).resolve()
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"没有缓存 PDF: {pdf_dir}")

    # 放宽阈值：_figure_candidates 在调用时读模块全局，改属性即生效。
    t4.MAX_FIGURES = args.max_figures
    t4.MIN_FIGURE_PIXELS = args.min_pixels

    papers: Dict[str, Any] = {}
    for pdf in pdfs:
        paper_id = pdf.stem
        out_dir = out_root / paper_id
        figures: List[Dict[str, Any]] = []
        try:
            with pymupdf.open(str(pdf)) as document:
                for figure_no, page_no, bbox, _xref, extension, width, height in t4._figure_candidates(document):
                    payload = document.extract_image(_xref)
                    path = out_dir / f"fig_{figure_no}.{extension}"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(payload["image"])
                    try:
                        recorded = path.resolve().relative_to(REPO_ROOT).as_posix()
                    except ValueError:
                        recorded = str(path)
                    figures.append(
                        {
                            "figure_no": figure_no,
                            "page": page_no,
                            "path": recorded,
                            "width": width,
                            "height": height,
                            "caption": t4._caption_for(document[page_no - 1], bbox),
                            "sha256_head": sha256_head(path),
                        }
                    )
        except Exception as exc:  # noqa: BLE001 - 单篇损坏不拖垮整批
            print(f"⚠️ 跳过损坏 PDF {paper_id}: {exc}")
            continue
        if figures:
            papers[paper_id] = {"figures": figures}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "figureqa_v2_extraction",
        "thresholds": {"max_figures": args.max_figures, "min_pixels": args.min_pixels},
        "pdf_dir": pdf_dir.as_posix(),
        "out_root": out_root.as_posix(),
        # 公开仓库纪律：只记录仓库相对路径（builder 以 image-root 为基准解析）
        "recorded_relative_to": ".",
        "papers": papers,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    n_figs = sum(len(p["figures"]) for p in papers.values())
    print("FIGURE_REEXTRACT_OK")
    print(f"papers with figures: {len(papers)} / {len(pdfs)} pdfs")
    print(f"figures extracted:   {n_figs}")
    print(f"output:              {output_path}")


if __name__ == "__main__":
    main()
