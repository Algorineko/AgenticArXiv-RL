#!/usr/bin/env python3
"""矢量图论文的图池扩展：定位 caption 文本块 → 渲染其上方图区。

背景：``tools/paper_figures_tool``（T4）只能抽内嵌位图，arXiv 的
matplotlib/矢量论文在它眼里 ``count:0``。本地 104 篇缓存 PDF 中 63 篇
属于这一类。这里用规则法补齐：

1. 每页找 ``Figure N:`` / ``Fig. N`` 开头的文本块（caption）；
2. 图区 = caption 上方的连续空白带（同栏），上界是再往上的最近文本块底边；
3. ``page.get_pixmap(clip=...)`` 渲染成 jpg，写入独立根目录
   ``output/pdf_figures_v2/<paper_id>/vec_fig_N.jpg``（与 seed/v2 位图零碰撞）；
4. 保守过滤：区高 ≥100pt、渲染后宽高 ≥150px、长宽比 0.2-5、
   墨水占比 1.5%-85%（防空图/纯文字）。

监督 caption 即定位到的文本块（去前缀在 builder 里做）。产物 sidecar
JSON 与 ``reextract_figures.py`` 同 schema，由
``build_figure_qa_v2_dataset.py`` 合并去重。

用法::

    python scripts/reextract_vector_figures.py [--sample 2608.14528v1]
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

CAPTION_RE = re.compile(r"^\(?(?:figure|fig\.?)\s*\d+\s*[:.\-]", re.IGNORECASE)
_INK_MIN, _INK_MAX = 0.015, 0.85


def sha256_head(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
    return digest.hexdigest()


def ink_ratio(jpeg_bytes: bytes) -> float:
    from PIL import Image

    image = Image.open(io.BytesIO(jpeg_bytes)).convert("L")
    pixels = list(image.getdata())
    if not pixels:
        return 0.0
    dark = sum(1 for v in pixels if v < 200)
    return dark / len(pixels)


def caption_blocks(page) -> List[Tuple[float, float, float, float, str]]:
    out = []
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        stripped = (text or "").strip()
        if CAPTION_RE.match(stripped):
            out.append((float(x0), float(y0), float(x1), float(y1), stripped))
    return out


def figure_rect(page, cap: Tuple[float, float, float, float]):
    """caption 上方的同栏图区；无效时返回 None。"""
    _, cy0, cx0, cx1 = cap[:4]
    page_w = page.rect.width
    cap_w = cx1 - cx0
    mid = page_w / 2
    if cap_w > page_w * 0.6:
        rx0, rx1 = 28.0, page_w - 28.0
    elif (cx0 + cx1) / 2 < mid:
        rx0, rx1 = 28.0, mid + 18.0
    else:
        rx0, rx1 = mid - 18.0, page_w - 28.0

    y_top = 40.0
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        if y1 <= cy0 - 2 and x0 < rx1 and x1 > rx0 and (text or "").strip():
            y_top = max(y_top, y1 + 4)
    height = cy0 - y_top - 2
    if height < 100:
        return None
    return (rx0, y_top, rx1, cy0 - 2)


def extract_paper(pdf_path: Path, out_dir: Path, dpi: int = 150) -> List[Dict[str, Any]]:
    import pymupdf

    figures: List[Dict[str, Any]] = []
    with pymupdf.open(str(pdf_path)) as document:
        for page_no, page in enumerate(document, start=1):
            for cap in caption_blocks(page):
                rect = figure_rect(page, cap)
                if rect is None:
                    continue
                pix = page.get_pixmap(dpi=dpi, clip=rect)
                if pix.width < 150 or pix.height < 150:
                    continue
                ratio = pix.width / max(1, pix.height)
                if not (0.2 <= ratio <= 5.0):
                    continue
                jpeg = pix.tobytes("jpeg")
                ink = ink_ratio(jpeg)
                if not (_INK_MIN <= ink <= _INK_MAX):
                    continue
                figure_no = len(figures) + 1
                path = out_dir / f"vec_fig_{figure_no}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(jpeg)
                try:
                    recorded = path.resolve().relative_to(REPO_ROOT).as_posix()
                except ValueError:
                    recorded = str(path)
                figures.append(
                    {
                        "figure_no": figure_no,
                        "page": page_no,
                        "path": recorded,
                        "width": pix.width,
                        "height": pix.height,
                        "caption": cap[4],
                        "sha256_head": sha256_head(path),
                        "method": "vector-render",
                    }
                )
    return figures


def main() -> None:
    parser = argparse.ArgumentParser(description="矢量图论文的图区渲染扩展")
    parser.add_argument("--pdf-dir", default="AgenticArxiv/output/pdf_raw")
    parser.add_argument("--output", default="data/vlm/figureqa_v2_extraction_vec.json")
    parser.add_argument("--out-root", default="AgenticArxiv/output/pdf_figures_v2")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--sample", default="", help="只处理指定 paper_id（目检用）")
    parser.add_argument("--exclude-with-raster", action="store_true", default=True,
                        help="位图抽取已有图的论文默认跳过（图池去重交给 builder）")
    args = parser.parse_args()

    import pymupdf

    pdf_dir = Path(args.pdf_dir)
    out_root = Path(args.out_root).resolve()
    raster_sidecar = Path("data/vlm/figureqa_v2_extraction.json")
    raster_papers: set = set()
    if raster_sidecar.exists() and args.exclude_with_raster:
        raster_papers = set(json.loads(raster_sidecar.read_text(encoding="utf-8")).get("papers", {}))

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if args.sample:
        pdfs = [pdf_dir / f"{args.sample}.pdf"]
    papers: Dict[str, Any] = {}
    for pdf in pdfs:
        paper_id = pdf.stem
        if paper_id in raster_papers:
            continue
        figures = extract_paper(pdf, out_root / paper_id, dpi=args.dpi)
        if figures:
            papers[paper_id] = {"figures": figures}

    payload = {
        "kind": "figureqa_v2_extraction_vec",
        "method": "caption-anchored vector region render",
        "dpi": args.dpi,
        "pdf_dir": pdf_dir.as_posix(),
        "out_root": out_root.as_posix(),
        "papers": papers,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    n_figs = sum(len(p["figures"]) for p in papers.values())
    print("VECTOR_FIGURE_REEXTRACT_OK")
    print(f"papers with vector figures: {len(papers)}")
    print(f"figures rendered:           {n_figs}")
    print(f"output:                     {output_path}")


if __name__ == "__main__":
    main()
