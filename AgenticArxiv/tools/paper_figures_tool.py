"""Deterministic figure extraction from a downloaded paper (README T4).

Like T2/T3 this is an *environment-side* capability: the policy decides whether
to extract, from which paper, and nothing else.  There is no VLM here — the tool
only pulls the embedded figure images and their nearby captions out of the PDF,
which keeps the observation identical on every replay.

The natural follow-up (T5 ``analyze_figure``) is deliberately **not** part of
this module: it needs a local VLM in the environment, and the README keeps that
behind an explicit multimodal switch so the text-only action space stays small.

A paper with no embedded raster figures is reported as ``count: 0`` rather than
raised as an error.  That is a fact about the paper, not a failed action, and
recording it in the snapshot is what lets offline replay answer identically to
the live run.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from config import settings
from models.store import store
from tools.tool_registry import registry


#: Vector-only papers can carry dozens of incidental glyph images; the first
#: few real figures are what a reader (or a follow-up VLM call) needs.
MAX_FIGURES = 8

#: Icons, logos and rules are not figures.  Filtering them out is what makes
#: ``count`` a meaningful signal instead of "how many XObjects the file has".
MIN_FIGURE_PIXELS = 96

_CAPTION_RE = re.compile(r"^\s*(?:figure|fig\.?)\s*(\d+)", re.IGNORECASE)

_EXTENSION_BY_NAME = {
    "png": "png",
    "jpeg": "jpg",
    "jpg": "jpg",
    "gif": "gif",
    "bmp": "bmp",
    "tiff": "tiff",
    "jpx": "jp2",
}


def figures_output_dir(paper_id: str) -> Path:
    """Per-paper directory for extracted figures (inside the sandboxed root)."""
    safe_id = str(paper_id).replace("/", "_")
    return Path(settings.figures_path) / safe_id


def _caption_for(page, bbox) -> Optional[str]:
    """Return the caption block belonging to a figure on *page*.

    Captions live below (occasionally above) the image.  Scanning the text
    blocks in reading order and taking the closest one that starts with
    "Figure N" / "Fig. N" is deterministic and good enough for the reward,
    which never grades caption quality.
    """
    try:
        blocks = page.get_text("blocks", sort=True)
    except Exception:
        return None

    left, top, right, bottom = bbox
    best = None
    best_distance = None
    for block in blocks:
        x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
        if not isinstance(text, str) or not _CAPTION_RE.match(text):
            continue
        # Horizontal overlap is what ties a caption to a figure in a
        # two-column layout; a caption in the other column is not ours.
        if x1 < left or x0 > right:
            continue
        distance = y0 - bottom if y0 >= bottom else top - y1
        if distance < 0:
            continue
        if best_distance is None or distance < best_distance:
            best, best_distance = text.strip(), distance
    return best


def _figure_candidates(document):
    """Yield ``(figure_no, page_no, bbox, xref, extension, width, height)``."""
    seen_xrefs = set()
    figure_no = 0
    for page_index, page in enumerate(document, start=1):
        try:
            placements = page.get_image_info(xrefs=True)
        except Exception:
            placements = []
        for placement in placements:
            xref = placement.get("xref")
            if not xref or xref in seen_xrefs:
                continue
            width = int(placement.get("width") or 0)
            height = int(placement.get("height") or 0)
            if min(width, height) < MIN_FIGURE_PIXELS:
                continue
            try:
                payload = document.extract_image(xref)
            except Exception:
                continue
            extension = _EXTENSION_BY_NAME.get(str(payload.get("ext", "")).lower())
            if extension is None or not payload.get("image"):
                continue
            seen_xrefs.add(xref)
            figure_no += 1
            yield (
                figure_no,
                page_index,
                tuple(placement.get("bbox") or (0, 0, 0, 0)),
                xref,
                extension,
                width,
                height,
            )
            if figure_no >= MAX_FIGURES:
                return


def extract_paper_figures(
    session_id: str = "default",
    ref: Union[str, int, None] = 1,
) -> Dict[str, Any]:
    """Extract embedded figures (image files + captions) from a paper PDF.

    The paper must be in the session's latest search results and its PDF must
    have been downloaded, exactly like ``get_paper_content``.
    """
    paper = store.resolve_paper(session_id, ref)
    if paper is None:
        raise ValueError("Paper not found; search for the paper and check the ref.")

    asset = store.get_pdf_asset(paper.id)
    if asset is None or asset.status != "READY" or not asset.local_path:
        raise ValueError(
            "Paper PDF is not downloaded; call download_arxiv_pdf first."
        )

    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Extracting paper figures requires PyMuPDF; "
            "install it with `pip install PyMuPDF`."
        ) from exc

    out_dir = figures_output_dir(paper.id)
    figures: List[Dict[str, Any]] = []

    try:
        with pymupdf.open(asset.local_path) as document:
            candidates = list(_figure_candidates(document))
            if candidates:
                out_dir.mkdir(parents=True, exist_ok=True)
            for figure_no, page_no, bbox, xref, extension, width, height in candidates:
                payload = document.extract_image(xref)
                path = out_dir / f"fig_{figure_no}.{extension}"
                path.write_bytes(payload["image"])
                figures.append(
                    {
                        "figure_no": figure_no,
                        "page": page_no,
                        "path": str(path),
                        "width": width,
                        "height": height,
                        "caption": _caption_for(document[page_no - 1], bbox),
                    }
                )
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"Failed to parse PDF: {exc}") from exc

    store.set_last_active_paper_id(session_id, paper.id)

    return {
        "paper_id": paper.id,
        "title": paper.title,
        "count": len(figures),
        "output_dir": str(out_dir),
        "figures": figures,
    }


PAPER_FIGURES_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "session_id": {
            "type": "string",
            "default": "default",
        },
        "ref": {
            "description": (
                "Paper reference: 1-based result index, arXiv id, or "
                "title fragment; null selects the most recently active paper."
            ),
            "anyOf": [
                {"type": "integer"},
                {"type": "string"},
                {"type": "null"},
            ],
        },
    },
    "required": [],
}


registry.register_tool(
    name="extract_paper_figures",
    description=(
        "Extract the embedded figures of a downloaded arXiv paper as image "
        "files, with page numbers and captions. The paper PDF must be "
        "downloaded first."
    ),
    parameter_schema=PAPER_FIGURES_TOOL_SCHEMA,
    func=extract_paper_figures,
)
