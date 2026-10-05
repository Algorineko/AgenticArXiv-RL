"""Deterministic reading of a translated paper, one page at a time.

``translate_arxiv_pdf`` only produces a file (pdf2zh's Chinese "mono" PDF), so
the translation never reached the agent context.  This tool closes that gap the
same way T2 does for the original paper: it is a plain reader with no
translation and no LLM inside, so the same file always yields the same
observation and an offline snapshot can replay it exactly.

The translation is addressed by **page**, not by section.  pdf2zh keeps the
original layout, so page N of the translation is page N of the paper, and page
1 normally holds the translated title and abstract.  Section headings, by
contrast, are translated text: the same "Introduction" comes back as 「引言」 or
「介绍」 depending on the service, and the English heading rules of
``get_paper_content`` silently pick the wrong block on Chinese text.  A page
number is the one address that survives translation unchanged.

Translation quality is not graded anywhere: like T3 summaries and T5 answers,
the text is an observation, not a target.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple, Union

from models.store import store
from tools.paper_content_tool import _normalize_text
from tools.tool_registry import registry


def validate_page(page: Any) -> int:
    """1-based page number; ``None`` means the first page."""
    if page is None:
        return 1
    if isinstance(page, bool):
        raise ValueError(f"page must be a positive integer; got {page!r}")
    try:
        value = int(page)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"page must be a positive integer; got {page!r}") from exc
    if value < 1:
        raise ValueError(f"page must be a positive integer; got {page!r}")
    return value


def _read_pdf_page(path: str, page: int) -> Tuple[str, int]:
    """Return ``(normalized text of one page, total page count)``."""
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Reading a translated paper requires PyMuPDF; "
            "install it with `pip install PyMuPDF`."
        ) from exc

    try:
        with pymupdf.open(path) as doc:
            total = doc.page_count
            raw = doc[page - 1].get_text("text", sort=True) if page <= total else None
    except Exception as exc:
        raise ValueError(f"Failed to parse PDF: {exc}") from exc

    if raw is None:
        raise ValueError(f"page must be between 1 and {total}; got {page}")

    text = _normalize_text(raw)
    if not text:
        raise ValueError(
            f"No text could be extracted from page {page}; it may be image-only."
        )
    return text, total


def get_translated_content(
    session_id: str = "default",
    ref: Union[str, int, None] = 1,
    page: int = 1,
) -> Dict[str, Any]:
    """Read one page of a paper's translated (Chinese mono) PDF.

    The paper must be in the session's latest search results and its
    translation must be READY, i.e. ``translate_arxiv_pdf`` has finished.
    ``ref=None`` points to the most recently operated paper, matching the
    other paper tools.
    """
    index = validate_page(page)

    paper = store.resolve_paper(session_id, ref)
    if paper is None:
        raise ValueError("Paper not found; search for the paper and check the ref.")

    asset = store.get_translate_asset(paper.id)
    if asset is None or asset.status != "READY" or not asset.output_mono_path:
        raise ValueError(
            "Translated PDF is not ready; call translate_arxiv_pdf first."
        )
    # The cache index can outlive the file (cleaned output dir, sandbox reset).
    # translate_arxiv_pdf re-translates when the file is gone, so say that
    # instead of surfacing a PDF parser error.
    if not os.path.isfile(asset.output_mono_path):
        raise ValueError(
            "Translated PDF file is missing; call translate_arxiv_pdf again to regenerate it."
        )

    content, total_pages = _read_pdf_page(asset.output_mono_path, index)

    store.set_last_active_paper_id(session_id, paper.id)

    # Short fields come before the page text so that a truncated observation
    # still shows which paper and page it is, and how many pages there are.
    return {
        "paper_id": paper.id,
        "title": paper.title,
        "page": index,
        "total_pages": total_pages,
        "content": content,
    }


TRANSLATED_CONTENT_TOOL_SCHEMA = {
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
        "page": {
            "description": (
                "1-based page of the translated PDF; page 1 usually holds "
                "the translated title and abstract."
            ),
            "type": "integer",
            "default": 1,
        },
    },
    "required": [],
}


registry.register_tool(
    name="get_translated_content",
    description=(
        "Read one page of a paper's Chinese translation (the PDF produced by "
        "translate_arxiv_pdf). The translation must be finished first. "
        "Returns the page text and the total page count."
    ),
    parameter_schema=TRANSLATED_CONTENT_TOOL_SCHEMA,
    func=get_translated_content,
)
