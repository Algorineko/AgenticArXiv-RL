"""Deterministic PDF-to-text paper reading tool.

No LLM is used here: text and section boundaries are derived solely from the
cached PDF so the same file always yields the same observation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Union

from models.store import store
from tools.tool_registry import registry


_SECTION_ALIASES = {
    "abstract": ("abstract",),
    "method": (
        "method",
        "methods",
        "methodology",
        "approach",
        "proposed method",
    ),
    "result": (
        "result",
        "results",
        "experiments",
        "experimental results",
        "evaluation",
    ),
    "conclusion": (
        "conclusion",
        "conclusions",
        "concluding remarks",
        "discussion and conclusion",
    ),
}


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _heading_name(line: str) -> Optional[str]:
    """Return a canonical section name when *line* looks like a section heading."""
    cleaned = re.sub(
        r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+)[.)]?\s*",
        "",
        line,
        flags=re.I,
    )
    cleaned = re.sub(r"[:.\s]+$", "", cleaned).strip().lower()

    if not cleaned or len(cleaned) > 80:
        return None

    for canonical, aliases in _SECTION_ALIASES.items():
        if cleaned in aliases:
            return canonical

    other_headings = {
        "introduction",
        "related work",
        "background",
        "preliminaries",
        "discussion",
        "limitations",
        "references",
        "acknowledgements",
        "acknowledgments",
        "appendix",
        "experiments and analysis",
    }

    if cleaned in other_headings:
        return "_other"

    return None


def _extract_section(text: str, section: str) -> str:
    wanted = section.lower().strip()

    if wanted not in _SECTION_ALIASES:
        raise ValueError(
            "section must be one of abstract/method/result/conclusion; "
            f"got {section!r}"
        )

    lines = text.splitlines()
    start = None

    for i, line in enumerate(lines):
        if _heading_name(line) == wanted:
            start = i + 1
            break

    if start is None:
        raise ValueError(f"section {wanted!r} was not found in the paper")

    end = len(lines)

    for i in range(start, len(lines)):
        if _heading_name(lines[i]) is not None:
            end = i
            break

    content = _normalize_text("\n".join(lines[start:end]))

    if not content:
        raise ValueError(f"section {wanted!r} is empty")

    return content


def _pdf_to_text(path: str) -> str:
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Reading paper content requires PyMuPDF; "
            "install it with `pip install PyMuPDF`."
        ) from exc

    try:
        with pymupdf.open(path) as doc:
            text = "\n".join(
                page.get_text("text", sort=True)
                for page in doc
            )
    except Exception as exc:
        raise ValueError(f"Failed to parse PDF: {exc}") from exc

    text = _normalize_text(text)

    if not text:
        raise ValueError(
            "No text could be extracted from the PDF; "
            "it may be image-only."
        )

    return text


def get_paper_content(
    session_id: str = "default",
    ref: Union[str, int, None] = 1,
    section: Optional[str] = None,
) -> Dict[str, Any]:
    """Read cached paper text, optionally restricting output to one section.

    The paper must first be present in the session's latest search results and
    its PDF must have been downloaded. ``ref=None`` points to the most recently
    operated paper, matching the other paper tools.
    """
    paper = store.resolve_paper(session_id, ref)

    if paper is None:
        raise ValueError(
            "Paper not found; search for the paper and check the ref."
        )

    asset = store.get_pdf_asset(paper.id)

    if asset is None or asset.status != "READY" or not asset.local_path:
        raise ValueError(
            "Paper PDF is not downloaded; call download_arxiv_pdf first."
        )

    text = _pdf_to_text(asset.local_path)

    canonical_section = (
        section.lower().strip()
        if isinstance(section, str)
        else None
    )

    # README T2 contract: the default observation is intentionally compact:
    # title + abstract. Full-paper text is parsed deterministically but is not
    # injected into the agent context unless a supported section is requested.
    if canonical_section:
        content = _extract_section(text, canonical_section)
    else:
        abstract = _extract_section(text, "abstract")
        content = f"{paper.title}\n\nAbstract\n{abstract}"

    store.set_last_active_paper_id(session_id, paper.id)

    return {
        "paper_id": paper.id,
        "title": paper.title,
        "section": canonical_section,
        "content": content,
    }


PAPER_CONTENT_TOOL_SCHEMA = {
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
        "section": {
            "description": (
                "Optional section. When omitted, returns title + abstract "
                "according to the README T2 contract."
            ),
            "anyOf": [
                {
                    "type": "string",
                    "enum": [
                        "abstract",
                        "method",
                        "result",
                        "conclusion",
                    ],
                },
                {"type": "null"},
            ],
            "default": None,
        },
    },
    "required": [],
}


registry.register_tool(
    name="get_paper_content",
    description=(
        "Read deterministic text from a downloaded arXiv paper. "
        "Returns title + abstract by default, or a requested "
        "abstract/method/result/conclusion section."
    ),
    parameter_schema=PAPER_CONTENT_TOOL_SCHEMA,
    func=get_paper_content,
)
