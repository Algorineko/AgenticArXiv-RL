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
        r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+)(?:[.)]\s*|\s+)",
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


#: 有些模板把标签和正文连在一起（"Abstract—We propose…"），这种行靠
#: `_heading_name` 认不出来，需要在恢复出的段落开头把它剥掉。
_INLINE_ABSTRACT_RE = re.compile(r"^\s*abstract\s*[:—–\-]+\s*", re.IGNORECASE)

#: 一段文字至少要有这么多词才可能是一个摘要（而不是标题、作者或单位）。
_MIN_ABSTRACT_WORDS = 25


def _abstract_without_heading(lines: list) -> Optional[str]:
    """Recover the abstract when the PDF typesets it without a heading line.

    A large share of arXiv papers use a template (CVPR/ICCV and friends) that
    renders the abstract with no visible "Abstract" label, so the heading
    search finds nothing and every reading task fails on an otherwise perfectly
    readable paper.

    The abstract is then, reliably, the **longest contiguous prose block before
    the first section heading**: title, author list and affiliations come
    earlier and are short by construction, while the abstract is a paragraph of
    a couple hundred words. Blocks are joined with spaces because the
    hard-wrapped lines are one paragraph, not separate ones.
    """
    cut = len(lines)

    for i, line in enumerate(lines):
        if _heading_name(line) is not None:
            cut = i
            break

    blocks: list = []
    current: list = []

    for line in lines[:cut]:
        if line.strip():
            current.append(line.strip())
        elif current:
            blocks.append(current)
            current = []

    if current:
        blocks.append(current)

    best: Optional[list] = None
    best_words = 0

    for block in blocks:
        words = len(" ".join(block).split())
        if words > best_words:
            best, best_words = block, words

    if best is None or best_words < _MIN_ABSTRACT_WORDS:
        return None

    joined = _normalize_text(" ".join(best))

    # The same templates often run the label into the text ("Abstract—Broad
    # public adoption…" / "Abstract: Behavior cloning…") so the heading search
    # misses it. Drop the label when it leads the recovered block; otherwise it
    # is just noise in front of the abstract.
    return _INLINE_ABSTRACT_RE.sub("", joined, count=1).strip()


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
        # Only the abstract gets this fallback. Method/result/conclusion really
        # can be absent from a given paper, and a task that explicitly asks for
        # one must keep getting a deterministic "not found" error rather than a
        # guess. An arXiv paper, by contrast, always has an abstract — a miss
        # here is a typesetting quirk, not a missing section.
        if wanted == "abstract":
            recovered = _abstract_without_heading(lines)
            if recovered:
                return recovered

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
