"""Paper summarisation as an *environment-side* tool (README T3).

Design contract (see README「工具集演进设计」):

* The policy only decides **whether** to summarise, **which** paper, and with
  which ``style`` / ``max_words``.  Every one of those is rule-checkable, so it
  fits the existing five-component verifiable reward untouched.
* The quality of the produced summary is deliberately **not** part of the
  reward.  Scoring it would need an LLM judge and open a new reward-hacking
  surface, so the summary text is treated as an observation, not a target.
* Summarisation is therefore deterministic by default (``extractive``): text is
  selected from the same PDF→text extraction T2 uses, with no sampling.  The
  same paper, style and budget always yield the same observation, which is what
  makes offline snapshot replay sound.

An optional ``local_model`` backend plugs in a local HF seq2seq/causal model
for richer text.  It is opt-in because it pulls model weights into the
environment; the reward does not look at the text either way.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Union

from models.store import store
from tools.paper_content_tool import (
    _extract_section,
    _pdf_to_text,
)
from tools.tool_registry import registry


# Free-form budgets would make the offline snapshot key space unbounded; the
# requested value is clamped to the smallest bucket that fits it.  Tasks declare
# a bucket value so the argument oracle stays exact.
MAX_WORDS_BUCKETS = (60, 120, 250)
DEFAULT_MAX_WORDS = 120

SUMMARY_STYLES = ("tldr", "structured", "bullet")

# Sections the ``structured`` style tries to fill, in presentation order.
_STRUCTURED_SECTIONS = ("abstract", "method", "result", "conclusion")

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_URL_RE = re.compile(r"https?://|www\.|arxiv:\s*\d", re.I)
_ARXIV_STAMP_RE = re.compile(r"^\s*arxiv:\d", re.I)


def normalize_style(style: Any) -> str:
    """Return the canonical style name, raising for anything unsupported."""
    if style is None:
        return "tldr"
    canonical = str(style).strip().lower()
    if canonical not in SUMMARY_STYLES:
        raise ValueError(
            "style must be one of " + "/".join(SUMMARY_STYLES) + f"; got {style!r}"
        )
    return canonical


def bucket_max_words(max_words: Any) -> int:
    """Clamp a requested word budget onto the supported bucket grid.

    Non-numeric input is a tool error rather than a silent default: the policy
    should be told its argument was wrong, not handed a plausible result.
    """
    if max_words is None:
        return DEFAULT_MAX_WORDS
    if isinstance(max_words, bool):
        raise ValueError(f"max_words must be a positive integer; got {max_words!r}")
    try:
        requested = int(max_words)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"max_words must be a positive integer; got {max_words!r}"
        ) from exc
    if requested <= 0:
        raise ValueError(f"max_words must be a positive integer; got {max_words!r}")
    for bucket in MAX_WORDS_BUCKETS:
        if requested <= bucket:
            return bucket
    return MAX_WORDS_BUCKETS[-1]


def _sentences(text: str) -> List[str]:
    """Split *text* into prose sentences, dropping reference-style noise."""
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []

    out: List[str] = []
    for raw in _SENTENCE_RE.split(cleaned):
        sentence = raw.strip()
        if not sentence:
            continue
        # Bibliographies and URL dumps are common in extracted PDF text and
        # add nothing to a summary; excluding them keeps the output stable
        # regardless of how much of the tail a paper happens to contain.
        if _URL_RE.search(sentence) or _ARXIV_STAMP_RE.match(sentence):
            continue
        if len(sentence.split()) < 5:
            continue
        out.append(sentence)
    return out


def _word_count(text: str) -> int:
    return len(text.split())


def _clip_words(text: str, limit: int) -> str:
    """Trim *text* to at most *limit* words, adding an ellipsis when cut."""
    words = text.split()
    if len(words) <= limit:
        return text.strip()
    return " ".join(words[:limit]).rstrip(",;:") + " …"


def _fit_sentences(sentences: Sequence[str], budget: int) -> str:
    """Greedily take whole sentences while they fit inside *budget* words."""
    chosen: List[str] = []
    used = 0
    for sentence in sentences:
        cost = _word_count(sentence)
        if chosen and used + cost > budget:
            break
        if not chosen and cost > budget:
            return _clip_words(sentence, budget)
        chosen.append(sentence)
        used += cost
    if not chosen:
        return ""
    text = " ".join(chosen)
    # A single oversized first sentence is already clipped above; otherwise the
    # loop stops on a whole-sentence boundary and stays under budget.
    return text if _word_count(text) <= budget else _clip_words(text, budget)


def _tldr(title: str, text: str, max_words: int) -> tuple[str, List[str]]:
    abstract = _extract_section(text, "abstract")
    # The figure/table caption block that often follows the abstract is not
    # prose; keeping the first sentences only is the stable, useful choice.
    summary = _fit_sentences(_sentences(abstract), max_words)
    if not summary:
        raise ValueError("abstract is empty")
    return summary, ["abstract"]


def _structured(title: str, text: str, max_words: int) -> tuple[str, List[str]]:
    available: List[tuple[str, str]] = []
    for section in _STRUCTURED_SECTIONS:
        try:
            available.append((section, _extract_section(text, section)))
        except ValueError:
            # Heterogeneous headings are expected; a missing section simply
            # does not contribute instead of failing the whole call.
            continue
    if not available:
        raise ValueError(
            "no summarisable section (abstract/method/result/conclusion) found"
        )

    # Share the budget proportionally so a section with less text does not
    # silently consume the whole allowance.
    shares = _allocate_budget(
        max_words, [_word_count(body) for _, body in available]
    )
    lines: List[str] = []
    used: List[str] = []
    for (section, body), share in zip(available, shares):
        chunk = _fit_sentences(_sentences(body), share)
        if not chunk:
            continue
        lines.append(f"[{section}] {chunk}")
        used.append(section)
    if not lines:
        raise ValueError("no summarisable section had usable text")
    return "\n".join(lines), used


def _allocate_budget(max_words: int, sizes: Sequence[int]) -> List[int]:
    """Split *max_words* across sections proportionally to their length."""
    total = sum(sizes) or 1
    floor = 12
    shares = [max(floor, int(max_words * size / total)) for size in sizes]

    # Trim from the largest share until the total fits again so the sum stays a
    # hard budget rather than a suggestion.
    while sum(shares) > max_words and max(shares) > floor:
        shares[shares.index(max(shares))] -= 1
    return shares


def _bullet(title: str, text: str, max_words: int) -> tuple[str, List[str]]:
    picked: List[tuple[int, str, str]] = []
    for order, section in enumerate(_STRUCTURED_SECTIONS):
        try:
            body = _extract_section(text, section)
        except ValueError:
            continue
        for position, sentence in enumerate(_sentences(body)):
            # Rank by section order first (abstract → method → result →
            # conclusion) then by how early the sentence appears, so the
            # ranking is fully determined by the text.
            picked.append((order * 1000 + position, section, sentence))

    if not picked:
        raise ValueError("no bullet-able section text found")

    picked.sort(key=lambda item: item[0])
    lines: List[str] = []
    used_sections: List[str] = []
    budget = max_words
    for _, section, sentence in picked:
        line = f"- {sentence}"
        cost = _word_count(line)
        if lines and cost > budget:
            break
        if not lines and cost > budget:
            lines.append(f"- {_clip_words(sentence, max(1, budget - 1))}")
            used_sections.append(section)
            break
        lines.append(line)
        used_sections.append(section)
        budget -= cost
    if not lines:
        raise ValueError("no bullet-able section text found")
    return "\n".join(lines), list(dict.fromkeys(used_sections))


_EXTRACTIVE_STYLES = {
    "tldr": _tldr,
    "structured": _structured,
    "bullet": _bullet,
}


def _summarize_extractive(
    title: str, text: str, style: str, max_words: int
) -> tuple[str, List[str]]:
    return _EXTRACTIVE_STYLES[style](title, text, max_words)


# --------------------------------------------------------------------------
# Optional local-model backend
# --------------------------------------------------------------------------

_LOCAL_MODEL_CACHE: Dict[str, Any] = {}

_LOCAL_MODEL_PROMPTS = {
    "tldr": (
        "Summarise the following paper in at most {n} words, as one paragraph.\n\n"
    ),
    "structured": (
        "Summarise the following paper in at most {n} words. "
        "Cover background, method, results and conclusion.\n\n"
    ),
    "bullet": (
        "Summarise the following paper as at most {n}-word bullet points, "
        "one point per line.\n\n"
    ),
}


def _load_local_model(model_path: str):
    """Load (and cache) the summarisation model named by ``SUMMARY_MODEL_PATH``."""
    cached = _LOCAL_MODEL_CACHE.get(model_path)
    if cached is not None:
        return cached

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "SUMMARY_BACKEND=local_model requires transformers and torch."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    )
    model.eval()
    _LOCAL_MODEL_CACHE[model_path] = (tokenizer, model)
    return tokenizer, model


def _summarize_with_local_model(
    title: str, text: str, style: str, max_words: int
) -> tuple[str, List[str]]:
    import torch

    model_path = os.getenv("SUMMARY_MODEL_PATH", "").strip()
    if not model_path:
        raise RuntimeError(
            "SUMMARY_BACKEND=local_model requires SUMMARY_MODEL_PATH to point "
            "at a local Hugging Face model directory."
        )

    tokenizer, model = _load_local_model(model_path)
    # Greedy decoding only: a sampled summary would break snapshot replay.
    body = _fit_sentences(_sentences(text), 900)
    prompt = _LOCAL_MODEL_PROMPTS[style].format(n=max_words)
    messages = [{"role": "user", "content": f"{prompt}Title: {title}\n\n{body}"}]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=int(max_words * 2.2) + 32,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    completion = generated[0][inputs["input_ids"].shape[-1]:]
    summary = _clip_words(
        tokenizer.decode(completion, skip_special_tokens=True).strip(), max_words
    )
    if not summary:
        raise ValueError("local summarisation model produced empty output")
    return summary, ["full_text"]


def summary_backend() -> str:
    """Return the configured backend name, validated eagerly."""
    name = os.getenv("SUMMARY_BACKEND", "extractive").strip().lower() or "extractive"
    if name not in {"extractive", "local_model"}:
        raise RuntimeError(
            "SUMMARY_BACKEND must be 'extractive' or 'local_model'; "
            f"got {name!r}"
        )
    return name


def summarize_paper(
    session_id: str = "default",
    ref: Union[str, int, None] = 1,
    style: Optional[str] = None,
    max_words: Optional[int] = None,
) -> Dict[str, Any]:
    """Summarise a downloaded paper through the environment-side backend.

    The paper must already be in the session's latest search results and its
    PDF must have been downloaded (T2's precondition).  ``ref=None`` selects the
    most recently operated paper, matching the other paper tools.
    """
    canonical_style = normalize_style(style)
    budget = bucket_max_words(max_words)

    paper = store.resolve_paper(session_id, ref)
    if paper is None:
        raise ValueError("Paper not found; search for the paper and check the ref.")

    asset = store.get_pdf_asset(paper.id)
    if asset is None or asset.status != "READY" or not asset.local_path:
        raise ValueError(
            "Paper PDF is not downloaded; call download_arxiv_pdf first."
        )

    text = _pdf_to_text(asset.local_path)
    backend = summary_backend()
    if backend == "local_model":
        summary, sections = _summarize_with_local_model(
            paper.title, text, canonical_style, budget
        )
    else:
        summary, sections = _summarize_extractive(
            paper.title, text, canonical_style, budget
        )

    store.set_last_active_paper_id(session_id, paper.id)

    return {
        "paper_id": paper.id,
        "title": paper.title,
        "style": canonical_style,
        "max_words": budget,
        "summary": summary,
        "sections_used": sections,
        "backend": backend,
    }


PAPER_SUMMARY_TOOL_SCHEMA = {
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
        "style": {
            "description": "Summary shape to produce.",
            "type": "string",
            "enum": list(SUMMARY_STYLES),
            "default": "tldr",
        },
        "max_words": {
            "description": (
                "Approximate word budget; values are rounded up to the "
                "nearest supported budget of "
                + ", ".join(str(b) for b in MAX_WORDS_BUCKETS)
                + "."
            ),
            "type": "integer",
            "default": DEFAULT_MAX_WORDS,
        },
    },
    "required": [],
}


registry.register_tool(
    name="summarize_paper",
    description=(
        "Summarise a downloaded arXiv paper from its extracted text. "
        "Choose style (tldr/structured/bullet) and a max_words budget; "
        "the paper PDF must be downloaded first."
    ),
    parameter_schema=PAPER_SUMMARY_TOOL_SCHEMA,
    func=summarize_paper,
)
