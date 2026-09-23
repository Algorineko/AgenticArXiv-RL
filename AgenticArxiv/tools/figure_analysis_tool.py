"""Figure analysis as an *environment-side* tool (README T5).

This is the multimodal end of the interpretation loop, and it keeps the same
boundary as T3: the policy is still a text-only small model.  What it decides
is *whether* to analyse a figure, *which* figure, and *what to ask*; the
looking is outsourced to a VLM that lives entirely inside the environment and
never enters the policy's action space.

Two consequences of that boundary, both deliberate:

* The answer text is an **observation, not a target**.  Grading it would need a
  rubric or an LLM judge and would write a third-party model's noise into the
  policy gradient.  The reward only checks that the call was correct and that a
  real answer came back.
* Answers must be reproducible, so snapshots record them at build time exactly
  like T3's summaries.  ``question`` is therefore a small enum rather than free
  text: a free-form question makes the offline key space unbounded and every
  unseen phrasing would miss the snapshot.

Backends mirror ``paper_summary_tool``: ``extractive`` (default) is
deterministic and needs no weights — it works from the caption T4 already
extracted and says so when the caption does not carry the answer.  ``vlm``
loads a local vision-language model and is what the design actually calls for;
it is opt-in because it puts model weights on the environment side, and because
snapshot construction then needs a GPU.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from models.store import store
from tools.paper_figures_tool import MAX_FIGURES, extract_paper_figures
from tools.tool_registry import registry


#: Supported question kinds.  Small on purpose: the policy is a ~1.5B text
#: model, and every extra member multiplies the offline snapshot's key space.
FIGURE_QUESTIONS = ("describe", "axes", "trend")

DEFAULT_QUESTION = "describe"

#: The longest side an image is downscaled to before it reaches the VLM.
#: Measured on a 4404×2351 figure: feeding the original took ~27s per answer
#: while the answer itself needed only 64 tokens — the cost is almost entirely
#: pixels, so the resize is the difference between a usable and an unusable
#: environment step.
VLM_MAX_IMAGE_SIDE = 1024

#: Upstream knob shared with paper_figures_tool; kept explicit so an out-of-range
#: figure number fails as a tool error rather than as a snapshot miss.
MAX_FIGURE_NO = MAX_FIGURES

_AXES_HINTS = (
    "axis", "x-", "y-", "versus", " vs", "vs.", "over time", "percentage",
    "percent", "%", "seconds", "minutes", "hours", "score", "accuracy",
    "latency", "throughput", "rate",
)
_TREND_HINTS = (
    "increase", "decrease", "higher", "lower", "improve", "outperform",
    "better", "worse", "reduc", "grow", "decline", "trend", "compared",
    "more than", "less than", "faster", "slower",
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_QUESTION_PROMPTS = {
    "describe": "Describe what this figure shows in one or two sentences.",
    "axes": "What quantities or axes does this figure use? Answer in one sentence.",
    "trend": "What trend or comparison does this figure convey? Answer in one sentence.",
}


def normalize_question(question: Any) -> str:
    """Return the canonical question kind, raising for anything unsupported."""
    if question is None:
        return DEFAULT_QUESTION
    canonical = str(question).strip().lower()
    if canonical not in FIGURE_QUESTIONS:
        raise ValueError(
            "question must be one of " + "/".join(FIGURE_QUESTIONS) + f"; got {question!r}"
        )
    return canonical


def validate_figure_no(figure_no: Any) -> int:
    """1-based figure index, bounded by what extraction can produce."""
    if isinstance(figure_no, bool):
        raise ValueError(f"figure_no must be a positive integer; got {figure_no!r}")
    try:
        value = int(figure_no)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"figure_no must be a positive integer; got {figure_no!r}"
        ) from exc
    if value < 1 or value > MAX_FIGURE_NO:
        raise ValueError(
            f"figure_no must be between 1 and {MAX_FIGURE_NO}; got {figure_no!r}"
        )
    return value


# --------------------------------------------------------------------------
# deterministic (extractive) backend
# --------------------------------------------------------------------------


def _sentences(text: str) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(cleaned) if part.strip()]


def _matching_sentences(caption: str, hints: Tuple[str, ...]) -> List[str]:
    return [s for s in _sentences(caption) if any(h in s.lower() for h in hints)]


def _extractive_answer(figure: Dict[str, Any], question: str) -> str:
    """Answer from the caption T4 already extracted, and admit what it cannot.

    Saying "the caption does not state this" is the honest deterministic result:
    it is still a grounded observation about the figure, and the alternative —
    inventing an answer without looking — is exactly the failure mode this
    backend exists to avoid.
    """
    caption = (figure.get("caption") or "").strip()

    if question == "describe":
        if caption:
            return caption
        return (
            f"Figure {figure.get('figure_no')} on page {figure.get('page')} "
            f"({figure.get('width')}×{figure.get('height')} pixels) has no "
            "extracted caption."
        )

    if not caption:
        return (
            f"The caption for figure {figure.get('figure_no')} was not "
            "extracted, so this cannot be answered from text."
        )

    hints = _AXES_HINTS if question == "axes" else _TREND_HINTS
    matches = _matching_sentences(caption, hints)
    if matches:
        return " ".join(matches)
    return (
        f"The caption for figure {figure.get('figure_no')} does not state this. "
        f"Caption: {caption}"
    )


def _load_figure(session_id: str, ref: Any, figure_no: int) -> Dict[str, Any]:
    """Return the extracted figure metadata for *figure_no* of the referenced paper.

    Reuses T4's extraction rather than reading files independently, so "the
    figure list" has exactly one definition.  It runs through the *caller's*
    session on purpose: a fresh session would have no paper list and the ref
    would not resolve.
    """
    result = extract_paper_figures(session_id=session_id, ref=ref)
    figures = result.get("figures") or []
    if figure_no > len(figures):
        raise ValueError(
            f"Paper {result.get('paper_id')} has {len(figures)} extracted "
            f"figure(s); figure_no={figure_no} is out of range."
        )
    figure = figures[figure_no - 1]
    if not Path(figure["path"]).is_file():
        raise ValueError(
            f"Figure file is missing on disk: {figure['path']}. "
            "Call extract_paper_figures again."
        )
    return figure


# --------------------------------------------------------------------------
# optional local-VLM backend
# --------------------------------------------------------------------------

_VLM_CACHE: Dict[str, Any] = {}


def _load_vlm(model_path: str):
    cached = _VLM_CACHE.get(model_path)
    if cached is not None:
        return cached

    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "FIGURE_ANALYSIS_BACKEND=vlm requires transformers with image-text-to-text "
            "support and torch."
        ) from exc

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16
    )
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    _VLM_CACHE[model_path] = (processor, model)
    return processor, model


def _vlm_answer(figure: Dict[str, Any], question: str) -> str:
    import torch
    from PIL import Image

    model_path = os.getenv("VLM_MODEL_PATH", "").strip()
    if not model_path:
        raise RuntimeError(
            "FIGURE_ANALYSIS_BACKEND=vlm requires VLM_MODEL_PATH to point at a "
            "local vision-language model directory."
        )

    processor, model = _load_vlm(model_path)

    image = Image.open(figure["path"]).convert("RGB")
    longest = max(image.size)
    if longest > VLM_MAX_IMAGE_SIDE:
        scale = VLM_MAX_IMAGE_SIDE / longest
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.LANCZOS,
        )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": _QUESTION_PROMPTS[question]},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        # Greedy: a sampled answer would make snapshot replay non-reproducible.
        generated = model.generate(**inputs, max_new_tokens=96, do_sample=False)

    completion = generated[:, inputs["input_ids"].shape[1]:]
    answer = processor.batch_decode(completion, skip_special_tokens=True)[0].strip()
    if not answer:
        raise ValueError("vision-language model produced an empty answer")
    return answer


def analysis_backend() -> str:
    """Return the configured backend name, validated eagerly."""
    name = os.getenv("FIGURE_ANALYSIS_BACKEND", "extractive").strip().lower() or "extractive"
    if name not in {"extractive", "vlm"}:
        raise RuntimeError(
            "FIGURE_ANALYSIS_BACKEND must be 'extractive' or 'vlm'; "
            f"got {name!r}"
        )
    return name


def analyze_figure(
    session_id: str = "default",
    ref: Union[str, int, None] = 1,
    figure_no: int = 1,
    question: Optional[str] = None,
) -> Dict[str, Any]:
    """Answer a question about one figure of a downloaded paper.

    The paper must have its figures extracted first (T4).  Errors stay
    deterministic: an unpublished/undownloaded paper, an out-of-range figure
    index and an unsupported question all raise rather than returning a
    plausible-looking answer.
    """
    canonical_question = normalize_question(question)
    index = validate_figure_no(figure_no)

    paper = store.resolve_paper(session_id, ref)
    if paper is None:
        raise ValueError("Paper not found; search for the paper and check the ref.")

    figure = _load_figure(session_id, ref, index)

    backend = analysis_backend()
    if backend == "vlm":
        answer = _vlm_answer(figure, canonical_question)
    else:
        answer = _extractive_answer(figure, canonical_question)

    store.set_last_active_paper_id(session_id, paper.id)

    return {
        "paper_id": paper.id,
        "figure_no": index,
        "page": figure.get("page"),
        "question": canonical_question,
        "answer": answer,
        "backend": backend,
        "image_path": figure["path"],
    }


FIGURE_ANALYSIS_TOOL_SCHEMA = {
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
        "figure_no": {
            "description": (
                "1-based index of the figure within the paper, in page order; "
                f"1 to {MAX_FIGURE_NO}."
            ),
            "type": "integer",
            "default": 1,
        },
        "question": {
            "description": "What to ask about the figure.",
            "type": "string",
            "enum": list(FIGURE_QUESTIONS),
            "default": DEFAULT_QUESTION,
        },
    },
    "required": [],
}


registry.register_tool(
    name="analyze_figure",
    description=(
        "Answer a question about one figure of a downloaded arXiv paper. "
        "The paper's figures must be extracted first. Choose question "
        "(describe/axes/trend); the answer comes from the environment, not "
        "from you."
    ),
    parameter_schema=FIGURE_ANALYSIS_TOOL_SCHEMA,
    func=analyze_figure,
)
