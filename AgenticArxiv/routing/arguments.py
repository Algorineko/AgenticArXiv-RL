"""Deterministic argument extraction for accepted external tool routes.

Jev is a typed decision model, not a free-form string generator.  Keep semantic
tool selection in the router, but compute identifiers, ordinals and explicit
options in code whenever the user already supplied them.  Ambiguous requests
defer to the policy model; this module never reads benchmark labels.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class RoutedArgumentResolution:
    """Argument resolution for one tool, rather than overall task completion.

    ``complete`` means its explicit references appear in earlier actions. Those
    actions may have failed, and other requested tools may still be needed.
    """

    status: str
    args: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


_ARXIV_ID = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", re.IGNORECASE)
_ORDINAL = re.compile(r"第\s*(\d+)\s*篇")
_QUERY = re.compile(
    r"\b(?:all|ti|au):\s*[^，,。；;\n]+", re.IGNORECASE
)


def resolve_routed_arguments(
    *,
    tool_name: str,
    task: str,
    history: str,
) -> RoutedArgumentResolution:
    """Resolve explicit arguments or report that this tool's refs are exhausted.

    A ``complete`` result cannot establish overall task success. The caller
    should restore the full policy context to decide what remains to be done.
    """
    clean_task = str(task or "")
    used_refs = _used_refs(history, tool_name)

    if tool_name == "search_arxiv_papers":
        match = _QUERY.search(clean_task)
        if not match:
            return _defer("keyword_query_not_explicit")
        query = match.group(0).strip().rstrip("：:")
        args: Dict[str, Any] = {"query": query}
        maximum = _first_int(
            clean_task,
            r"(?:最多(?:返回)?|返回)\s*(\d+)\s*篇",
        )
        if maximum is not None:
            args["max_results"] = maximum
        days = _first_int(clean_task, r"(?:最近|近)\s*(\d+)\s*天")
        if days is not None:
            args["days"] = days
        return _resolved(args, "explicit_arxiv_query")

    if tool_name == "get_recently_submitted_cs_papers":
        args = {}
        category = re.search(r"(?:cs\.)?([A-Z]{2})\b", clean_task)
        if category:
            args["aspect"] = category.group(1).upper()
        days = _first_int(clean_task, r"(?:最近|近)\s*(\d+)\s*天")
        maximum = _first_int(clean_task, r"(?:最多(?:返回)?|返回)\s*(\d+)\s*篇")
        if days is not None:
            args["days"] = days
        if maximum is not None:
            args["max_results"] = maximum
        return _resolved(args, "explicit_recent_search_options") if args else _defer(
            "recent_search_options_not_explicit"
        )

    if tool_name in {
        "download_arxiv_pdf",
        "translate_arxiv_pdf",
        "get_paper_cache_status",
        "get_paper_content",
        "summarize_paper",
        "extract_paper_figures",
        "analyze_figure",
    }:
        reference = _next_reference(clean_task, used_refs)
        if reference.status == "complete":
            return reference
        if reference.status != "resolved":
            return reference
        args = dict(reference.args)

        if tool_name == "translate_arxiv_pdf":
            service = re.search(r"\b(google|bing|deepl)\b", clean_task, re.IGNORECASE)
            if service:
                args["service"] = service.group(1).lower()
            if "强制" in clean_task:
                args["force"] = True
            threads = _first_int(clean_task, r"(\d+)\s*(?:线程|个线程)")
            if threads is not None:
                args["threads"] = threads
            if "双语" in clean_task:
                args["keep_dual"] = True
        elif tool_name == "download_arxiv_pdf" and "强制" in clean_task:
            args["force"] = True
        elif tool_name == "get_paper_content":
            for section in ("abstract", "method", "result", "conclusion"):
                if section in clean_task.lower():
                    args["section"] = section
                    break
        elif tool_name == "summarize_paper":
            for style in ("tldr", "structured", "bullet"):
                if style in clean_task.lower():
                    args["style"] = style
                    break
            words = _first_int(clean_task, r"(\d+)\s*(?:词|words?)")
            if words is not None:
                args["max_words"] = words
        elif tool_name == "analyze_figure":
            figure_no = _first_int(clean_task, r"(?:图|figure)\s*(\d+)")
            if figure_no is not None:
                args["figure_no"] = figure_no
            question_map = {
                "坐标": "axes",
                "趋势": "trend",
                "describe": "describe",
                "axes": "axes",
                "trend": "trend",
            }
            for marker, value in question_map.items():
                if marker in clean_task.lower():
                    args["question"] = value
                    break
        return _resolved(args, "explicit_reference_and_options")

    return _defer("unsupported_tool")


def _next_reference(task: str, used_refs: List[Any]) -> RoutedArgumentResolution:
    paper_id = _ARXIV_ID.search(task)
    if paper_id:
        ref = paper_id.group(0)
        if ref in used_refs:
            return _complete("explicit_reference_already_used")
        return _resolved({"ref": ref}, "explicit_arxiv_id")

    refs = [int(value) for value in _ORDINAL.findall(task)]
    if refs:
        if any(ref < 1 for ref in refs):
            return _blocked("non_positive_reference")
        for ref in refs:
            if ref not in used_refs:
                return _resolved({"ref": ref}, "explicit_ordinal")
        return _complete("all_explicit_references_used")

    active_markers = (
        "刚操作",
        "刚下载",
        "刚才那篇",
        "最近操作",
        "上一条操作",
        "上一次操作",
    )
    if any(marker in task for marker in active_markers):
        if None in used_refs:
            return _complete("active_reference_already_used")
        return _resolved({"ref": None}, "active_reference")
    return _defer("paper_reference_not_explicit")


def _used_refs(history: str, tool_name: str) -> List[Any]:
    used: List[Any] = []
    for match in re.finditer(
        r"Action:\s*(\{[^\n]*\})\s*\nObservation:",
        str(history or ""),
    ):
        try:
            action = json.loads(match.group(1))
        except (TypeError, ValueError):
            continue
        if action.get("name") == tool_name:
            used.append((action.get("args") or {}).get("ref"))
    return used


def _first_int(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _resolved(args: Dict[str, Any], reason: str) -> RoutedArgumentResolution:
    return RoutedArgumentResolution("resolved", args, reason)


def _complete(reason: str) -> RoutedArgumentResolution:
    return RoutedArgumentResolution("complete", {}, reason)


def _blocked(reason: str) -> RoutedArgumentResolution:
    return RoutedArgumentResolution("blocked", {}, reason)


def _defer(reason: str) -> RoutedArgumentResolution:
    return RoutedArgumentResolution("defer", {}, reason)
