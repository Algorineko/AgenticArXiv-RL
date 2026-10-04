"""TypeSafe Jev adapter used only for optional inference-time routing."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Mapping, Sequence

import requests

from routing.base import RouteDecision


FINISH_CRITERION = (
    "Do not call a tool. Select this when the task is already complete OR cannot "
    "be executed safely: required session/paper context is missing, a paper "
    "numeric/ordinal reference is invalid or outside the available candidate list, "
    "the requested action is unsupported, or no available tool can satisfy it. "
    "An explicit arXiv ID, URL, or title is a valid direct reference for tools that "
    "support one and MUST NOT be rejected merely because it is absent from the "
    "current candidate list."
)


ROUTING_CRITERIA = {
    "get_recently_submitted_cs_papers": (
        "Fetch recent/new computer-science submissions by cs category and a recent "
        "time window. Use this only for requests framed as recent/latest papers in "
        "a cs.* area, not for arbitrary keyword, title, author, or all: queries."
    ),
    "search_arxiv_papers": (
        "Search arXiv by explicit keywords, title, author, or an arXiv query such as "
        "all:, ti:, or au:. A days filter does not turn a keyword query into the "
        "recent-category tool."
    ),
    "download_arxiv_pdf": (
        "Download a paper PDF. A numeric ref needs a valid candidate list; an "
        "explicit arXiv ID, URL, or title can be used directly even when it is not "
        "present in the current candidate list."
    ),
    "translate_arxiv_pdf": (
        "Translate a selected or directly identified paper PDF. Use for translation "
        "requests and preserve options such as service, threads, force, or keep_dual."
    ),
    "get_paper_cache_status": (
        "Inspect local download/translation cache state for a paper. Use for status, "
        "cached, downloaded, raw, mono, or translation-state questions."
    ),
    "get_paper_content": "Read the abstract or a named text section of a paper.",
    "summarize_paper": "Summarize a paper in the requested style and word budget.",
    "extract_paper_figures": "Extract figures from a downloaded paper.",
    "analyze_figure": "Describe or analyze a particular extracted paper figure.",
}


class JevToolRouter:
    """Ask Jev for a typed next-tool choice and otherwise defer to the policy.

    The adapter never executes a tool and never generates tool arguments. Network
    errors, malformed responses, unknown choices and low confidence all produce a
    non-accepted decision, allowing :class:`BaseAgent` to retain its original
    Qwen/policy path.
    """

    name = "jev"
    transient_statuses = {429, 500, 502, 503, 504}

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "jev-latest",
        base_url: str = "https://api.typesafe.ai",
        min_confidence: float = 0.80,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        backoff_s: float = 0.75,
        max_state_chars: int = 12000,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.strip():
            raise ValueError("TYPESAFE_API_KEY is required when TOOL_ROUTER=jev")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("JEV_MIN_CONFIDENCE must be between 0 and 1")
        self.api_key = api_key.strip()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.min_confidence = float(min_confidence)
        self.timeout_s = float(timeout_s)
        self.max_retries = max(0, int(max_retries))
        self.backoff_s = max(0.0, float(backoff_s))
        self.max_state_chars = max(1000, int(max_state_chars))
        self.session = session or requests.Session()
        self.sleep = sleep

    def route(
        self,
        *,
        task: str,
        history: str,
        tools: Sequence[Mapping[str, Any]],
    ) -> RouteDecision:
        started = time.perf_counter()
        criteria = {}
        for tool in tools:
            name = str(tool.get("name") or "")
            if not name:
                continue
            description = str(tool.get("description") or name)
            distinction = ROUTING_CRITERIA.get(name)
            criteria[name] = (
                f"{distinction} Project tool description: {description}"
                if distinction else description
            )
        criteria["FINISH"] = FINISH_CRITERION
        payload = {
            "state": {
                "agent": "AgenticArXiv ReAct research assistant",
                "task": task,
                "history_and_environment": self._bounded_history(history),
                "decision": (
                    "Select only the immediate next action. Select FINISH instead "
                    "of a tool when required context is absent, a reference is "
                    "invalid/out of range, or the operation is unsupported."
                ),
            },
            "model": self.model,
            "questions": {
                "next_tool": {
                    "type": "choice",
                    "instructions": (
                        "Which single tool should the agent call next? Choose "
                        "FINISH when no valid tool call should be made."
                    ),
                    "criteria": criteria,
                }
            },
        }
        try:
            response = self._post(payload)
            answer = response["answers"]["next_tool"]
            selected = str(answer["choice"])
            confidence = float(answer.get("confidence", 0.0))
            probabilities = {
                str(name): float(probability)
                for name, probability in (answer.get("probabilities") or {}).items()
            }
            usage = {
                key: int(value)
                for key, value in (response.get("usage") or {}).items()
                if isinstance(value, (int, float))
            }
        except Exception as exc:
            return self._defer(
                started,
                reason=f"api_error:{type(exc).__name__}",
            )

        if selected not in criteria:
            return self._defer(started, reason="unknown_tool")
        if confidence < self.min_confidence:
            return RouteDecision(
                selected_tool=selected,
                confidence=confidence,
                accepted=False,
                source=self.name,
                reason="low_confidence",
                latency_ms=self._elapsed_ms(started),
                probabilities=probabilities,
                usage=usage,
            )
        return RouteDecision(
            selected_tool=selected,
            confidence=confidence,
            accepted=True,
            source=self.name,
            latency_ms=self._elapsed_ms(started),
            probabilities=probabilities,
            usage=usage,
        )

    def _post(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}/v1/systemone",
                    headers=headers,
                    json=dict(payload),
                    timeout=(10, self.timeout_s),
                )
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.SSLError,
            ) as exc:
                last_error = exc
            else:
                if response.status_code not in self.transient_statuses:
                    response.raise_for_status()
                    data = response.json()
                    if not isinstance(data, dict):
                        raise ValueError("Jev response must be a JSON object")
                    return data
                last_error = RuntimeError(f"HTTP {response.status_code}")
            if attempt < self.max_retries:
                self.sleep(self.backoff_s * (2**attempt))
        if last_error is None:
            raise RuntimeError("Jev request failed")
        raise last_error

    def _bounded_history(self, history: str) -> str:
        clean = history.strip() or "(no prior tool observations)"
        if len(clean) <= self.max_state_chars:
            return clean
        return "[earlier context truncated]\n" + clean[-self.max_state_chars :]

    def _defer(self, started: float, *, reason: str) -> RouteDecision:
        return RouteDecision(
            accepted=False,
            source=self.name,
            reason=reason,
            latency_ms=self._elapsed_ms(started),
        )

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 2)
