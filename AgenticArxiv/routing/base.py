"""Small, provider-neutral contracts for optional external tool routers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class RouteDecision:
    """One next-tool decision.

    ``accepted=False`` means that the normal policy model must choose the tool.
    A terminal decision uses ``selected_tool="FINISH"``.
    """

    selected_tool: str | None = None
    confidence: float = 0.0
    accepted: bool = False
    source: str = "policy"
    reason: str | None = None
    latency_ms: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=dict)
    usage: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ToolRouter(Protocol):
    """Interface implemented by optional next-tool routers."""

    name: str

    def route(
        self,
        *,
        task: str,
        history: str,
        tools: Sequence[Mapping[str, Any]],
    ) -> RouteDecision:
        """Choose the immediate next tool or defer to the policy model."""
