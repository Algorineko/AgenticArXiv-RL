"""Environment-driven construction for the optional inference router."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env.local", override=False)
except Exception:
    pass

from routing.jev import JevToolRouter


def build_tool_router_from_env():
    """Return no router for the default policy mode, or a configured Jev router."""
    mode = os.getenv("TOOL_ROUTER", "policy").strip().lower()
    if mode in {"", "policy", "qwen", "none", "off"}:
        return None
    if mode != "jev":
        raise ValueError(
            f"Unsupported TOOL_ROUTER={mode!r}; choose 'policy' or 'jev'"
        )
    api_key = os.getenv("TYPESAFE_API_KEY", "")
    return JevToolRouter(
        api_key=api_key,
        model=os.getenv("JEV_MODEL", "jev-latest"),
        base_url=os.getenv("JEV_BASE_URL", "https://api.typesafe.ai"),
        min_confidence=float(os.getenv("JEV_MIN_CONFIDENCE", "0.80")),
        timeout_s=float(os.getenv("JEV_TIMEOUT_S", "30")),
        max_retries=int(os.getenv("JEV_MAX_RETRIES", "2")),
        backoff_s=float(os.getenv("JEV_BACKOFF_S", "0.75")),
    )
