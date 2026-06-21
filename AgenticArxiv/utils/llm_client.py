# AgenticArxiv/utils/llm_client.py
import os
from typing import Any, Dict, List, Optional
import requests


class LLMClient:
    """
    对接 OpenAI-compatible /v1/chat/completions 接口
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_s: int = 60,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s

    def chat_completions(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1000,
        stream: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if extra:
            payload.update(extra)

        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()


def get_env_llm_client() -> LLMClient:
    """
    从环境变量读取：
    - LLM_BASE_URL   默认: https://antigravity.byssted.cn
    - LLM_API_KEY    必填
    """
    base_url = os.getenv("LLM_BASE_URL", "https://antigravity.byssted.cn")
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("Missing env: LLM_API_KEY (请在 .env 或 shell 环境中设置)")
    return LLMClient(base_url=base_url, api_key=api_key)
