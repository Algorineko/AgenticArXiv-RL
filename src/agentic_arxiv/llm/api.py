"""OpenAI-compatible API 请求与有界重试。"""

import os
import random
import time
from typing import Any, Dict, List, Optional

import requests
from loguru import logger


class LLMClient:
    """对接 OpenAI-compatible /v1/chat/completions 接口。

    可重试的连接错误、超时、429 和 5xx 使用有界指数退避；
    其余 4xx 直接抛错，总尝试次数为 1 + max_retries。
    """

    MAX_BACKOFF_S = 30.0

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_s: int = 60,
        max_retries: int = 3,
        backoff_s: float = 1.0,
        backoff_factor: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max(0, int(max_retries))
        self.backoff_s = float(backoff_s)
        self.backoff_factor = float(backoff_factor)

    def chat_completions(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1000,
        stream: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.model_name = model
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

        last_error: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                self._sleep_before_retry(attempt, f"{type(exc).__name__}: {exc}")
                continue
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                if attempt >= self.max_retries:
                    resp.raise_for_status()
                retry_after = self._parse_retry_after(resp)
                self._sleep_before_retry(attempt, f"HTTP {resp.status_code}", retry_after)
                continue
            resp.raise_for_status()
            return resp.json()
        assert last_error is not None
        raise last_error

    def _retry_delay(self, attempt: int, retry_after: Optional[float] = None) -> float:
        """第 attempt 次重试（从 0 计）前的等待秒数。"""
        if retry_after is not None:
            delay = min(float(retry_after), self.MAX_BACKOFF_S)
        else:
            delay = min(
                self.backoff_s * (self.backoff_factor ** attempt),
                self.MAX_BACKOFF_S,
            )
        return delay * random.uniform(0.5, 1.5)

    def _sleep_before_retry(
        self,
        attempt: int,
        reason: str,
        retry_after: Optional[float] = None,
    ) -> None:
        delay = self._retry_delay(attempt, retry_after)
        logger.warning(
            "LLM 请求失败（{}），{:.1f}s 后进行第 {}/{} 次重试",
            reason, delay, attempt + 1, self.max_retries,
        )
        time.sleep(delay)

    @staticmethod
    def _parse_retry_after(resp: requests.Response) -> Optional[float]:
        """读取 429 响应的 Retry-After 秒数；缺失或非法时返回 None。"""
        value = resp.headers.get("Retry-After")
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return None


def get_env_llm_client() -> LLMClient:
    """读取显式操作已初始化的环境，不在导入时加载 .env。"""
    base_url = os.getenv("LLM_BASE_URL")
    if not base_url:
        raise RuntimeError(
            "Missing env: LLM_BASE_URL (请在 .env 或 shell 环境中设置为 OpenAI 兼容服务根地址)"
        )
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("Missing env: LLM_API_KEY (请在 .env 或 shell 环境中设置)")
    return LLMClient(
        base_url=base_url,
        api_key=api_key,
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
        backoff_s=float(os.getenv("LLM_BACKOFF_S", "1.0")),
    )
