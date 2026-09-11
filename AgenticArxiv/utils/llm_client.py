# AgenticArxiv/utils/llm_client.py
import os
import random
import time
from hashlib import sha256
from typing import Any, Dict, List, Optional
import requests
from loguru import logger


class LLMClient:
    """
    对接 OpenAI-compatible /v1/chat/completions 接口

    对可重试的失败（连接错误、超时、429、5xx）做有界指数退避重试；
    其余 4xx 直接抛错。总尝试次数 = 1 + max_retries。
    """

    #: 单次重试等待的上限，避免服务端持续故障时睡过头
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
        # 循环内所有分支都会 return 或 raise，这里仅作防御
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
        # 抖动避免并发 rollout 在同一时刻集体重试（惊群）
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
        """读取 429 响应的 Retry-After 头（秒）；缺失或非法时返回 None。"""
        value = resp.headers.get("Retry-After")
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return None


class TransformersLLMClient:
    """Expose a local Hugging Face causal LM through the small client contract.

    Imports are intentionally lazy so benchmark and data-processing utilities do
    not need to import the training stack unless local inference is requested.
    The returned object mirrors the subset of OpenAI's chat-completions response
    consumed by :class:`BaseAgent`.
    """

    def __init__(
        self,
        model: str,
        device: str = "auto",
        dtype: str = "auto",
        seed: Optional[int] = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model
        self.seed = seed
        self._stream_seed = seed
        self._calls = 0
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype {dtype!r}; choose from {sorted(dtype_map)}")
        load_kwargs: Dict[str, Any] = {"dtype": dtype_map[dtype]}
        self.device = device
        if device == "auto":
            load_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model, **load_kwargs)
        if device != "auto":
            self.model.to(device)
        self.model.eval()

    def start_generation_stream(self, stream_key: str) -> None:
        """Start an order-independent deterministic RNG stream for one rollout.

        A global ``seed + call_count`` stream makes a later task's samples depend
        on how many ReAct turns every earlier task happened to use. Deriving the
        stream from task/trial identity keeps paired Base/SFT/RL evaluations
        comparable even when an environment change alters earlier trajectories.
        """
        self._calls = 0
        if self.seed is None:
            self._stream_seed = None
            return
        digest = sha256(str(stream_key).encode("utf-8")).digest()
        offset = int.from_bytes(digest[:8], "big") % (2**31)
        self._stream_seed = self.seed + offset

    def chat_completions(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1000,
        stream: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        import torch

        if stream:
            raise ValueError("TransformersLLMClient does not support streaming")
        extra = dict(extra or {})
        stop = extra.pop("stop", []) or []
        template_kwargs = extra.pop("chat_template_kwargs", {}) or {}
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        target = next(self.model.parameters()).device
        inputs = {key: value.to(target) for key, value in inputs.items()}
        do_sample = temperature is not None and temperature > 0
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            **extra,
        }
        # Transformers supports string-level stopping when the tokenizer is
        # supplied to generate().  Keep the decode-time split below as a final
        # guard, but stop generation eagerly so a local model does not waste up
        # to max_tokens hallucinating its own Observation and later turns.
        if stop:
            generation_kwargs["stop_strings"] = list(stop)
            generation_kwargs["tokenizer"] = self.tokenizer
        if do_sample:
            generation_kwargs["temperature"] = temperature
        if self.seed is not None:
            stream_seed = getattr(self, "_stream_seed", self.seed)
            torch.manual_seed(stream_seed + self._calls)
        self._calls += 1
        with torch.inference_mode():
            output = self.model.generate(**inputs, **generation_kwargs)
        prompt_tokens = inputs["input_ids"].shape[-1]
        completion_ids = output[0, prompt_tokens:]
        content = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        for marker in stop:
            if marker in content:
                content = content.split(marker, 1)[0]
        completion_tokens = int(completion_ids.shape[-1])
        return {
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": completion_tokens,
                "total_tokens": int(prompt_tokens) + completion_tokens,
            },
        }


def get_env_llm_client() -> LLMClient:
    """
    从环境变量读取：
    - LLM_BASE_URL   默认: https://antigravity.byssted.cn
    - LLM_API_KEY    必填
    - LLM_MAX_RETRIES  可选, 默认 3（0 表示不重试）
    - LLM_BACKOFF_S    可选, 首次重试等待秒数, 默认 1.0
    """
    base_url = os.getenv("LLM_BASE_URL", "https://antigravity.byssted.cn")
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("Missing env: LLM_API_KEY (请在 .env 或 shell 环境中设置)")
    return LLMClient(
        base_url=base_url,
        api_key=api_key,
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
        backoff_s=float(os.getenv("LLM_BACKOFF_S", "1.0")),
    )
