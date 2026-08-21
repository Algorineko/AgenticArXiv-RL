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
        load_kwargs: Dict[str, Any] = {"torch_dtype": dtype_map[dtype]}
        self.device = device
        if device == "auto":
            load_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model, **load_kwargs)
        if device != "auto":
            self.model.to(device)
        self.model.eval()

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
        if do_sample:
            generation_kwargs["temperature"] = temperature
        if self.seed is not None:
            # A different, reproducible stream per call gives DPO rollouts
            # diversity without making repeated dataset builds irreproducible.
            torch.manual_seed(self.seed + self._calls)
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
    """
    base_url = os.getenv("LLM_BASE_URL", "https://antigravity.byssted.cn")
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("Missing env: LLM_API_KEY (请在 .env 或 shell 环境中设置)")
    return LLMClient(base_url=base_url, api_key=api_key)
