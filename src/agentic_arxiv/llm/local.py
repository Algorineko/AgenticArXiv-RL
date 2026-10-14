"""本地 Hugging Face 客户端，不从训练入口取得模型加载能力。"""

from hashlib import sha256
from typing import Any, Dict, List, Optional

from agentic_arxiv.config import resolve_model_source

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
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "本地推理需要 torch 和 transformers。"
                "请安装：pip install 'agentic-arxiv[local]'。"
            ) from exc

        load_source = resolve_model_source(model, local_only=True)
        self.model_name = model
        self.seed = seed
        self._stream_seed = seed
        self._calls = 0
        self.tokenizer = AutoTokenizer.from_pretrained(load_source)
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
        self.model = AutoModelForCausalLM.from_pretrained(load_source, **load_kwargs)
        if device != "auto":
            self.model.to(device)
        self.model.eval()

    @classmethod
    def from_loaded(
        cls, model: Any, tokenizer: Any, model_name: str = ""
    ) -> "TransformersLLMClient":
        """绑定已加载对象，不重复加载，也不改变模型的训练模式。"""
        client = cls.__new__(cls)
        client.model = model
        client.tokenizer = tokenizer
        client.model_name = model_name
        client.seed = None
        client._stream_seed = None
        client._calls = 0
        client.device = "loaded"
        return client

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
        raw_model_content = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        content = raw_model_content
        matched_stop = any(marker in raw_model_content for marker in stop)
        for marker in stop:
            if marker in content:
                content = content.split(marker, 1)[0]
        completion_tokens = int(completion_ids.shape[-1])
        # EOS 以实际 generate 覆盖/模型配置为准，不能把 tokenizer 的 EOS
        # 猜成模型结束条件。配置不可得时保留未知，不据长度臆测结束原因。
        unknown = object()
        generation_config = generation_kwargs.get(
            "generation_config", getattr(self.model, "generation_config", None)
        )
        eos_ids = generation_kwargs.get("eos_token_id", unknown)
        if eos_ids is unknown:
            config = generation_config if generation_config is not None else getattr(self.model, "config", None)
            eos_ids = getattr(config, "eos_token_id", unknown)
        eos_known = eos_ids is None or (
            isinstance(eos_ids, int) and not isinstance(eos_ids, bool)
        ) or (isinstance(eos_ids, (list, tuple)) and all(
            isinstance(token, int) and not isinstance(token, bool) for token in eos_ids
        ))
        eos_values = (
            {eos_ids} if isinstance(eos_ids, int) and not isinstance(eos_ids, bool)
            else set(eos_ids or []) if eos_known else set()
        )
        ended_on_eos = bool(eos_values.intersection(completion_ids.tolist()))
        finish_reason = None
        if matched_stop or ended_on_eos:
            finish_reason = "stop"
        elif eos_known and completion_tokens >= int(generation_kwargs["max_new_tokens"]):
            finish_reason = "length"
        return {
            "choices": [{"message": {"role": "assistant", "content": content},
                         "finish_reason": finish_reason}],
            "raw_model_content": raw_model_content,
            "usage": {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": completion_tokens,
                "total_tokens": int(prompt_tokens) + completion_tokens,
            },
        }
