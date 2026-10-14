"""统一客户端导出；API 调用方不加载本地客户端模块。"""

from agentic_arxiv.llm.api import LLMClient, get_env_llm_client

# 本地类通过 PEP 562 的 __getattr__ 延迟导出，保持 API 导入轻量。
__all__ = ["LLMClient", "TransformersLLMClient", "get_env_llm_client"]  # noqa: F822


def __getattr__(name: str):
    if name == "TransformersLLMClient":
        from agentic_arxiv.llm.local import TransformersLLMClient

        return TransformersLLMClient
    raise AttributeError(name)
