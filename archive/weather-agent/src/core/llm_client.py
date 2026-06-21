import os
import requests
import json
from typing import Dict, Any, Optional
from loguru import logger


class LLMClient:
    """LLM客户端，用于调用自部署的VLLM服务"""

    def __init__(self, api_url: str = None, model: str = None):
        self.api_url = api_url or os.getenv("LLM_API_URL", "http://10.26.69.92:8000/v1/chat/completions")
        self.model = model or os.getenv("LLM_MODEL", "qwen2.5-7b-instruct")

    def generate(self,
                 messages: list,
                 temperature: float = 0.1,
                 max_tokens: int = 512,
                 stream: bool = False) -> str:
        """调用LLM生成回复

        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大token数
            stream: 是否流式输出

        Returns:
            LLM生成的回复文本
        """
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": stream,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            logger.debug(f"发送请求到LLM: {self.api_url}")
            logger.debug(f"请求参数: {json.dumps(payload, ensure_ascii=False, indent=2)}")

            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()
                logger.debug(f"LLM响应: {content[:200]}...")
                return content
            else:
                error_msg = f"LLM API错误: {response.status_code}, {response.text}"
                logger.error(error_msg)
                return f"Error: {error_msg}"

        except requests.exceptions.Timeout:
            error_msg = "LLM API请求超时"
            logger.error(error_msg)
            return f"Error: {error_msg}"
        except requests.exceptions.ConnectionError:
            error_msg = "无法连接到LLM服务"
            logger.error(error_msg)
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"LLM调用失败: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"

    def chat(self, user_message: str, history: list = None) -> tuple:
        """简单的对话接口

        Args:
            user_message: 用户消息
            history: 历史对话记录

        Returns:
            (回复内容, 更新后的历史记录)
        """
        if history is None:
            history = []

        messages = history + [{"role": "user", "content": user_message}]
        response = self.generate(messages)

        # 更新历史记录
        new_history = messages + [{"role": "assistant", "content": response}]

        return response, new_history
