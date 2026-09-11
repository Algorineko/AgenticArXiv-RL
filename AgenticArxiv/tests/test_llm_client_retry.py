#!/usr/bin/env python3
"""LLMClient 重试行为的测试。

不需要 LLM、网络或 GPU —— 用假响应与 mock 的 requests.post / time.sleep /
random.uniform 覆盖各重试路径，并验证等待时长符合指数退避约定。

运行：
    cd AgenticArxiv && python tests/test_llm_client_retry.py
"""

import os
import sys
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("STORE_BACKEND", "memory")

import requests  # noqa: E402

from utils.llm_client import LLMClient  # noqa: E402


class FakeResponse:
    """最小可用的 requests.Response 替身。"""

    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"{self.status_code} error", response=self
            )


def make_client(**kwargs):
    return LLMClient(base_url="http://fake", api_key="k", **kwargs)


class TestRetry(unittest.TestCase):
    def setUp(self):
        # 冻结随机抖动并记录 sleep 时长，保证断言确定性且不真的等待
        uniform_patcher = mock.patch(
            "utils.llm_client.random.uniform", return_value=1.0
        )
        sleep_patcher = mock.patch("utils.llm_client.time.sleep")
        self.mock_uniform = uniform_patcher.start()
        self.mock_sleep = sleep_patcher.start()
        self.addCleanup(uniform_patcher.stop)
        self.addCleanup(sleep_patcher.stop)

    def chat(self, client, **kwargs):
        return client.chat_completions(
            model="m", messages=[{"role": "user", "content": "hi"}], **kwargs
        )

    def test_success_on_first_attempt(self):
        with mock.patch(
            "utils.llm_client.requests.post",
            return_value=FakeResponse(200, {"ok": True}),
        ) as post:
            result = make_client().chat_completions(
                model="m", messages=[{"role": "user", "content": "hi"}]
            )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(post.call_count, 1)
        self.mock_sleep.assert_not_called()

    def test_retry_on_429_then_success(self):
        responses = [
            FakeResponse(429, headers={"Retry-After": "2"}),
            FakeResponse(200, {"ok": True}),
        ]
        with mock.patch(
            "utils.llm_client.requests.post", side_effect=responses
        ) as post:
            result = make_client().chat_completions(
                model="m", messages=[{"role": "user", "content": "hi"}]
            )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(post.call_count, 2)
        self.mock_sleep.assert_called_once_with(2.0)  # 尊重 Retry-After

    def test_retry_on_5xx_then_success(self):
        responses = [
            FakeResponse(503),
            FakeResponse(502),
            FakeResponse(200, {"ok": True}),
        ]
        with mock.patch(
            "utils.llm_client.requests.post", side_effect=responses
        ) as post:
            result = make_client().chat_completions(
                model="m", messages=[{"role": "user", "content": "hi"}]
            )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(post.call_count, 3)
        self.assertEqual(
            [call.args[0] for call in self.mock_sleep.call_args_list],
            [1.0, 2.0],  # 指数退避: backoff_s * factor**attempt
        )

    def test_retry_on_connection_error_then_success(self):
        with mock.patch(
            "utils.llm_client.requests.post",
            side_effect=[requests.exceptions.ConnectionError("boom"), FakeResponse(200, {"ok": True})],
        ) as post:
            result = make_client().chat_completions(
                model="m", messages=[{"role": "user", "content": "hi"}]
            )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(post.call_count, 2)
        self.assertEqual(self.mock_sleep.call_count, 1)

    def test_retry_on_timeout_then_success(self):
        with mock.patch(
            "utils.llm_client.requests.post",
            side_effect=[requests.exceptions.Timeout("slow"), FakeResponse(200, {"ok": True})],
        ) as post:
            result = make_client().chat_completions(
                model="m", messages=[{"role": "user", "content": "hi"}]
            )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(post.call_count, 2)

    def test_client_error_is_not_retried(self):
        with mock.patch(
            "utils.llm_client.requests.post",
            return_value=FakeResponse(400, {"error": "bad request"}),
        ) as post:
            with self.assertRaises(requests.exceptions.HTTPError):
                make_client().chat_completions(
                    model="m", messages=[{"role": "user", "content": "hi"}]
                )
        self.assertEqual(post.call_count, 1)  # 4xx（除 429）不重试
        self.mock_sleep.assert_not_called()

    def test_exhausted_retries_raise_last_http_error(self):
        responses = [FakeResponse(429)] * 4
        with mock.patch(
            "utils.llm_client.requests.post", side_effect=responses
        ) as post:
            with self.assertRaises(requests.exceptions.HTTPError):
                make_client(max_retries=3).chat_completions(
                    model="m", messages=[{"role": "user", "content": "hi"}]
                )
        self.assertEqual(post.call_count, 4)  # 1 + max_retries
        self.assertEqual(self.mock_sleep.call_count, 3)

    def test_exhausted_retries_raise_last_connection_error(self):
        with mock.patch(
            "utils.llm_client.requests.post",
            side_effect=[requests.exceptions.ConnectionError("x")] * 4,
        ) as post:
            with self.assertRaises(requests.exceptions.ConnectionError):
                make_client(max_retries=3).chat_completions(
                    model="m", messages=[{"role": "user", "content": "hi"}]
                )
        self.assertEqual(post.call_count, 4)

    def test_max_retries_zero_means_single_attempt(self):
        with mock.patch(
            "utils.llm_client.requests.post",
            side_effect=requests.exceptions.ConnectionError("x"),
        ) as post:
            with self.assertRaises(requests.exceptions.ConnectionError):
                make_client(max_retries=0).chat_completions(
                    model="m", messages=[{"role": "user", "content": "hi"}]
                )
        self.assertEqual(post.call_count, 1)
        self.mock_sleep.assert_not_called()

    def test_backoff_is_capped(self):
        client = make_client(max_retries=10, backoff_s=1.0, backoff_factor=10.0)
        responses = [FakeResponse(500)] * 3 + [FakeResponse(200, {"ok": True})]
        with mock.patch(
            "utils.llm_client.requests.post", side_effect=responses
        ):
            client.chat_completions(
                model="m", messages=[{"role": "user", "content": "hi"}]
            )
        delays = [call.args[0] for call in self.mock_sleep.call_args_list]
        self.assertEqual(delays, [1.0, 10.0, 30.0])  # 100s 被压到上限 30s

    def test_retry_after_is_capped(self):
        responses = [
            FakeResponse(429, headers={"Retry-After": "120"}),
            FakeResponse(200, {"ok": True}),
        ]
        with mock.patch(
            "utils.llm_client.requests.post", side_effect=responses
        ):
            make_client().chat_completions(
                model="m", messages=[{"role": "user", "content": "hi"}]
            )
        self.mock_sleep.assert_called_once_with(30.0)  # MAX_BACKOFF_S 上限

    def test_invalid_retry_after_falls_back_to_backoff(self):
        responses = [
            FakeResponse(429, headers={"Retry-After": "not-a-number"}),
            FakeResponse(200, {"ok": True}),
        ]
        with mock.patch(
            "utils.llm_client.requests.post", side_effect=responses
        ):
            make_client().chat_completions(
                model="m", messages=[{"role": "user", "content": "hi"}]
            )
        self.mock_sleep.assert_called_once_with(1.0)  # 退化为指数退避

    def test_jitter_is_applied(self):
        self.mock_uniform.return_value = 0.5
        responses = [FakeResponse(500), FakeResponse(200, {"ok": True})]
        with mock.patch(
            "utils.llm_client.requests.post", side_effect=responses
        ):
            make_client(backoff_s=2.0).chat_completions(
                model="m", messages=[{"role": "user", "content": "hi"}]
            )
        self.mock_sleep.assert_called_once_with(1.0)  # 2.0 * 0.5


if __name__ == "__main__":
    unittest.main(verbosity=2)
