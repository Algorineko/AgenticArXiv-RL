"""PDF 下载的重试契约。

arXiv 的 PDF 端点会间歇性悬住：同一批并发请求里有的以 MB/s 传完，有的长时间
一个字节都不回。build_snapshot 是整条离线流水线唯一联网的一步，一次抖动让
一篇论文永久缺失，就得重跑一次全量预取。
"""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from utils.pdf_downloader import (
    DEFAULT_DOWNLOAD_RETRIES,
    download_pdf,
    normalize_arxiv_pdf_url,
    safe_filename,
)

_PDF_BYTES = b"%PDF-1.4\n" + b"x" * 4096 + b"\n%%EOF\n"


class _FakeResponse:
    def __init__(self, payload: bytes = _PDF_BYTES):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=1024):
        yield self.payload


class DownloadRetryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dest = Path(self.tmp.name) / "paper.pdf"

    def tearDown(self):
        self.tmp.cleanup()

    def test_success_on_the_first_attempt_does_not_retry(self):
        with mock.patch(
            "utils.pdf_downloader.requests.get", return_value=_FakeResponse()
        ) as get:
            size, sha = download_pdf("https://example.invalid/p.pdf", str(self.dest))

        self.assertEqual(get.call_count, 1)
        self.assertEqual(size, len(_PDF_BYTES))
        self.assertEqual(sha, hashlib.sha256(_PDF_BYTES).hexdigest())
        self.assertEqual(self.dest.read_bytes(), _PDF_BYTES)
        self.assertFalse(Path(str(self.dest) + ".part").exists())

    def test_a_transient_failure_is_retried_and_succeeds(self):
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("connection reset")
            return _FakeResponse()

        with mock.patch("utils.pdf_downloader.requests.get", side_effect=flaky), \
             mock.patch("utils.pdf_downloader.time.sleep"):
            size, _ = download_pdf("https://example.invalid/p.pdf", str(self.dest))

        self.assertEqual(calls["n"], 2)
        self.assertEqual(size, len(_PDF_BYTES))
        self.assertTrue(self.dest.exists())

    def test_persistent_failure_raises_and_leaves_no_partial_file(self):
        with mock.patch(
            "utils.pdf_downloader.requests.get",
            side_effect=ConnectionError("hang"),
        ) as get, mock.patch("utils.pdf_downloader.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "PDF 下载失败"):
                download_pdf("https://example.invalid/p.pdf", str(self.dest))

        self.assertEqual(get.call_count, DEFAULT_DOWNLOAD_RETRIES)
        self.assertFalse(self.dest.exists())
        self.assertFalse(Path(str(self.dest) + ".part").exists())

    def test_a_non_pdf_body_is_rejected_after_retrying(self):
        """重定向到错误页时会拿到一份 HTML，它必须在重试后仍然被拒绝。"""
        with mock.patch(
            "utils.pdf_downloader.requests.get",
            return_value=_FakeResponse(b"<html>not a pdf</html>"),
        ) as get, mock.patch("utils.pdf_downloader.time.sleep"):
            with self.assertRaises(RuntimeError):
                download_pdf("https://example.invalid/p.pdf", str(self.dest))

        self.assertEqual(get.call_count, DEFAULT_DOWNLOAD_RETRIES)
        self.assertFalse(self.dest.exists())

    def test_read_timeout_stays_bounded(self):
        from utils.pdf_downloader import DEFAULT_READ_TIMEOUT_S

        self.assertGreater(DEFAULT_READ_TIMEOUT_S, 0)
        self.assertLess(DEFAULT_READ_TIMEOUT_S, 120)


class UrlHelpersTest(unittest.TestCase):
    def test_normalize_appends_the_pdf_suffix(self):
        self.assertEqual(
            normalize_arxiv_pdf_url("https://arxiv.org/pdf/2601.00001v1"),
            "https://arxiv.org/pdf/2601.00001v1.pdf",
        )

    def test_safe_filename_strips_path_separators(self):
        self.assertEqual(safe_filename("2601.00001/v1"), "2601.00001_v1")


if __name__ == "__main__":
    unittest.main()
