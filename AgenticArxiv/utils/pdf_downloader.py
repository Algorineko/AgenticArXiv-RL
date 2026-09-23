# AgenticArxiv/utils/pdf_downloader.py
from __future__ import annotations

import hashlib
import os
import re
import time
from typing import Optional, Tuple
from urllib.parse import urlparse, urlunparse

import requests


_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str) -> str:
    return _SAFE_FILENAME_RE.sub("_", name).strip("_")


def normalize_arxiv_pdf_url(url: str) -> str:
    """
    arXiv 的 pdf_url 有时不带 .pdf，统一归一化为带 .pdf 的路径
    """
    u = (url or "").strip()
    if not u:
        raise ValueError("pdf_url 为空")
    p = urlparse(u)
    path = p.path.rstrip("/")
    if not path.endswith(".pdf"):
        path = path + ".pdf"
    return urlunparse(p._replace(path=path))


def acquire_lock(lock_path: str, retries: int = 150, delay_s: float = 0.2) -> None:
    """
    简单文件锁：创建 lock 文件，存在则等待一会儿重试
    """
    for _ in range(retries):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return
        except FileExistsError:
            time.sleep(delay_s)
    raise RuntimeError(f"获取下载锁失败: {lock_path}（可能有其他下载在进行）")


def release_lock(lock_path: str) -> None:
    try:
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except Exception:
        pass


def _looks_like_pdf(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(5)
        return head.startswith(b"%PDF")
    except Exception:
        return False


#: 读超时是「两次读到字节之间的间隔」而不是总时长，所以它可以压得比总量级
#: 低得多：一条健康的连接不会几十秒一个字节都没有。arXiv 会间歇性地让连接
#: 悬住（实测同一批 URL 里有的 4.4MB/s 传完、有的完全不动），把 120s 缩短到
#: 30s 能让每次挂起少等 90 秒。
DEFAULT_READ_TIMEOUT_S = 30
DEFAULT_DOWNLOAD_RETRIES = 3


def _download_once(
    url: str, tmp_path: str, timeout: Tuple[int, int]
) -> Tuple[int, str]:
    """One transfer attempt into *tmp_path*; validates the PDF magic number."""
    sha = hashlib.sha256()
    size = 0

    headers = {"User-Agent": "AgenticArxiv/0.1 (+pdf downloader)"}
    with requests.get(
        url, stream=True, allow_redirects=True, headers=headers, timeout=timeout
    ) as response:
        response.raise_for_status()

        with open(tmp_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                sha.update(chunk)
                size += len(chunk)

    if size < 1024 or not _looks_like_pdf(tmp_path):
        raise RuntimeError(
            "下载结果不像有效 PDF（content 可能是 HTML/重定向页/错误页）"
        )

    return size, sha.hexdigest()


def download_pdf(
    url: str,
    dest_path: str,
    timeout: Tuple[int, int] = (10, DEFAULT_READ_TIMEOUT_S),
    retries: int = DEFAULT_DOWNLOAD_RETRIES,
) -> Tuple[int, str]:
    """
    下载 PDF 到 dest_path，使用 .part 临时文件，完成后原子替换
    返回 (size_bytes, sha256_hex)

    arXiv 的 PDF 端点是间歇性不稳定的：同一批并发请求里，一部分以 MB/s 传完，
    另一部分长时间一个字节都不回。没有重试时，一次抖动就等于这篇论文永久缺失，
    而 build_snapshot 是整条离线流水线唯一联网的一步 —— 缺一篇就要重跑一次。
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    tmp_path = dest_path + ".part"
    attempts = max(1, int(retries))
    last_error: Optional[Exception] = None

    for attempt in range(attempts):
        try:
            if os.path.exists(tmp_path):
                # 避免上次异常残留
                os.remove(tmp_path)
            size, digest = _download_once(url, tmp_path, timeout)
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)
            continue

        os.replace(tmp_path, dest_path)
        return size, digest

    # 全部重试都失败：清掉残片，让调用方看到的是「这篇没下下来」而不是半个文件
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except OSError:
        pass

    raise RuntimeError(f"PDF 下载失败（已重试 {attempts} 次）: {last_error}") from last_error
