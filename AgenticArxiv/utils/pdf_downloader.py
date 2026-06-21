# AgenticArxiv/utils/pdf_downloader.py
from __future__ import annotations

import hashlib
import os
import re
import time
from typing import Tuple
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


def download_pdf(url: str, dest_path: str, timeout: Tuple[int, int] = (10, 120)) -> Tuple[int, str]:
    """
    下载 PDF 到 dest_path，使用 .part 临时文件，完成后原子替换
    返回 (size_bytes, sha256_hex)
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    tmp_path = dest_path + ".part"
    if os.path.exists(tmp_path):
        # 避免上次异常残留
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    sha = hashlib.sha256()
    size = 0

    headers = {"User-Agent": "AgenticArxiv/0.1 (+pdf downloader)"}
    with requests.get(url, stream=True, allow_redirects=True, headers=headers, timeout=timeout) as r:
        r.raise_for_status()

        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                sha.update(chunk)
                size += len(chunk)

    # 基础校验：PDF 魔数
    if size < 1024 or not _looks_like_pdf(tmp_path):
        # 保留 part 便于排查？这里直接删掉避免污染
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise RuntimeError("下载结果不像有效 PDF（content 可能是 HTML/重定向页/错误页）")

    os.replace(tmp_path, dest_path)
    return size, sha.hexdigest()
