# AgenticArxiv/tools/pdf_download_tool.py
from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Any, Optional, Union

from tools.tool_registry import registry
from models.store import store
from models.schemas import Paper
from models.schemas import PdfAsset
from utils.pdf_downloader import (
    normalize_arxiv_pdf_url,
    safe_filename,
    acquire_lock,
    release_lock,
    download_pdf,
)
from config import settings


def _fallback_pdf_url(paper_id: str) -> str:
    return f"https://arxiv.org/pdf/{paper_id}.pdf"


def download_arxiv_pdf(
    session_id: str = "default",
    ref: Union[str, int, None] = 1,
    force: bool = False,
) -> Dict[str, Any]:
    """
    按 session 的短期记忆 + ref(1-based/arxiv id/title子串) 下载 PDF 到 settings.pdf_raw_path
    支持 ref = None（JSON null）：表示“最近一次操作的论文”
    文件命名：{paper.id}.pdf （canonical）
    同时更新 output/pdf_cache.json
    """
    paper: Optional[Paper] = None
    paper_id: Optional[str] = None
    pdf_url: Optional[str] = None

    if ref is None:
        # 指代：最近一次操作的论文
        last_id = store.get_last_active_paper_id(session_id)
        if not last_id:
            raise ValueError("未找到指代对象：请先下载/翻译/查状态某篇论文，或明确提供 ref（序号/id/标题）")
        paper_id = last_id
        paper = store.resolve_paper(session_id, paper_id)  # 可能为 None（last_papers 过期）
        pdf_url = (paper.pdf_url if paper else None) or _fallback_pdf_url(paper_id)
    else:
        paper = store.resolve_paper(session_id, ref)
        if paper is None:
            raise ValueError("未找到论文：请先调用 /arxiv/recent 写入 session 记忆，或检查 ref 是否正确")
        paper_id = paper.id
        pdf_url = paper.pdf_url or _fallback_pdf_url(paper_id)

    # 一旦确定 paper_id，就更新 last_active（不区分下载/翻译，只要“操作了”就更新）
    store.set_last_active_paper_id(session_id, paper_id)

    pdf_url = normalize_arxiv_pdf_url(pdf_url)

    filename = safe_filename(paper_id) + ".pdf"
    local_path = os.path.join(settings.pdf_raw_path, filename)

    # 1) 若已存在且不强制，直接返回 READY
    existed = os.path.exists(local_path) and os.path.getsize(local_path) > 0
    asset = store.get_pdf_asset(paper_id)

    if existed and not force:
        if asset is None:
            asset = PdfAsset(
                paper_id=paper_id,
                pdf_url=pdf_url,
                local_path=local_path,
                status="READY",
                size_bytes=os.path.getsize(local_path),
                downloaded_at=datetime.now(),
            )
            store.upsert_pdf_asset(asset)
        else:
            # 如果索引里不是 READY，纠正一下
            if asset.status != "READY":
                store.update_pdf_asset(
                    paper_id,
                    status="READY",
                    local_path=local_path,
                    pdf_url=pdf_url,
                    size_bytes=os.path.getsize(local_path),
                    downloaded_at=asset.downloaded_at or datetime.now(),
                    error=None,
                )
        return {
            "session_id": session_id,
            "paper_id": paper_id,
            "pdf_url": pdf_url,
            "local_path": local_path,
            "status": "READY",
            "existed": True,
        }

    # 2) 需要下载：加锁避免并发重复下载
    lock_path = local_path + ".lock"
    acquire_lock(lock_path)

    try:
        # 标记 DOWNLOADING
        if asset is None:
            asset = PdfAsset(
                paper_id=paper_id,
                pdf_url=pdf_url,
                local_path=local_path,
                status="DOWNLOADING",
                size_bytes=0,
                sha256=None,
                downloaded_at=None,
                error=None,
            )
            store.upsert_pdf_asset(asset)
        else:
            store.update_pdf_asset(
                paper_id,
                status="DOWNLOADING",
                pdf_url=pdf_url,
                local_path=local_path,
                error=None,
            )

        # 真正下载
        size_bytes, sha256_hex = download_pdf(pdf_url, local_path)

        # 标记 READY
        store.update_pdf_asset(
            paper_id,
            status="READY",
            size_bytes=size_bytes,
            sha256=sha256_hex,
            downloaded_at=datetime.now(),
            error=None,
        )

        return {
            "session_id": session_id,
            "paper_id": paper_id,
            "pdf_url": pdf_url,
            "local_path": local_path,
            "status": "READY",
            "existed": False,
            "size_bytes": size_bytes,
            "sha256": sha256_hex,
        }

    except Exception as e:
        store.update_pdf_asset(paper_id, status="FAILED", error=str(e))
        raise
    finally:
        release_lock(lock_path)


PDF_DOWNLOAD_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "session_id": {
            "type": "string",
            "description": "会话ID，用于从短期记忆中解析 ref 或读取 last_active_paper_id",
            "default": "default",
        },
        "ref": {
            "description": "论文引用：1-based序号 或 arxiv id 或 title子串；也支持 null 表示最近一次操作的论文",
            "anyOf": [{"type": "integer"}, {"type": "string"}, {"type": "null"}],
        },
        "force": {
            "type": "boolean",
            "description": "是否强制重新下载（覆盖本地文件）",
            "default": False,
        },
    },
    "required": [],
}

registry.register_tool(
    name="download_arxiv_pdf",
    description="根据 session 的短期记忆和 ref 下载 arXiv 论文 PDF，并维护 pdf_cache.json 索引（ref 支持 null 指代最近操作论文）",
    parameter_schema=PDF_DOWNLOAD_TOOL_SCHEMA,
    func=download_arxiv_pdf,
)