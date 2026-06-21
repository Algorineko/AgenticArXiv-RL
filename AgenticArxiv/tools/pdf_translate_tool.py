# AgenticArxiv/tools/pdf_translate_tool.py
from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Any, Optional, Union, Callable

from tools.tool_registry import registry
from models.store import store
from models.schemas import PdfAsset, TranslateAsset
from utils.pdf_downloader import (
    normalize_arxiv_pdf_url,
    safe_filename,
    acquire_lock,
    release_lock,
    download_pdf,
)
from utils.pdf_translator import run_pdf2zh_translate
from config import settings


def _fallback_pdf_url(paper_id: str) -> str:
    return f"https://arxiv.org/pdf/{paper_id}.pdf"


def _ensure_pdf_downloaded_by_id(
    paper_id: str,
    pdf_url: Optional[str],
    force: bool,
) -> str:
    """
    不依赖 session 短期记忆：只靠 paper_id/pdf_url 确保 raw pdf 存在，并同步更新 pdf_cache.json
    """
    url = normalize_arxiv_pdf_url(pdf_url or _fallback_pdf_url(paper_id))
    local_path = os.path.join(settings.pdf_raw_path, safe_filename(paper_id) + ".pdf")

    existed = os.path.exists(local_path) and os.path.getsize(local_path) > 0
    asset = store.get_pdf_asset(paper_id)

    if existed and not force:
        if asset is None:
            asset = PdfAsset(
                paper_id=paper_id,
                pdf_url=url,
                local_path=local_path,
                status="READY",
                size_bytes=os.path.getsize(local_path),
                downloaded_at=datetime.now(),
            )
            store.upsert_pdf_asset(asset)
        elif asset.status != "READY":
            store.update_pdf_asset(
                paper_id,
                status="READY",
                local_path=local_path,
                pdf_url=url,
                size_bytes=os.path.getsize(local_path),
                downloaded_at=asset.downloaded_at or datetime.now(),
                error=None,
            )
        return local_path

    # 需要下载：加锁避免重复
    lock_path = local_path + ".lock"
    acquire_lock(lock_path)
    try:
        if asset is None:
            asset = PdfAsset(
                paper_id=paper_id,
                pdf_url=url,
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
                pdf_url=url,
                local_path=local_path,
                error=None,
            )

        size_bytes, sha256_hex = download_pdf(url, local_path)

        store.update_pdf_asset(
            paper_id,
            status="READY",
            size_bytes=size_bytes,
            sha256=sha256_hex,
            downloaded_at=datetime.now(),
            error=None,
        )
        return local_path
    except Exception as e:
        store.update_pdf_asset(paper_id, status="FAILED", error=str(e))
        raise
    finally:
        release_lock(lock_path)


def translate_arxiv_pdf(
    session_id: str = "default",
    ref: Union[str, int, None] = None,
    force: bool = False,
    service: str = None,
    threads: int = None,
    keep_dual: bool = False,
    # 依赖 session 的参数（用于 TranslateTask 执行）
    paper_id: Optional[str] = None,
    pdf_url: Optional[str] = None,
    input_pdf_path: Optional[str] = None,
    #  内部进度回调（异步任务用；HTTP 同步接口不传也不影响）
    progress_cb: Optional[Callable[[float, Optional[Dict[str, Any]]], None]] = None,
) -> Dict[str, Any]:
    """
    翻译 PDF(全文)，输出到 settings.pdf_translated_path
    默认只保留 mono(中文单语), dual 会被删除(除非 keep_dual=True)
    并维护 translate_cache.json

    支持 ref = None（JSON null）：表示“最近一次操作的论文”
    """
    service = service or settings.pdf2zh_service
    threads = int(threads or settings.pdf2zh_threads)

    def _emit(p: float, detail: Optional[Dict[str, Any]] = None) -> None:
        if not progress_cb:
            return
        try:
            if p < 0:
                p = 0.0
            if p > 1:
                p = 1.0
            progress_cb(float(p), detail or {})
        except Exception:
            pass

    _emit(0.0, {"stage": "prepare", "msg": "start resolve inputs"})

    # 允许三种入口：
    # 1) input_pdf_path
    # 2) paper_id/pdf_url
    # 3) ref（含 ref=null -> last_active）
    if ref is None and not paper_id and not input_pdf_path:
        last_id = store.get_last_active_paper_id(session_id)
        if not last_id:
            raise ValueError("未找到指代对象：请先下载/翻译/查状态某篇论文，或明确提供 ref（序号/id/标题）")
        paper_id = last_id

        pdf_asset = store.get_pdf_asset(paper_id)
        if pdf_asset and pdf_asset.pdf_url:
            pdf_url = pdf_asset.pdf_url
        else:
            paper = store.resolve_paper(session_id, paper_id)
            pdf_url = (paper.pdf_url if paper else None) or _fallback_pdf_url(paper_id)

    # 1) 确定 paper_id / pdf_url / input_pdf_path
    if input_pdf_path and os.path.exists(input_pdf_path):
        in_path = input_pdf_path
        if not paper_id:
            paper_id = os.path.splitext(os.path.basename(in_path))[0]
        _emit(0.0, {"stage": "pdf_ready", "msg": "use local input_pdf_path"})
    else:
        if not paper_id:
            paper = store.resolve_paper(session_id, ref)
            if paper is None:
                raise ValueError(
                    "未找到论文：请先调用 /arxiv/recent 写入 session 记忆，或传 paper_id/pdf_url/input_pdf_path"
                )
            paper_id = paper.id
            pdf_url = paper.pdf_url or _fallback_pdf_url(paper_id)

        in_path = _ensure_pdf_downloaded_by_id(paper_id, pdf_url, force=force)

    store.set_last_active_paper_id(session_id, paper_id)

    # 2) 输出路径
    os.makedirs(settings.pdf_translated_path, exist_ok=True)
    mono_path = os.path.join(settings.pdf_translated_path, f"{paper_id}-mono.pdf")
    dual_path = os.path.join(settings.pdf_translated_path, f"{paper_id}-dual.pdf")
    os.makedirs(settings.pdf_translated_log_path, exist_ok=True)
    log_path = os.path.join(settings.pdf_translated_log_path, f"{paper_id}.pdf2zh.log")

    existed = os.path.exists(mono_path) and os.path.getsize(mono_path) > 0
    asset = store.get_translate_asset(paper_id)

    # 3) 若已存在且不强制，直接返回（异步任务跳过：runner 已有 fast-path）
    if existed and not force and not progress_cb:
        if asset is None:
            asset = TranslateAsset(
                paper_id=paper_id,
                input_pdf_path=in_path,
                output_mono_path=mono_path,
                output_dual_path=dual_path if (keep_dual and os.path.exists(dual_path)) else None,
                status="READY",
                service=service,
                threads=threads,
                translated_at=datetime.now(),
                error=None,
            )
            store.upsert_translate_asset(asset)
        elif asset.status != "READY":
            store.update_translate_asset(
                paper_id,
                status="READY",
                input_pdf_path=in_path,
                output_mono_path=mono_path,
                output_dual_path=dual_path if (keep_dual and os.path.exists(dual_path)) else None,
                service=service,
                threads=threads,
                translated_at=asset.translated_at or datetime.now(),
                error=None,
            )
        _emit(1.0, {"stage": "done", "msg": "cache hit"})
        return {
            "session_id": session_id,
            "paper_id": paper_id,
            "input_pdf_path": in_path,
            "output_pdf_path": mono_path,
            "status": "READY",
            "existed": True,
            "service": service,
            "threads": threads,
            "log_path": log_path,
        }

    # 4) 翻译：加锁避免并发重复翻译
    lock_path = mono_path + ".lock"
    acquire_lock(lock_path)
    try:
        if asset is None:
            asset = TranslateAsset(
                paper_id=paper_id,
                input_pdf_path=in_path,
                output_mono_path=mono_path,
                output_dual_path=None,
                status="TRANSLATING",
                service=service,
                threads=threads,
                translated_at=None,
                error=None,
            )
            store.upsert_translate_asset(asset)
        else:
            store.update_translate_asset(
                paper_id,
                status="TRANSLATING",
                input_pdf_path=in_path,
                output_mono_path=mono_path,
                service=service,
                threads=threads,
                error=None,
            )

        _emit(0.0, {"stage": "translating", "msg": "pdf2zh start"})

        def _on_pdf2zh(p: float, detail: Optional[Dict[str, Any]] = None) -> None:
            _emit(p, detail)

        res = run_pdf2zh_translate(
            pdf2zh_bin=settings.pdf2zh_bin,
            input_pdf=in_path,
            out_dir=settings.pdf_translated_path,
            service=service,
            threads=threads,
            keep_dual=keep_dual,
            log_path=log_path,
            progress_cb=_on_pdf2zh,
        )

        _emit(1.0, {"stage": "finalizing", "msg": "rename outputs"})

        # pdf2zh 实际输出名可能是 stem-mono / stem-zh，这里统一成 {paper_id}-mono.pdf
        if res.mono_path != mono_path:
            if os.path.exists(mono_path):
                os.remove(mono_path)
            os.replace(res.mono_path, mono_path)

        # dual：如果保留且存在，也统一命名
        kept_dual = None
        if keep_dual and res.dual_path and os.path.exists(res.dual_path):
            kept_dual = dual_path
            if res.dual_path != dual_path:
                if os.path.exists(dual_path):
                    os.remove(dual_path)
                os.replace(res.dual_path, dual_path)

        store.update_translate_asset(
            paper_id,
            status="READY",
            output_mono_path=mono_path,
            output_dual_path=kept_dual,
            translated_at=datetime.now(),
            error=None,
        )

        return {
            "session_id": session_id,
            "paper_id": paper_id,
            "input_pdf_path": in_path,
            "output_pdf_path": mono_path,
            "status": "READY",
            "existed": False,
            "service": service,
            "threads": threads,
            "log_path": log_path,
        }

    except Exception as e:
        store.update_translate_asset(paper_id, status="FAILED", error=str(e))
        raise
    finally:
        release_lock(lock_path)


PDF_TRANSLATE_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "session_id": {"type": "string", "default": "default"},
        "ref": {
            "description": "论文引用：1-based序号 或 arxiv id 或 title子串；也支持 null 表示最近一次操作的论文",
            "anyOf": [{"type": "integer"}, {"type": "string"}, {"type": "null"}],
        },
        "force": {"type": "boolean", "default": False},
        "service": {"type": "string", "description": "翻译服务，如 bing/deepl/google", "default": "bing"},
        "threads": {"type": "integer", "description": "线程数", "default": 4, "minimum": 1, "maximum": 32},
        "keep_dual": {"type": "boolean", "description": "是否保留双语 PDF", "default": False},
        "paper_id": {"type": "string", "description": "可选：直接指定 paper_id（不依赖 session）"},
        "pdf_url": {"type": "string", "description": "可选：直接指定 pdf_url（不依赖 session）"},
        "input_pdf_path": {"type": "string", "description": "可选：直接指定本地 PDF 路径（不依赖 session）"},
    },
    "required": [],
}

registry.register_tool(
    name="translate_arxiv_pdf",
    description="调用 pdf2zh 翻译 PDF 全文，默认只保留中文单语 mono PDF，并维护 translate_cache.json 索引（ref 支持 null 指代最近操作论文）",
    parameter_schema=PDF_TRANSLATE_TOOL_SCHEMA,
    func=translate_arxiv_pdf,
)