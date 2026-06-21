# AgenticArxiv/api/endpoints.py
from __future__ import annotations

import json
import os
import queue
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Literal

from tools.tool_registry import registry
from models.schemas import (
    Paper, TranslateTask, PdfAsset, TranslateAsset,
    ChatLogItem, AgentStepItem, LogSessionSummary,
)
from models.store import store
from utils.logger import log

from services.runtime import event_bus, translate_runner
from services.log_service import log_service
from config import settings


router = APIRouter()


class HealthResponse(BaseModel):
    status: str = "ok"


class ToolInfo(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class ListToolsResponse(BaseModel):
    tools: List[ToolInfo]


class ExecuteToolRequest(BaseModel):
    name: str = Field(..., description="工具名称")
    args: Dict[str, Any] = Field(default_factory=dict, description="工具参数(JSON对象)")


class ExecuteToolResponse(BaseModel):
    name: str
    result: Any


class ArxivRecentRequest(BaseModel):
    session_id: str = Field(default="default", description="会话ID，用于短期记忆")
    max_results: int = Field(default=15, ge=1, le=100)
    aspect: str = Field(default="*", description="cs子领域，如 AI/LG/CL/... 或 *")
    days: int = Field(default=7, ge=1, le=30)
    output_path: Optional[str] = Field(
        default=None,
        description="可选：保存到指定文件路径；不传则用项目 output/recent_cs_papers.txt",
    )
    save_to_file: bool = Field(default=True, description="是否写入到文件")


class ArxivRecentResponse(BaseModel):
    session_id: str
    count: int
    papers: List[Paper]


class SessionPapersResponse(BaseModel):
    session_id: str
    papers: List[Paper]


class CreateTranslateTaskRequest(BaseModel):
    session_id: str = "default"
    ref: str | int | None = Field(
        default=None,
        description="论文引用: 1-based序号 或 arxiv id 或 title子串; 也支持 null 表示最近一次操作的论文",
    )
    force: bool = Field(default=False, description="是否强制重新翻译")
    service: str = Field(default="bing", description="翻译服务，如 bing/deepl/google")
    threads: int = Field(default=4, ge=1, le=32, description="线程数")
    keep_dual: bool = Field(
        default=False, description="是否保留双语 PDF（默认否，只保留中文）"
    )


class CreateTranslateTaskResponse(BaseModel):
    task: TranslateTask


class GetTaskResponse(BaseModel):
    task: TranslateTask


class AgentRunRequest(BaseModel):
    session_id: str = "default"
    task: str = Field(..., description="给Agent的任务描述")
    agent_model: Optional[str] = Field(default=None, description="可选：指定模型名")


class AgentRunResponse(BaseModel):
    task: str
    history: List[Dict[str, str]]
    final_observation: str


class DownloadPdfRequest(BaseModel):
    session_id: str = Field(default="default", description="会话ID，用于短期记忆")
    ref: str | int | None = Field(
        default=None,
        description="论文引用: 1-based序号 或 arxiv id 或 title子串; 也支持 null 表示最近一次操作的论文",
    )
    force: bool = Field(default=False, description="是否强制重新下载")


class DownloadPdfResponse(BaseModel):
    session_id: str
    paper_id: str
    pdf_url: str
    local_path: str
    status: str
    existed: bool
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None


class TranslatePdfRequest(BaseModel):
    session_id: str = Field(default="default", description="会话ID, 用于短期记忆")
    ref: str | int | None = Field(
        default=None,
        description="论文引用:1-based序号 或 arxiv id 或 title子串; 也支持 null 表示最近一次操作的论文",
    )
    force: bool = Field(default=False, description="是否强制重新翻译（覆盖本地 mono）")
    service: str = Field(default="bing", description="翻译服务，如 bing/deepl/google")
    threads: int = Field(
        default=4, ge=1, le=32, description="线程数（服务器 4 核可设 4）"
    )
    keep_dual: bool = Field(
        default=False, description="是否保留双语 PDF（默认否，只保留中文）"
    )


class TranslatePdfResponse(BaseModel):
    session_id: str
    paper_id: str
    input_pdf_path: str
    output_pdf_path: str
    status: str
    existed: bool
    service: str
    threads: int
    log_path: Optional[str] = None


class ChatRequest(BaseModel):
    session_id: str = Field(default="default")
    message: str = Field(..., description="用户对 Agent 的一句话/一个任务")
    agent_model: Optional[str] = None
    agent_type: str = Field(
        default="regex",
        description="Agent 架构: regex | mcp | skill_cli",
    )


class ChatResponse(BaseModel):
    session_id: str
    message: str
    reply: str
    history: List[Dict[str, str]]
    papers: List[Paper]
    pdf_assets: List[PdfAsset]
    translate_assets: List[TranslateAsset]
    tasks: List[TranslateTask] = Field(
        default_factory=list, description="该 session 的最近任务（可选）"
    )
    agent_type: str = Field(default="regex", description="使用的 Agent 架构")


class PdfAssetsResponse(BaseModel):
    assets: List[PdfAsset]


class TranslateAssetsResponse(BaseModel):
    assets: List[TranslateAsset]


DeleteKind = Literal["pdf", "translate"]


class DeleteAssetResponse(BaseModel):
    kind: DeleteKind
    paper_id: str
    removed_cache: bool
    deleted_files: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


# ---------------- SSE ----------------
def _sse_pack(event_json: str) -> str:
    try:
        obj = json.loads(event_json)
        et = obj.get("type", "message")
    except Exception:
        et = "message"
    return f"event: {et}\ndata: {event_json}\n\n"


@router.get("/events")
def events(session_id: str = Query(default="default")):
    sid = session_id or "default"
    sub_id, q = event_bus.subscribe(sid)

    def gen():
        try:
            hello = json.dumps(
                {"type": "connected", "session_id": sid}, ensure_ascii=False
            )
            yield _sse_pack(hello)

            while True:
                try:
                    data = q.get(timeout=15)
                    yield _sse_pack(data)
                except queue.Empty:
                    yield ": ping\n\n"
        finally:
            event_bus.unsubscribe(sid, sub_id)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


# ---------------- helpers ----------------
def _is_under_root(path: str, root: str) -> bool:
    ap = os.path.abspath(path)
    ar = os.path.abspath(root)
    try:
        return os.path.commonpath([ap, ar]) == ar
    except Exception:
        return False


def _inline_pdf_response(file_path: str, filename: str) -> FileResponse:
    """
    返回浏览器原生 inline 预览的 PDF 响应。
    - FileResponse (Starlette) 通常支持 Range 请求，翻页/跳页体验更好。
    """
    # RFC 5987 filename* (UTF-8)
    safe_name = filename or os.path.basename(file_path)
    disp = f"inline; filename*=UTF-8''{quote(safe_name)}"
    headers = {
        "Content-Disposition": disp,
        "Cache-Control": "no-store",
    }
    return FileResponse(file_path, media_type="application/pdf", headers=headers)


def _has_active_task_for_paper(paper_id: str) -> bool:
    tasks = store.list_tasks(session_id=None, limit=10000)
    for t in tasks:
        if t.paper_id == paper_id and t.status in ("PENDING", "RUNNING"):
            return True
    return False


def _safe_remove_file(
    path: Optional[str], root: str, deleted: List[str], warnings: List[str]
) -> None:
    if not path:
        return
    ap = os.path.abspath(path)
    if not _is_under_root(ap, root):
        raise HTTPException(
            status_code=400, detail=f"拒绝删除：路径不在允许目录内: {ap}"
        )
    if not os.path.exists(ap):
        return
    try:
        if os.path.isdir(ap):
            raise HTTPException(
                status_code=400, detail=f"拒绝删除目录（仅允许文件）: {ap}"
            )
        os.remove(ap)
        deleted.append(ap)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文件失败: {ap}, err={e}")


# ---------------- basic ----------------
@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/tools", response_model=ListToolsResponse)
def list_tools() -> ListToolsResponse:
    tools = registry.list_tools()
    return ListToolsResponse(tools=[ToolInfo(**t) for t in tools])


@router.post("/tools/execute", response_model=ExecuteToolResponse)
def execute_tool(req: ExecuteToolRequest) -> ExecuteToolResponse:
    try:
        result = registry.execute_tool(req.name, req.args)
        return ExecuteToolResponse(name=req.name, result=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"工具执行失败: {str(e)}")


@router.post("/arxiv/recent", response_model=ArxivRecentResponse)
def arxiv_recent(req: ArxivRecentRequest) -> ArxivRecentResponse:
    tool_name = "get_recently_submitted_cs_papers"

    args: Dict[str, Any] = {
        "max_results": req.max_results,
        "aspect": req.aspect,
        "days": req.days,
        "save_to_file": req.save_to_file,
    }
    if req.output_path is not None:
        args["output_path"] = req.output_path

    try:
        result = registry.execute_tool(tool_name, args)
        if not isinstance(result, list):
            raise RuntimeError(f"工具返回类型异常: {type(result)}")

        papers_obj = [Paper(**p) for p in result]
        store.set_last_papers(req.session_id, papers_obj)

        return ArxivRecentResponse(
            session_id=req.session_id,
            count=len(papers_obj),
            papers=papers_obj,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取论文失败: {str(e)}")


@router.get("/sessions/{session_id}/papers", response_model=SessionPapersResponse)
def get_session_papers(session_id: str) -> SessionPapersResponse:
    papers = store.get_last_papers(session_id)
    return SessionPapersResponse(session_id=session_id, papers=papers)


# ---------------- translate tasks (create -> auto-run) ----------------
@router.post("/translate/tasks", response_model=CreateTranslateTaskResponse)
def create_translate_task(
    req: CreateTranslateTaskRequest,
) -> CreateTranslateTaskResponse:
    try:
        t = translate_runner.enqueue(
            session_id=req.session_id,
            ref=req.ref,
            force=req.force,
            service=req.service,
            threads=req.threads,
            keep_dual=req.keep_dual,
        )
        return CreateTranslateTaskResponse(task=t)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建翻译任务失败: {str(e)}")


@router.get("/translate/tasks/{task_id}", response_model=GetTaskResponse)
def get_translate_task(task_id: str) -> GetTaskResponse:
    t = store.get_task(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return GetTaskResponse(task=t)


# ---------------- agent ----------------
@router.post("/agent/run", response_model=AgentRunResponse)
def run_agent(req: AgentRunRequest) -> AgentRunResponse:
    try:
        from utils.llm_client import get_env_llm_client
        from agents.agent_engine import ReActAgent

        llm_client = get_env_llm_client()
        agent = ReActAgent(llm_client)
        result = agent.run(
            task=req.task, agent_model=req.agent_model, session_id=req.session_id
        )
        return AgentRunResponse(**result)
    except Exception as e:
        log.error(f"Agent运行失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent运行失败: {str(e)}")


# ---------------- pdf download ----------------
@router.post("/pdf/download", response_model=DownloadPdfResponse)
def pdf_download(req: DownloadPdfRequest) -> DownloadPdfResponse:
    tool_name = "download_arxiv_pdf"
    try:
        result = registry.execute_tool(
            tool_name,
            {"session_id": req.session_id, "ref": req.ref, "force": req.force},
        )
        if not isinstance(result, dict):
            raise RuntimeError(f"工具返回类型异常: {type(result)}")
        return DownloadPdfResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"下载PDF失败: {str(e)}")


# ---------------- pdf translate (sync 保留) ----------------
@router.post("/pdf/translate", response_model=TranslatePdfResponse)
def pdf_translate(req: TranslatePdfRequest) -> TranslatePdfResponse:
    tool_name = "translate_arxiv_pdf"
    try:
        result = registry.execute_tool(
            tool_name,
            {
                "session_id": req.session_id,
                "ref": req.ref,
                "force": req.force,
                "service": req.service,
                "threads": req.threads,
                "keep_dual": req.keep_dual,
            },
        )
        if not isinstance(result, dict):
            raise RuntimeError(f"工具返回类型异常: {type(result)}")
        return TranslatePdfResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"翻译PDF失败: {str(e)}")


@router.post("/pdf/translate/async", response_model=CreateTranslateTaskResponse)
def pdf_translate_async(req: TranslatePdfRequest) -> CreateTranslateTaskResponse:
    try:
        t = translate_runner.enqueue(
            session_id=req.session_id,
            ref=req.ref,
            force=req.force,
            service=req.service,
            threads=req.threads,
            keep_dual=req.keep_dual,
        )
        return CreateTranslateTaskResponse(task=t)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建异步翻译任务失败: {str(e)}")


# ---------------- chat ----------------
@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        from utils.llm_client import get_env_llm_client

        llm_client = get_env_llm_client()

        if req.agent_type == "mcp":
            from mcp_protocol.mcp_agent import MCPAgent
            agent = MCPAgent(llm_client)
        elif req.agent_type == "skill_cli":
            from skill_cli.skill_agent import SkillAgent
            agent = SkillAgent(llm_client)
        else:
            from agents.agent_engine import ReActAgent
            agent = ReActAgent(llm_client)

        result = agent.run(
            task=req.message, agent_model=req.agent_model, session_id=req.session_id
        )

        papers = store.get_last_papers(req.session_id)
        pdf_assets = store.list_pdf_assets()
        translate_assets = store.list_translate_assets()
        tasks = store.list_tasks(session_id=req.session_id, limit=50)

        reply = ""
        for step in reversed(result.get("history", [])):
            if step.get("action") not in ("FINISH", "FORCE_STOP", "ERROR"):
                reply = step.get("observation", "")
                break
        reply = reply or result.get("final_observation", "")

        return ChatResponse(
            session_id=req.session_id,
            message=req.message,
            reply=reply,
            history=result.get("history", []),
            papers=papers,
            pdf_assets=list(pdf_assets),
            translate_assets=list(translate_assets),
            tasks=tasks,
            agent_type=req.agent_type,
        )
    except Exception as e:
        log.error(f"/chat 失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"chat失败: {str(e)}")


@router.get("/pdf/assets", response_model=PdfAssetsResponse)
def list_pdf_assets() -> PdfAssetsResponse:
    return PdfAssetsResponse(assets=store.list_pdf_assets())


@router.get("/translate/assets", response_model=TranslateAssetsResponse)
def list_translate_assets() -> TranslateAssetsResponse:
    return TranslateAssetsResponse(assets=store.list_translate_assets())


# ---------------- PDF view endpoints ----------------
@router.get("/pdf/view/raw/{paper_id}")
def view_raw_pdf(
    paper_id: str,
    session_id: str = Query(
        default="default", description="用于更新 last_active（可选）"
    ),
):
    """
    预览已下载 raw PDF（浏览器原生 inline 打开）
    """
    sid = session_id or "default"

    asset = store.get_pdf_asset(paper_id)
    # 允许索引缺失但文件存在：fallback 到 canonical 路径
    canonical = os.path.join(settings.pdf_raw_path, f"{paper_id}.pdf")
    path = asset.local_path if asset and asset.local_path else canonical

    if asset and asset.status != "READY":
        raise HTTPException(
            status_code=409, detail=f"PDF 未处于 READY，当前状态={asset.status}"
        )

    if not path or not os.path.exists(path) or os.path.getsize(path) <= 0:
        raise HTTPException(status_code=404, detail="raw PDF 文件不存在")

    if not _is_under_root(path, settings.pdf_raw_path):
        raise HTTPException(status_code=400, detail="拒绝预览：文件不在允许目录内")

    # 更新 last_active（预览也算一次操作）
    store.set_last_active_paper_id(sid, paper_id)

    filename = f"{paper_id}.pdf"
    return _inline_pdf_response(path, filename)


@router.get("/pdf/view/translated/{paper_id}")
def view_translated_pdf(
    paper_id: str,
    variant: Literal["mono", "dual"] = Query(default="mono"),
    session_id: str = Query(
        default="default", description="用于更新 last_active（可选）"
    ),
):
    """
    预览已翻译 PDF（默认 mono；若保留 dual 可用 variant=dual）
    """
    sid = session_id or "default"

    asset = store.get_translate_asset(paper_id)
    if asset and asset.status != "READY":
        raise HTTPException(
            status_code=409, detail=f"翻译未处于 READY，当前状态={asset.status}"
        )

    # 路径选择：优先读索引；索引缺失时 fallback 到 canonical
    mono_canonical = os.path.join(settings.pdf_translated_path, f"{paper_id}-mono.pdf")
    dual_canonical = os.path.join(settings.pdf_translated_path, f"{paper_id}-dual.pdf")

    if variant == "dual":
        path = (
            asset.output_dual_path
            if asset and asset.output_dual_path
            else dual_canonical
        )
        filename = f"{paper_id}-dual.pdf"
    else:
        path = (
            asset.output_mono_path
            if asset and asset.output_mono_path
            else mono_canonical
        )
        filename = f"{paper_id}-mono.pdf"

    if not path or not os.path.exists(path) or os.path.getsize(path) <= 0:
        raise HTTPException(status_code=404, detail="translated PDF 文件不存在")

    if not _is_under_root(path, settings.pdf_translated_path):
        raise HTTPException(status_code=400, detail="拒绝预览：文件不在允许目录内")

    store.set_last_active_paper_id(sid, paper_id)
    return _inline_pdf_response(path, filename)


# ---------------- delete endpoints (你已实现的删除功能保持不变) ----------------
@router.delete("/pdf/assets/{paper_id}", response_model=DeleteAssetResponse)
def delete_pdf_asset(
    paper_id: str,
    session_id: str = Query(
        default="default", description="用于 SSE 广播删除事件（可选）"
    ),
) -> DeleteAssetResponse:
    sid = session_id or "default"
    asset = store.get_pdf_asset(paper_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="pdf asset 不存在")

    if asset.status == "DOWNLOADING" or _has_active_task_for_paper(paper_id):
        raise HTTPException(
            status_code=409, detail="该论文存在下载/翻译任务进行中，暂不可删除"
        )

    local_path = asset.local_path
    lock_path = (local_path or "") + ".lock"
    part_path = (local_path or "") + ".part"

    if lock_path and os.path.exists(lock_path):
        raise HTTPException(
            status_code=409, detail="检测到锁文件，可能仍在下载/处理中，暂不可删除"
        )

    deleted_files: List[str] = []
    warnings: List[str] = []

    _safe_remove_file(local_path, settings.pdf_raw_path, deleted_files, warnings)
    _safe_remove_file(part_path, settings.pdf_raw_path, deleted_files, warnings)
    _safe_remove_file(lock_path, settings.pdf_raw_path, deleted_files, warnings)

    removed_cache = store.delete_pdf_asset(paper_id)

    try:
        event_bus.publish(
            sid,
            {
                "type": "asset_deleted",
                "kind": "pdf",
                "paper_id": paper_id,
                "deleted_files": deleted_files,
            },
        )
    except Exception:
        pass

    return DeleteAssetResponse(
        kind="pdf",
        paper_id=paper_id,
        removed_cache=bool(removed_cache),
        deleted_files=deleted_files,
        warnings=warnings,
    )


@router.delete("/translate/assets/{paper_id}", response_model=DeleteAssetResponse)
def delete_translate_asset(
    paper_id: str,
    session_id: str = Query(
        default="default", description="用于 SSE 广播删除事件（可选）"
    ),
) -> DeleteAssetResponse:
    sid = session_id or "default"
    asset = store.get_translate_asset(paper_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="translate asset 不存在")

    if asset.status == "TRANSLATING" or _has_active_task_for_paper(paper_id):
        raise HTTPException(
            status_code=409, detail="该论文存在翻译任务进行中，暂不可删除"
        )

    mono_path = asset.output_mono_path
    dual_path = asset.output_dual_path
    mono_lock = (mono_path or "") + ".lock"
    log_path = os.path.join(settings.pdf_translated_log_path, f"{paper_id}.pdf2zh.log")

    if mono_lock and os.path.exists(mono_lock):
        raise HTTPException(
            status_code=409, detail="检测到锁文件，可能仍在翻译/处理中，暂不可删除"
        )

    deleted_files: List[str] = []
    warnings: List[str] = []

    _safe_remove_file(mono_path, settings.pdf_translated_path, deleted_files, warnings)
    if dual_path:
        _safe_remove_file(
            dual_path, settings.pdf_translated_path, deleted_files, warnings
        )
    _safe_remove_file(mono_lock, settings.pdf_translated_path, deleted_files, warnings)

    _safe_remove_file(
        log_path, settings.pdf_translated_log_path, deleted_files, warnings
    )

    removed_cache = store.delete_translate_asset(paper_id)

    try:
        event_bus.publish(
            sid,
            {
                "type": "asset_deleted",
                "kind": "translate",
                "paper_id": paper_id,
                "deleted_files": deleted_files,
            },
        )
    except Exception:
        pass

    return DeleteAssetResponse(
        kind="translate",
        paper_id=paper_id,
        removed_cache=bool(removed_cache),
        deleted_files=deleted_files,
        warnings=warnings,
    )


# ---------------- logs ----------------

class LogSessionsResponse(BaseModel):
    sessions: List[LogSessionSummary]


class LogMessagesResponse(BaseModel):
    messages: List[ChatLogItem]


class LogStepsResponse(BaseModel):
    steps: List[AgentStepItem]


@router.get("/logs/sessions", response_model=LogSessionsResponse)
def get_log_sessions(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> LogSessionsResponse:
    return LogSessionsResponse(sessions=log_service.list_sessions(limit, offset))


@router.get("/logs/sessions/{session_id}/messages", response_model=LogMessagesResponse)
def get_log_messages(
    session_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> LogMessagesResponse:
    return LogMessagesResponse(messages=log_service.list_messages(session_id, limit, offset))


@router.get("/logs/messages/{msg_id}/steps", response_model=LogStepsResponse)
def get_log_steps(msg_id: str) -> LogStepsResponse:
    return LogStepsResponse(steps=log_service.get_steps(msg_id))
