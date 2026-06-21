# AgenticArxiv/models/schemas.py
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict
from datetime import datetime


class Paper(BaseModel):
    id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
    published: Optional[str] = None
    updated: Optional[str] = None
    pdf_url: Optional[str] = None
    primary_category: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    comment: Optional[str] = None
    links: List[str] = Field(default_factory=list)


class SessionState(BaseModel):
    session_id: str
    last_papers: List[Paper] = Field(default_factory=list)
    last_active_paper_id: Optional[str] = None
    last_active_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=datetime.now)


TaskStatus = Literal["PENDING", "RUNNING", "SUCCEEDED", "FAILED"]


class TranslateTask(BaseModel):
    task_id: str
    session_id: str
    paper_id: str
    status: TaskStatus = "PENDING"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    progress: float = 0.0
    input_pdf_url: Optional[str] = None
    input_pdf_path: Optional[str] = None
    output_pdf_path: Optional[str] = None
    error: Optional[str] = None
    meta: Dict[str, str] = Field(default_factory=dict)


# --- PDF / Translate asset schemas (formerly in pdf_cache.py / translate_cache.py) ---

PdfAssetStatus = Literal["NOT_DOWNLOADED", "DOWNLOADING", "READY", "FAILED"]


class PdfAsset(BaseModel):
    paper_id: str
    pdf_url: str = ""
    local_path: str = ""
    status: PdfAssetStatus = "NOT_DOWNLOADED"
    size_bytes: int = 0
    sha256: Optional[str] = None
    downloaded_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = None


TranslateAssetStatus = Literal["NOT_TRANSLATED", "TRANSLATING", "READY", "FAILED"]


class TranslateAsset(BaseModel):
    paper_id: str
    input_pdf_path: str = ""
    output_mono_path: str = ""
    output_dual_path: Optional[str] = None
    status: TranslateAssetStatus = "NOT_TRANSLATED"
    service: Optional[str] = None
    threads: int = 0
    translated_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = None


# --- Logging schemas ---

class ChatLogItem(BaseModel):
    msg_id: str
    session_id: str
    role: str
    content: Optional[str] = None
    model: Optional[str] = None
    agent_type: Optional[str] = None
    created_at: Optional[datetime] = None


class AgentStepItem(BaseModel):
    step_index: int
    thought: Optional[str] = None
    action_name: Optional[str] = None
    action_args: Optional[str] = None
    observation: Optional[str] = None
    llm_latency_ms: Optional[int] = None
    tool_latency_ms: Optional[int] = None
    created_at: Optional[datetime] = None


class LogSessionSummary(BaseModel):
    session_id: str
    message_count: int
    last_active_at: Optional[datetime] = None
