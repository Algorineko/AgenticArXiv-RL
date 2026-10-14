# AgenticArxiv/models/schemas.py
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
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
    source: Optional[str] = None


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
    source: Optional[str] = None
