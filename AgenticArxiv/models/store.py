# AgenticArxiv/models/store.py
from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

from sqlalchemy import func

from config import settings
from models.db import get_sync_session
from models.orm import (
    PdfAssetRow, TranslateAssetRow, SessionRow, SessionPaperRow, TranslateTaskRow,
)
from models.schemas import (
    Paper, PdfAsset, TranslateAsset, TranslateTask,
)

_REF_RE = re.compile(r"(?:第)?\s*(\d+)\s*(?:篇)?")


def _pdf_row_to_schema(row: PdfAssetRow) -> PdfAsset:
    return PdfAsset(
        paper_id=row.paper_id,
        pdf_url=row.pdf_url or "",
        local_path=row.local_path or "",
        status=row.status or "NOT_DOWNLOADED",
        size_bytes=row.size_bytes or 0,
        sha256=row.sha256,
        downloaded_at=row.downloaded_at,
        updated_at=row.updated_at or datetime.now(),
        error=row.error,
    )


def _translate_row_to_schema(row: TranslateAssetRow) -> TranslateAsset:
    return TranslateAsset(
        paper_id=row.paper_id,
        input_pdf_path=row.input_pdf_path or "",
        output_mono_path=row.output_mono_path or "",
        output_dual_path=row.output_dual_path,
        status=row.status or "NOT_TRANSLATED",
        service=row.service,
        threads=row.threads or 0,
        translated_at=row.translated_at,
        updated_at=row.updated_at or datetime.now(),
        error=row.error,
    )


def _task_row_to_schema(row: TranslateTaskRow) -> TranslateTask:
    meta = {}
    if row.meta:
        try:
            meta = json.loads(row.meta)
        except Exception:
            pass
    return TranslateTask(
        task_id=row.task_id,
        session_id=row.session_id,
        paper_id=row.paper_id,
        status=row.status or "PENDING",
        progress=row.progress or 0.0,
        input_pdf_url=row.input_pdf_url,
        input_pdf_path=row.input_pdf_path,
        output_pdf_path=row.output_pdf_path,
        error=row.error,
        meta=meta,
        created_at=row.created_at or datetime.now(),
        updated_at=row.updated_at or datetime.now(),
    )


def _paper_row_to_schema(row: SessionPaperRow) -> Paper:
    def _load_json_list(s):
        if not s:
            return []
        try:
            return json.loads(s)
        except Exception:
            return []

    return Paper(
        id=row.paper_id,
        title=row.title or "",
        authors=_load_json_list(row.authors),
        summary=row.summary,
        published=row.published,
        updated=row.updated,
        pdf_url=row.pdf_url,
        primary_category=row.primary_category,
        categories=_load_json_list(row.categories),
        comment=row.comment,
        links=_load_json_list(row.links),
    )


class Store:
    """MySQL-backed store (synchronous, thread-safe via SQLAlchemy connection pool)."""

    def __init__(self, ttl_minutes: int = 60, max_papers: int = 50):
        self.ttl = timedelta(minutes=ttl_minutes)
        self.max_papers = max_papers

        # Ensure output directories exist
        os.makedirs(settings.pdf_raw_path, exist_ok=True)
        os.makedirs(settings.pdf_translated_path, exist_ok=True)

    # -------- session memory --------

    def set_last_papers(self, session_id: str, papers: List[Paper]) -> None:
        with get_sync_session() as db:
            # Ensure session row exists
            row = db.query(SessionRow).filter_by(session_id=session_id).first()
            if not row:
                row = SessionRow(session_id=session_id)
                db.add(row)
            row.updated_at = datetime.now()

            # Replace session_papers
            db.query(SessionPaperRow).filter_by(session_id=session_id).delete()
            for i, p in enumerate(papers[: self.max_papers]):
                db.add(SessionPaperRow(
                    session_id=session_id,
                    paper_id=p.id,
                    title=p.title,
                    authors=json.dumps(p.authors, ensure_ascii=False),
                    summary=p.summary,
                    published=p.published,
                    updated=p.updated,
                    pdf_url=p.pdf_url,
                    primary_category=p.primary_category,
                    categories=json.dumps(p.categories, ensure_ascii=False),
                    comment=p.comment,
                    links=json.dumps(p.links, ensure_ascii=False),
                    position=i,
                ))
            db.commit()

    def get_last_papers(self, session_id: str) -> List[Paper]:
        with get_sync_session() as db:
            row = db.query(SessionRow).filter_by(session_id=session_id).first()
            if not row:
                return []
            if row.updated_at and (datetime.now() - row.updated_at) > self.ttl:
                return []
            paper_rows = (
                db.query(SessionPaperRow)
                .filter_by(session_id=session_id)
                .order_by(SessionPaperRow.position)
                .all()
            )
            return [_paper_row_to_schema(r) for r in paper_rows]

    # -------- last active paper --------

    def set_last_active_paper_id(self, session_id: str, paper_id: str) -> None:
        if not paper_id:
            return
        with get_sync_session() as db:
            row = db.query(SessionRow).filter_by(session_id=session_id).first()
            if not row:
                row = SessionRow(session_id=session_id)
                db.add(row)
            row.last_active_paper_id = paper_id
            row.last_active_at = datetime.now()
            row.updated_at = datetime.now()
            db.commit()

    def get_last_active_paper_id(self, session_id: str) -> Optional[str]:
        with get_sync_session() as db:
            row = db.query(SessionRow).filter_by(session_id=session_id).first()
            if not row:
                return None
            if row.last_active_at and (datetime.now() - row.last_active_at) > self.ttl:
                row.last_active_paper_id = None
                row.last_active_at = None
                db.commit()
                return None
            return row.last_active_paper_id

    def resolve_paper(
        self, session_id: str, ref: Union[str, int, None]
    ) -> Optional[Paper]:
        papers = self.get_last_papers(session_id)
        if not papers:
            return None

        if ref is None:
            last_id = self.get_last_active_paper_id(session_id)
            if not last_id:
                return None
            for p in papers:
                if p.id == last_id:
                    return p
            return None

        if isinstance(ref, int):
            idx = ref - 1
            return papers[idx] if 0 <= idx < len(papers) else None

        s = str(ref).strip()
        m = _REF_RE.fullmatch(s)
        if m:
            idx = int(m.group(1)) - 1
            return papers[idx] if 0 <= idx < len(papers) else None

        for p in papers:
            if p.id == s:
                return p

        low = s.lower()
        for p in papers:
            if low in (p.title or "").lower():
                return p
        return None

    # -------- PDF cache --------

    def get_pdf_asset(self, paper_id: str) -> Optional[PdfAsset]:
        with get_sync_session() as db:
            row = db.query(PdfAssetRow).filter_by(paper_id=paper_id).first()
            return _pdf_row_to_schema(row) if row else None

    def upsert_pdf_asset(self, asset: PdfAsset) -> PdfAsset:
        with get_sync_session() as db:
            row = db.query(PdfAssetRow).filter_by(paper_id=asset.paper_id).first()
            if not row:
                row = PdfAssetRow(paper_id=asset.paper_id)
                db.add(row)
            row.pdf_url = asset.pdf_url
            row.local_path = asset.local_path
            row.status = asset.status
            row.size_bytes = asset.size_bytes
            row.sha256 = asset.sha256
            row.downloaded_at = asset.downloaded_at
            row.updated_at = datetime.now()
            row.error = asset.error
            db.commit()
            db.refresh(row)
            return _pdf_row_to_schema(row)

    def update_pdf_asset(self, paper_id: str, **kwargs) -> Optional[PdfAsset]:
        with get_sync_session() as db:
            row = db.query(PdfAssetRow).filter_by(paper_id=paper_id).first()
            if not row:
                return None
            for k, v in kwargs.items():
                if hasattr(row, k):
                    setattr(row, k, v)
            row.updated_at = datetime.now()
            db.commit()
            db.refresh(row)
            return _pdf_row_to_schema(row)

    def delete_pdf_asset(self, paper_id: str) -> bool:
        with get_sync_session() as db:
            row = db.query(PdfAssetRow).filter_by(paper_id=paper_id).first()
            if not row:
                return False
            db.delete(row)
            db.commit()
            return True

    def list_pdf_assets(self) -> List[PdfAsset]:
        with get_sync_session() as db:
            rows = db.query(PdfAssetRow).order_by(PdfAssetRow.updated_at.desc()).all()
            return [_pdf_row_to_schema(r) for r in rows]

    # -------- Translate cache --------

    def get_translate_asset(self, paper_id: str) -> Optional[TranslateAsset]:
        with get_sync_session() as db:
            row = db.query(TranslateAssetRow).filter_by(paper_id=paper_id).first()
            return _translate_row_to_schema(row) if row else None

    def upsert_translate_asset(self, asset: TranslateAsset) -> TranslateAsset:
        with get_sync_session() as db:
            row = db.query(TranslateAssetRow).filter_by(paper_id=asset.paper_id).first()
            if not row:
                row = TranslateAssetRow(paper_id=asset.paper_id)
                db.add(row)
            row.input_pdf_path = asset.input_pdf_path
            row.output_mono_path = asset.output_mono_path
            row.output_dual_path = asset.output_dual_path
            row.status = asset.status
            row.service = asset.service
            row.threads = asset.threads
            row.translated_at = asset.translated_at
            row.updated_at = datetime.now()
            row.error = asset.error
            db.commit()
            db.refresh(row)
            return _translate_row_to_schema(row)

    def update_translate_asset(self, paper_id: str, **kwargs) -> Optional[TranslateAsset]:
        with get_sync_session() as db:
            row = db.query(TranslateAssetRow).filter_by(paper_id=paper_id).first()
            if not row:
                return None
            for k, v in kwargs.items():
                if hasattr(row, k):
                    setattr(row, k, v)
            row.updated_at = datetime.now()
            db.commit()
            db.refresh(row)
            return _translate_row_to_schema(row)

    def delete_translate_asset(self, paper_id: str) -> bool:
        with get_sync_session() as db:
            row = db.query(TranslateAssetRow).filter_by(paper_id=paper_id).first()
            if not row:
                return False
            db.delete(row)
            db.commit()
            return True

    def list_translate_assets(self) -> List[TranslateAsset]:
        with get_sync_session() as db:
            rows = db.query(TranslateAssetRow).order_by(TranslateAssetRow.updated_at.desc()).all()
            return [_translate_row_to_schema(r) for r in rows]

    # -------- tasks --------

    def create_translate_task(
        self,
        session_id: str,
        paper_id: str,
        input_pdf_url: Optional[str] = None,
        meta: Optional[Dict[str, str]] = None,
    ) -> TranslateTask:
        task_id = uuid.uuid4().hex
        now = datetime.now()
        with get_sync_session() as db:
            row = TranslateTaskRow(
                task_id=task_id,
                session_id=session_id,
                paper_id=paper_id,
                status="PENDING",
                input_pdf_url=input_pdf_url,
                meta=json.dumps(meta or {}, ensure_ascii=False),
                created_at=now,
                updated_at=now,
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            return _task_row_to_schema(row)

    def get_task(self, task_id: str) -> Optional[TranslateTask]:
        with get_sync_session() as db:
            row = db.query(TranslateTaskRow).filter_by(task_id=task_id).first()
            return _task_row_to_schema(row) if row else None

    def update_task(self, task_id: str, **kwargs) -> Optional[TranslateTask]:
        with get_sync_session() as db:
            row = db.query(TranslateTaskRow).filter_by(task_id=task_id).first()
            if not row:
                return None
            for k, v in kwargs.items():
                if k == "meta" and isinstance(v, dict):
                    row.meta = json.dumps(v, ensure_ascii=False)
                elif hasattr(row, k):
                    setattr(row, k, v)
            row.updated_at = datetime.now()
            db.commit()
            db.refresh(row)
            return _task_row_to_schema(row)

    def list_tasks(
        self, session_id: Optional[str] = None, limit: int = 50
    ) -> List[TranslateTask]:
        with get_sync_session() as db:
            q = db.query(TranslateTaskRow)
            if session_id:
                q = q.filter_by(session_id=session_id)
            rows = q.order_by(TranslateTaskRow.updated_at.desc()).limit(max(1, limit)).all()
            return [_task_row_to_schema(r) for r in rows]

    # -------- startup validation --------

    def validate_local_paths(self) -> None:
        """Mark READY assets as FAILED if their local files are missing."""
        with get_sync_session() as db:
            for row in db.query(PdfAssetRow).filter_by(status="READY").all():
                if not row.local_path or not os.path.exists(row.local_path):
                    row.status = "FAILED"
                    row.error = f"local file missing: {row.local_path}"
                    row.updated_at = datetime.now()
            for row in db.query(TranslateAssetRow).filter_by(status="READY").all():
                if not row.output_mono_path or not os.path.exists(row.output_mono_path):
                    row.status = "FAILED"
                    row.error = f"local file missing: {row.output_mono_path}"
                    row.updated_at = datetime.now()
            db.commit()


store = Store()
