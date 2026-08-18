# AgenticArxiv/models/store_memory.py
"""进程内内存版 Store，与 MySQL 版 API 完全一致。

用途：离线 RL rollout / 单元测试 / 无数据库演示。
- 不依赖 sqlalchemy / pymysql
- 单进程内有效，进程退出即丢失
- TTL 语义与 MySQL 版保持一致，便于复现同样的"会话记忆过期"行为
"""
from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

from config import settings
from models.schemas import Paper, PdfAsset, TranslateAsset, TranslateTask

_REF_RE = re.compile(r"(?:第)?\s*(\d+)\s*(?:篇)?")


class _SessionState:
    __slots__ = ("papers", "last_active_paper_id", "last_active_at", "updated_at")

    def __init__(self) -> None:
        self.papers: List[Paper] = []
        self.last_active_paper_id: Optional[str] = None
        self.last_active_at: Optional[datetime] = None
        self.updated_at: datetime = datetime.now()


class MemoryStore:
    """In-memory store (single process, no external dependency)."""

    def __init__(self, ttl_minutes: int = 60, max_papers: int = 50, make_dirs: bool = True):
        self.ttl = timedelta(minutes=ttl_minutes)
        self.max_papers = max_papers
        self._sessions: Dict[str, _SessionState] = {}
        self._pdf_assets: Dict[str, PdfAsset] = {}
        self._translate_assets: Dict[str, TranslateAsset] = {}
        self._tasks: Dict[str, TranslateTask] = {}

        if make_dirs:
            os.makedirs(settings.pdf_raw_path, exist_ok=True)
            os.makedirs(settings.pdf_translated_path, exist_ok=True)

    # -------- internal --------

    def _session(self, session_id: str) -> _SessionState:
        st = self._sessions.get(session_id)
        if st is None:
            st = _SessionState()
            self._sessions[session_id] = st
        return st

    def reset(self) -> None:
        """清空所有状态（RL rollout 之间隔离用）。"""
        self._sessions.clear()
        self._pdf_assets.clear()
        self._translate_assets.clear()
        self._tasks.clear()

    # -------- session memory --------

    def set_last_papers(self, session_id: str, papers: List[Paper]) -> None:
        st = self._session(session_id)
        st.papers = list(papers[: self.max_papers])
        st.updated_at = datetime.now()

    def get_last_papers(self, session_id: str) -> List[Paper]:
        st = self._sessions.get(session_id)
        if not st:
            return []
        if st.updated_at and (datetime.now() - st.updated_at) > self.ttl:
            return []
        return list(st.papers)

    # -------- last active paper --------

    def set_last_active_paper_id(self, session_id: str, paper_id: str) -> None:
        if not paper_id:
            return
        st = self._session(session_id)
        st.last_active_paper_id = paper_id
        st.last_active_at = datetime.now()
        st.updated_at = datetime.now()

    def get_last_active_paper_id(self, session_id: str) -> Optional[str]:
        st = self._sessions.get(session_id)
        if not st:
            return None
        if st.last_active_at and (datetime.now() - st.last_active_at) > self.ttl:
            st.last_active_paper_id = None
            st.last_active_at = None
            return None
        return st.last_active_paper_id

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
        return self._pdf_assets.get(paper_id)

    def upsert_pdf_asset(self, asset: PdfAsset) -> PdfAsset:
        asset.updated_at = datetime.now()
        self._pdf_assets[asset.paper_id] = asset
        return asset

    def update_pdf_asset(self, paper_id: str, **kwargs) -> Optional[PdfAsset]:
        asset = self._pdf_assets.get(paper_id)
        if not asset:
            return None
        for k, v in kwargs.items():
            if hasattr(asset, k):
                setattr(asset, k, v)
        asset.updated_at = datetime.now()
        return asset

    def delete_pdf_asset(self, paper_id: str) -> bool:
        return self._pdf_assets.pop(paper_id, None) is not None

    def list_pdf_assets(self) -> List[PdfAsset]:
        return sorted(
            self._pdf_assets.values(), key=lambda a: a.updated_at, reverse=True
        )

    # -------- Translate cache --------

    def get_translate_asset(self, paper_id: str) -> Optional[TranslateAsset]:
        return self._translate_assets.get(paper_id)

    def upsert_translate_asset(self, asset: TranslateAsset) -> TranslateAsset:
        asset.updated_at = datetime.now()
        self._translate_assets[asset.paper_id] = asset
        return asset

    def update_translate_asset(self, paper_id: str, **kwargs) -> Optional[TranslateAsset]:
        asset = self._translate_assets.get(paper_id)
        if not asset:
            return None
        for k, v in kwargs.items():
            if hasattr(asset, k):
                setattr(asset, k, v)
        asset.updated_at = datetime.now()
        return asset

    def delete_translate_asset(self, paper_id: str) -> bool:
        return self._translate_assets.pop(paper_id, None) is not None

    def list_translate_assets(self) -> List[TranslateAsset]:
        return sorted(
            self._translate_assets.values(), key=lambda a: a.updated_at, reverse=True
        )

    # -------- tasks --------

    def create_translate_task(
        self,
        session_id: str,
        paper_id: str,
        input_pdf_url: Optional[str] = None,
        meta: Optional[Dict[str, str]] = None,
    ) -> TranslateTask:
        task = TranslateTask(
            task_id=uuid.uuid4().hex,
            session_id=session_id,
            paper_id=paper_id,
            status="PENDING",
            input_pdf_url=input_pdf_url,
            meta=meta or {},
        )
        self._tasks[task.task_id] = task
        return task

    def get_task(self, task_id: str) -> Optional[TranslateTask]:
        return self._tasks.get(task_id)

    def update_task(self, task_id: str, **kwargs) -> Optional[TranslateTask]:
        task = self._tasks.get(task_id)
        if not task:
            return None
        for k, v in kwargs.items():
            if hasattr(task, k):
                setattr(task, k, v)
        task.updated_at = datetime.now()
        return task

    def list_tasks(
        self, session_id: Optional[str] = None, limit: int = 50
    ) -> List[TranslateTask]:
        items = list(self._tasks.values())
        if session_id:
            items = [t for t in items if t.session_id == session_id]
        items.sort(key=lambda t: t.updated_at, reverse=True)
        return items[: max(1, limit)]

    # -------- startup validation --------

    def validate_local_paths(self) -> None:
        for asset in self._pdf_assets.values():
            if asset.status == "READY" and (
                not asset.local_path or not os.path.exists(asset.local_path)
            ):
                asset.status = "FAILED"
                asset.error = f"local file missing: {asset.local_path}"
                asset.updated_at = datetime.now()
        for asset in self._translate_assets.values():
            if asset.status == "READY" and (
                not asset.output_mono_path or not os.path.exists(asset.output_mono_path)
            ):
                asset.status = "FAILED"
                asset.error = f"local file missing: {asset.output_mono_path}"
                asset.updated_at = datetime.now()
