# AgenticArxiv/models/orm.py
from datetime import datetime

from sqlalchemy import (
    Column, Integer, BigInteger, Float, String, Text, DateTime, Index,
)

from models.db import Base


class PdfAssetRow(Base):
    __tablename__ = "pdf_assets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_id = Column(String(64), unique=True, nullable=False, index=True)
    pdf_url = Column(String(512), default="")
    local_path = Column(String(512), default="")
    status = Column(String(32), default="NOT_DOWNLOADED")
    size_bytes = Column(BigInteger, default=0)
    sha256 = Column(String(128), nullable=True)
    downloaded_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    error = Column(Text, nullable=True)


class TranslateAssetRow(Base):
    __tablename__ = "translate_assets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_id = Column(String(64), unique=True, nullable=False, index=True)
    input_pdf_path = Column(String(512), default="")
    output_mono_path = Column(String(512), default="")
    output_dual_path = Column(String(512), nullable=True)
    status = Column(String(32), default="NOT_TRANSLATED")
    service = Column(String(64), nullable=True)
    threads = Column(Integer, default=0)
    translated_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    error = Column(Text, nullable=True)


class SessionRow(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(128), unique=True, nullable=False, index=True)
    last_active_paper_id = Column(String(64), nullable=True)
    last_active_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class SessionPaperRow(Base):
    __tablename__ = "session_papers"
    __table_args__ = (
        Index("ix_session_papers_sid_pos", "session_id", "position"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(128), nullable=False, index=True)
    paper_id = Column(String(64), nullable=False)
    title = Column(String(512), default="")
    authors = Column(Text, default="[]")        # JSON array
    summary = Column(Text, nullable=True)
    published = Column(String(64), nullable=True)
    updated = Column(String(64), nullable=True)
    pdf_url = Column(String(512), nullable=True)
    primary_category = Column(String(32), nullable=True)
    categories = Column(Text, default="[]")     # JSON array
    comment = Column(Text, nullable=True)
    links = Column(Text, default="[]")          # JSON array
    position = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.now)


class TranslateTaskRow(Base):
    __tablename__ = "translate_tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(64), unique=True, nullable=False, index=True)
    session_id = Column(String(128), nullable=False, index=True)
    paper_id = Column(String(64), nullable=False)
    status = Column(String(32), default="PENDING")
    progress = Column(Float, default=0.0)
    input_pdf_url = Column(String(512), nullable=True)
    input_pdf_path = Column(String(512), nullable=True)
    output_pdf_path = Column(String(512), nullable=True)
    error = Column(Text, nullable=True)
    meta = Column(Text, default="{}")           # JSON dict
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


# --- Logging tables ---

class ChatLogRow(Base):
    __tablename__ = "chat_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(128), nullable=False, index=True)
    msg_id = Column(String(64), unique=True, nullable=False, index=True)
    role = Column(String(16), nullable=False)       # 'user' | 'assistant'
    content = Column(Text, nullable=True)
    model = Column(String(128), nullable=True)
    agent_type = Column(String(32), nullable=True)  # regex | mcp | skill_cli
    created_at = Column(DateTime, default=datetime.now)


class AgentStepRow(Base):
    __tablename__ = "agent_steps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    msg_id = Column(String(64), nullable=False, index=True)
    step_index = Column(Integer, nullable=False)
    thought = Column(Text, nullable=True)
    action_name = Column(String(128), nullable=True)
    action_args = Column(Text, nullable=True)       # JSON string
    observation = Column(Text, nullable=True)
    llm_latency_ms = Column(Integer, nullable=True)
    tool_latency_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
