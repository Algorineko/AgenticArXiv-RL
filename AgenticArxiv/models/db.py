# AgenticArxiv/models/db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import settings


class Base(DeclarativeBase):
    pass


engine = create_engine(
    settings.mysql_uri,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    echo=False,
)

SyncSessionLocal = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


def get_sync_session() -> Session:
    return SyncSessionLocal()


def init_db():
    """Create all tables (safe to call repeatedly)."""
    import models.orm  # noqa: F401 — ensure all models are registered
    Base.metadata.create_all(bind=engine)
