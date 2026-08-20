"""SQLAlchemy engine and session (§5).

One engine per process, created lazily from `Settings` rather than at import time -- so
importing this module in a test never opens a socket, and `get_settings()` (which reads
`.env`) can be overridden before the first real connection is made.
"""

from __future__ import annotations

from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from fcesapi.config import get_settings


class Base(DeclarativeBase):
    pass


@lru_cache
def get_engine():
    return create_engine(get_settings().database_url, pool_pre_ping=True)


@lru_cache
def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: one session per request, closed after."""
    db = get_sessionmaker()()
    try:
        yield db
    finally:
        db.close()
