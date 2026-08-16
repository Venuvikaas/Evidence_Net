"""Database setup and session management for EVIDENCE-Net Metadata Store (Phase 12).

Provides SQLite/SQLAlchemy database connection management, session creation,
and table initialization for experiment, run, artifact, and review metadata.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DEFAULT_DB_PATH = Path("data/metadata.db")


class Base(DeclarativeBase):
    """Base class for SQLAlchemy ORM models."""

    pass


def get_db_url(db_path: Path | str | None = None) -> str:
    """Return database URL string. Defaults to SQLite at data/metadata.db."""
    if db_path is None:
        path = DEFAULT_DB_PATH
    else:
        path = Path(db_path)

    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.resolve()}"


def init_db(db_path: Path | str | None = None) -> sessionmaker[Session]:
    """Initialize database tables and return sessionmaker instance."""
    url = get_db_url(db_path)
    engine = create_engine(url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db_session(session_factory: sessionmaker[Session]) -> Generator[Session, None, None]:
    """Yield a database session and close it afterwards."""
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
