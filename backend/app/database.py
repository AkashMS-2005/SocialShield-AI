"""
database.py – SQLAlchemy async engine + session factory.

The database stores scan history. Tables are created synchronously on startup
(using stdlib sqlite3 to avoid event-loop conflicts in tests), while all
request-time queries use the async engine.

Swap DATABASE_URL in .env to PostgreSQL without changing this file.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from app.config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:  # type: ignore[override]
    """FastAPI dependency that yields a database session."""
    async with AsyncSessionLocal() as session:
        yield session


def init_db_sync() -> None:
    """
    Create tables synchronously using stdlib sqlite3.

    Called once at startup. This avoids running an async event loop
    during Starlette's lifespan thread (which causes deadlocks with
    aiosqlite on Windows in test environments).
    """
    db_path = settings.database_url.replace("sqlite+aiosqlite:///", "")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scan_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            username        TEXT    NOT NULL,
            platform        TEXT    NOT NULL,
            classification  TEXT    NOT NULL,
            risk_score      REAL,
            scanned_at      TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()
