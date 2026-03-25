"""Local SQLite adapter that mirrors the interface of storage/dynamo.py.
Used when STORAGE_BACKEND=local — no AWS credentials required.

Database file is stored at: {local_storage_path}/metadata.db
"""

import asyncio
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from src.config import settings

# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    user_id     TEXT NOT NULL,
    doc_id      TEXT NOT NULL,
    name        TEXT NOT NULL,
    source_type TEXT NOT NULL,
    status      TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (user_id, doc_id)
)
"""


def _db_path() -> Path:
    p = Path(settings.local_storage_path)
    p.mkdir(parents=True, exist_ok=True)
    return p / "metadata.db"


@contextmanager
def _connect() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(str(_db_path()), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(_CREATE_TABLE)
        conn.commit()
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _run(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


# ---------------------------------------------------------------------------
# Public interface — matches dynamo.py exactly
# ---------------------------------------------------------------------------


def _put_document_sync(
    user_id: str,
    doc_id: str,
    name: str,
    source_type: str,
    status: str,
    chunk_count: int,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO documents
                (user_id, doc_id, name, source_type, status, chunk_count, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, doc_id) DO UPDATE SET
                name        = excluded.name,
                source_type = excluded.source_type,
                status      = excluded.status,
                chunk_count = excluded.chunk_count,
                updated_at  = excluded.updated_at
            """,
            (user_id, doc_id, name, source_type, status, chunk_count, _now_iso(), _now_iso()),
        )
        conn.commit()


async def put_document(
    user_id: str,
    doc_id: str,
    name: str,
    source_type: str,
    status: str,
    chunk_count: int = 0,
) -> None:
    await _run(_put_document_sync, user_id, doc_id, name, source_type, status, chunk_count)


def _get_document_sync(user_id: str, doc_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE user_id = ? AND doc_id = ?",
            (user_id, doc_id),
        ).fetchone()
        return dict(row) if row else None


async def get_document_config(user_id: str, doc_id: str) -> dict | None:
    return await _run(_get_document_sync, user_id, doc_id)


def _list_documents_sync(user_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM documents WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


async def list_documents(user_id: str) -> list[dict]:
    return await _run(_list_documents_sync, user_id)


def _update_document_status_sync(
    user_id: str,
    doc_id: str,
    status: str,
    chunk_count: int | None,
    expected_status: str | None,
) -> None:
    with _connect() as conn:
        if expected_status is not None:
            conn.execute(
                """
                UPDATE documents
                SET status = ?, chunk_count = COALESCE(?, chunk_count), updated_at = ?
                WHERE user_id = ? AND doc_id = ? AND status = ?
                """,
                (status, chunk_count, _now_iso(), user_id, doc_id, expected_status),
            )
        else:
            conn.execute(
                """
                UPDATE documents
                SET status = ?, chunk_count = COALESCE(?, chunk_count), updated_at = ?
                WHERE user_id = ? AND doc_id = ?
                """,
                (status, chunk_count, _now_iso(), user_id, doc_id),
            )
        conn.commit()


async def update_document_status(
    user_id: str,
    doc_id: str,
    status: str,
    chunk_count: int | None = None,
    expected_status: str | None = None,
) -> None:
    await _run(_update_document_status_sync, user_id, doc_id, status, chunk_count, expected_status)


def _delete_document_sync(user_id: str, doc_id: str) -> None:
    with _connect() as conn:
        conn.execute(
            "DELETE FROM documents WHERE user_id = ? AND doc_id = ?",
            (user_id, doc_id),
        )
        conn.commit()


async def delete_document(user_id: str, doc_id: str) -> None:
    await _run(_delete_document_sync, user_id, doc_id)
