"""Integration tests for the FastAPI routes (Phase 7).

Uses httpx AsyncClient with ASGITransport so real FastAPI middleware/validation
runs, but all external I/O (Qdrant, DynamoDB, S3, LLM) is mocked.
"""
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app
from src.auth import UserContext

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FAKE_USER = UserContext(user_id="user-test-001", email="test@example.com")


def _auth_override():
    """Override the JWT dependency so tests don't need a real token."""
    return FAKE_USER


@pytest.fixture
async def client():
    app.dependency_overrides = {}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides = {}


@pytest.fixture
async def auth_client():
    """Client with JWT auth bypassed."""
    from src.api.deps import get_current_user
    app.dependency_overrides[get_current_user] = _auth_override
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides = {}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# 1. Upload without auth → 401/403
# ---------------------------------------------------------------------------


async def test_upload_without_auth(client):
    response = await client.post(
        "/documents/upload",
        files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    # FastAPI OAuth2 scheme returns 403 when no token provided
    assert response.status_code in (401, 403)


# ---------------------------------------------------------------------------
# 2. Upload with bad extension → 400
# ---------------------------------------------------------------------------


async def test_upload_bad_extension(auth_client):
    response = await auth_client.post(
        "/documents/upload",
        files={"file": ("malware.exe", b"MZ\x90\x00", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 3. Upload happy path → 202 processing
# ---------------------------------------------------------------------------


async def test_upload_happy_path(auth_client):
    fake_pdf = b"%PDF-1.4 fake content"

    with (
        patch("src.api.routes.documents.put_document", new=AsyncMock()),
        patch("src.api.routes.documents._run_file_ingestion", new=AsyncMock()),
    ):
        response = await auth_client.post(
            "/documents/upload",
            files={"file": ("report.pdf", fake_pdf, "application/pdf")},
        )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "processing"
    assert "doc_id" in body


# ---------------------------------------------------------------------------
# 4. List documents → returns mapped DocumentResponse list
# ---------------------------------------------------------------------------


async def test_list_documents(auth_client):
    fake_items = [
        {
            "doc_id": "doc-1",
            "name": "report.pdf",
            "source_type": "pdf",
            "status": "ready",
            "chunk_count": 42,
            "created_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "doc_id": "doc-2",
            "name": "notes.docx",
            "source_type": "docx",
            "status": "processing",
            "chunk_count": 0,
            "created_at": "2026-01-02T00:00:00+00:00",
        },
    ]

    with patch("src.api.routes.documents.list_documents", new=AsyncMock(return_value=fake_items)):
        response = await auth_client.get("/documents")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert body[0]["doc_id"] == "doc-1"
    assert body[1]["status"] == "processing"


# ---------------------------------------------------------------------------
# 5. Delete non-existent document → 404
# ---------------------------------------------------------------------------


async def test_delete_nonexistent_document(auth_client):
    with patch("src.api.routes.documents.get_document_config", new=AsyncMock(return_value=None)):
        response = await auth_client.delete("/documents/no-such-id")

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# 6. Delete existing document → 204
# ---------------------------------------------------------------------------


async def test_delete_existing_document(auth_client):
    fake_doc = {
        "doc_id": "doc-1",
        "name": "report.pdf",
        "source_type": "pdf",
        "status": "ready",
        "chunk_count": 5,
        "created_at": "2026-01-01T00:00:00+00:00",
    }

    with (
        patch("src.api.routes.documents.get_document_config", new=AsyncMock(return_value=fake_doc)),
        patch("src.api.routes.documents.delete_by_doc_id"),
        patch("src.api.routes.documents.delete_document", new=AsyncMock()),
        patch("src.api.routes.documents.delete_file", new=AsyncMock()),
    ):
        response = await auth_client.delete("/documents/doc-1")

    assert response.status_code == 204


# ---------------------------------------------------------------------------
# 7. Query happy path → QueryResponse structure
# ---------------------------------------------------------------------------


async def test_query_happy_path(auth_client):
    fake_result = {
        "answer": "RAG stands for Retrieval-Augmented Generation.",
        "sources": [{"doc_id": "doc-1", "chunk_index": 0, "page_number": 1, "score": 0.9}],
        "confidence": 0.9,
    }

    with patch("src.api.routes.query.answer_question", new=AsyncMock(return_value=fake_result)):
        response = await auth_client.post("/query", json={"question": "What is RAG?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == fake_result["answer"]
    assert body["confidence"] == pytest.approx(0.9)
    assert len(body["sources"]) == 1


# ---------------------------------------------------------------------------
# 8. Query — question too long → 422
# ---------------------------------------------------------------------------


async def test_query_question_too_long(auth_client):
    response = await auth_client.post("/query", json={"question": "x" * 2001})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 9. Query — top_k out of range → 422
# ---------------------------------------------------------------------------


async def test_query_top_k_out_of_range(auth_client):
    response = await auth_client.post(
        "/query", json={"question": "What is RAG?", "top_k": 99}
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 10. DB source — invalid table name → 422
# ---------------------------------------------------------------------------


async def test_db_source_invalid_table_name(auth_client):
    response = await auth_client.post(
        "/documents/db",
        json={
            "connection_string": "postgresql://localhost/mydb",
            "table_name": "my table; DROP TABLE users",
            "content_columns": ["body"],
            "row_id_column": "id",
        },
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 11. DB source — invalid connection string scheme → 422
# ---------------------------------------------------------------------------


async def test_db_source_invalid_connection_scheme(auth_client):
    response = await auth_client.post(
        "/documents/db",
        json={
            "connection_string": "sqlite:///local.db",
            "table_name": "articles",
            "content_columns": ["body"],
            "row_id_column": "id",
        },
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 12. Evaluate without auth → 401/403
# ---------------------------------------------------------------------------


async def test_evaluate_without_auth(client):
    response = await client.post(
        "/evaluate",
        json={"question": "What is RAG?", "answer": "It retrieves docs.", "contexts": ["doc text"]},
    )
    assert response.status_code in (401, 403)
