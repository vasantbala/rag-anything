# tests/test_retriever.py
import pytest
from unittest.mock import patch, MagicMock
from qdrant_client.models import ScoredPoint, SparseVector, Payload

from src.retriever import retrieve, RetrievalResult
from src.prompt_builder import build_prompt, RetrievedChunk


# ── Helpers ────────────────────────────────────────────────────────────────────

FAKE_DENSE = [0.1] * 1024
FAKE_SPARSE = SparseVector(indices=[0, 1, 2], values=[0.5, 0.3, 0.2])
THRESHOLD = 0.1  # must match settings.reranker_score_threshold


def _make_point(i: int, user_id: str = "user-1", doc_id: str = "doc-1") -> ScoredPoint:
    return ScoredPoint(
        id=str(i),
        version=0,
        score=0.9,
        payload={
            "text": f"chunk text {i}",
            "doc_id": doc_id,
            "user_id": user_id,
            "source_type": "pdf",
            "chunk_index": i,
            "page_number": i + 1,
        },
        vector=None,
    )


def _make_rerank_results(points, score=0.9):
    """Simulate flashrank output: list of dicts with id + score."""
    return [{"id": i, "score": score} for i in range(len(points))]


# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@patch("src.retriever._ranker")
@patch("src.retriever.client")
@patch("src.retriever.embed_sparse", return_value=[FAKE_SPARSE])
@patch("src.retriever.embed_texts", return_value=[FAKE_DENSE])
async def test_happy_path(mock_embed, mock_sparse, mock_client, mock_ranker):
    points = [_make_point(i) for i in range(10)]
    mock_client.query_points.return_value = MagicMock(points=points)
    mock_ranker.rerank.return_value = _make_rerank_results(points, score=0.9)

    result = await retrieve("what is RAG?", user_id="user-1")

    assert isinstance(result, RetrievalResult)
    assert result.sufficient is True
    assert len(result.chunks) == 5  # default retrieval_top_k


@pytest.mark.asyncio
@patch("src.retriever._ranker")
@patch("src.retriever.client")
@patch("src.retriever.embed_sparse", return_value=[FAKE_SPARSE])
@patch("src.retriever.embed_texts", return_value=[FAKE_DENSE])
async def test_no_results(mock_embed, mock_sparse, mock_client, mock_ranker):
    mock_client.query_points.return_value = MagicMock(points=[])

    result = await retrieve("what is RAG?", user_id="user-1")

    assert result.sufficient is False
    assert result.chunks == []
    mock_ranker.rerank.assert_not_called()


@pytest.mark.asyncio
@patch("src.retriever._ranker")
@patch("src.retriever.client")
@patch("src.retriever.embed_sparse", return_value=[FAKE_SPARSE])
@patch("src.retriever.embed_texts", return_value=[FAKE_DENSE])
async def test_low_confidence(mock_embed, mock_sparse, mock_client, mock_ranker):
    points = [_make_point(i) for i in range(5)]
    mock_client.query_points.return_value = MagicMock(points=points)
    mock_ranker.rerank.return_value = _make_rerank_results(points, score=0.01)  # below threshold

    result = await retrieve("what is RAG?", user_id="user-1")

    assert result.sufficient is False


@pytest.mark.asyncio
@patch("src.retriever._ranker")
@patch("src.retriever.client")
@patch("src.retriever.embed_sparse", return_value=[FAKE_SPARSE])
@patch("src.retriever.embed_texts", return_value=[FAKE_DENSE])
async def test_user_id_isolation(mock_embed, mock_sparse, mock_client, mock_ranker):
    points = [_make_point(i, user_id="user-abc") for i in range(5)]
    mock_client.query_points.return_value = MagicMock(points=points)
    mock_ranker.rerank.return_value = _make_rerank_results(points)

    await retrieve("question", user_id="user-abc")

    call_kwargs = mock_client.query_points.call_args.kwargs
    must_conditions = call_kwargs["query_filter"].must
    user_id_condition = next(c for c in must_conditions if c.key == "user_id")
    assert user_id_condition.match.value == "user-abc"


@pytest.mark.asyncio
@patch("src.retriever._ranker")
@patch("src.retriever.client")
@patch("src.retriever.embed_sparse", return_value=[FAKE_SPARSE])
@patch("src.retriever.embed_texts", return_value=[FAKE_DENSE])
async def test_doc_ids_filter(mock_embed, mock_sparse, mock_client, mock_ranker):
    points = [_make_point(i) for i in range(5)]
    mock_client.query_points.return_value = MagicMock(points=points)
    mock_ranker.rerank.return_value = _make_rerank_results(points)

    await retrieve("question", user_id="user-1", doc_ids=["doc-abc"])

    call_kwargs = mock_client.query_points.call_args.kwargs
    must_conditions = call_kwargs["query_filter"].must
    doc_id_condition = next(c for c in must_conditions if c.key == "doc_id")
    assert "doc-abc" in doc_id_condition.match.any


def test_prompt_builder():
    chunks = [
        RetrievedChunk(text=f"text {i}", doc_id=f"doc-{i}", source_type="pdf",
                       chunk_index=i, page_number=i + 1, reranker_score=0.9)
        for i in range(3)
    ]
    messages = build_prompt("what is RAG?", chunks)

    system = messages[0]["content"]
    assert messages[1]["content"] == "what is RAG?"
    for chunk in chunks:
        assert chunk.text in system
        assert chunk.doc_id in system