from __future__ import annotations

import flashrank
from dataclasses import dataclass, field
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, MatchAny,
    Prefetch, FusionQuery, Fusion, ScoredPoint, SparseVector,
)

from src.config import settings
from src.embedder import embed_texts
from src.sparse_embedder import embed_sparse
from src.vector_store import _client as client

# Module-level — loaded once, not per request
_ranker = flashrank.Ranker(model_name="ms-marco-MiniLM-L-12-v2")

@dataclass
class RetrievedChunk:
    text: str
    doc_id: str
    source_type: str
    chunk_index: int
    page_number: int | None
    reranker_score: float
    metadata: dict = field(default_factory=dict)

@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    sufficient: bool    


async def _embed_question(question: str) -> tuple[list[float], SparseVector]:
    dense = (await embed_texts([question]))[0]
    sparse = embed_sparse([question.lower()])[0]  # must match ingestion normalization
    return dense, sparse

def _qdrant_hybrid_search(
    dense_vec: list[float],
    sparse_vec: SparseVector,
    user_id: str,
    doc_ids: list[str] | None = None,
    limit: int = 20,
) -> list[ScoredPoint]:
    must = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
    if doc_ids:
        must.append(FieldCondition(key="doc_id", match=MatchAny(any=doc_ids)))

    f = Filter(must=must)
    response = client.query_points(
        collection_name=settings.qdrant.collection_name,
        prefetch=[
            Prefetch(query=dense_vec, using="dense", limit=limit, filter=f),
            Prefetch(query=sparse_vec, using="sparse", limit=limit, filter=f),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return response.points


def _rerank(
    question: str,
    points: list[ScoredPoint],
    top_k: int,
) -> list[tuple[ScoredPoint, float]]:
    passages = [{"id": i, "text": p.payload["text"]} for i, p in enumerate(points)]
    results = _ranker.rerank(flashrank.RerankRequest(query=question, passages=passages))

    reranked = [(points[r["id"]], r["score"]) for r in results]
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:top_k]


async def retrieve(
    question: str,
    user_id: str,
    doc_ids: list[str] | None = None,
    top_k: int | None = None,
) -> RetrievalResult:
    dense_vec, sparse_vec = await _embed_question(question)

    points = _qdrant_hybrid_search(dense_vec, sparse_vec, user_id, doc_ids)
    if not points:
        return RetrievalResult(chunks=[], sufficient=False)

    reranked = _rerank(question, points, top_k=top_k or settings.retrieval_top_k)

    sufficient = reranked[0][1] >= settings.reranker_score_threshold

    chunks = []
    for point, score in reranked:
        p = point.payload
        known_keys = {"text", "doc_id", "source_type", "chunk_index", "page_number", "user_id"}
        chunks.append(RetrievedChunk(
            text=p["text"],
            doc_id=p["doc_id"],
            source_type=p.get("source_type", ""),
            chunk_index=p.get("chunk_index", 0),
            page_number=p.get("page_number"),
            reranker_score=float(score),
            metadata={k: v for k, v in p.items() if k not in known_keys},
        ))

    return RetrievalResult(chunks=chunks, sufficient=sufficient)