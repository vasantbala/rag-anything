import uuid

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    Filter,
    FilterSelector,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
    HnswConfigDiff
)

from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from qdrant_client.http.exceptions import UnexpectedResponse
from src.config import settings

"""
No HNSW tuning
Qdrant's default HNSW index parameters (m=16, ef_construct=100) are conservative. 
For 1M+ points, production systems tune these for their recall/latency tradeoff. 
Not relevant at this scale, but worth knowing the knob exists 
in VectorParams(hnsw_config=HnswConfigDiff(...)).
"""

_client = QdrantClient(url=settings.qdrant.url, api_key=settings.qdrant.api_key)

UPSERT_BATCH = 100

def ensure_collection() -> None:
    # Catch 409 race condition — two Lambda instances starting simultaneously
    try:
        if not _client.collection_exists(settings.qdrant.collection_name):
            _client.create_collection(
                collection_name=settings.qdrant.collection_name,
                vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
                },
                hnsw_config=HnswConfigDiff(payload_m=16, m=16),
            )
    except UnexpectedResponse as e:
        if e.status_code != 409:
            raise

    # Payload indexes — must exist before any query/delete that filters on these fields
    for field in ("user_id", "doc_id"):
        _client.create_payload_index(
            collection_name=settings.qdrant.collection_name,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
        )

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((UnexpectedResponse, httpx.ConnectError, httpx.TimeoutException)),
)
def _upsert_batch(points: list[PointStruct]) -> None:
    _client.upsert(collection_name=settings.qdrant.collection_name, points=points)
            
def upsert_chunks(
    chunks: list[tuple[str, dict]],
    dense_vectors: list[list[float]],
    sparse_vectors: list,
    doc_id: str,
    user_id: str,
    source_type: str,
) -> None:
    points = []
    for i, ((chunk_text, chunk_metadata), dense, sparse) in enumerate(
        zip(chunks, dense_vectors, sparse_vectors)
    ):
        chunk_index = chunk_metadata.get("chunk_index", i)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{user_id}_{doc_id}_{chunk_index}"))

        points.append(
            PointStruct(
                id=point_id,
                vector={"dense": dense, "sparse": sparse},
                payload={
                    "user_id": user_id,
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "source_type": source_type,
                    **chunk_metadata,
                },
            )
        )

    # Batch upserts — Qdrant recommends max 100 points per request
    for i in range(0, len(points), UPSERT_BATCH):
        _upsert_batch(points[i : i + UPSERT_BATCH])

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((UnexpectedResponse, httpx.ConnectError, httpx.TimeoutException)),
)
def delete_by_doc_id(doc_id: str, user_id: str) -> None:
    _client.delete(
        collection_name=settings.qdrant.collection_name,
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                ]
            )
        ),
    )