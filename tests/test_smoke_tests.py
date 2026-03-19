import asyncio
import shutil
from src.ingestor import ingest_file
from src.storage.dynamo import put_document
from src.vector_store import _client as qdrant_client, ensure_collection, delete_by_doc_id
from src.embedder import get_embedder
from src.config import settings

PDF_SRC = "tests/Uber-selfdriving.pdf"
PDF_COPY = "tests/Uber-selfdriving-run.pdf"
USER_ID = "smoke-user-001"
DOC_ID  = "smoke-doc-001"

async def main():
    # 0. Ensure collection exists with correct config
    ensure_collection()
    print("Collection ready.")

    # 1. Sanity-check embedding dimensions before ingestion
    embedder = get_embedder()
    sample_vectors = await embedder.embed_texts(["hello world"])
    actual_dims = len(sample_vectors[0])
    print(f"Embedding provider: {embedder.__class__.__name__}")
    print(f"Embedding dimensions: {actual_dims} (expected: {settings.embedding_dimensions})")
    assert actual_dims == settings.embedding_dimensions, (
        f"Dimension mismatch! Model returns {actual_dims}, collection expects {settings.embedding_dimensions}"
    )

    # 2. Clean slate — delete any vectors from a previous run
    delete_by_doc_id(DOC_ID, USER_ID)
    print("Cleaned up previous vectors.")

    # 3. Copy PDF so ingest_file's cleanup doesn't destroy the original
    shutil.copy(PDF_SRC, PDF_COPY)

    # 4. Create DynamoDB record (required by ingest_file's error handler)
    await put_document(
        user_id=USER_ID,
        doc_id=DOC_ID,
        name="Uber-selfdriving.pdf",
        source_type="pdf",
        status="processing",
    )
    print("DynamoDB record created.")

    # 5. Ingest — loads, chunks, embeds (OpenRouter), upserts to Qdrant
    result = await ingest_file(
        file_path=PDF_COPY,
        filename="Uber-selfdriving.pdf",
        doc_id=DOC_ID,
        user_id=USER_ID,
        source_type="pdf",
    )
    print(f"Ingestion result: {result}")

    # 6. Verify Qdrant has the correct number of vectors for this doc
    count_result = qdrant_client.count(
        collection_name=settings.qdrant.collection_name,
        count_filter={
            "must": [
                {"key": "doc_id", "match": {"value": DOC_ID}},
                {"key": "user_id", "match": {"value": USER_ID}},
            ]
        },
        exact=True,
    )
    print(f"Qdrant vectors for doc: {count_result.count}")
    assert count_result.count == result["chunk_count"], (
        f"Expected {result['chunk_count']} vectors in Qdrant, found {count_result.count}"
    )
    print("All checks passed.")

if __name__ == "__main__":
    asyncio.run(main())

