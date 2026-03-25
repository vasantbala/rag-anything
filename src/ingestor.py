import os
from src.chunker import chunk_documents
from src.embedder import embed_texts
from src.sparse_embedder import embed_sparse
from src.storage.factory import delete_file, update_document_status
from src.vector_store import upsert_chunks
from src.api.loaders.pdf_loader import PdfLoader
from src.api.loaders.docx_loader import DocxLoader
#from src.api.loaders.txt_loader import TxtLoader
#from src.api.loaders.md_loader import MdLoader
from src.config import settings

_LOADERS = {
    ".pdf": PdfLoader,
    ".docx": DocxLoader,
    #".txt": TxtLoader,
    #".md": MdLoader,
}

async def ingest_file(
    file_path: str,
    filename: str,
    doc_id: str,
    user_id: str,
    source_type: str,
) -> dict:
    # File size guard — reject before doing any expensive work
    file_size = os.path.getsize(file_path)
    if file_size > settings.max_file_size_bytes:
        raise ValueError(
            f"File exceeds maximum allowed size of {settings.max_file_size_bytes // (1024 * 1024)}MB"
        )

    ext = os.path.splitext(filename)[1].lower()
    loader_class = _LOADERS.get(ext)
    if loader_class is None:
        raise ValueError(f"Unsupported file type: {ext}")

    try:
        loader = loader_class()
        pages = loader.load(file_path)

        if not pages:
            raise ValueError(f"No extractable text found in {filename}")

        chunks = chunk_documents(pages)
        texts = [c[0] for c in chunks]

        dense_vectors = await embed_texts(texts)
        sparse_vectors = embed_sparse(texts)

        upsert_chunks(chunks, dense_vectors, sparse_vectors, doc_id, user_id, source_type)

    except Exception:
        # Mark document as failed in DynamoDB so it doesn't stay stuck in "processing"
        await update_document_status(user_id, doc_id, status="failed", expected_status="processing")
        # Also delete the raw file from S3 - no orphaned files on failed ingestion
        await delete_file(user_id, doc_id, filename)
        raise

    finally:
        # Always clean up the temp file — Lambda /tmp persists across warm invocations
        if os.path.exists(file_path):
            os.remove(file_path)

    return {"doc_id": doc_id, "chunk_count": len(chunks)}