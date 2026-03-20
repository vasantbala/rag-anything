import asyncio
import logging
import os
import tempfile
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, status

from src.api.deps import get_current_user
from src.api.models import DbSourceRequest, DocumentResponse, IngestResponse
from src.auth import UserContext
from src.ingestor import ingest_file
from src.storage.dynamo import (
    delete_document,
    get_document_config,
    list_documents,
    put_document,
    update_document_status,
)
from src.storage.s3 import delete_file, upload_file
from src.vector_store import delete_by_doc_id

logger = logging.getLogger(__name__)

router = APIRouter()

_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
_FILE_SOURCE_TYPES = {".pdf", ".docx", ".txt", ".md"}


# ---------------------------------------------------------------------------
# Background helpers
# ---------------------------------------------------------------------------


async def _run_file_ingestion(
    tmp_path: str,
    filename: str,
    file_bytes: bytes,
    doc_id: str,
    user_id: str,
    source_type: str,
) -> None:
    try:
        result = await ingest_file(
            file_path=tmp_path,
            filename=filename,
            doc_id=doc_id,
            user_id=user_id,
            source_type=source_type,
        )
        await upload_file(file_bytes, user_id, doc_id, filename)
        await update_document_status(
            user_id, doc_id, "ready", chunk_count=result["chunk_count"]
        )
    except Exception:
        logger.exception("Ingestion failed for doc_id=%s user_id=%s", doc_id, user_id)
        await update_document_status(user_id, doc_id, "failed")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/upload", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    current_user: UserContext = Depends(get_current_user),
) -> IngestResponse:
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()

    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(_ALLOWED_EXTENSIONS)}",
        )

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    doc_id = str(uuid.uuid4())

    # Write to a named temp file — ingest_file expects a path, not bytes
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(file_bytes)
    tmp.flush()
    tmp.close()

    await put_document(
        user_id=current_user.user_id,
        doc_id=doc_id,
        name=filename,
        source_type=ext.lstrip("."),
        status="processing",
    )

    background_tasks.add_task(
        _run_file_ingestion,
        tmp.name,
        filename,
        file_bytes,
        doc_id,
        current_user.user_id,
        ext.lstrip("."),
    )

    return IngestResponse(doc_id=doc_id, status="processing")


@router.post("/db", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def register_db_source(
    config: DbSourceRequest,
    background_tasks: BackgroundTasks,
    current_user: UserContext = Depends(get_current_user),
) -> IngestResponse:
    from src.ingestor import ingest_db_table  # deferred — optional dependency

    doc_id = str(uuid.uuid4())

    # Encrypt connection string before storing
    try:
        from src.storage.secrets import encrypt_secret  # noqa: PLC0415
        encrypted_payload = encrypt_secret(config.connection_string)
    except ImportError:
        logger.warning("secrets module not available — storing connection_string unencrypted (dev only)")
        encrypted_payload = {"ciphertext": config.connection_string, "encrypted": False}

    await put_document(
        user_id=current_user.user_id,
        doc_id=doc_id,
        name=config.table_name,
        source_type="db_table",
        status="processing",
    )

    background_tasks.add_task(
        _run_db_ingestion,
        config,
        encrypted_payload,
        doc_id,
        current_user.user_id,
    )

    return IngestResponse(doc_id=doc_id, status="processing")


async def _run_db_ingestion(config: DbSourceRequest, encrypted_payload: dict, doc_id: str, user_id: str) -> None:
    try:
        from src.ingestor import ingest_db_table  # noqa: PLC0415
        result = await ingest_db_table(
            connection_string=config.connection_string,
            table_name=config.table_name,
            content_columns=config.content_columns,
            metadata_columns=config.metadata_columns,
            row_id_column=config.row_id_column,
            doc_id=doc_id,
            user_id=user_id,
        )
        await update_document_status(user_id, doc_id, "ready", chunk_count=result["chunk_count"])
    except Exception:
        logger.exception("DB ingestion failed for doc_id=%s user_id=%s", doc_id, user_id)
        await update_document_status(user_id, doc_id, "failed")


@router.get("", response_model=list[DocumentResponse])
async def list_user_documents(
    current_user: UserContext = Depends(get_current_user),
) -> list[DocumentResponse]:
    items = await list_documents(user_id=current_user.user_id)
    return [
        DocumentResponse(
            doc_id=item["doc_id"],
            name=item["name"],
            source_type=item["source_type"],
            status=item["status"],
            chunk_count=int(item.get("chunk_count", 0)),
            created_at=item["created_at"],
        )
        for item in items
    ]


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_document(
    doc_id: str,
    current_user: UserContext = Depends(get_current_user),
) -> None:
    doc = await get_document_config(current_user.user_id, doc_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )

    # Remove vectors from Qdrant
    await asyncio.to_thread(delete_by_doc_id, doc_id, current_user.user_id)

    # Remove metadata from DynamoDB
    await delete_document(current_user.user_id, doc_id)

    # Remove raw file from S3 if it's a file-based document
    if doc.get("source_type") in {st.lstrip(".") for st in _FILE_SOURCE_TYPES}:
        await delete_file(current_user.user_id, doc_id, doc["name"])
