"""Local filesystem adapter that mirrors the interface of storage/s3.py.
Used when STORAGE_BACKEND=local — no AWS credentials required.

Files are stored at:  {local_storage_path}/files/{user_id}/{doc_id}/{filename}
"""

import asyncio
from pathlib import Path

from src.config import settings

# Reuse the same validation logic as the real S3 adapter
_MAGIC_BYTES: dict[bytes, str] = {
    b"%PDF": "application/pdf",
    b"PK\x03\x04": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

_TEXT_EXTENSIONS = {".txt", ".md"}


def _validate_mime(file_bytes: bytes, filename: str) -> str:
    ext = f".{filename.rsplit('.', 1)[-1].lower()}"
    if ext in _TEXT_EXTENSIONS:
        return "text/plain" if ext == ".txt" else "text/markdown"
    for magic, content_type in _MAGIC_BYTES.items():
        if file_bytes[: len(magic)] == magic:
            return content_type
    raise ValueError("File content does not match a supported type. Allowed: PDF, DOCX, TXT, MD")


def _file_path(user_id: str, doc_id: str, filename: str) -> Path:
    return Path(settings.local_storage_path) / "files" / user_id / doc_id / filename


async def upload_file(file_bytes: bytes, user_id: str, doc_id: str, filename: str) -> str:
    _validate_mime(file_bytes, filename)
    path = _file_path(user_id, doc_id, filename)

    loop = asyncio.get_event_loop()

    def _write() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(file_bytes)

    await loop.run_in_executor(None, _write)
    return f"{user_id}/{doc_id}/{filename}"


async def delete_file(user_id: str, doc_id: str, filename: str) -> None:
    path = _file_path(user_id, doc_id, filename)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: path.unlink(missing_ok=True))


async def generate_presigned_url(
    user_id: str,
    doc_id: str,
    filename: str,
    expires_in: int = 300,  # noqa: ARG001 — no expiry concept for local files
) -> dict:
    """Returns the same {url, fields} shape as the real S3 presigned POST.

    The returned URL points to the local FastAPI upload endpoint
    (POST /internal/upload/{user_id}/{doc_id}/{filename}) which accepts
    the identical multipart form that the browser would send directly to S3.
    """
    url = f"{settings.app_base_url}/internal/upload/{user_id}/{doc_id}/{filename}"
    return {"url": url, "fields": {}}
