import asyncio
import boto3
from src.config import settings

_s3 = boto3.client(
    "s3",
    region_name=settings.aws_region,
    aws_access_key_id=settings.aws_access_key_id,
    aws_secret_access_key=settings.aws_secret_access_key,
)

# Magic bytes for allowed file types — checked before any upload
_MAGIC_BYTES: dict[bytes, str] = {
    b"%PDF": "application/pdf",
    b"PK\x03\x04": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

_TEXT_EXTENSIONS = {".txt", ".md"}

async def _run(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

def _validate_mime(file_bytes: bytes, filename: str) -> str:
    ext = f".{filename.rsplit('.', 1)[-1].lower()}"

    # Text files have no reliable magic bytes — trust extension only
    if ext in _TEXT_EXTENSIONS:
        return "text/plain" if ext == ".txt" else "text/markdown"

    # Binary formats: check magic bytes
    for magic, content_type in _MAGIC_BYTES.items():
        if file_bytes[:len(magic)] == magic:
            return content_type

    raise ValueError(
        f"File content does not match a supported type. "
        f"Allowed: PDF, DOCX, TXT, MD"
    )

async def upload_file(file_bytes: bytes, user_id: str, doc_id: str, filename: str) -> str:
    content_type = _validate_mime(file_bytes, filename)   # Validate before upload

    key = f"{user_id}/{doc_id}/{filename}"
    await _run(
        _s3.put_object,
        Bucket=settings.s3_bucket_name,
        Key=key,
        Body=file_bytes,
        ContentType=content_type,
        ServerSideEncryption="aws:kms",
    )
    return key

async def delete_file(user_id: str, doc_id: str, filename: str) -> None:
    key = f"{user_id}/{doc_id}/{filename}"
    await _run(
        _s3.delete_object,
        Bucket=settings.s3_bucket_name,
        Key=key,
    )    

# Pre-signed URL for direct client-to-S3 uploads — bypasses Lambda/API GW payload limits
async def generate_presigned_url(
    user_id: str,
    doc_id: str,
    filename: str,
    expires_in: int = 300,
) -> dict:
    key = f"{user_id}/{doc_id}/{filename}"
    # presigned_post returns {"url": ..., "fields": ...}
    # The client POSTs directly to S3 using these — no Lambda in the upload path
    response = await _run(
        _s3.generate_presigned_post,
        Bucket=settings.s3_bucket_name,
        Key=key,
        Conditions=[
            ["content-length-range", 1, settings.max_file_size_bytes],
        ],
        ExpiresIn=expires_in,
    )
    return response    