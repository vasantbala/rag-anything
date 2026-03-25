"""Local-only upload endpoint — handles the multipart POST that the browser
sends directly to S3 when using presigned URLs.  In local mode,
generate_presigned_url() returns a URL that points here instead of S3.

Only registered in the FastAPI app when STORAGE_BACKEND=local.
"""

from fastapi import APIRouter, HTTPException, UploadFile, status

from src.storage.local_s3 import upload_file

router = APIRouter(tags=["local-dev"])


@router.post("/internal/upload/{user_id}/{doc_id}/{filename}")
async def local_presigned_upload(
    user_id: str,
    doc_id: str,
    filename: str,
    file: UploadFile,
) -> dict:
    """Accepts the same multipart form the browser would POST directly to S3."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file.")
    await upload_file(data, user_id, doc_id, filename)
    return {"status": "ok"}
