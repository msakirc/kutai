"""File upload recipe — multipart upload + S3-compat/local storage + metadata.

RECIPE_PARAM markers:
  # RECIPE_PARAM:STORAGE_BACKEND=local
  # RECIPE_PARAM:UPLOAD_DIR=./uploads
  # RECIPE_PARAM:MAX_FILE_BYTES=10485760
  # RECIPE_PARAM:ALLOWED_MIME_TYPES=image/png,image/jpeg,application/pdf
  # RECIPE_PARAM:S3_BUCKET_ENV=UPLOAD_S3_BUCKET

Routes:
  POST /upload           — multipart upload → returns {file_id, sha256}
  GET  /files/{id}/meta  — file metadata (no content download in v1)

Storage adapters:
  local  — content-addressable filesystem under UPLOAD_DIR
  s3     — S3-compatible storage, bucket from env var S3_BUCKET_ENV
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

STORAGE_BACKEND = "local"  # RECIPE_PARAM:STORAGE_BACKEND=local
UPLOAD_DIR = "./uploads"  # RECIPE_PARAM:UPLOAD_DIR=./uploads
MAX_FILE_BYTES = 10485760  # RECIPE_PARAM:MAX_FILE_BYTES=10485760
ALLOWED_MIME_TYPES = "image/png,image/jpeg,application/pdf"  # RECIPE_PARAM:ALLOWED_MIME_TYPES=image/png,image/jpeg,application/pdf
S3_BUCKET_ENV = "UPLOAD_S3_BUCKET"  # RECIPE_PARAM:S3_BUCKET_ENV=UPLOAD_S3_BUCKET

_ALLOWED_MIME_SET = {m.strip() for m in ALLOWED_MIME_TYPES.split(",") if m.strip()}

router = APIRouter(tags=["files"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    file_id: int
    sha256: str
    original_name: str
    mime_type: str
    byte_size: int


class FileMeta(BaseModel):
    id: int
    original_name: str
    mime_type: str
    byte_size: int
    sha256: str
    created_at: str
    owner_user_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Streaming read + sha256 helper
# ---------------------------------------------------------------------------

async def _read_and_hash(upload: UploadFile, max_bytes: int) -> tuple[bytes, str]:
    """Stream-read the upload, hash with sha256, enforce max_bytes.

    Raises HTTPException(413) if file exceeds max_bytes.
    """
    h = hashlib.sha256()
    chunks: list[bytes] = []
    total = 0
    chunk_size = 65536
    while True:
        chunk = await upload.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: maximum {max_bytes} bytes allowed",
            )
        h.update(chunk)
        chunks.append(chunk)
    return b"".join(chunks), h.hexdigest()


# ---------------------------------------------------------------------------
# Storage adapters
# ---------------------------------------------------------------------------

async def _store_local(data: bytes, sha256: str, ext: str) -> str:
    """Write to content-addressable local path. Returns storage_path."""
    import aiofiles  # T6: confirm aiofiles in requirements.txt

    prefix = sha256[:2]
    target_dir = Path(UPLOAD_DIR) / prefix
    target_dir.mkdir(parents=True, exist_ok=True)
    storage_path = str(target_dir / f"{sha256}{ext}")
    if not Path(storage_path).exists():
        async with aiofiles.open(storage_path, "wb") as f:
            await f.write(data)
    return storage_path


async def _store_s3(data: bytes, sha256: str, ext: str) -> str:
    """Write to S3-compatible storage. Returns storage_path (s3://bucket/key)."""
    import asyncio
    import boto3  # T6: confirm boto3 in requirements.txt

    bucket = os.environ.get(S3_BUCKET_ENV, "")
    if not bucket:
        raise HTTPException(
            status_code=500,
            detail=f"S3 bucket not configured: env var {S3_BUCKET_ENV} is unset",
        )
    key = f"uploads/{sha256[:2]}/{sha256}{ext}"
    client = boto3.client("s3")  # T6: inject client from app state, not per-request
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: client.put_object(Bucket=bucket, Key=key, Body=data),
    )
    return f"s3://{bucket}/{key}"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

async def _db_find_by_sha256(db, sha256: str) -> Optional[dict]:
    """Return existing file record by sha256, or None."""
    cur = await db.execute(
        "SELECT id, original_name, mime_type, byte_size, sha256, storage_path, created_at "
        "FROM files WHERE sha256 = ? LIMIT 1",
        (sha256,),
    )
    row = await cur.fetchone()
    if row is None:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


async def _db_insert_file(
    db, original_name: str, mime_type: str, byte_size: int,
    storage_path: str, sha256: str, owner_user_id: Optional[int],
) -> int:
    """Insert file metadata row. Returns new row id."""
    cur = await db.execute(
        "INSERT INTO files (owner_user_id, original_name, mime_type, byte_size, storage_path, sha256) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (owner_user_id, original_name, mime_type, byte_size, storage_path, sha256),
    )
    await db.commit()
    return cur.lastrowid


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_file(
    file: UploadFile = File(...),
    owner_user_id: Optional[int] = None,  # T6: replace with Depends(get_current_user)
) -> UploadResponse:
    """Multipart file upload. Returns file_id and sha256 for integrity verification."""
    from src.infra.db import get_db  # T6: swap for project db getter

    # MIME type validation
    content_type = (file.content_type or "").split(";")[0].strip()
    if content_type not in _ALLOWED_MIME_SET:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {content_type}. Allowed: {ALLOWED_MIME_TYPES}",
        )

    data, sha256 = await _read_and_hash(file, MAX_FILE_BYTES)
    byte_size = len(data)

    db = await get_db()

    # Deduplication: reuse existing storage if sha256 matches
    existing = await _db_find_by_sha256(db, sha256)
    if existing:
        return UploadResponse(
            file_id=existing["id"],
            sha256=sha256,
            original_name=existing["original_name"],
            mime_type=existing["mime_type"],
            byte_size=existing["byte_size"],
        )

    # Determine file extension from original name for storage path
    original_name = file.filename or "upload"
    ext = Path(original_name).suffix.lower()

    if STORAGE_BACKEND == "s3":
        storage_path = await _store_s3(data, sha256, ext)
    else:
        storage_path = await _store_local(data, sha256, ext)

    file_id = await _db_insert_file(
        db,
        original_name=original_name,
        mime_type=content_type,
        byte_size=byte_size,
        storage_path=storage_path,
        sha256=sha256,
        owner_user_id=owner_user_id,
    )

    return UploadResponse(
        file_id=file_id,
        sha256=sha256,
        original_name=original_name,
        mime_type=content_type,
        byte_size=byte_size,
    )


@router.get("/files/{file_id}/meta", response_model=FileMeta)
async def get_file_meta(file_id: int) -> FileMeta:
    """Return file metadata. Does NOT expose storage_path."""
    from src.infra.db import get_db  # T6: swap for project db getter

    db = await get_db()
    cur = await db.execute(
        "SELECT id, owner_user_id, original_name, mime_type, byte_size, sha256, created_at "
        "FROM files WHERE id = ?",
        (file_id,),
    )
    row = await cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found")
    cols = [d[0] for d in cur.description]
    data = dict(zip(cols, row))
    return FileMeta(
        id=data["id"],
        original_name=data["original_name"],
        mime_type=data["mime_type"],
        byte_size=data["byte_size"],
        sha256=data["sha256"],
        created_at=str(data.get("created_at", "")),
        owner_user_id=data.get("owner_user_id"),
    )
