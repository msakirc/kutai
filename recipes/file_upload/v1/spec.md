# File Upload Recipe (v1)

## Scope
Multipart file upload with two storage adapters (local filesystem and S3-compatible),
file metadata persisted to a DB table, and a metadata retrieval endpoint.

## When to pick
- Users or agents need to upload files (images, PDFs, documents)
- Local adapter: development, single-instance deployments without cloud storage
- S3 adapter: multi-instance or cloud deployments, large file volumes

## When NOT to pick
- Streaming media (video/audio): S3 presigned URLs for direct browser upload are better
- Files >1GB: multipart streaming via presigned URL is more appropriate
- Temporary scratch files that don't need metadata persistence

## Shape
- `POST /upload` — multipart/form-data, returns `{"file_id": N, "sha256": "..."}`
- `GET /files/{id}/meta` — returns file metadata (no content download in v1)
- File content is NOT served via the API in v1 — add a download route post-instantiation
- `files` table: id, owner_user_id FK, original_name, mime_type, byte_size, storage_path, sha256, created_at

### Local storage adapter
- Writes to `UPLOAD_DIR/<sha256[:2]>/<sha256>.<ext>` (content-addressable layout)
- Deduplication: if sha256 already exists, returns existing file_id without re-writing

### S3 adapter
- Bucket name from env var `S3_BUCKET_ENV` (default: `UPLOAD_S3_BUCKET`)
- Key: `uploads/<sha256[:2]>/<sha256>` — content-addressable
- Uses `boto3` with async executor wrapper (not aioboto3 — avoids extra dep)

## Tradeoffs
- No streaming upload in v1 — entire file is read into memory before write. Set `MAX_FILE_BYTES` to cap memory usage.
- `ALLOWED_MIME_TYPES` is enforced server-side via Content-Type header only — not a magic-byte check. Add python-magic for production hardening.
- sha256 deduplication is content-based; different filenames with identical content share one storage slot (intended).
- S3 writes are synchronous in the request path (via executor). Offload to a task queue for large files.
