# File upload recipe — FastAPI stack-specific notes

- Declare the upload parameter as `file: UploadFile = File(...)` — not `bytes`. UploadFile gives streaming access; `bytes` reads the whole body eagerly.
- Add `MAX_FILE_BYTES` check during streaming read: read in chunks, count bytes, raise `HTTPException(413)` on overflow.
- `python-multipart` is required by FastAPI for file uploads. Add to `requirements.txt`.
- For local storage, use `aiofiles` for async writes to avoid blocking the event loop.
- For S3, wrap `boto3.client.put_object` in `asyncio.get_event_loop().run_in_executor(None, ...)` — aioboto3 is an optional dep, not default.
- Return `sha256` in the upload response so the client can verify integrity without a follow-up GET.
- The `GET /files/{id}/meta` route must NOT return the storage_path — that's an internal implementation detail. Return only id, original_name, mime_type, byte_size, sha256, created_at.
- Set `Content-Disposition: attachment` on any download route you add post-instantiation — browsers may otherwise inline PDFs.
