# File upload recipe — gotchas

- `Content-Type` header is user-supplied and trivially spoofed. `ALLOWED_MIME_TYPES` filtering on Content-Type is a UX guardrail, not a security boundary. Add `python-magic` byte sniffing for production.
- FastAPI `UploadFile.read()` loads the entire file into memory. Enforce `MAX_FILE_BYTES` by reading in chunks and aborting at the limit — don't read-then-check.
- `python-multipart` must be installed; FastAPI silently fails to parse multipart if it's missing.
- sha256 over the raw bytes is fine for deduplication but adds CPU for large files. Stream into a temp file while hashing concurrently.
- S3 `put_object` in a thread executor blocks the event loop thread pool. For >10MB files, consider pre-signed URL + client-direct upload.
- `UPLOAD_DIR` must be an absolute path in production — relative paths depend on the process working directory which can shift under supervisors.
- Local content-addressable layout `<sha256[:2]>/<sha256>` keeps directory entry counts manageable; flat `uploads/` dir hits inode limits at ~100k files on most filesystems.
- File metadata row insertion should happen BEFORE writing to storage so a crash during write doesn't leave orphan storage objects with no DB record.
- `owner_user_id` FK should allow NULL for unauthenticated upload scenarios. Don't add NOT NULL without confirming the auth model.
- Deleting files: the recipe doesn't add a DELETE route in v1. Deduplication means a file_id delete must check reference count before removing from storage.
- Mime type check on `application/octet-stream`: many browsers send this for unknown types. Handle as allowed or rejected explicitly via `ALLOWED_MIME_TYPES` — don't silently drop.
