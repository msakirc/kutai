"""File upload recipe — backend smoke tests.

Runs POST-instantiation against the instantiated recipe in the mission
workspace. Recipe sources are scaffolds (.template suffix).
"""
from __future__ import annotations

import pytest


async def test_upload_returns_file_id_and_sha256():
    """POST /upload with valid file returns file_id (int) and sha256 (hex string)."""
    # T6 WILL FILL — POST multipart with small PNG, assert file_id > 0 and sha256 is 64 hex chars
    pass


async def test_upload_deduplication_returns_same_file_id():
    """Uploading the same file twice returns the same file_id."""
    # T6 WILL FILL — upload same bytes twice, assert file_id matches
    pass


async def test_upload_rejects_disallowed_mime_type():
    """POST /upload with disallowed MIME type returns 415."""
    # T6 WILL FILL — upload text/plain file, assert status_code == 415
    pass


async def test_upload_rejects_oversized_file():
    """POST /upload with file > MAX_FILE_BYTES returns 413."""
    # T6 WILL FILL — upload MAX_FILE_BYTES+1 bytes, assert status_code == 413
    pass


async def test_get_file_meta_returns_metadata():
    """GET /files/{id}/meta returns expected fields after upload."""
    # T6 WILL FILL — upload, then GET meta; assert original_name, mime_type, byte_size, sha256 match
    pass


async def test_get_file_meta_not_found_returns_404():
    """GET /files/99999/meta returns 404 for non-existent file."""
    # T6 WILL FILL — assert status_code == 404
    pass
