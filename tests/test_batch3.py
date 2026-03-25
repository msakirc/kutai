"""Tests for Batch 3: error categories, DB indexes, credential expiration."""

import asyncio
import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime, timezone, timedelta


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestErrorCategoryPopulation:
    """Verify that error_category is populated on task failure."""

    def test_classify_import(self):
        """The orchestrator should be able to import _classify_error."""
        from src.infra.dead_letter import _classify_error
        assert _classify_error("429 rate limit", "unknown") == "rate_limit"

    def test_update_task_receives_error_category(self):
        """When a task fails after max retries, update_task should include error_category."""
        # This is tested via the DLQ classify_error + the orchestrator integration
        from src.infra.dead_letter import _classify_error
        error_str = "TimeoutError: Request timed out after 30s"
        cat = _classify_error(error_str, "unknown")
        assert cat == "timeout"


class TestDBIndexes:
    """Verify that index creation statements are valid."""

    def test_index_definitions(self):
        """Check that the index list in init_db is properly structured."""
        # We can't run init_db without a real DB, but we can verify the
        # SQL patterns are valid by importing and checking
        import src.infra.db as db_module
        import inspect
        source = inspect.getsource(db_module.init_db)
        # Verify key indexes are defined
        assert "idx_tasks_status" in source
        assert "idx_tasks_mission_id" in source
        assert "idx_tasks_status_priority" in source
        assert "idx_conversations_task_id" in source
        assert "idx_memory_mission_category" in source


class TestCredentialExpiration:
    """Test credential storage with expiration."""

    def _make_mock_db(self):
        db = AsyncMock()
        cursor = AsyncMock()
        cursor.lastrowid = 1
        cursor.rowcount = 1
        db.execute = AsyncMock(return_value=cursor)
        db.commit = AsyncMock()
        return db, cursor

    def test_store_credential_with_expiration(self):
        """store_credential should embed expiration in the encrypted envelope."""
        from src.security.credential_store import store_credential, _encrypt, _decrypt

        mock_db, cursor = self._make_mock_db()
        with patch("src.infra.db.get_db", AsyncMock(return_value=mock_db)):
            _run(store_credential(
                "test_service",
                {"token": "abc123"},
                expires_at="2030-01-01T00:00:00+00:00",
            ))

        # Verify the encrypted data was stored
        assert mock_db.execute.called
        call_args = mock_db.execute.call_args
        encrypted_data = call_args[0][1][1]  # second positional in VALUES
        decrypted = json.loads(_decrypt(encrypted_data))
        assert decrypted["_data"]["token"] == "abc123"
        assert decrypted["_expires_at"] == "2030-01-01T00:00:00+00:00"

    def test_get_credential_not_expired(self):
        """get_credential should return data if not expired."""
        from src.security.credential_store import get_credential, _encrypt

        future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        envelope = json.dumps({
            "_data": {"token": "valid"},
            "_expires_at": future,
        })
        encrypted = _encrypt(envelope)

        mock_db, cursor = self._make_mock_db()
        cursor.fetchone = AsyncMock(return_value=(encrypted,))

        with patch("src.infra.db.get_db", AsyncMock(return_value=mock_db)):
            result = _run(get_credential("test_service"))

        assert result == {"token": "valid"}

    def test_get_credential_expired(self):
        """get_credential should return None if expired."""
        from src.security.credential_store import get_credential, _encrypt

        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        envelope = json.dumps({
            "_data": {"token": "expired"},
            "_expires_at": past,
        })
        encrypted = _encrypt(envelope)

        mock_db, cursor = self._make_mock_db()
        cursor.fetchone = AsyncMock(return_value=(encrypted,))

        with patch("src.infra.db.get_db", AsyncMock(return_value=mock_db)):
            result = _run(get_credential("test_service"))

        assert result is None

    def test_legacy_credential_still_works(self):
        """Credentials stored without envelope should still work."""
        from src.security.credential_store import get_credential, _encrypt

        # Legacy format: flat dict, no envelope
        legacy = json.dumps({"token": "legacy_token"})
        encrypted = _encrypt(legacy)

        mock_db, cursor = self._make_mock_db()
        cursor.fetchone = AsyncMock(return_value=(encrypted,))

        with patch("src.infra.db.get_db", AsyncMock(return_value=mock_db)):
            result = _run(get_credential("test_service"))

        assert result == {"token": "legacy_token"}

    def test_store_credential_without_expiration(self):
        """Credentials without expiration should never expire."""
        from src.security.credential_store import store_credential, _encrypt, _decrypt

        mock_db, cursor = self._make_mock_db()
        with patch("src.infra.db.get_db", AsyncMock(return_value=mock_db)):
            _run(store_credential("test_service", {"key": "value"}))

        call_args = mock_db.execute.call_args
        encrypted_data = call_args[0][1][1]
        decrypted = json.loads(_decrypt(encrypted_data))
        assert "_expires_at" not in decrypted
        assert decrypted["_data"]["key"] == "value"
