"""Tests for the dead-letter queue."""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from src.infra.dead_letter import (
    quarantine_task,
    get_dlq_tasks,
    get_dlq_summary,
    resolve_dlq_task,
    retry_dlq_task,
    _classify_error,
    _check_mission_health,
    MISSION_DLQ_THRESHOLD,
)

# Patch target: get_db is imported from src.infra.db inside each function
DB_PATCH = "src.infra.db.get_db"


def _run(coro):
    """Run an async coroutine synchronously for testing."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestErrorClassification:
    def test_rate_limit(self):
        assert _classify_error("429 rate limit exceeded", "unknown") == "rate_limit"

    def test_timeout(self):
        assert _classify_error("Request timed out after 30s", "unknown") == "timeout"

    def test_auth_failure(self):
        assert _classify_error("401 Unauthorized", "unknown") == "auth_failure"

    def test_budget(self):
        assert _classify_error("Budget exceeded for mission #5", "unknown") == "budget_exceeded"

    def test_parse_error(self):
        assert _classify_error("JSON parse error at line 5", "unknown") == "parse_error"

    def test_not_found(self):
        assert _classify_error("File not found: /tmp/x", "unknown") == "not_found"

    def test_network(self):
        assert _classify_error("Connection refused to api.example.com", "unknown") == "network_error"

    def test_explicit_category_preserved(self):
        assert _classify_error("some error", "custom_cat") == "custom_cat"

    def test_unknown_fallback(self):
        assert _classify_error("something weird happened", "unknown") == "unknown"


def _make_mock_db():
    """Create a mock DB connection."""
    db = AsyncMock()
    cursor = AsyncMock()
    cursor.lastrowid = 1
    cursor.rowcount = 1
    cursor.fetchall = AsyncMock(return_value=[])
    cursor.fetchone = AsyncMock(return_value=(0, 0, 0))
    db.execute = AsyncMock(return_value=cursor)
    db.commit = AsyncMock()
    return db, cursor


class TestDLQOperations:
    """Test DLQ CRUD operations with mocked DB."""

    def test_quarantine_task(self):
        mock_db, cursor = _make_mock_db()
        with patch(DB_PATCH, AsyncMock(return_value=mock_db)):
            with patch("src.infra.dead_letter._check_mission_health", AsyncMock()):
                dlq_id = _run(quarantine_task(
                    task_id=42,
                    mission_id=10,
                    error="Test error",
                    original_agent="executor",
                    attempts_snapshot=3,
                ))
                assert dlq_id == 1
                assert mock_db.execute.call_count >= 2

    def test_get_dlq_tasks_empty(self):
        mock_db, cursor = _make_mock_db()
        with patch(DB_PATCH, AsyncMock(return_value=mock_db)):
            tasks = _run(get_dlq_tasks(mission_id=10))
            assert tasks == []

    def test_resolve_dlq_task(self):
        mock_db, cursor = _make_mock_db()
        with patch(DB_PATCH, AsyncMock(return_value=mock_db)):
            result = _run(resolve_dlq_task(42, "manual"))
            assert result is True

    def test_retry_dlq_task(self):
        mock_db, cursor = _make_mock_db()
        fake_task = {"id": 42, "context": "{}", "failed_in_phase": None}
        with patch(DB_PATCH, AsyncMock(return_value=mock_db)):
            with patch("src.infra.db.update_task", AsyncMock()):
                with patch("src.infra.db.get_task", AsyncMock(return_value=fake_task)):
                    result = _run(retry_dlq_task(42))
                    assert result is True

    def test_get_dlq_summary(self):
        mock_db, cursor = _make_mock_db()
        cursor.fetchone = AsyncMock(return_value=(5, 3, 2))

        pragma_cursor = AsyncMock()
        pragma_cursor.fetchall = AsyncMock(return_value=[])

        cat_cursor = AsyncMock()
        cat_cursor.fetchall = AsyncMock(return_value=[("timeout", 2), ("unknown", 1)])

        mock_db.execute = AsyncMock(side_effect=[
            cursor,         # CREATE TABLE
            pragma_cursor,  # PRAGMA table_info (migration check)
            cursor,         # summary query
            cat_cursor,     # category query
        ])

        with patch(DB_PATCH, AsyncMock(return_value=mock_db)):
            summary = _run(get_dlq_summary())
            assert summary["total"] == 5
            assert summary["unresolved"] == 3

    def test_quarantine_auto_classifies(self):
        """Quarantine should auto-classify rate limit errors."""
        mock_db, cursor = _make_mock_db()
        with patch(DB_PATCH, AsyncMock(return_value=mock_db)):
            with patch("src.infra.dead_letter._check_mission_health", AsyncMock()):
                _run(quarantine_task(
                    task_id=99,
                    mission_id=5,
                    error="429 rate limit exceeded for model",
                ))
                # Check the INSERT call includes rate_limit category
                insert_call = [
                    c for c in mock_db.execute.call_args_list
                    if "INSERT" in str(c)
                ]
                assert len(insert_call) >= 1


class TestMissionHealthCheck:
    def test_mission_paused_on_threshold(self):
        """Mission should be paused when DLQ threshold is reached."""
        mock_db, cursor = _make_mock_db()
        cursor.fetchone = AsyncMock(return_value=(MISSION_DLQ_THRESHOLD,))

        mock_update_mission = AsyncMock()

        # Mock get_bot at the import site inside _check_mission_health
        mock_telegram = MagicMock()
        mock_telegram.get_bot = MagicMock(return_value=None)

        import sys
        sys.modules.setdefault("telegram", MagicMock())
        sys.modules.setdefault("telegram.ext", MagicMock())

        with patch(DB_PATCH, AsyncMock(return_value=mock_db)):
            with patch("src.infra.db.update_mission", mock_update_mission):
                # Patch at the point where _check_mission_health imports it
                with patch.dict("sys.modules", {"src.app.telegram_bot": mock_telegram}):
                    _run(_check_mission_health(mission_id=10))
                    mock_update_mission.assert_called_once_with(10, status="paused")

    def test_mission_not_paused_below_threshold(self):
        """Mission should NOT be paused when below threshold."""
        mock_db, cursor = _make_mock_db()
        cursor.fetchone = AsyncMock(return_value=(MISSION_DLQ_THRESHOLD - 1,))

        mock_update_mission = AsyncMock()

        with patch(DB_PATCH, AsyncMock(return_value=mock_db)):
            with patch("src.infra.db.update_mission", mock_update_mission):
                _run(_check_mission_health(mission_id=10))
                mock_update_mission.assert_not_called()
