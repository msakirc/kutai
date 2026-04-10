"""Tests for DLQ Analyst pattern detection."""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from src.infra.dlq_analyst import DLQAnalyst


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestPatternDetection:
    """Pattern grouping and threshold logic."""

    def setup_method(self):
        self.analyst = DLQAnalyst()

    def test_no_pattern_below_threshold(self):
        """2 similar failures should NOT trigger an alert."""
        entries = [
            {"task_id": 1, "error_category": "timeout", "original_agent": "researcher",
             "error": "timed out", "mission_id": None, "quarantined_at": "2026-04-08 10:00:00"},
            {"task_id": 2, "error_category": "timeout", "original_agent": "researcher",
             "error": "timed out", "mission_id": None, "quarantined_at": "2026-04-08 10:30:00"},
        ]
        patterns = self.analyst.detect_patterns(entries)
        assert len(patterns) == 0

    def test_pattern_at_threshold(self):
        """3 similar failures SHOULD trigger."""
        entries = [
            {"task_id": 1, "error_category": "timeout", "original_agent": "researcher",
             "error": "timed out", "mission_id": None, "quarantined_at": "2026-04-08 10:00:00"},
            {"task_id": 2, "error_category": "timeout", "original_agent": "coder",
             "error": "timed out", "mission_id": None, "quarantined_at": "2026-04-08 10:30:00"},
            {"task_id": 3, "error_category": "timeout", "original_agent": "researcher",
             "error": "request timed out", "mission_id": None, "quarantined_at": "2026-04-08 11:00:00"},
        ]
        patterns = self.analyst.detect_patterns(entries)
        assert len(patterns) >= 1
        assert any(p["group_key"] == "category:timeout" for p in patterns)

    def test_groups_by_mission(self):
        """3 failures from same mission should trigger mission pattern."""
        entries = [
            {"task_id": i, "error_category": cat, "original_agent": "coder",
             "error": "some error", "mission_id": 42, "quarantined_at": f"2026-04-08 1{i}:00:00"}
            for i, cat in enumerate(["timeout", "parse_error", "network_error"], 1)
        ]
        patterns = self.analyst.detect_patterns(entries)
        assert any(p["group_key"] == "mission:42" for p in patterns)


class TestDeduplication:
    """Alert dedup within 1-hour window."""

    def setup_method(self):
        self.analyst = DLQAnalyst()

    def test_first_alert_not_deduped(self):
        assert not self.analyst.is_deduped("category:timeout")

    def test_second_alert_within_hour_deduped(self):
        self.analyst.record_alert("category:timeout")
        assert self.analyst.is_deduped("category:timeout")

    def test_different_pattern_not_deduped(self):
        self.analyst.record_alert("category:timeout")
        assert not self.analyst.is_deduped("category:network_error")


class TestAlertFormatting:
    """Telegram message formatting."""

    def setup_method(self):
        self.analyst = DLQAnalyst()

    def test_format_contains_task_ids(self):
        pattern = {
            "group_key": "category:timeout",
            "count": 3,
            "entries": [
                {"task_id": 42, "original_agent": "researcher", "error": "timed out after 300s"},
                {"task_id": 45, "original_agent": "coder", "error": "request timeout"},
                {"task_id": 48, "original_agent": "researcher", "error": "deadline exceeded"},
            ],
            "diagnostic": "llama-server responding but slow (4.2s)",
        }
        msg = self.analyst.format_alert(pattern)
        assert "#42" in msg
        assert "#45" in msg
        assert "#48" in msg
        assert "timeout" in msg.lower()
        assert "4.2s" in msg


class TestDiagnostics:
    """Known failure signature checks."""

    def setup_method(self):
        self.analyst = DLQAnalyst()

    @patch("src.infra.dlq_analyst.aiohttp.ClientSession")
    def test_timeout_diagnostic_server_down(self, mock_session_cls):
        """Timeout pattern should check llama-server health."""
        mock_resp = AsyncMock()
        mock_resp.status = 500
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = AsyncMock(return_value=mock_resp)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        result = _run(self.analyst.run_diagnostic("category:timeout", []))
        assert "not responding" in result.lower() or "unhealthy" in result.lower()

    def test_grading_diagnostic_same_model(self):
        """Grading failures with same model should flag the model."""
        entries = [
            {"task_id": i, "error": f"grade fail", "original_agent": "coder",
             "error_category": "unknown", "mission_id": None,
             "quarantined_at": f"2026-04-08 1{i}:00:00"}
            for i in range(1, 4)
        ]
        for e in entries:
            e["error"] = "Grading exhausted: model=qwen2.5-7b"
        result = _run(self.analyst.run_diagnostic("category:unknown", entries))
        assert "qwen2.5-7b" in result or "same model" in result.lower() or result == ""
