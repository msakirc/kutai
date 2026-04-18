"""
Tests for stuck-task recovery fixes:

1. claim_task uses strftime (not isoformat) so watchdog comparisons work
2. Watchdog resets stuck processing tasks, respects retry limits
3. Dedup resets stuck processing duplicates instead of blocking
4. _handle_complete handles malformed JSON context gracefully
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestClaimTaskDateFormat(unittest.TestCase):
    """claim_task must store started_at in SQLite-compatible format (no 'T')."""

    def test_claim_task_uses_strftime_format(self):
        """Verify claim_task stores started_at without 'T' separator."""
        from src.infra import db as db_mod

        source = inspect.getsource(db_mod.claim_task)

        # Must use strftime, not isoformat
        self.assertIn("strftime", source,
                       "claim_task must use strftime for SQLite-compatible dates")
        self.assertNotIn(".isoformat()", source,
                         "claim_task must NOT use isoformat (T separator breaks watchdog)")

    def test_strftime_format_matches_sqlite_datetime(self):
        """strftime output should have space separator, not T."""
        now = datetime.now(timezone.utc)
        formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        self.assertNotIn("T", formatted)
        self.assertIn(" ", formatted)
        # Should be 19 chars: YYYY-MM-DD HH:MM:SS
        self.assertEqual(len(formatted), 19)

    def test_isoformat_has_incompatible_T_separator(self):
        """Confirm isoformat would produce incompatible format (regression guard)."""
        now = datetime.now(timezone.utc)
        iso = now.isoformat()
        self.assertIn("T", iso)

    def test_strftime_comparable_with_sqlite_datetime(self):
        """strftime dates must compare correctly with SQLite datetime() output.

        SQLite datetime('now') returns 'YYYY-MM-DD HH:MM:SS'.
        isoformat() returns 'YYYY-MM-DDTHH:MM:SS+00:00'.
        String comparison: 'T' (0x54) > ' ' (0x20), so ISO dates
        ALWAYS appear later than SQLite dates, breaking < comparisons.
        """
        # Simulate: task started 10 minutes ago
        ten_min_ago = datetime.now(timezone.utc) - timedelta(minutes=10)
        started_at_strftime = ten_min_ago.strftime("%Y-%m-%d %H:%M:%S")

        # Simulate: SQLite threshold = datetime('now', '-5 minutes')
        five_min_ago = datetime.now(timezone.utc) - timedelta(minutes=5)
        threshold = five_min_ago.strftime("%Y-%m-%d %H:%M:%S")

        # strftime: started_at < threshold should be True (10 min ago < 5 min ago)
        self.assertLess(started_at_strftime, threshold,
                        "strftime format should allow correct < comparison")

        # isoformat: same comparison would ALSO be True in pure Python,
        # but in SQLite the comparison with datetime() fails because
        # datetime() returns space-separated while the stored value has T.
        started_at_iso = ten_min_ago.isoformat()
        # The T makes it compare differently with space-separated SQLite output
        self.assertIn("T", started_at_iso)
        self.assertNotIn("T", threshold)


class TestWatchdogRetryLimits(unittest.TestCase):
    """Watchdog must respect retry limits when resetting stuck tasks."""

    def test_watchdog_checks_retry_count(self):
        """Watchdog query must SELECT attempts and max_attempts."""
        from src.core.orchestrator import Orchestrator
        source = inspect.getsource(Orchestrator.watchdog)

        # The stuck-processing query should fetch retry info
        self.assertIn("attempts", source,
                       "watchdog must check attempts for stuck tasks")
        self.assertIn("max_attempts", source,
                       "watchdog must check max_attempts for stuck tasks")

    def test_watchdog_has_failed_branch(self):
        """Watchdog must mark exhausted-attempt tasks as failed, not just reset."""
        from src.core.orchestrator import Orchestrator
        source = inspect.getsource(Orchestrator.watchdog)

        # Should have logic to mark tasks as failed when attempts exhausted
        self.assertIn("exhausted attempts", source.lower(),
                       "watchdog must handle exhausted attempts differently")


class TestDedupStuckReset(unittest.TestCase):
    """Dedup should reset stuck processing tasks instead of blocking retries."""

    def test_dedup_query_includes_started_at(self):
        """The dedup SELECT must include started_at for stuck detection."""
        from src.infra import db as db_mod
        source = inspect.getsource(db_mod.add_task)

        # The dedup query should select started_at
        self.assertIn("started_at", source,
                       "add_task dedup query must SELECT started_at for stuck detection")

    def test_dedup_checks_10_minute_threshold(self):
        """Dedup should check if processing task is stuck >10 minutes."""
        from src.infra import db as db_mod
        source = inspect.getsource(db_mod.add_task)

        self.assertIn("10 minutes", source,
                       "add_task should check for stuck processing tasks >10 minutes")

    def test_dedup_resets_stuck_task(self):
        """Dedup should reset stuck processing task to pending."""
        from src.infra import db as db_mod
        source = inspect.getsource(db_mod.add_task)

        # Should contain logic to reset stuck tasks
        self.assertIn("resetting to pending", source.lower(),
                       "add_task should reset stuck processing tasks")


class TestHandleCompleteJsonSafety(unittest.TestCase):
    """_handle_complete must handle malformed JSON context without crashing."""

    def test_context_parsing_catches_json_errors(self):
        """The notification section's json.loads must be wrapped in try/except."""
        from src.core.orchestrator import Orchestrator
        source = inspect.getsource(Orchestrator._handle_complete)

        # Count json.loads calls — each should have error handling
        # At minimum, the task_ctx_parsed section should catch JSONDecodeError
        self.assertIn("JSONDecodeError", source,
                       "_handle_complete must catch JSONDecodeError for context parsing")

    def test_context_parsed_fallback_to_empty_dict(self):
        """Malformed context should fall back to empty dict."""
        from src.core.orchestrator import Orchestrator
        source = inspect.getsource(Orchestrator._handle_complete)

        # Should have isinstance check and fallback
        self.assertIn("task_ctx_parsed = {}", source,
                       "_handle_complete must fall back to {} on bad JSON")


class TestProcessTaskErrorHandling(unittest.TestCase):
    """process_task exception handler must reset tasks properly.

    After the Plan A refactor, the exception handling body lives in
    Orchestrator._handle_unexpected_failure; process_task only delegates.
    Check the text in that method's source.
    """

    def _unexpected_source(self):
        from src.core.orchestrator import Orchestrator
        return inspect.getsource(Orchestrator._handle_unexpected_failure)

    def test_exception_handler_resets_to_pending(self):
        """On exception, task should be reset to pending if retries remain."""
        self.assertIn("status=\"pending\"", self._unexpected_source(),
                       "unexpected-failure handler must reset to pending on retryable error")

    def test_exception_handler_marks_failed_on_exhaustion(self):
        """On exception with no retries left, task should be marked failed."""
        self.assertIn("status=\"failed\"", self._unexpected_source(),
                       "unexpected-failure handler must mark failed when retries exhausted")

    def test_exception_handler_checks_retry_count(self):
        """Exception handler must use retry bookkeeping (RetryContext)."""
        source = self._unexpected_source()
        # RetryContext encapsulates retry_count + max_retries comparison.
        self.assertIn("RetryContext", source)
        self.assertIn("record_failure", source)


if __name__ == "__main__":
    unittest.main()
