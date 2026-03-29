# tests/test_source_quality.py
"""Tests for web source quality tracking."""

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use in-memory DB for tests
os.environ.setdefault("DB_PATH", ":memory:")

# Single event loop for the whole module (aiosqlite connection is loop-bound)
_loop = asyncio.new_event_loop()


def _run(coro):
    """Helper to run async code in tests on the shared event loop."""
    return _loop.run_until_complete(coro)


# Module-level setup: init DB once, keep connection alive for all tests
_db_initialized = False


def _ensure_db():
    global _db_initialized
    if not _db_initialized:
        from src.infra import db as _db
        _run(_db.close_db())
        _run(_db.init_db())
        _db_initialized = True


class TestSourceQuality(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _ensure_db()

    def test_record_success_inserts(self):
        """record_source_quality creates a new row on first call."""
        from src.infra.db import record_source_quality, get_source_quality

        _run(record_source_quality("example.com", success=True, relevance=0.8))
        result = _run(get_source_quality(["example.com"]))

        self.assertIn("example.com", result)
        row = result["example.com"]
        self.assertEqual(row["success_count"], 1)
        self.assertEqual(row["fail_count"], 0)
        self.assertEqual(row["block_count"], 0)
        self.assertAlmostEqual(row["avg_relevance"], 0.8, places=2)
        self.assertIsNotNone(row["last_success"])

    def test_record_success_updates(self):
        """Repeated successes increment count and update avg_relevance."""
        from src.infra.db import record_source_quality, get_source_quality

        _run(record_source_quality("update-test.com", success=True, relevance=0.6))
        _run(record_source_quality("update-test.com", success=True, relevance=1.0))
        result = _run(get_source_quality(["update-test.com"]))

        row = result["update-test.com"]
        self.assertEqual(row["success_count"], 2)
        # avg_relevance should be incremental mean: (0.6*1 + 1.0) / 2 = 0.8
        self.assertAlmostEqual(row["avg_relevance"], 0.8, places=2)

    def test_record_failure(self):
        """record_source_quality increments fail_count on failure."""
        from src.infra.db import record_source_quality, get_source_quality

        _run(record_source_quality("fail-test.com", success=False))
        _run(record_source_quality("fail-test.com", success=False))
        result = _run(get_source_quality(["fail-test.com"]))

        row = result["fail-test.com"]
        self.assertEqual(row["fail_count"], 2)
        self.assertEqual(row["success_count"], 0)
        self.assertIsNotNone(row["last_failure"])

    def test_record_blocked(self):
        """record_source_quality increments block_count when blocked=True."""
        from src.infra.db import record_source_quality, get_source_quality

        _run(record_source_quality("blocked-test.com", success=False, blocked=True))
        result = _run(get_source_quality(["blocked-test.com"]))

        row = result["blocked-test.com"]
        self.assertEqual(row["block_count"], 1)
        self.assertEqual(row["fail_count"], 0)
        self.assertIsNotNone(row["last_block"])

    def test_get_source_quality_returns_empty_for_unknown(self):
        """get_source_quality returns empty dict for domains not in DB."""
        from src.infra.db import get_source_quality

        result = _run(get_source_quality(["never-seen-domain.xyz"]))
        self.assertEqual(result, {})

    def test_get_source_quality_multiple_domains(self):
        """get_source_quality returns data for multiple domains at once."""
        from src.infra.db import record_source_quality, get_source_quality

        _run(record_source_quality("multi-a.com", success=True, relevance=0.5))
        _run(record_source_quality("multi-b.com", success=False))
        result = _run(get_source_quality(["multi-a.com", "multi-b.com", "multi-c.com"]))

        self.assertIn("multi-a.com", result)
        self.assertIn("multi-b.com", result)
        self.assertNotIn("multi-c.com", result)

    def test_get_source_quality_empty_list(self):
        """get_source_quality handles empty input gracefully."""
        from src.infra.db import get_source_quality

        result = _run(get_source_quality([]))
        self.assertEqual(result, {})


class TestReorderUrlsByQuality(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _ensure_db()
        from src.infra.db import record_source_quality
        # Seed quality data for reorder tests
        _run(record_source_quality("good-domain.com", success=True, relevance=0.9))
        _run(record_source_quality("good-domain.com", success=True, relevance=0.9))
        _run(record_source_quality("bad-domain.com", success=False))
        _run(record_source_quality("bad-domain.com", success=False))
        _run(record_source_quality("bad-domain.com", success=False, blocked=True))

    def test_good_domains_come_first(self):
        """URLs from high-quality domains should be sorted before bad ones."""
        from src.tools.web_search import _reorder_urls_by_quality

        urls = [
            "https://bad-domain.com/page1",
            "https://good-domain.com/page2",
            "https://unknown-domain.com/page3",
        ]
        result = _run(_reorder_urls_by_quality(urls))

        # good-domain should be first (high success rate + relevance)
        self.assertEqual(result[0], "https://good-domain.com/page2")
        # unknown should be before bad (neutral 0.5 > bad's low score)
        self.assertEqual(result[1], "https://unknown-domain.com/page3")
        self.assertEqual(result[2], "https://bad-domain.com/page1")

    def test_unknown_domains_get_neutral_score(self):
        """Unknown domains should get 0.5 score (middle of range)."""
        from src.tools.web_search import _reorder_urls_by_quality

        urls = [
            "https://totally-new-a.com/page1",
            "https://totally-new-b.com/page2",
        ]
        result = _run(_reorder_urls_by_quality(urls))
        # Both unknown — order preserved since both score 0.5
        self.assertEqual(len(result), 2)

    def test_empty_urls(self):
        """Reordering empty list returns empty list."""
        from src.tools.web_search import _reorder_urls_by_quality

        result = _run(_reorder_urls_by_quality([]))
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
