"""Tests for reliable artifact persistence (DB-first store, flush, warm, retry)."""

from __future__ import annotations

import asyncio
import sqlite3
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.workflows.engine.artifacts import ArtifactStore


class TestDBFirstStore(unittest.TestCase):
    """ArtifactStore.store() writes DB first; cache updates only on success."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_db_failure_prevents_cache_update(self):
        """When use_db=True and DB write fails, the cache must NOT be updated."""
        store = ArtifactStore(use_db=True)

        with patch(
            "src.collaboration.blackboard.update_blackboard_entry",
            new_callable=AsyncMock,
            side_effect=RuntimeError("DB down"),
        ):
            with self.assertRaises(RuntimeError):
                self._run(store.store(1, "spec", "content"))

        # Cache should be empty because DB write failed before cache update
        self.assertIsNone(self._run(store.retrieve(1, "spec")))
        self.assertEqual(store._cache, {})

    def test_db_success_updates_cache(self):
        """When use_db=True and DB write succeeds, cache is updated."""
        store = ArtifactStore(use_db=True)

        with patch(
            "src.collaboration.blackboard.update_blackboard_entry",
            new_callable=AsyncMock,
        ):
            self._run(store.store(1, "spec", "content"))

        self.assertEqual(self._run(store.retrieve(1, "spec")), "content")

    def test_use_db_false_skips_db(self):
        """When use_db=False, store writes only to cache — no DB interaction."""
        store = ArtifactStore(use_db=False)

        # No mocking needed — use_db=False should never touch blackboard
        self._run(store.store(1, "spec", "content"))
        self.assertEqual(self._run(store.retrieve(1, "spec")), "content")


class TestFlushCache(unittest.TestCase):
    """ArtifactStore.flush_cache() persists all cached entries to DB."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_flush_writes_all_entries_to_db(self):
        """flush_cache iterates cache and writes each entry to the blackboard."""
        store = ArtifactStore(use_db=True)

        mock_update = AsyncMock()
        with patch(
            "src.collaboration.blackboard.update_blackboard_entry",
            new_callable=AsyncMock,
        ) as mock_store_update:
            # Populate cache via normal store (which also calls DB)
            self._run(store.store(1, "a", "aaa"))
            self._run(store.store(1, "b", "bbb"))
            self._run(store.store(2, "c", "ccc"))
            mock_store_update.reset_mock()

            # Now flush
            self._run(store.flush_cache())

            # Should have written all 3 entries
            self.assertEqual(mock_store_update.call_count, 3)
            call_args = {
                (c.args[0], c.args[1], c.args[2], c.args[3])
                for c in mock_store_update.call_args_list
            }
            self.assertIn((1, "artifacts", "a", "aaa"), call_args)
            self.assertIn((1, "artifacts", "b", "bbb"), call_args)
            self.assertIn((2, "artifacts", "c", "ccc"), call_args)

    def test_flush_logs_failures_without_raising(self):
        """flush_cache logs errors but does not raise."""
        store = ArtifactStore(use_db=False)

        # Populate cache directly (use_db=False skips DB)
        self._run(store.store(1, "x", "val"))

        # Now switch to DB mode and flush with a failing DB
        store._use_db = True

        with patch(
            "src.collaboration.blackboard.update_blackboard_entry",
            new_callable=AsyncMock,
            side_effect=RuntimeError("DB gone"),
        ):
            # Should NOT raise
            self._run(store.flush_cache())

    def test_flush_noop_when_use_db_false(self):
        """flush_cache does nothing when use_db=False."""
        store = ArtifactStore(use_db=False)
        self._run(store.store(1, "x", "val"))

        with patch(
            "src.collaboration.blackboard.update_blackboard_entry",
            new_callable=AsyncMock,
        ) as mock_update:
            self._run(store.flush_cache())
            mock_update.assert_not_called()


class TestWarmCache(unittest.TestCase):
    """ArtifactStore.warm_cache() loads artifacts from DB into cache."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_warm_cache_loads_from_db(self):
        """warm_cache populates the in-memory cache from the blackboard."""
        store = ArtifactStore(use_db=True)

        db_artifacts = {"spec": "from_db", "plan": "plan_db"}

        with patch(
            "src.collaboration.blackboard.read_blackboard",
            new_callable=AsyncMock,
            return_value=db_artifacts,
        ):
            self._run(store.warm_cache(1))

        # Cache should now have those values (no DB call needed for retrieve)
        self.assertEqual(store._cache["1"], {"spec": "from_db", "plan": "plan_db"})

    def test_warm_cache_no_artifacts_section(self):
        """warm_cache does nothing when blackboard has no artifacts."""
        store = ArtifactStore(use_db=True)

        with patch(
            "src.collaboration.blackboard.read_blackboard",
            new_callable=AsyncMock,
            return_value=None,
        ):
            self._run(store.warm_cache(1))

        self.assertEqual(store._cache, {})

    def test_warm_cache_noop_when_use_db_false(self):
        """warm_cache does nothing when use_db=False."""
        store = ArtifactStore(use_db=False)

        with patch(
            "src.collaboration.blackboard.read_blackboard",
            new_callable=AsyncMock,
        ) as mock_read:
            self._run(store.warm_cache(1))
            mock_read.assert_not_called()

    def test_warm_cache_merges_with_existing(self):
        """warm_cache merges DB data with any existing cache entries."""
        store = ArtifactStore(use_db=False)
        self._run(store.store(1, "local", "local_val"))

        store._use_db = True

        with patch(
            "src.collaboration.blackboard.read_blackboard",
            new_callable=AsyncMock,
            return_value={"db_art": "db_val"},
        ):
            self._run(store.warm_cache(1))

        self.assertEqual(store._cache["1"]["local"], "local_val")
        self.assertEqual(store._cache["1"]["db_art"], "db_val")


class TestPersistRetry(unittest.TestCase):
    """blackboard._persist() retries once on SQLite busy errors."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_retry_on_sqlite_busy(self):
        """_persist retries once when SQLite reports 'database is locked'."""
        from src.collaboration.blackboard import _persist, clear_cache

        mock_db = AsyncMock()
        call_count = 0

        async def mock_execute(sql, params=None):
            nonlocal call_count
            if "INSERT OR REPLACE" in sql:
                call_count += 1
                if call_count == 1:
                    raise sqlite3.OperationalError("database is locked")
            return AsyncMock()

        mock_db.execute = mock_execute
        mock_db.commit = AsyncMock()

        with patch("src.collaboration.blackboard.get_db", return_value=mock_db):
            with patch("src.collaboration.blackboard._ensure_table", new_callable=AsyncMock):
                with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                    self._run(_persist(1, {"test": "data"}))

                    # Should have retried once with 100ms delay
                    mock_sleep.assert_called_once_with(0.1)
                    self.assertEqual(call_count, 2)

    def test_non_busy_operational_error_propagates(self):
        """Non-busy OperationalError propagates immediately without retry."""
        from src.collaboration.blackboard import _persist

        mock_db = AsyncMock()

        async def mock_execute(sql, params=None):
            if "INSERT OR REPLACE" in sql:
                raise sqlite3.OperationalError("disk I/O error")
            return AsyncMock()

        mock_db.execute = mock_execute

        with patch("src.collaboration.blackboard.get_db", return_value=mock_db):
            with patch("src.collaboration.blackboard._ensure_table", new_callable=AsyncMock):
                with self.assertRaises(sqlite3.OperationalError) as ctx:
                    self._run(_persist(1, {"test": "data"}))
                self.assertIn("disk I/O", str(ctx.exception))

    def test_non_sqlite_error_propagates(self):
        """Non-SQLite exceptions propagate immediately."""
        from src.collaboration.blackboard import _persist

        mock_db = AsyncMock()

        async def mock_execute(sql, params=None):
            if "INSERT OR REPLACE" in sql:
                raise ValueError("unexpected")
            return AsyncMock()

        mock_db.execute = mock_execute

        with patch("src.collaboration.blackboard.get_db", return_value=mock_db):
            with patch("src.collaboration.blackboard._ensure_table", new_callable=AsyncMock):
                with self.assertRaises(ValueError):
                    self._run(_persist(1, {"test": "data"}))


if __name__ == "__main__":
    unittest.main()
