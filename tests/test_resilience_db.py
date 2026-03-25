# tests/test_resilience_db.py
"""
Tests for Resilience DB changes (Task 1):
  A) approval_requests table
  B) insert_tasks_atomically()
  C) propagate_skips()
  D) get_ready_tasks() with skipped deps
"""
import asyncio
import json
import os
import sys
import tempfile
import unittest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_async(coro):
    """Run an async coroutine synchronously for tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


try:
    import aiosqlite
    HAS_AIOSQLITE = True
except ImportError:
    HAS_AIOSQLITE = False


class _DBTestBase(unittest.TestCase):
    """Shared setUp/tearDown for all DB-backed test classes."""

    def setUp(self):
        if not HAS_AIOSQLITE:
            self.skipTest("aiosqlite not installed")

        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.tmp.name
        self.tmp.close()

        from src.infra import db as db_mod
        from src.app import config

        self._orig_config_path = config.DB_PATH
        self._orig_db_path = db_mod.DB_PATH
        self.db_mod = db_mod

        # Patch BOTH copies of DB_PATH and reset singleton
        config.DB_PATH = self.db_path
        db_mod.DB_PATH = self.db_path
        db_mod._db_connection = None

        run_async(db_mod.init_db())

    def tearDown(self):
        run_async(self.db_mod.close_db())
        from src.app import config
        config.DB_PATH = self._orig_config_path
        self.db_mod.DB_PATH = self._orig_db_path
        try:
            os.unlink(self.db_path)
        except OSError:
            pass
        for suffix in ("-wal", "-shm"):
            try:
                os.unlink(self.db_path + suffix)
            except OSError:
                pass

    async def _create_mission(self, title="Test Mission"):
        db = await self.db_mod.get_db()
        cursor = await db.execute(
            "INSERT INTO missions (title) VALUES (?)", (title,)
        )
        await db.commit()
        return cursor.lastrowid

    async def _create_task(self, mission_id, title, status="pending",
                           depends_on=None, agent_type="executor"):
        db = await self.db_mod.get_db()
        task_hash = self.db_mod.compute_task_hash(
            title, "", agent_type, mission_id, None
        )
        cursor = await db.execute(
            """INSERT INTO tasks
               (mission_id, title, description, agent_type, status,
                depends_on, task_hash)
               VALUES (?, ?, '', ?, ?, ?, ?)""",
            (mission_id, title, agent_type, status,
             json.dumps(depends_on or []), task_hash)
        )
        await db.commit()
        return cursor.lastrowid

    async def _get_task_status(self, task_id):
        db = await self.db_mod.get_db()
        cursor = await db.execute(
            "SELECT status, error FROM tasks WHERE id = ?", (task_id,)
        )
        row = await cursor.fetchone()
        return (row[0], row[1]) if row else (None, None)


# ─── A) approval_requests table ──────────────────────────────────────────────

class TestApprovalRequestsTable(_DBTestBase):

    def test_table_exists_after_init(self):
        async def _test():
            db = await self.db_mod.get_db()
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='approval_requests'"
            )
            row = await cursor.fetchone()
            self.assertIsNotNone(row)
        run_async(_test())

    def test_insert_and_query(self):
        async def _test():
            db = await self.db_mod.get_db()
            await db.execute(
                """INSERT INTO approval_requests (task_id, mission_id, title, details)
                   VALUES (1, 1, 'Deploy to prod', 'Needs review')"""
            )
            await db.commit()

            cursor = await db.execute(
                "SELECT * FROM approval_requests WHERE task_id = 1"
            )
            row = await cursor.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], 1)  # task_id
            self.assertEqual(row[2], "Deploy to prod")  # title
            self.assertEqual(row[4], "pending")  # default status
        run_async(_test())

    def test_status_default_is_pending(self):
        async def _test():
            db = await self.db_mod.get_db()
            await db.execute(
                "INSERT INTO approval_requests (task_id) VALUES (42)"
            )
            await db.commit()
            cursor = await db.execute(
                "SELECT status FROM approval_requests WHERE task_id = 42"
            )
            row = await cursor.fetchone()
            self.assertEqual(row[0], "pending")
        run_async(_test())


# ─── B) insert_tasks_atomically() ────────────────────────────────────────────

class TestInsertTasksAtomically(_DBTestBase):

    def test_basic_insert(self):
        async def _test():
            mission_id = await self._create_mission()
            tasks = [
                {"title": "Task A", "description": "Desc A", "agent_type": "researcher"},
                {"title": "Task B", "description": "Desc B", "agent_type": "executor"},
            ]
            ids = await self.db_mod.insert_tasks_atomically(tasks, mission_id)
            self.assertEqual(len(ids), 2)
            self.assertTrue(all(i > 0 for i in ids))

            # Verify tasks exist in DB
            db = await self.db_mod.get_db()
            for tid in ids:
                cursor = await db.execute("SELECT * FROM tasks WHERE id = ?", (tid,))
                row = await cursor.fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(dict(row)["mission_id"], mission_id)
        run_async(_test())

    def test_dedup_within_batch(self):
        async def _test():
            mission_id = await self._create_mission()
            tasks = [
                {"title": "Same Task", "description": "Same Desc", "agent_type": "executor"},
                {"title": "Same Task", "description": "Same Desc", "agent_type": "executor"},
            ]
            ids = await self.db_mod.insert_tasks_atomically(tasks, mission_id)
            self.assertEqual(len(ids), 2)
            self.assertGreater(ids[0], 0)
            self.assertEqual(ids[1], -1)  # deduped
        run_async(_test())

    def test_dedup_against_existing(self):
        async def _test():
            mission_id = await self._create_mission()
            # Create a task first
            first_ids = await self.db_mod.insert_tasks_atomically(
                [{"title": "Existing", "description": "Desc", "agent_type": "executor"}],
                mission_id
            )
            self.assertGreater(first_ids[0], 0)

            # Try to insert the same task again
            second_ids = await self.db_mod.insert_tasks_atomically(
                [{"title": "Existing", "description": "Desc", "agent_type": "executor"}],
                mission_id
            )
            self.assertEqual(second_ids[0], -1)  # deduped
        run_async(_test())

    def test_depends_on_stored(self):
        async def _test():
            mission_id = await self._create_mission()
            tasks = [
                {"title": "T1", "description": "", "agent_type": "executor",
                 "depends_on": [10, 20]},
            ]
            ids = await self.db_mod.insert_tasks_atomically(tasks, mission_id)
            db = await self.db_mod.get_db()
            cursor = await db.execute(
                "SELECT depends_on FROM tasks WHERE id = ?", (ids[0],)
            )
            row = await cursor.fetchone()
            self.assertEqual(json.loads(row[0]), [10, 20])
        run_async(_test())

    def test_context_stored(self):
        async def _test():
            mission_id = await self._create_mission()
            ctx = {"key": "value", "nested": {"a": 1}}
            tasks = [
                {"title": "T1", "description": "", "agent_type": "executor",
                 "context": ctx},
            ]
            ids = await self.db_mod.insert_tasks_atomically(tasks, mission_id)
            db = await self.db_mod.get_db()
            cursor = await db.execute(
                "SELECT context FROM tasks WHERE id = ?", (ids[0],)
            )
            row = await cursor.fetchone()
            self.assertEqual(json.loads(row[0]), ctx)
        run_async(_test())

    def test_empty_list(self):
        async def _test():
            mission_id = await self._create_mission()
            ids = await self.db_mod.insert_tasks_atomically([], mission_id)
            self.assertEqual(ids, [])
        run_async(_test())

    def test_defaults(self):
        async def _test():
            mission_id = await self._create_mission()
            tasks = [{"title": "Minimal"}]  # no description, agent_type, etc.
            ids = await self.db_mod.insert_tasks_atomically(tasks, mission_id)
            self.assertEqual(len(ids), 1)
            self.assertGreater(ids[0], 0)

            db = await self.db_mod.get_db()
            cursor = await db.execute("SELECT * FROM tasks WHERE id = ?", (ids[0],))
            row = dict(await cursor.fetchone())
            self.assertEqual(row["agent_type"], "executor")
            self.assertEqual(row["tier"], "auto")
            self.assertEqual(row["priority"], 5)
        run_async(_test())


# ─── C) propagate_skips() ────────────────────────────────────────────────────

class TestPropagateSkips(_DBTestBase):

    def test_no_skipped_deps_no_change(self):
        """Tasks with no skipped deps should remain pending."""
        async def _test():
            mission_id = await self._create_mission()
            t1 = await self._create_task(mission_id, "T1", status="completed")
            t2 = await self._create_task(mission_id, "T2", depends_on=[t1])

            count = await self.db_mod.propagate_skips(mission_id)
            self.assertEqual(count, 0)

            status, _ = await self._get_task_status(t2)
            self.assertEqual(status, "pending")
        run_async(_test())

    def test_all_deps_skipped_propagates(self):
        """Task with all deps skipped should be auto-skipped."""
        async def _test():
            mission_id = await self._create_mission()
            t1 = await self._create_task(mission_id, "T1", status="skipped")
            t2 = await self._create_task(mission_id, "T2", depends_on=[t1])

            count = await self.db_mod.propagate_skips(mission_id)
            self.assertEqual(count, 1)

            status, error = await self._get_task_status(t2)
            self.assertEqual(status, "skipped")
            self.assertEqual(error, "dependency_skipped")
        run_async(_test())

    def test_mixed_deps_not_skipped(self):
        """Task with one completed and one skipped dep should NOT be skipped."""
        async def _test():
            mission_id = await self._create_mission()
            t1 = await self._create_task(mission_id, "T1", status="completed")
            t2 = await self._create_task(mission_id, "T2", status="skipped")
            t3 = await self._create_task(mission_id, "T3", depends_on=[t1, t2])

            count = await self.db_mod.propagate_skips(mission_id)
            self.assertEqual(count, 0)

            status, _ = await self._get_task_status(t3)
            self.assertEqual(status, "pending")
        run_async(_test())

    def test_transitive_propagation(self):
        """Skipping should cascade: T1(skipped) -> T2(skip) -> T3(skip)."""
        async def _test():
            mission_id = await self._create_mission()
            t1 = await self._create_task(mission_id, "T1", status="skipped")
            t2 = await self._create_task(mission_id, "T2", depends_on=[t1])
            t3 = await self._create_task(mission_id, "T3", depends_on=[t2])

            count = await self.db_mod.propagate_skips(mission_id)
            self.assertEqual(count, 2)

            status2, error2 = await self._get_task_status(t2)
            self.assertEqual(status2, "skipped")
            self.assertEqual(error2, "dependency_skipped")

            status3, error3 = await self._get_task_status(t3)
            self.assertEqual(status3, "skipped")
            self.assertEqual(error3, "dependency_skipped")
        run_async(_test())

    def test_pending_deps_block_propagation(self):
        """If a dep is still pending, don't skip even if another is skipped."""
        async def _test():
            mission_id = await self._create_mission()
            t1 = await self._create_task(mission_id, "T1", status="skipped")
            t2 = await self._create_task(mission_id, "T2", status="pending")
            t3 = await self._create_task(mission_id, "T3", depends_on=[t1, t2])

            count = await self.db_mod.propagate_skips(mission_id)
            self.assertEqual(count, 0)

            status, _ = await self._get_task_status(t3)
            self.assertEqual(status, "pending")
        run_async(_test())

    def test_no_tasks_returns_zero(self):
        """Propagation on a mission with no tasks returns 0."""
        async def _test():
            mission_id = await self._create_mission()
            count = await self.db_mod.propagate_skips(mission_id)
            self.assertEqual(count, 0)
        run_async(_test())


# ─── D) get_ready_tasks() with skipped deps ──────────────────────────────────

class TestGetReadyTasksSkippedDeps(_DBTestBase):

    def test_all_deps_completed_still_ready(self):
        """Original behavior: all deps completed -> task is ready."""
        async def _test():
            mission_id = await self._create_mission()
            t1 = await self._create_task(mission_id, "T1", status="completed")
            t2 = await self._create_task(mission_id, "T2", depends_on=[t1])

            ready = await self.db_mod.get_ready_tasks(limit=10)
            ready_ids = [t["id"] for t in ready]
            self.assertIn(t2, ready_ids)
        run_async(_test())

    def test_mixed_completed_skipped_is_ready(self):
        """Task with some completed + some skipped deps is ready."""
        async def _test():
            mission_id = await self._create_mission()
            t1 = await self._create_task(mission_id, "T1", status="completed")
            t2 = await self._create_task(mission_id, "T2", status="skipped")
            t3 = await self._create_task(mission_id, "T3", depends_on=[t1, t2])

            ready = await self.db_mod.get_ready_tasks(limit=10)
            ready_ids = [t["id"] for t in ready]
            self.assertIn(t3, ready_ids)
        run_async(_test())

    def test_all_deps_skipped_auto_skips_task(self):
        """Task with ALL deps skipped is auto-skipped (not returned as ready)."""
        async def _test():
            mission_id = await self._create_mission()
            t1 = await self._create_task(mission_id, "T1", status="skipped")
            t2 = await self._create_task(mission_id, "T2", status="skipped")
            t3 = await self._create_task(mission_id, "T3", depends_on=[t1, t2])

            ready = await self.db_mod.get_ready_tasks(limit=10)
            ready_ids = [t["id"] for t in ready]
            self.assertNotIn(t3, ready_ids)

            # Verify it was marked skipped in DB
            status, error = await self._get_task_status(t3)
            self.assertEqual(status, "skipped")
            self.assertEqual(error, "dependency_skipped")
        run_async(_test())

    def test_no_deps_still_ready(self):
        """Tasks with no dependencies are still returned as ready."""
        async def _test():
            mission_id = await self._create_mission()
            t1 = await self._create_task(mission_id, "T1")

            ready = await self.db_mod.get_ready_tasks(limit=10)
            ready_ids = [t["id"] for t in ready]
            self.assertIn(t1, ready_ids)
        run_async(_test())

    def test_failed_deps_still_block(self):
        """Tasks with failed deps are still blocked (not ready)."""
        async def _test():
            mission_id = await self._create_mission()
            t1 = await self._create_task(mission_id, "T1", status="failed")
            t2 = await self._create_task(mission_id, "T2", depends_on=[t1])

            ready = await self.db_mod.get_ready_tasks(limit=10)
            ready_ids = [t["id"] for t in ready]
            self.assertNotIn(t2, ready_ids)
        run_async(_test())

    def test_pending_dep_still_blocks(self):
        """Tasks with pending deps remain blocked."""
        async def _test():
            mission_id = await self._create_mission()
            t1 = await self._create_task(mission_id, "T1", status="pending")
            t2 = await self._create_task(mission_id, "T2", depends_on=[t1])

            ready = await self.db_mod.get_ready_tasks(limit=10)
            ready_ids = [t["id"] for t in ready]
            # t1 is ready (no deps), t2 is blocked (dep pending)
            self.assertIn(t1, ready_ids)
            self.assertNotIn(t2, ready_ids)
        run_async(_test())


if __name__ == "__main__":
    unittest.main()
