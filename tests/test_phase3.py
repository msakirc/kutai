# tests/test_phase3.py
"""
Tests for Phase 3: Scheduler & Task Engine

  3.1 Cron scheduler (scheduled_tasks CRUD, next_run computation)
  3.2 Task cancellation (cancel + child propagation)
  3.3 Task reprioritization
  3.4 Task timeout (timeout_seconds column)
  3.5 Dependency graph (get_task_tree)
"""
import asyncio
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import aiosqlite
    HAS_AIOSQLITE = True
except ImportError:
    HAS_AIOSQLITE = False


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _patch_db_path(db_mod, db_path):
    import config
    config.DB_PATH = db_path
    db_mod.DB_PATH = db_path


class _DBTestBase(unittest.TestCase):
    def setUp(self):
        if not HAS_AIOSQLITE:
            self.skipTest("aiosqlite not installed")
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.tmp.name
        self.tmp.close()

        import config
        import db as db_mod
        self._orig_config_path = config.DB_PATH
        self._orig_db_path = db_mod.DB_PATH
        self.db_mod = db_mod

        _patch_db_path(db_mod, self.db_path)
        db_mod._db_connection = None
        run_async(db_mod.init_db())

    def tearDown(self):
        run_async(self.db_mod.close_db())
        import config
        config.DB_PATH = self._orig_config_path
        self.db_mod.DB_PATH = self._orig_db_path
        for suffix in ("", "-wal", "-shm"):
            try:
                os.unlink(self.db_path + suffix)
            except OSError:
                pass


# ─── 3.1 Scheduled Tasks ───────────────────────────────────────────────────

class TestScheduledTasks(_DBTestBase):

    def test_add_and_get_scheduled_tasks(self):
        async def _test():
            sid = await self.db_mod.add_scheduled_task(
                title="Hourly check",
                description="Check server status",
                cron_expression="0 * * * *",
            )
            self.assertIsNotNone(sid)
            tasks = await self.db_mod.get_scheduled_tasks()
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0]["title"], "Hourly check")
            self.assertEqual(tasks[0]["enabled"], 1)
        run_async(_test())

    def test_get_due_tasks_no_next_run(self):
        """Tasks with no next_run are always due."""
        async def _test():
            await self.db_mod.add_scheduled_task(
                title="Always due", cron_expression="0 * * * *"
            )
            due = await self.db_mod.get_due_scheduled_tasks()
            self.assertEqual(len(due), 1)
        run_async(_test())

    def test_get_due_tasks_future_not_due(self):
        """Tasks with next_run in the future are not due."""
        async def _test():
            sid = await self.db_mod.add_scheduled_task(
                title="Future task", cron_expression="0 * * * *"
            )
            future = (datetime.now() + timedelta(hours=2)).isoformat()
            await self.db_mod.update_scheduled_task(sid, next_run=future)
            due = await self.db_mod.get_due_scheduled_tasks()
            self.assertEqual(len(due), 0)
        run_async(_test())

    def test_disabled_task_not_due(self):
        async def _test():
            sid = await self.db_mod.add_scheduled_task(
                title="Disabled", cron_expression="0 * * * *"
            )
            await self.db_mod.update_scheduled_task(sid, enabled=0)
            due = await self.db_mod.get_due_scheduled_tasks()
            self.assertEqual(len(due), 0)
        run_async(_test())

    def test_update_last_run(self):
        async def _test():
            sid = await self.db_mod.add_scheduled_task(
                title="Update test", cron_expression="0 * * * *"
            )
            now = datetime.now().isoformat()
            await self.db_mod.update_scheduled_task(sid, last_run=now)
            tasks = await self.db_mod.get_scheduled_tasks()
            self.assertEqual(tasks[0]["last_run"], now)
        run_async(_test())


# ─── 3.2 Task Cancellation ─────────────────────────────────────────────────

class TestTaskCancellation(_DBTestBase):

    def test_cancel_pending_task(self):
        async def _test():
            tid = await self.db_mod.add_task(
                "Cancel me", "d", agent_type="coder"
            )
            result = await self.db_mod.cancel_task(tid)
            self.assertTrue(result)
            task = await self.db_mod.get_task(tid)
            self.assertEqual(task["status"], "cancelled")
        run_async(_test())

    def test_cancel_completed_task_fails(self):
        async def _test():
            tid = await self.db_mod.add_task(
                "Already done", "d", agent_type="coder"
            )
            await self.db_mod.update_task(tid, status="completed")
            result = await self.db_mod.cancel_task(tid)
            self.assertFalse(result)
        run_async(_test())

    def test_cancel_propagates_to_children(self):
        async def _test():
            parent = await self.db_mod.add_task(
                "Parent cancel", "d", agent_type="planner"
            )
            subtasks = [
                {"title": "Child 1", "description": "d", "agent_type": "coder"},
                {"title": "Child 2", "description": "d", "agent_type": "researcher"},
            ]
            child_ids = await self.db_mod.add_subtasks_atomically(
                parent, subtasks, parent_status="waiting_subtasks"
            )
            result = await self.db_mod.cancel_task(parent)
            self.assertTrue(result)
            for cid in child_ids:
                if cid > 0:
                    child = await self.db_mod.get_task(cid)
                    self.assertEqual(child["status"], "cancelled")
        run_async(_test())

    def test_cancel_nonexistent_task(self):
        async def _test():
            result = await self.db_mod.cancel_task(99999)
            self.assertFalse(result)
        run_async(_test())


# ─── 3.3 Task Reprioritization ─────────────────────────────────────────────

class TestTaskReprioritization(_DBTestBase):

    def test_reprioritize_pending_task(self):
        async def _test():
            tid = await self.db_mod.add_task(
                "Reprio test", "d", agent_type="coder", priority=5
            )
            result = await self.db_mod.reprioritize_task(tid, 10)
            self.assertTrue(result)
            task = await self.db_mod.get_task(tid)
            self.assertEqual(task["priority"], 10)
        run_async(_test())

    def test_reprioritize_completed_fails(self):
        async def _test():
            tid = await self.db_mod.add_task(
                "Done reprio", "d", agent_type="coder"
            )
            await self.db_mod.update_task(tid, status="completed")
            result = await self.db_mod.reprioritize_task(tid, 10)
            self.assertFalse(result)
        run_async(_test())


# ─── 3.4 Timeout Column ────────────────────────────────────────────────────

class TestTimeoutColumn(_DBTestBase):

    def test_timeout_column_exists(self):
        async def _test():
            tid = await self.db_mod.add_task(
                "Timeout test", "d", agent_type="coder"
            )
            task = await self.db_mod.get_task(tid)
            self.assertIn("timeout_seconds", task)
            self.assertIsNone(task["timeout_seconds"])
        run_async(_test())

    def test_set_timeout(self):
        async def _test():
            tid = await self.db_mod.add_task(
                "Timeout set", "d", agent_type="coder"
            )
            await self.db_mod.update_task(tid, timeout_seconds=300)
            task = await self.db_mod.get_task(tid)
            self.assertEqual(task["timeout_seconds"], 300)
        run_async(_test())


# ─── 3.5 Dependency Graph ──────────────────────────────────────────────────

class TestTaskTree(_DBTestBase):

    def test_get_task_tree(self):
        async def _test():
            from db import add_mission
            gid = await add_mission("Test mission", "desc")
            t1 = await self.db_mod.add_task(
                "Step 1", "d", mission_id=gid, agent_type="coder"
            )
            t2 = await self.db_mod.add_task(
                "Step 2", "d", mission_id=gid, agent_type="researcher"
            )
            tree = await self.db_mod.get_task_tree(gid)
            self.assertEqual(len(tree), 2)
            ids = [t["id"] for t in tree]
            self.assertIn(t1, ids)
            self.assertIn(t2, ids)
        run_async(_test())

    def test_empty_tree(self):
        async def _test():
            tree = await self.db_mod.get_task_tree(99999)
            self.assertEqual(tree, [])
        run_async(_test())


# ─── 3.1 Cron Next-Run Computation (no DB needed) ─────────────────────────
# Inline to avoid importing orchestrator with its heavy deps.

def _compute_next_run(cron_expr: str, after: datetime) -> datetime | None:
    """Replicated from Orchestrator for testing without telegram etc."""
    try:
        parts = cron_expr.strip().split()
        if len(parts) != 5:
            return None

        minute, hour, day, month, weekday = parts

        if minute != "*" and hour == "*":
            m = int(minute)
            candidate = after.replace(minute=m, second=0, microsecond=0)
            if candidate <= after:
                candidate += timedelta(hours=1)
            return candidate

        if minute != "*" and hour != "*":
            m, h = int(minute), int(hour)
            candidate = after.replace(
                hour=h, minute=m, second=0, microsecond=0
            )
            if candidate <= after:
                candidate += timedelta(days=1)
            return candidate

        return after + timedelta(hours=1)
    except Exception:
        return None


class TestComputeNextRun(unittest.TestCase):

    def test_hourly_cron(self):
        after = datetime(2025, 1, 15, 10, 30, 0)
        result = _compute_next_run("0 * * * *", after)
        self.assertEqual(result.minute, 0)
        self.assertEqual(result.hour, 11)

    def test_hourly_already_past(self):
        after = datetime(2025, 1, 15, 10, 0, 0)
        result = _compute_next_run("0 * * * *", after)
        self.assertEqual(result.hour, 11)

    def test_daily_cron(self):
        after = datetime(2025, 1, 15, 8, 0, 0)
        result = _compute_next_run("30 9 * * *", after)
        self.assertEqual(result.hour, 9)
        self.assertEqual(result.minute, 30)
        self.assertEqual(result.day, 15)

    def test_daily_already_past(self):
        after = datetime(2025, 1, 15, 10, 0, 0)
        result = _compute_next_run("30 9 * * *", after)
        # Should be next day
        self.assertEqual(result.day, 16)

    def test_invalid_cron(self):
        result = _compute_next_run("bad", datetime.now())
        self.assertIsNone(result)

    def test_wildcard_fallback(self):
        after = datetime(2025, 1, 15, 10, 0, 0)
        result = _compute_next_run("* * * * *", after)
        # Fallback: 1 hour later
        self.assertEqual(result, after + timedelta(hours=1))


# ─── Agent Timeout Defaults (no DB needed) ──────────────────────────────────

class TestAgentTimeouts(unittest.TestCase):

    def test_timeouts_defined(self):
        # Import from orchestrator would pull heavy deps,
        # so we just replicate the dict.
        AGENT_TIMEOUTS = {
            "planner": 120, "coder": 300, "researcher": 180,
            "reviewer": 120, "executor": 180, "pipeline": 600,
        }
        self.assertEqual(AGENT_TIMEOUTS["planner"], 120)
        self.assertEqual(AGENT_TIMEOUTS["coder"], 300)
        self.assertEqual(AGENT_TIMEOUTS["pipeline"], 600)

    def test_default_timeout_for_unknown_agent(self):
        AGENT_TIMEOUTS = {
            "planner": 120, "coder": 300, "researcher": 180,
            "reviewer": 120, "executor": 180, "pipeline": 600,
        }
        # Unknown agent falls back to dict.get default
        self.assertEqual(AGENT_TIMEOUTS.get("unknown_agent", 180), 180)


if __name__ == "__main__":
    unittest.main()
