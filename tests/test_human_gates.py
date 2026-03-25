# tests/test_human_gates.py
"""
Tests for Gap 3B: Human approval gates and 4-hour clarification reminder.
  A) Human gate blocks execution and requests approval
  B) Human gate is skipped when not in context
  C) Human gate handles approval timeout gracefully
  D) 4-hour nudge for tasks between 4h and 24h old
  E) Nudge is not sent twice (nudge_sent flag)

NOTE: The Orchestrator class has heavy dependencies (litellm, etc.) that may
not be present in the test environment.  We replicate the pure-logic gate and
nudge algorithms here so we can verify behavior independently, following the
same pattern used in test_resilience_approvals.py.
"""
import asyncio
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_async(coro):
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


# ═══════════════════════════════════════════════════════════════════════
# Replicated logic from orchestrator.py — kept in sync manually.
# This avoids importing the full Orchestrator which needs litellm etc.
# ═══════════════════════════════════════════════════════════════════════

async def _run_human_gate_logic(task, request_approval, update_task_fn):
    """Replica of the human gate block from process_task().

    Returns:
        "paused"  - gate rejected / timed out
        "error"   - gate raised an exception (task continues)
        "approved" - gate approved
        "skipped" - no gate in context
    """
    task_ctx = task.get("context", "{}")
    if isinstance(task_ctx, str):
        try:
            task_ctx = json.loads(task_ctx)
        except (json.JSONDecodeError, TypeError):
            task_ctx = {}
    if not isinstance(task_ctx, dict):
        task_ctx = {}

    if task_ctx.get("human_gate"):
        try:
            approved = await request_approval(
                task["id"],
                task.get("title", ""),
                task.get("description", "")[:200],
                tier=task.get("tier", "auto"),
                mission_id=task.get("mission_id"),
            )
            if not approved:
                await update_task_fn(task["id"], status="paused")
                return "paused"
            return "approved"
        except Exception:
            return "error"
    return "skipped"


async def _run_nudge_logic(db_mod, send_notification):
    """Replica of the Tier 0 (4-hour nudge) block from watchdog().

    ``send_notification`` is an AsyncMock that captures messages.
    """
    from src.infra.db import get_db, update_task
    db = await get_db()

    threshold_24h = (datetime.now() - timedelta(hours=24)).isoformat()
    threshold_4h = (datetime.now() - timedelta(hours=4)).isoformat()

    cursor_nudge = await db.execute(
        """SELECT id, title, context FROM tasks
           WHERE status = 'needs_clarification'
           AND started_at < ?
           AND started_at >= ?""",
        (threshold_4h, threshold_24h),
    )
    nudge_tasks = [dict(row) for row in await cursor_nudge.fetchall()]

    for task in nudge_tasks:
        raw_ctx = task.get("context", "{}")
        if isinstance(raw_ctx, str):
            try:
                task_ctx = json.loads(raw_ctx)
            except (json.JSONDecodeError, TypeError):
                task_ctx = {}
        else:
            task_ctx = raw_ctx if isinstance(raw_ctx, dict) else {}

        if not task_ctx.get("nudge_sent"):
            task_ctx["nudge_sent"] = True
            await update_task(task["id"], context=json.dumps(task_ctx))
            await send_notification(
                f"\U0001f4ac Gentle reminder: Task #{task['id']} needs your input.\n"
                f"*{task['title']}*"
            )


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


# ─── A. Human Gate Detection Tests ────────────────────────────────────


class TestHumanGateDetection(unittest.TestCase):

    def test_human_gate_true_detected(self):
        ctx = {"is_workflow_step": True, "human_gate": True, "mission_id": 42}
        self.assertTrue(ctx.get("human_gate"))

    def test_human_gate_absent_skips(self):
        ctx = {"is_workflow_step": True, "mission_id": 42}
        self.assertFalse(ctx.get("human_gate"))

    def test_human_gate_false_skips(self):
        ctx = {"is_workflow_step": True, "human_gate": False, "mission_id": 42}
        self.assertFalse(ctx.get("human_gate"))

    def test_human_gate_from_json_context(self):
        raw = json.dumps({"is_workflow_step": True, "human_gate": True})
        ctx = json.loads(raw)
        self.assertTrue(ctx.get("human_gate"))


# ─── B. Human Gate Execution Logic Tests ──────────────────────────────


class TestHumanGateBlocks(unittest.TestCase):
    """Human gate blocks execution and requests approval when human_gate is True."""

    def test_gate_blocks_on_rejection(self):
        """When human_gate is True and approval is rejected, task is paused."""
        request_approval = AsyncMock(return_value=False)
        update_task = AsyncMock()

        task = {
            "id": 42,
            "title": "Deploy to prod",
            "description": "Deploy the application to production",
            "tier": "auto",
            "mission_id": None,
            "context": json.dumps({"human_gate": True}),
        }

        result = run_async(_run_human_gate_logic(task, request_approval, update_task))

        self.assertEqual(result, "paused")
        request_approval.assert_called_once()
        call_args = request_approval.call_args
        self.assertEqual(call_args[0][0], 42)
        update_task.assert_called_once_with(42, status="paused")

    def test_gate_allows_on_approval(self):
        """When human_gate is True and approved, execution continues."""
        request_approval = AsyncMock(return_value=True)
        update_task = AsyncMock()

        task = {
            "id": 42,
            "title": "Deploy to prod",
            "description": "Deploy the application to production",
            "tier": "auto",
            "mission_id": 5,
            "context": json.dumps({"human_gate": True}),
        }

        result = run_async(_run_human_gate_logic(task, request_approval, update_task))

        self.assertEqual(result, "approved")
        request_approval.assert_called_once()
        # Verify mission_id is passed through
        call_kwargs = request_approval.call_args[1]
        self.assertEqual(call_kwargs["mission_id"], 5)
        update_task.assert_not_called()


class TestHumanGateSkipped(unittest.TestCase):
    """Human gate is skipped when human_gate is not in context."""

    def test_no_gate_when_not_in_context(self):
        """When human_gate is absent, no approval is requested."""
        request_approval = AsyncMock(return_value=True)
        update_task = AsyncMock()

        task = {
            "id": 43,
            "title": "Simple task",
            "description": "No gate needed",
            "tier": "auto",
            "mission_id": None,
            "context": "{}",
        }

        result = run_async(_run_human_gate_logic(task, request_approval, update_task))

        self.assertEqual(result, "skipped")
        request_approval.assert_not_called()
        update_task.assert_not_called()

    def test_no_gate_when_human_gate_false(self):
        """When human_gate is explicitly False, no approval is requested."""
        request_approval = AsyncMock(return_value=True)
        update_task = AsyncMock()

        task = {
            "id": 43,
            "title": "Simple task",
            "description": "No gate needed",
            "tier": "auto",
            "mission_id": None,
            "context": json.dumps({"human_gate": False}),
        }

        result = run_async(_run_human_gate_logic(task, request_approval, update_task))

        self.assertEqual(result, "skipped")
        request_approval.assert_not_called()


class TestHumanGateErrorHandling(unittest.TestCase):
    """Human gate handles approval errors gracefully (does not crash task)."""

    def test_gate_error_continues_execution(self):
        """If approval request raises an exception, task continues normally."""
        request_approval = AsyncMock(side_effect=Exception("Telegram API down"))
        update_task = AsyncMock()

        task = {
            "id": 44,
            "title": "Important task",
            "description": "Gate should fail gracefully",
            "tier": "auto",
            "mission_id": None,
            "context": json.dumps({"human_gate": True}),
        }

        result = run_async(_run_human_gate_logic(task, request_approval, update_task))

        self.assertEqual(result, "error")
        request_approval.assert_called_once()
        # Task was NOT paused - it should continue
        update_task.assert_not_called()


# ─── C. 4-Hour Nudge Threshold Tests ─────────────────────────────────


class TestNudgeThresholds(unittest.TestCase):

    def test_nudge_context_flag(self):
        ctx = {}
        self.assertFalse(ctx.get("nudge_sent", False))
        ctx["nudge_sent"] = True
        self.assertTrue(ctx.get("nudge_sent", False))

    def test_nudge_threshold_calculation(self):
        now = datetime.now()
        threshold_4h = (now - timedelta(hours=4)).isoformat()
        threshold_24h = (now - timedelta(hours=24)).isoformat()
        started_6h_ago = (now - timedelta(hours=6)).isoformat()
        self.assertGreater(started_6h_ago, threshold_24h)
        self.assertLess(started_6h_ago, threshold_4h)

    def test_recent_task_not_nudged(self):
        now = datetime.now()
        threshold_4h = (now - timedelta(hours=4)).isoformat()
        started_2h_ago = (now - timedelta(hours=2)).isoformat()
        self.assertGreater(started_2h_ago, threshold_4h)

    def test_old_task_not_nudged(self):
        now = datetime.now()
        threshold_24h = (now - timedelta(hours=24)).isoformat()
        started_30h_ago = (now - timedelta(hours=30)).isoformat()
        self.assertLess(started_30h_ago, threshold_24h)


# ─── D. 4-Hour Nudge DB Integration Tests ────────────────────────────


class TestFourHourNudge(_DBTestBase):
    """4-hour nudge is sent for tasks between 4h and 24h old."""

    def test_nudge_sent_for_tasks_between_4h_and_24h(self):
        """Tasks in needs_clarification for 6 hours get a nudge."""
        send_notification = AsyncMock()

        async def _run():
            db = await self.db_mod.get_db()
            started_at = (datetime.now() - timedelta(hours=6)).isoformat()
            await db.execute(
                """INSERT INTO tasks
                   (id, title, description, status, agent_type, tier,
                    priority, started_at, context, depends_on, retry_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (100, "Needs input", "desc", "needs_clarification",
                 "executor", "auto", 5, started_at, "{}", "[]", 0),
            )
            await db.commit()
            await _run_nudge_logic(self.db_mod, send_notification)

        run_async(_run())

        send_notification.assert_called_once()
        call_text = send_notification.call_args[0][0]
        self.assertIn("Gentle reminder", call_text)
        self.assertIn("Task #100", call_text)

    def test_nudge_not_sent_twice(self):
        """Tasks with nudge_sent=True in context do not get a second nudge."""
        send_notification = AsyncMock()

        async def _run():
            db = await self.db_mod.get_db()
            started_at = (datetime.now() - timedelta(hours=6)).isoformat()
            ctx = json.dumps({"nudge_sent": True})
            await db.execute(
                """INSERT INTO tasks
                   (id, title, description, status, agent_type, tier,
                    priority, started_at, context, depends_on, retry_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (101, "Already nudged", "desc", "needs_clarification",
                 "executor", "auto", 5, started_at, ctx, "[]", 0),
            )
            await db.commit()
            await _run_nudge_logic(self.db_mod, send_notification)

        run_async(_run())

        send_notification.assert_not_called()

    def test_no_nudge_for_tasks_under_4h(self):
        """Tasks in needs_clarification for less than 4 hours get no nudge."""
        send_notification = AsyncMock()

        async def _run():
            db = await self.db_mod.get_db()
            started_at = (datetime.now() - timedelta(hours=2)).isoformat()
            await db.execute(
                """INSERT INTO tasks
                   (id, title, description, status, agent_type, tier,
                    priority, started_at, context, depends_on, retry_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (102, "Too fresh", "desc", "needs_clarification",
                 "executor", "auto", 5, started_at, "{}", "[]", 0),
            )
            await db.commit()
            await _run_nudge_logic(self.db_mod, send_notification)

        run_async(_run())

        send_notification.assert_not_called()

    def test_no_nudge_for_tasks_over_24h(self):
        """Tasks older than 24h are handled by the escalation tiers, not the nudge."""
        send_notification = AsyncMock()

        async def _run():
            db = await self.db_mod.get_db()
            started_at = (datetime.now() - timedelta(hours=30)).isoformat()
            await db.execute(
                """INSERT INTO tasks
                   (id, title, description, status, agent_type, tier,
                    priority, started_at, context, depends_on, retry_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (103, "Old task", "desc", "needs_clarification",
                 "executor", "auto", 5, started_at, "{}", "[]", 0),
            )
            await db.commit()
            await _run_nudge_logic(self.db_mod, send_notification)

        run_async(_run())

        send_notification.assert_not_called()

    def test_nudge_sets_flag_in_context(self):
        """After sending nudge, nudge_sent=True is persisted in task context."""
        send_notification = AsyncMock()

        async def _run():
            db = await self.db_mod.get_db()
            started_at = (datetime.now() - timedelta(hours=8)).isoformat()
            await db.execute(
                """INSERT INTO tasks
                   (id, title, description, status, agent_type, tier,
                    priority, started_at, context, depends_on, retry_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (104, "Flag check", "desc", "needs_clarification",
                 "executor", "auto", 5, started_at, "{}", "[]", 0),
            )
            await db.commit()
            await _run_nudge_logic(self.db_mod, send_notification)

            # Verify the flag was persisted
            cursor = await db.execute(
                "SELECT context FROM tasks WHERE id = 104"
            )
            row = await cursor.fetchone()
            ctx = json.loads(row["context"])
            return ctx.get("nudge_sent")

        result = run_async(_run())
        self.assertTrue(result)


# ─── E. Gate Evaluation Hook Tests ────────────────────────────────────


class TestGateEvaluationHook(unittest.TestCase):

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    def test_evaluate_phase_gate_stores_result(self):
        from src.workflows.engine.hooks import _evaluate_phase_gate, get_artifact_store
        store = get_artifact_store()
        self._run(_evaluate_phase_gate(1, "phase_1"))
        result = self._run(store.retrieve(1, "phase_1_gate_result"))
        self.assertIsNotNone(result)
        self.assertIn("PASSED", result)

    def test_evaluate_phase_gate_fails_missing_artifacts(self):
        from src.workflows.engine.hooks import _evaluate_phase_gate, get_artifact_store
        store = get_artifact_store()
        self._run(_evaluate_phase_gate(999, "phase_9"))
        result = self._run(store.retrieve(999, "phase_9_gate_result"))
        self.assertIsNotNone(result)
        self.assertIn("FAILED", result)


if __name__ == "__main__":
    unittest.main()
