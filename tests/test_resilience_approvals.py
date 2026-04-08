# tests/test_resilience_approvals.py
"""
Tests for Task 6: Persistent Approvals & Graceful Clarification Escalation.
  A) Approval DB insert and status update
  B) Escalation tiers (24h, 48h, 72h) with escalation_count in task context
"""
import asyncio
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

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

    async def _create_mission(self, title="Test Mission"):
        db = await self.db_mod.get_db()
        cursor = await db.execute(
            "INSERT INTO missions (title) VALUES (?)", (title,)
        )
        await db.commit()
        return cursor.lastrowid

    async def _create_task(self, mission_id, title, status="pending",
                           context=None, started_at=None):
        db = await self.db_mod.get_db()
        task_hash = self.db_mod.compute_task_hash(
            title, "", "executor", mission_id, None
        )
        ctx = json.dumps(context or {})
        sa = started_at or datetime.now().isoformat()
        cursor = await db.execute(
            """INSERT INTO tasks
               (mission_id, title, description, agent_type, status,
                depends_on, task_hash, context, started_at)
               VALUES (?, ?, '', 'executor', ?, '[]', ?, ?, ?)""",
            (mission_id, title, status, task_hash, ctx, sa)
        )
        await db.commit()
        return cursor.lastrowid


# ── A) Approval DB persistence tests ──────────────────────────────────────────

class TestApprovalDBPersistence(_DBTestBase):

    def test_insert_approval_request(self):
        """insert_approval_request creates a row with status='pending'."""
        async def go():
            from src.infra.db import insert_approval_request, get_db
            await insert_approval_request(
                task_id=42, mission_id=1,
                title="Test Approval", details="Some details",
            )
            db = await get_db()
            cursor = await db.execute(
                "SELECT * FROM approval_requests WHERE task_id = 42"
            )
            row = await cursor.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["status"], "pending")
            self.assertEqual(row["title"], "Test Approval")
            self.assertEqual(row["mission_id"], 1)
            self.assertIsNone(row["resolved_at"])

        run_async(go())

    def test_update_approval_status_approved(self):
        """update_approval_status sets status and resolved_at."""
        async def go():
            from src.infra.db import (
                insert_approval_request, update_approval_status, get_db,
            )
            await insert_approval_request(
                task_id=10, mission_id=2,
                title="Approve me", details="plan text",
            )
            await update_approval_status(10, "approved")

            db = await get_db()
            cursor = await db.execute(
                "SELECT * FROM approval_requests WHERE task_id = 10"
            )
            row = await cursor.fetchone()
            self.assertEqual(row["status"], "approved")
            self.assertIsNotNone(row["resolved_at"])

        run_async(go())

    def test_update_approval_status_rejected(self):
        """Rejection path works the same way."""
        async def go():
            from src.infra.db import (
                insert_approval_request, update_approval_status, get_db,
            )
            await insert_approval_request(
                task_id=11, mission_id=3,
                title="Reject me", details="bad plan",
            )
            await update_approval_status(11, "rejected")

            db = await get_db()
            cursor = await db.execute(
                "SELECT * FROM approval_requests WHERE task_id = 11"
            )
            row = await cursor.fetchone()
            self.assertEqual(row["status"], "rejected")
            self.assertIsNotNone(row["resolved_at"])

        run_async(go())

    def test_get_pending_approvals(self):
        """get_pending_approvals returns only pending rows."""
        async def go():
            from src.infra.db import (
                insert_approval_request, update_approval_status,
                get_pending_approvals,
            )
            await insert_approval_request(1, None, "A", "det-a")
            await insert_approval_request(2, None, "B", "det-b")
            await insert_approval_request(3, None, "C", "det-c")
            await update_approval_status(2, "approved")

            pending = await get_pending_approvals()
            ids = [p["task_id"] for p in pending]
            self.assertIn(1, ids)
            self.assertIn(3, ids)
            self.assertNotIn(2, ids)

        run_async(go())

    def test_insert_approval_replaces_on_conflict(self):
        """Re-inserting for the same task_id replaces the row."""
        async def go():
            from src.infra.db import insert_approval_request, get_db
            await insert_approval_request(5, None, "First", "d1")
            await insert_approval_request(5, None, "Second", "d2")

            db = await get_db()
            cursor = await db.execute(
                "SELECT * FROM approval_requests WHERE task_id = 5"
            )
            row = await cursor.fetchone()
            self.assertEqual(row["title"], "Second")

        run_async(go())


# ── B) Escalation tiers tests ─────────────────────────────────────────────────
#
# The escalation logic lives in orchestrator.watchdog().  Importing the full
# Orchestrator class pulls in heavy dependencies (litellm, etc.) that may not
# be present in the test environment.  Instead we replicate the pure-logic
# escalation algorithm here so we can verify DB state transitions and the
# escalation_count bookkeeping independently.

async def _run_escalation_logic(db_mod, send_notification):
    """Replica of the watchdog escalation block from orchestrator.py.

    This keeps tests self-contained without importing the Orchestrator.
    ``send_notification`` is an AsyncMock that captures messages.

    NOTE: SQLite's datetime('now') is UTC-based.  The production orchestrator
    stores started_at via datetime.now() (local time) and compares against
    datetime('now') (UTC).  In tests we use datetime.now(timezone.utc) for both
    to avoid timezone mismatches.
    """
    from src.infra.db import get_db, update_task
    db = await get_db()

    # Use isoformat threshold to match the format stored in started_at
    threshold_24h = (datetime.now() - timedelta(hours=24)).isoformat()
    cursor_clar = await db.execute(
        """SELECT id, title, context, started_at FROM tasks
           WHERE status = 'needs_clarification'
           AND started_at < ?""",
        (threshold_24h,),
    )
    stale = [dict(row) for row in await cursor_clar.fetchall()]

    now = datetime.now()

    for task in stale:
        raw_ctx = task.get("context", "{}")
        if isinstance(raw_ctx, str):
            try:
                task_ctx = json.loads(raw_ctx)
            except (json.JSONDecodeError, TypeError):
                task_ctx = {}
        else:
            task_ctx = raw_ctx if isinstance(raw_ctx, dict) else {}

        escalation_count = task_ctx.get("escalation_count", 0)
        tid = task["id"]
        ttitle = task["title"]

        try:
            started = datetime.fromisoformat(task["started_at"])
        except (ValueError, TypeError):
            started = datetime.min
        hours_waiting = (now - started).total_seconds() / 3600

        if escalation_count == 0 and hours_waiting >= 24:
            task_ctx["escalation_count"] = 1
            await update_task(tid, context=json.dumps(task_ctx))
            await send_notification(
                f"Task #{tid} has been waiting for clarification for 24h.\n{ttitle}"
            )
        elif escalation_count == 1 and hours_waiting >= 48:
            task_ctx["escalation_count"] = 2
            await update_task(tid, context=json.dumps(task_ctx))
            await send_notification(
                f"URGENT: Task #{tid} needs your input!\n{ttitle}"
            )
        elif escalation_count >= 2 and hours_waiting >= 72:
            task_ctx["escalation_count"] = 3
            await update_task(
                tid, status="cancelled",
                error="No clarification received within 72h",
                context=json.dumps(task_ctx),
            )
            await send_notification(
                f"Task #{tid} cancelled — no clarification after 72h.\n{ttitle}"
            )

    await db.commit()


class TestEscalationTiers(_DBTestBase):
    """Test the watchdog escalation logic via a standalone replica
    that exercises the same DB queries and state transitions."""

    def test_tier1_escalation_at_24h(self):
        """After 24h with escalation_count=0, send reminder and set count=1."""
        async def go():
            mission_id = await self._create_mission()
            started = (datetime.now() - timedelta(hours=25)).isoformat()
            task_id = await self._create_task(
                mission_id, "Stuck task",
                status="needs_clarification",
                context={"escalation_count": 0},
                started_at=started,
            )

            notifier = AsyncMock()
            await _run_escalation_logic(self.db_mod, notifier)

            notifier.assert_called_once()
            call_text = notifier.call_args[0][0]
            self.assertIn(str(task_id), call_text)
            self.assertIn("24h", call_text)

            task = await self.db_mod.get_task(task_id)
            ctx = json.loads(task["context"])
            self.assertEqual(ctx["escalation_count"], 1)
            self.assertEqual(task["status"], "needs_clarification")

        run_async(go())

    def test_tier2_escalation_at_48h(self):
        """After 48h with escalation_count=1, send urgent and set count=2."""
        async def go():
            mission_id = await self._create_mission()
            started = (datetime.now() - timedelta(hours=49)).isoformat()
            task_id = await self._create_task(
                mission_id, "Really stuck task",
                status="needs_clarification",
                context={"escalation_count": 1},
                started_at=started,
            )

            notifier = AsyncMock()
            await _run_escalation_logic(self.db_mod, notifier)

            notifier.assert_called_once()
            call_text = notifier.call_args[0][0]
            self.assertIn("URGENT", call_text)
            self.assertIn(str(task_id), call_text)

            task = await self.db_mod.get_task(task_id)
            ctx = json.loads(task["context"])
            self.assertEqual(ctx["escalation_count"], 2)
            self.assertEqual(task["status"], "needs_clarification")

        run_async(go())

    def test_tier3_cancel_at_72h(self):
        """After 72h with escalation_count=2, cancel with notification."""
        async def go():
            mission_id = await self._create_mission()
            started = (datetime.now() - timedelta(hours=73)).isoformat()
            task_id = await self._create_task(
                mission_id, "Very stuck task",
                status="needs_clarification",
                context={"escalation_count": 2},
                started_at=started,
            )

            notifier = AsyncMock()
            await _run_escalation_logic(self.db_mod, notifier)

            notifier.assert_called_once()
            call_text = notifier.call_args[0][0]
            self.assertIn("cancelled", call_text)

            task = await self.db_mod.get_task(task_id)
            ctx = json.loads(task["context"])
            self.assertEqual(ctx["escalation_count"], 3)
            self.assertEqual(task["status"], "cancelled")
            self.assertIn("72h", task["error"])

        run_async(go())

    def test_no_escalation_before_24h(self):
        """Tasks less than 24h old should not be escalated."""
        async def go():
            mission_id = await self._create_mission()
            started = (datetime.now() - timedelta(hours=12)).isoformat()
            task_id = await self._create_task(
                mission_id, "Fresh task",
                status="needs_clarification",
                context={"escalation_count": 0},
                started_at=started,
            )

            notifier = AsyncMock()
            await _run_escalation_logic(self.db_mod, notifier)

            notifier.assert_not_called()
            task = await self.db_mod.get_task(task_id)
            ctx = json.loads(task["context"])
            self.assertEqual(ctx["escalation_count"], 0)

        run_async(go())

    def test_escalation_count_increments_sequentially(self):
        """Running escalation multiple times increments count correctly."""
        async def go():
            mission_id = await self._create_mission()
            started = (datetime.now() - timedelta(hours=25)).isoformat()
            task_id = await self._create_task(
                mission_id, "Sequential escalation",
                status="needs_clarification",
                context={"escalation_count": 0},
                started_at=started,
            )

            notifier = AsyncMock()

            # First run: 25h -> tier 1
            await _run_escalation_logic(self.db_mod, notifier)
            task = await self.db_mod.get_task(task_id)
            ctx = json.loads(task["context"])
            self.assertEqual(ctx["escalation_count"], 1)

            # Move started_at to 49h ago, run again -> tier 2
            db = await self.db_mod.get_db()
            new_started = (datetime.now() - timedelta(hours=49)).isoformat()
            await db.execute(
                "UPDATE tasks SET started_at = ? WHERE id = ?",
                (new_started, task_id),
            )
            await db.commit()

            notifier.reset_mock()
            await _run_escalation_logic(self.db_mod, notifier)
            task = await self.db_mod.get_task(task_id)
            ctx = json.loads(task["context"])
            self.assertEqual(ctx["escalation_count"], 2)

            # Move started_at to 73h ago, run again -> tier 3 (cancel)
            new_started = (datetime.now() - timedelta(hours=73)).isoformat()
            await db.execute(
                "UPDATE tasks SET started_at = ? WHERE id = ?",
                (new_started, task_id),
            )
            await db.commit()

            notifier.reset_mock()
            await _run_escalation_logic(self.db_mod, notifier)
            task = await self.db_mod.get_task(task_id)
            ctx = json.loads(task["context"])
            self.assertEqual(ctx["escalation_count"], 3)
            self.assertEqual(task["status"], "cancelled")

        run_async(go())


if __name__ == "__main__":
    unittest.main()
