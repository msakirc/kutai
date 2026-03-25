# tests/test_phase1.py
"""
Tests for Phase 1: Data Layer Hardening
  1.1 WAL mode & connection pool (singleton)
  1.2 Task locking (claim_task)
  1.3 Transaction safety (add_subtasks_atomically)
  1.4 Task checkpointing (save/load/clear)
  1.5 Idempotency keys (_tool_idempotency_key)
  1.6 Task deduplication (compute_task_hash + add_task dedup)
"""
import asyncio
import hashlib
import json
import os
import sys
import tempfile
import unittest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Helpers ────────────────────────────────────────────────────────────────

def run_async(coro):
    """Run an async coroutine synchronously for tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _patch_db_path(db_mod, db_path):
    """Patch the DB_PATH in both config and the db module (which captures
    its own copy via ``from config import DB_PATH``)."""
    import config
    config.DB_PATH = db_path
    db_mod.DB_PATH = db_path


# ─── 1.5 Idempotency Key (no DB needed) ────────────────────────────────────
# Inline to avoid importing agents.base which pulls in litellm.

def _tool_idempotency_key(tool_name: str, tool_args: dict) -> str:
    """Replicated from BaseAgent for testing without litellm."""
    raw = f"{tool_name}|{json.dumps(tool_args, sort_keys=True)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class TestToolIdempotencyKey(unittest.TestCase):

    def test_deterministic(self):
        k1 = _tool_idempotency_key("shell", {"command": "ls -la"})
        k2 = _tool_idempotency_key("shell", {"command": "ls -la"})
        self.assertEqual(k1, k2)

    def test_different_tool_different_key(self):
        k1 = _tool_idempotency_key("shell", {"command": "ls"})
        k2 = _tool_idempotency_key("write_file", {"command": "ls"})
        self.assertNotEqual(k1, k2)

    def test_different_args_different_key(self):
        k1 = _tool_idempotency_key("shell", {"command": "ls"})
        k2 = _tool_idempotency_key("shell", {"command": "pwd"})
        self.assertNotEqual(k1, k2)

    def test_arg_order_independent(self):
        """sorted keys ensures {a:1, b:2} == {b:2, a:1}."""
        k1 = _tool_idempotency_key("shell", {"a": 1, "b": 2})
        k2 = _tool_idempotency_key("shell", {"b": 2, "a": 1})
        self.assertEqual(k1, k2)

    def test_length_is_16(self):
        k = _tool_idempotency_key("shell", {"command": "ls"})
        self.assertEqual(len(k), 16)


# ─── 1.6 Task Deduplication Hash (no DB needed) ────────────────────────────
# Inline from db.py

def compute_task_hash(title, description, agent_type,
                      mission_id=None, parent_task_id=None):
    raw = f"{title or ''}|{description or ''}|{agent_type or ''}|{mission_id or ''}|{parent_task_id or ''}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


class TestComputeTaskHash(unittest.TestCase):

    def test_same_inputs_same_hash(self):
        h1 = compute_task_hash("Build API", "REST", "coder", 1, None)
        h2 = compute_task_hash("Build API", "REST", "coder", 1, None)
        self.assertEqual(h1, h2)

    def test_different_agent_type_different_hash(self):
        h1 = compute_task_hash("Build API", "REST", "coder", 1, None)
        h2 = compute_task_hash("Build API", "REST", "researcher", 1, None)
        self.assertNotEqual(h1, h2)

    def test_different_parent_different_hash(self):
        h1 = compute_task_hash("Build API", "REST", "coder", 1, 10)
        h2 = compute_task_hash("Build API", "REST", "coder", 1, 20)
        self.assertNotEqual(h1, h2)

    def test_hash_length_32(self):
        h = compute_task_hash("t", "d", "a", 1, 2)
        self.assertEqual(len(h), 32)

    def test_none_values_consistent(self):
        h1 = compute_task_hash("t", "d", "a", None, None)
        h2 = compute_task_hash("t", "d", "a", None, None)
        self.assertEqual(h1, h2)


# ─── DB-backed tests (WAL, claim_task, add_subtasks, checkpoint) ───────────
# These use a real temp SQLite database via aiosqlite.

try:
    import aiosqlite
    HAS_AIOSQLITE = True
except ImportError:
    HAS_AIOSQLITE = False


class _DBTestBase(unittest.TestCase):
    """Shared setUp/tearDown for all DB-backed test classes."""

    INIT_DB = True  # override to False if you don't need schema

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

        # Patch BOTH copies of DB_PATH and reset singleton
        _patch_db_path(db_mod, self.db_path)
        db_mod._db_connection = None

        if self.INIT_DB:
            run_async(db_mod.init_db())

    def tearDown(self):
        run_async(self.db_mod.close_db())
        import config
        config.DB_PATH = self._orig_config_path
        self.db_mod.DB_PATH = self._orig_db_path
        try:
            os.unlink(self.db_path)
        except OSError:
            pass
        # Also clean WAL/SHM files
        for suffix in ("-wal", "-shm"):
            try:
                os.unlink(self.db_path + suffix)
            except OSError:
                pass


class TestWALAndConnectionPool(_DBTestBase):
    """Test WAL mode and singleton connection pool."""
    INIT_DB = False  # we only need raw get_db()

    def test_singleton_returns_same_connection(self):
        async def _test():
            conn1 = await self.db_mod.get_db()
            conn2 = await self.db_mod.get_db()
            self.assertIs(conn1, conn2)
        run_async(_test())

    def test_wal_mode_enabled(self):
        async def _test():
            db = await self.db_mod.get_db()
            cursor = await db.execute("PRAGMA journal_mode")
            row = await cursor.fetchone()
            self.assertEqual(row[0], "wal")
        run_async(_test())

    def test_close_and_reopen(self):
        async def _test():
            conn1 = await self.db_mod.get_db()
            await self.db_mod.close_db()
            self.assertIsNone(self.db_mod._db_connection)
            conn2 = await self.db_mod.get_db()
            self.assertIsNotNone(conn2)
            self.assertIsNot(conn1, conn2)
        run_async(_test())

    def test_busy_timeout_set(self):
        async def _test():
            db = await self.db_mod.get_db()
            cursor = await db.execute("PRAGMA busy_timeout")
            row = await cursor.fetchone()
            self.assertEqual(row[0], 5000)
        run_async(_test())


class TestClaimTask(_DBTestBase):
    """Test atomic task locking via claim_task."""

    def test_claim_pending_task_succeeds(self):
        async def _test():
            task_id = await self.db_mod.add_task(
                "Claim test 1", "desc", agent_type="coder"
            )
            self.assertIsNotNone(task_id)
            result = await self.db_mod.claim_task(task_id)
            self.assertTrue(result)
            task = await self.db_mod.get_task(task_id)
            self.assertEqual(task["status"], "processing")
        run_async(_test())

    def test_double_claim_fails(self):
        async def _test():
            task_id = await self.db_mod.add_task(
                "Claim test 2", "desc", agent_type="coder"
            )
            self.assertIsNotNone(task_id)
            first = await self.db_mod.claim_task(task_id)
            second = await self.db_mod.claim_task(task_id)
            self.assertTrue(first)
            self.assertFalse(second)
        run_async(_test())

    def test_claim_nonexistent_task(self):
        async def _test():
            result = await self.db_mod.claim_task(99999)
            self.assertFalse(result)
        run_async(_test())

    def test_claim_sets_started_at(self):
        async def _test():
            task_id = await self.db_mod.add_task(
                "Claim started_at", "d", agent_type="coder"
            )
            await self.db_mod.claim_task(task_id)
            task = await self.db_mod.get_task(task_id)
            self.assertIsNotNone(task["started_at"])
        run_async(_test())


class TestAddSubtasksAtomically(_DBTestBase):
    """Test transaction-safe subtask creation."""

    def test_creates_subtasks_and_updates_parent(self):
        async def _test():
            parent_id = await self.db_mod.add_task(
                "Parent A", "desc", agent_type="planner"
            )
            subtasks = [
                {"title": "Sub A1", "description": "d1", "agent_type": "coder"},
                {"title": "Sub A2", "description": "d2", "agent_type": "researcher"},
            ]
            ids = await self.db_mod.add_subtasks_atomically(
                parent_id, subtasks, parent_status="waiting_subtasks"
            )
            self.assertEqual(len(ids), 2)
            self.assertTrue(all(i > 0 for i in ids))

            parent = await self.db_mod.get_task(parent_id)
            self.assertEqual(parent["status"], "waiting_subtasks")

            for sub_id in ids:
                sub = await self.db_mod.get_task(sub_id)
                self.assertEqual(sub["parent_task_id"], parent_id)
                self.assertEqual(sub["status"], "pending")
        run_async(_test())

    def test_dedup_within_transaction(self):
        async def _test():
            parent_id = await self.db_mod.add_task(
                "Parent B", "desc", agent_type="planner"
            )
            subtasks = [
                {"title": "Same B", "description": "same", "agent_type": "coder"},
                {"title": "Same B", "description": "same", "agent_type": "coder"},
            ]
            ids = await self.db_mod.add_subtasks_atomically(
                parent_id, subtasks, parent_status="waiting_subtasks"
            )
            self.assertTrue(ids[0] > 0)
            self.assertEqual(ids[1], -1)
        run_async(_test())

    def test_parent_result_updated(self):
        async def _test():
            parent_id = await self.db_mod.add_task(
                "Parent C", "desc", agent_type="planner"
            )
            subtasks = [
                {"title": "Sub C", "description": "d", "agent_type": "coder"},
            ]
            await self.db_mod.add_subtasks_atomically(
                parent_id, subtasks,
                parent_status="waiting_subtasks",
                parent_result="Decomposed into subtasks"
            )
            parent = await self.db_mod.get_task(parent_id)
            self.assertEqual(parent["result"], "Decomposed into subtasks")
        run_async(_test())


class TestTaskCheckpointing(_DBTestBase):
    """Test save/load/clear checkpoint."""

    def test_save_and_load(self):
        async def _test():
            task_id = await self.db_mod.add_task(
                "Ckpt save", "d", agent_type="coder"
            )
            state = {"iteration": 3, "messages": [{"role": "user", "content": "hi"}]}
            await self.db_mod.save_task_checkpoint(task_id, state)
            loaded = await self.db_mod.load_task_checkpoint(task_id)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["iteration"], 3)
            self.assertEqual(len(loaded["messages"]), 1)
        run_async(_test())

    def test_load_no_checkpoint_returns_none(self):
        async def _test():
            task_id = await self.db_mod.add_task(
                "Ckpt none", "d", agent_type="coder"
            )
            loaded = await self.db_mod.load_task_checkpoint(task_id)
            self.assertIsNone(loaded)
        run_async(_test())

    def test_clear_checkpoint(self):
        async def _test():
            task_id = await self.db_mod.add_task(
                "Ckpt clear", "d", agent_type="coder"
            )
            await self.db_mod.save_task_checkpoint(task_id, {"iteration": 5})
            await self.db_mod.clear_task_checkpoint(task_id)
            loaded = await self.db_mod.load_task_checkpoint(task_id)
            self.assertIsNone(loaded)
        run_async(_test())

    def test_overwrite_checkpoint(self):
        async def _test():
            task_id = await self.db_mod.add_task(
                "Ckpt overwrite", "d", agent_type="coder"
            )
            await self.db_mod.save_task_checkpoint(task_id, {"iteration": 1})
            await self.db_mod.save_task_checkpoint(task_id, {"iteration": 5})
            loaded = await self.db_mod.load_task_checkpoint(task_id)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["iteration"], 5)
        run_async(_test())


class TestTaskDedup(_DBTestBase):
    """Test add_task dedup skips duplicate pending tasks."""

    def test_duplicate_returns_none(self):
        async def _test():
            id1 = await self.db_mod.add_task(
                "Dedup A", "REST", agent_type="coder"
            )
            id2 = await self.db_mod.add_task(
                "Dedup A", "REST", agent_type="coder"
            )
            self.assertIsNotNone(id1)
            self.assertIsNone(id2)
        run_async(_test())

    def test_different_tasks_not_deduped(self):
        async def _test():
            id1 = await self.db_mod.add_task(
                "Dedup B", "REST", agent_type="coder"
            )
            id2 = await self.db_mod.add_task(
                "Dedup C", "Debug", agent_type="coder"
            )
            self.assertIsNotNone(id1)
            self.assertIsNotNone(id2)
            self.assertNotEqual(id1, id2)
        run_async(_test())

    def test_completed_task_not_blocking(self):
        async def _test():
            id1 = await self.db_mod.add_task(
                "Dedup D", "REST", agent_type="coder"
            )
            self.assertIsNotNone(id1)
            await self.db_mod.update_task(id1, status="completed")
            id2 = await self.db_mod.add_task(
                "Dedup D", "REST", agent_type="coder"
            )
            self.assertIsNotNone(id2)
        run_async(_test())


if __name__ == "__main__":
    unittest.main()
