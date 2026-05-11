# tests/security/test_z6_t2c_audit.py
"""Z6 T2C — credential_access_log audit trail."""

import asyncio
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


try:
    import aiosqlite  # noqa: F401

    HAS_AIOSQLITE = True
except ImportError:
    HAS_AIOSQLITE = False


class _BaseAudit(unittest.TestCase):
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

        import src.security.credential_store as cs_mod
        from src.security import credential_audit, _audit_context
        from src.security import credential_schemas as schemas
        self.cs_mod = cs_mod
        self.audit = credential_audit
        self.audit_ctx = _audit_context
        self.schemas = schemas
        schemas.reset_cache()
        cs_mod._fernet = None
        cs_mod._MASTER_KEY = None
        self._orig_key = os.environ.get("KUTAY_MASTER_KEY")
        os.environ["KUTAY_MASTER_KEY"] = "z6-t2c-test-key"

        run_async(db_mod.init_db())

    def tearDown(self):
        run_async(self.db_mod.close_db())
        from src.app import config
        config.DB_PATH = self._orig_config_path
        self.db_mod.DB_PATH = self._orig_db_path
        self.cs_mod._fernet = None
        self.cs_mod._MASTER_KEY = None
        if self._orig_key is None:
            os.environ.pop("KUTAY_MASTER_KEY", None)
        else:
            os.environ["KUTAY_MASTER_KEY"] = self._orig_key
        for suffix in ("", "-wal", "-shm"):
            try:
                os.unlink(self.db_path + suffix)
            except OSError:
                pass


class TestAuditTableExists(_BaseAudit):
    def test_table_present(self):
        async def _t():
            db = await self.db_mod.get_db()
            cur = await db.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='credential_access_log'"
            )
            row = await cur.fetchone()
            self.assertIsNotNone(row)
        run_async(_t())


class TestAuditLogsStoreAndRead(_BaseAudit):
    def test_store_then_read_emits_two_events(self):
        async def _t():
            await self.cs_mod.store_credential("github", {"token": "ghp_x"})
            await self.cs_mod.get_credential("github")

            events = await self.audit.recent_events("github", limit=10)
            self.assertGreaterEqual(len(events), 2)
            actions = [e["action"] for e in events]
            # newest first → read came after write
            self.assertEqual(actions[0], "read")
            self.assertIn("write", actions)
            self.assertTrue(all(e["success"] == 1 for e in events))
        run_async(_t())

    def test_resave_logs_rotate(self):
        async def _t():
            await self.cs_mod.store_credential("github", {"token": "a"})
            await self.cs_mod.store_credential("github", {"token": "b"})

            events = await self.audit.recent_events("github")
            actions = [e["action"] for e in events]
            self.assertIn("rotate", actions)
            self.assertIn("write", actions)
        run_async(_t())

    def test_delete_logged(self):
        async def _t():
            await self.cs_mod.store_credential("github", {"token": "x"})
            ok = await self.cs_mod.delete_credential("github")
            self.assertTrue(ok)
            events = await self.audit.recent_events("github")
            self.assertEqual(events[0]["action"], "delete")
            self.assertEqual(events[0]["success"], 1)
        run_async(_t())

    def test_missing_credential_read_logged_as_failure(self):
        async def _t():
            await self.cs_mod.get_credential("nonexistent")
            events = await self.audit.recent_events("nonexistent")
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["action"], "read")
            self.assertEqual(events[0]["success"], 0)
            self.assertEqual(events[0]["error"], "not_found")
        run_async(_t())


class TestAuditContextPropagation(_BaseAudit):
    def test_context_var_captured(self):
        async def _t():
            async with self.audit_ctx.audit_context(
                mission_id=42, task_id=101, agent="executor", model_id="qwen2.5-7b",
            ):
                await self.cs_mod.store_credential("github", {"token": "ghp_x"})
                await self.cs_mod.get_credential("github")

            events = await self.audit.recent_events("github")
            # read event newest
            read = events[0]
            self.assertEqual(read["mission_id"], 42)
            self.assertEqual(read["task_id"], 101)
            self.assertEqual(read["agent"], "executor")
            self.assertEqual(read["model_id"], "qwen2.5-7b")
        run_async(_t())

    def test_no_context_leaves_nulls(self):
        async def _t():
            self.audit_ctx.reset()
            await self.cs_mod.store_credential("github", {"token": "ghp_x"})
            events = await self.audit.recent_events("github")
            self.assertIsNone(events[0]["mission_id"])
            self.assertIsNone(events[0]["agent"])
        run_async(_t())

    def test_nested_context_merges(self):
        async def _t():
            async with self.audit_ctx.audit_context(mission_id=7):
                async with self.audit_ctx.audit_context(agent="planner"):
                    await self.cs_mod.store_credential(
                        "github", {"token": "ghp_x"}
                    )
            events = await self.audit.recent_events("github")
            self.assertEqual(events[0]["mission_id"], 7)
            self.assertEqual(events[0]["agent"], "planner")
        run_async(_t())


class TestRecentEventsLimit(_BaseAudit):
    def test_limit_applied(self):
        async def _t():
            for _ in range(5):
                await self.cs_mod.store_credential("github", {"token": "x"})
            events = await self.audit.recent_events("github", limit=2)
            self.assertEqual(len(events), 2)
        run_async(_t())

    def test_global_query_returns_all_services(self):
        async def _t():
            await self.cs_mod.store_credential("github", {"token": "x"})
            await self.cs_mod.store_credential("vercel", {"token": "y"})
            events = await self.audit.recent_events(None, limit=20)
            services = {e["service_name"] for e in events}
            self.assertEqual(services, {"github", "vercel"})
        run_async(_t())


if __name__ == "__main__":
    unittest.main()
