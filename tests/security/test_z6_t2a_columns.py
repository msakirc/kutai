# tests/security/test_z6_t2a_columns.py
"""Z6 T2A — credentials hardening schema columns.

Verifies:
  * Migration is idempotent (init_db twice is a no-op).
  * `scope`, `rotated_at`, `expires_at`, `key_version`, `schema_id` columns
    exist on the `credentials` table.
  * `store_credential()` writes `expires_at` to BOTH the indexed column
    AND the encrypted envelope.
  * `get_credential()` honours expiration from the column (pre-check) and
    from the envelope (tamper-proof).
  * `scope` defaults to `read_write` for legacy-style stores.
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

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


class _BaseT2A(unittest.TestCase):
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
        self.cs_mod = cs_mod
        cs_mod._fernet = None
        cs_mod._MASTER_KEY = None
        self._orig_key = os.environ.get("KUTAY_MASTER_KEY")
        os.environ["KUTAY_MASTER_KEY"] = "z6-t2a-test-key"

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


class TestColumnsExist(_BaseT2A):
    def test_columns_present(self):
        async def _t():
            db = await self.db_mod.get_db()
            cur = await db.execute("PRAGMA table_info(credentials)")
            cols = {row[1] for row in await cur.fetchall()}
            for expected in {"scope", "rotated_at", "expires_at",
                             "key_version", "schema_id"}:
                self.assertIn(expected, cols)
        run_async(_t())


class TestMigrationIdempotent(_BaseT2A):
    def test_second_init_is_noop(self):
        async def _t():
            # init_db() in setUp already ran. Call again — should not raise.
            await self.db_mod.init_db()
            await self.db_mod.init_db()
            db = await self.db_mod.get_db()
            cur = await db.execute(
                "SELECT COUNT(*) FROM schema_migrations "
                "WHERE version = '2026-05-11-credentials-hardening'"
            )
            row = await cur.fetchone()
            self.assertEqual(row[0], 1)
        run_async(_t())


class TestExpiresAtSync(_BaseT2A):
    def test_expires_at_written_to_column_and_envelope(self):
        async def _t():
            future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
            await self.cs_mod.store_credential(
                "github", {"token": "ghp_x"}, expires_at=future
            )

            db = await self.db_mod.get_db()
            cur = await db.execute(
                "SELECT expires_at, encrypted_data, scope FROM credentials "
                "WHERE service_name = ?",
                ("github",),
            )
            row = await cur.fetchone()
            self.assertEqual(row[0], future)  # column has it
            # envelope also has it (decrypt + assert)
            plaintext = self.cs_mod._decrypt(row[1])
            envelope = json.loads(plaintext)
            self.assertEqual(envelope["_expires_at"], future)
            # scope defaults to read_write
            self.assertEqual(row[2], "read_write")
        run_async(_t())

    def test_no_expires_at_keeps_column_null(self):
        async def _t():
            await self.cs_mod.store_credential("github", {"token": "y"})
            db = await self.db_mod.get_db()
            cur = await db.execute(
                "SELECT expires_at FROM credentials WHERE service_name = ?",
                ("github",),
            )
            row = await cur.fetchone()
            self.assertIsNone(row[0])
        run_async(_t())


class TestExpiryEnforced(_BaseT2A):
    def test_expired_via_column_returns_none(self):
        async def _t():
            past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
            await self.cs_mod.store_credential(
                "vercel", {"token": "stale"}, expires_at=past
            )
            result = await self.cs_mod.get_credential("vercel")
            self.assertIsNone(result)
        run_async(_t())

    def test_expired_via_envelope_only_returns_none(self):
        # Simulate column tampering: clear the column but leave envelope expired
        async def _t():
            past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
            await self.cs_mod.store_credential(
                "tamper", {"token": "bad"}, expires_at=past
            )
            db = await self.db_mod.get_db()
            await db.execute(
                "UPDATE credentials SET expires_at = NULL "
                "WHERE service_name = ?",
                ("tamper",),
            )
            await db.commit()
            result = await self.cs_mod.get_credential("tamper")
            self.assertIsNone(result)
        run_async(_t())

    def test_unexpired_returns_data(self):
        async def _t():
            future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
            await self.cs_mod.store_credential(
                "ok", {"token": "fresh"}, expires_at=future
            )
            result = await self.cs_mod.get_credential("ok")
            self.assertIsNotNone(result)
            self.assertEqual(result["token"], "fresh")
        run_async(_t())


class TestScopeAndSchemaIdPersisted(_BaseT2A):
    def test_scope_and_schema_id_round_trip(self):
        async def _t():
            await self.cs_mod.store_credential(
                "stripe",
                {"secret_key": "sk_test_x"},
                scope="read_only",
                schema_id="stripe",
            )
            db = await self.db_mod.get_db()
            cur = await db.execute(
                "SELECT scope, schema_id, key_version FROM credentials "
                "WHERE service_name = ?",
                ("stripe",),
            )
            row = await cur.fetchone()
            self.assertEqual(row[0], "read_only")
            self.assertEqual(row[1], "stripe")
            self.assertEqual(row[2], 1)  # default key_version
        run_async(_t())

    def test_upsert_preserves_scope_when_not_passed(self):
        async def _t():
            await self.cs_mod.store_credential(
                "github", {"token": "a"}, scope="admin"
            )
            # Second store without scope must NOT reset to default
            await self.cs_mod.store_credential("github", {"token": "b"})
            db = await self.db_mod.get_db()
            cur = await db.execute(
                "SELECT scope FROM credentials WHERE service_name = ?",
                ("github",),
            )
            row = await cur.fetchone()
            self.assertEqual(row[0], "admin")
        run_async(_t())


if __name__ == "__main__":
    unittest.main()
