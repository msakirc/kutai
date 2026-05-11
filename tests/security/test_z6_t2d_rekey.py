# tests/security/test_z6_t2d_rekey.py
"""Z6 T2D — credential rekey CLI + key_version mismatch guard."""

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


def _b64key(seed: bytes) -> str:
    """Build a 32-byte Fernet-compatible base64 key from a seed."""
    import base64
    pad = (seed + b"\x00" * 32)[:32]
    return base64.urlsafe_b64encode(pad).decode()


class _BaseRekey(unittest.TestCase):
    def setUp(self):
        if not HAS_AIOSQLITE:
            self.skipTest("aiosqlite not installed")

        try:
            from cryptography.fernet import Fernet  # noqa: F401
        except ImportError:
            self.skipTest("cryptography not installed")

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
        from src.security import credential_schemas as schemas
        self.cs_mod = cs_mod
        self.schemas = schemas
        schemas.reset_cache()

        # Stash env, then set v1 only initially
        self._env_backup = {
            k: os.environ.get(k)
            for k in ("KUTAY_MASTER_KEY", "KUTAY_MASTER_KEY_v1",
                     "KUTAY_MASTER_KEY_v2", "KUTAY_MASTER_KEY_v3")
        }
        for k in self._env_backup:
            os.environ.pop(k, None)
        os.environ["KUTAY_MASTER_KEY_v1"] = _b64key(b"rekey-test-v1-key")

        cs_mod._reset_key_state()
        run_async(db_mod.init_db())

    def tearDown(self):
        run_async(self.db_mod.close_db())
        from src.app import config
        config.DB_PATH = self._orig_config_path
        self.db_mod.DB_PATH = self._orig_db_path
        self.cs_mod._reset_key_state()
        for k, v in self._env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for suffix in ("", "-wal", "-shm"):
            try:
                os.unlink(self.db_path + suffix)
            except OSError:
                pass


class TestRekeyRoundTrip(_BaseRekey):
    def test_v1_to_v2_migration(self):
        async def _t():
            # Store under v1
            await self.cs_mod.store_credential("github", {"token": "ghp_x"})
            got = await self.cs_mod.get_credential("github")
            self.assertEqual(got["token"], "ghp_x")

            # Add v2 and rekey
            os.environ["KUTAY_MASTER_KEY_v2"] = _b64key(b"rekey-test-v2-key")
            self.cs_mod._reset_key_state()

            from src.security.rekey import rekey

            stats = await rekey(1, 2)
            self.assertEqual(stats["rekeyed"], 1)
            self.assertEqual(stats["errors"], 0)

            # Verify row now at version 2
            db = await self.db_mod.get_db()
            cur = await db.execute(
                "SELECT key_version FROM credentials WHERE service_name = ?",
                ("github",),
            )
            row = await cur.fetchone()
            self.assertEqual(row[0], 2)

            # Decrypt still works
            got = await self.cs_mod.get_credential("github")
            self.assertEqual(got["token"], "ghp_x")
        run_async(_t())

    def test_dry_run_does_not_modify(self):
        async def _t():
            await self.cs_mod.store_credential("github", {"token": "ghp_x"})

            os.environ["KUTAY_MASTER_KEY_v2"] = _b64key(b"rekey-test-v2-key")
            self.cs_mod._reset_key_state()

            from src.security.rekey import rekey

            stats = await rekey(1, 2, dry_run=True)
            self.assertEqual(stats["rekeyed"], 1)

            db = await self.db_mod.get_db()
            cur = await db.execute(
                "SELECT key_version FROM credentials WHERE service_name = ?",
                ("github",),
            )
            row = await cur.fetchone()
            self.assertEqual(row[0], 1)  # still v1
        run_async(_t())

    def test_already_at_target_is_skipped(self):
        async def _t():
            os.environ["KUTAY_MASTER_KEY_v2"] = _b64key(b"rekey-test-v2-key")
            self.cs_mod._reset_key_state()
            # Stored fresh now under v2 (current = max version = 2)
            await self.cs_mod.store_credential("github", {"token": "ghp_x"})

            from src.security.rekey import rekey

            stats = await rekey(1, 2)
            self.assertEqual(stats["rekeyed"], 0)
            self.assertEqual(stats["skipped"], 1)
        run_async(_t())


class TestMismatchGuard(_BaseRekey):
    def test_refuses_write_when_existing_version_unavailable(self):
        async def _t():
            # Store under v1
            await self.cs_mod.store_credential("github", {"token": "ghp_x"})

            # Drop v1, introduce v2 only — existing row references v1 which
            # we no longer have AND the active version (2) doesn't match.
            os.environ.pop("KUTAY_MASTER_KEY_v1", None)
            os.environ["KUTAY_MASTER_KEY_v2"] = _b64key(b"rekey-test-v2-key")
            self.cs_mod._reset_key_state()

            with self.assertRaises(RuntimeError) as cm:
                await self.cs_mod.store_credential(
                    "github", {"token": "new"}
                )
            self.assertIn("rekey", str(cm.exception).lower())
        run_async(_t())

    def test_refuses_when_versions_mismatch_but_both_available(self):
        async def _t():
            await self.cs_mod.store_credential("github", {"token": "ghp_x"})

            os.environ["KUTAY_MASTER_KEY_v2"] = _b64key(b"rekey-test-v2-key")
            self.cs_mod._reset_key_state()

            with self.assertRaises(RuntimeError) as cm:
                await self.cs_mod.store_credential(
                    "github", {"token": "new"}
                )
            self.assertIn("rekey", str(cm.exception).lower())
        run_async(_t())


class TestCliArgs(unittest.TestCase):
    def test_main_rejects_same_version(self):
        try:
            from cryptography.fernet import Fernet  # noqa: F401
        except ImportError:
            self.skipTest("cryptography not installed")
        from src.security.rekey import main
        # Same version → ValueError → exit 2
        rc = main(["--from-version", "1", "--to-version", "1"])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
