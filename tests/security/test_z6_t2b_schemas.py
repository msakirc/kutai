# tests/security/test_z6_t2b_schemas.py
"""Z6 T2B — per-vendor credential schemas.

Verifies:
  * Schemas load from `credential_schemas/<service>.json`.
  * `validate_payload()` accepts well-formed payloads and rejects missing
    required fields, unknown fields, and unknown scopes.
  * `store_credential()` enforces schema; `unsafe=True` bypasses.
  * Auto-schema_id is set when a schema exists for the service.
  * `describe_schema()` returns a non-empty summary for known services.
"""

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


class TestSchemaLoader(unittest.TestCase):
    def setUp(self):
        from src.security import credential_schemas as cs

        self.cs = cs
        cs.reset_cache()

    def test_known_services_includes_wired_vendors(self):
        services = set(self.cs.known_services())
        for expected in {"github", "vercel", "railway", "stripe",
                         "sendgrid", "cloudflare", "sentry", "supabase"}:
            self.assertIn(expected, services)

    def test_load_schema_returns_dict_with_required_keys(self):
        s = self.cs.load_schema("stripe")
        self.assertIsNotNone(s)
        for key in ("service_name", "required_fields", "optional_fields",
                    "scopes", "default_scope"):
            self.assertIn(key, s)
        self.assertEqual(s["service_name"], "stripe")

    def test_load_schema_unknown_returns_none(self):
        self.assertIsNone(self.cs.load_schema("not-a-real-service"))


class TestValidatePayload(unittest.TestCase):
    def setUp(self):
        from src.security import credential_schemas as cs

        self.cs = cs
        cs.reset_cache()

    def test_valid_payload_accepted(self):
        ok, errors = self.cs.validate_payload(
            "stripe",
            {"secret_key": "sk_test_x", "publishable_key": "pk_test_y"},
        )
        self.assertTrue(ok, msg=errors)
        self.assertEqual(errors, [])

    def test_missing_required_rejected(self):
        ok, errors = self.cs.validate_payload(
            "stripe", {"secret_key": "sk_test_x"}
        )
        self.assertFalse(ok)
        self.assertTrue(any("publishable_key" in e for e in errors))

    def test_empty_required_rejected(self):
        ok, errors = self.cs.validate_payload(
            "stripe", {"secret_key": "", "publishable_key": "pk_x"}
        )
        self.assertFalse(ok)
        self.assertTrue(any("secret_key" in e for e in errors))

    def test_unknown_field_rejected(self):
        ok, errors = self.cs.validate_payload(
            "github", {"token": "ghp_x", "weird": "extra"}
        )
        self.assertFalse(ok)
        self.assertTrue(any("weird" in e for e in errors))

    def test_unknown_scope_rejected(self):
        ok, errors = self.cs.validate_payload(
            "github", {"token": "ghp_x"}, scope="superuser"
        )
        self.assertFalse(ok)
        self.assertTrue(any("scope" in e for e in errors))

    def test_known_scope_accepted(self):
        ok, errors = self.cs.validate_payload(
            "github", {"token": "ghp_x"}, scope="read_only"
        )
        self.assertTrue(ok, msg=errors)

    def test_unknown_service_accepts_anything(self):
        ok, errors = self.cs.validate_payload(
            "no-schema-svc", {"anything": "goes"}
        )
        self.assertTrue(ok)
        self.assertEqual(errors, [])


class TestStoreEnforcesSchema(unittest.TestCase):
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
        from src.security import credential_schemas as schemas
        self.cs_mod = cs_mod
        self.schemas = schemas
        cs_mod._fernet = None
        cs_mod._MASTER_KEY = None
        schemas.reset_cache()
        self._orig_key = os.environ.get("KUTAY_MASTER_KEY")
        os.environ["KUTAY_MASTER_KEY"] = "z6-t2b-test-key"

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

    def test_store_rejects_invalid_payload(self):
        async def _t():
            with self.assertRaises(self.cs_mod.CredentialSchemaError):
                await self.cs_mod.store_credential(
                    "stripe", {"secret_key": "only_one"}
                )
        run_async(_t())

    def test_unsafe_bypass_allows_invalid(self):
        async def _t():
            await self.cs_mod.store_credential(
                "stripe", {"secret_key": "only_one"}, unsafe=True
            )
            got = await self.cs_mod.get_credential("stripe")
            self.assertEqual(got["secret_key"], "only_one")
        run_async(_t())

    def test_auto_schema_id_set_when_schema_exists(self):
        async def _t():
            await self.cs_mod.store_credential(
                "github", {"token": "ghp_x"}
            )
            db = await self.db_mod.get_db()
            cur = await db.execute(
                "SELECT schema_id FROM credentials WHERE service_name = ?",
                ("github",),
            )
            row = await cur.fetchone()
            self.assertEqual(row[0], "github")
        run_async(_t())

    def test_no_schema_no_schema_id(self):
        async def _t():
            await self.cs_mod.store_credential(
                "myrandomvendor", {"k": "v"}
            )
            db = await self.db_mod.get_db()
            cur = await db.execute(
                "SELECT schema_id FROM credentials WHERE service_name = ?",
                ("myrandomvendor",),
            )
            row = await cur.fetchone()
            self.assertIsNone(row[0])
        run_async(_t())

    def test_store_rejects_invalid_scope(self):
        async def _t():
            with self.assertRaises(self.cs_mod.CredentialSchemaError):
                await self.cs_mod.store_credential(
                    "github", {"token": "ghp_x"}, scope="megasuperadmin"
                )
        run_async(_t())


class TestDescribeSchema(unittest.TestCase):
    def setUp(self):
        from src.security import credential_schemas as cs

        self.cs = cs
        cs.reset_cache()

    def test_describe_known(self):
        out = self.cs.describe_schema("stripe")
        self.assertIn("stripe", out)
        self.assertIn("Required", out)
        self.assertIn("secret_key", out)

    def test_describe_unknown(self):
        out = self.cs.describe_schema("nope")
        self.assertIn("No schema", out)


if __name__ == "__main__":
    unittest.main()
