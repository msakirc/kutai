# tests/test_credential_store.py
"""
Tests for the encrypted credential store (Gap 5).
  - Encrypt/decrypt roundtrip
  - Store and retrieve credentials
  - Delete credentials
  - List credentials
  - Missing master key fallback
"""

import asyncio
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


class _CredDBTestBase(unittest.TestCase):
    """Shared setUp/tearDown for credential store DB tests."""

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

        # Patch DB_PATH and reset singleton
        config.DB_PATH = self.db_path
        db_mod.DB_PATH = self.db_path
        db_mod._db_connection = None

        # Reset fernet singleton so each test starts fresh
        import src.security.credential_store as cs_mod
        self.cs_mod = cs_mod
        cs_mod._fernet = None
        cs_mod._MASTER_KEY = None

        run_async(db_mod.init_db())

    def tearDown(self):
        run_async(self.db_mod.close_db())
        from src.app import config
        config.DB_PATH = self._orig_config_path
        self.db_mod.DB_PATH = self._orig_db_path

        # Reset fernet singleton
        self.cs_mod._fernet = None
        self.cs_mod._MASTER_KEY = None

        try:
            os.unlink(self.db_path)
        except OSError:
            pass
        for suffix in ("-wal", "-shm"):
            try:
                os.unlink(self.db_path + suffix)
            except OSError:
                pass


class TestEncryptDecryptRoundtrip(unittest.TestCase):
    """Test encryption/decryption without DB."""

    def setUp(self):
        import src.security.credential_store as cs_mod
        self.cs_mod = cs_mod
        cs_mod._fernet = None
        cs_mod._MASTER_KEY = None
        # Set a test master key
        self._orig_key = os.environ.get("KUTAY_MASTER_KEY")
        os.environ["KUTAY_MASTER_KEY"] = "test-secret-key-for-unit-tests"

    def tearDown(self):
        self.cs_mod._fernet = None
        self.cs_mod._MASTER_KEY = None
        if self._orig_key is not None:
            os.environ["KUTAY_MASTER_KEY"] = self._orig_key
        else:
            os.environ.pop("KUTAY_MASTER_KEY", None)

    def test_encrypt_decrypt_roundtrip(self):
        """Encrypting and decrypting should return the original text."""
        original = '{"token": "ghp_abc123", "org": "myorg"}'
        encrypted = self.cs_mod._encrypt(original)

        # Encrypted should differ from original
        self.assertNotEqual(encrypted, original)

        # Decrypted should match original
        decrypted = self.cs_mod._decrypt(encrypted)
        self.assertEqual(decrypted, original)

    def test_encrypt_empty_string(self):
        """Empty string should roundtrip correctly."""
        encrypted = self.cs_mod._encrypt("")
        decrypted = self.cs_mod._decrypt(encrypted)
        self.assertEqual(decrypted, "")

    def test_encrypt_unicode(self):
        """Unicode content should roundtrip correctly."""
        original = '{"key": "value with unicode chars"}'
        encrypted = self.cs_mod._encrypt(original)
        decrypted = self.cs_mod._decrypt(encrypted)
        self.assertEqual(decrypted, original)


class TestCredentialStoreFallback(unittest.TestCase):
    """Test fallback when KUTAY_MASTER_KEY is not set."""

    def setUp(self):
        import src.security.credential_store as cs_mod
        self.cs_mod = cs_mod
        cs_mod._fernet = None
        cs_mod._MASTER_KEY = None
        self._orig_key = os.environ.get("KUTAY_MASTER_KEY")
        # Remove master key to test fallback
        os.environ.pop("KUTAY_MASTER_KEY", None)

    def tearDown(self):
        self.cs_mod._fernet = None
        self.cs_mod._MASTER_KEY = None
        if self._orig_key is not None:
            os.environ["KUTAY_MASTER_KEY"] = self._orig_key

    def test_fallback_still_works(self):
        """Without master key, encrypt/decrypt should still work (with warning)."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            original = '{"token": "test123"}'
            encrypted = self.cs_mod._encrypt(original)
            decrypted = self.cs_mod._decrypt(encrypted)
            self.assertEqual(decrypted, original)

            # Check that a warning was issued (cryptography may or may not be present)
            # If cryptography is installed, we expect a fallback warning
            if self.cs_mod._HAS_CRYPTOGRAPHY:
                self.assertTrue(
                    any("KUTAY_MASTER_KEY" in str(warning.message) for warning in w),
                    "Expected a warning about missing KUTAY_MASTER_KEY"
                )


class TestStoreAndRetrieve(_CredDBTestBase):
    """Test storing and retrieving credentials from DB."""

    def test_store_and_get(self):
        """Store a credential and retrieve it."""
        from src.security.credential_store import store_credential, get_credential

        async def _test():
            data = {"token": "ghp_abc123", "org": "myorg"}
            await store_credential("github", data)

            result = await get_credential("github")
            self.assertIsNotNone(result)
            self.assertEqual(result["token"], "ghp_abc123")
            self.assertEqual(result["org"], "myorg")

        run_async(_test())

    def test_get_nonexistent(self):
        """Getting a nonexistent credential returns None."""
        from src.security.credential_store import get_credential

        async def _test():
            result = await get_credential("nonexistent")
            self.assertIsNone(result)

        run_async(_test())

    def test_upsert(self):
        """Storing twice for same service should update."""
        from src.security.credential_store import store_credential, get_credential

        async def _test():
            await store_credential("github", {"token": "old_token"})
            await store_credential("github", {"token": "new_token"})

            result = await get_credential("github")
            self.assertEqual(result["token"], "new_token")

        run_async(_test())


class TestDeleteCredential(_CredDBTestBase):
    """Test credential deletion."""

    def test_delete_existing(self):
        """Deleting an existing credential returns True."""
        from src.security.credential_store import (
            store_credential, delete_credential, get_credential
        )

        async def _test():
            await store_credential("github", {"token": "abc"})
            result = await delete_credential("github")
            self.assertTrue(result)

            # Verify it's gone
            cred = await get_credential("github")
            self.assertIsNone(cred)

        run_async(_test())

    def test_delete_nonexistent(self):
        """Deleting a nonexistent credential returns False."""
        from src.security.credential_store import delete_credential

        async def _test():
            result = await delete_credential("nonexistent")
            self.assertFalse(result)

        run_async(_test())


class TestListCredentials(_CredDBTestBase):
    """Test listing credential service names."""

    def test_list_empty(self):
        """Empty DB returns empty list."""
        from src.security.credential_store import list_credentials

        async def _test():
            result = await list_credentials()
            self.assertEqual(result, [])

        run_async(_test())

    def test_list_multiple(self):
        """List returns all service names sorted."""
        from src.security.credential_store import store_credential, list_credentials

        async def _test():
            await store_credential("vercel", {"token": "v1"})
            await store_credential("github", {"token": "g1"})
            await store_credential("railway", {"token": "r1"})

            result = await list_credentials()
            self.assertEqual(result, ["github", "railway", "vercel"])

        run_async(_test())


if __name__ == "__main__":
    unittest.main()
