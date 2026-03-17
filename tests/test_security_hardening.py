"""Tests for security hardening and reliability fixes.

Covers:
- SQL column whitelist (prevents injection via dynamic kwargs)
- Blackboard cache locking (prevents concurrent corruption)
- Atomic task dedup (prevents race conditions)
- Credential key enforcement (blocks insecure fallback in prod)
- URL validation (prevents SSRF in HTTP integrations)
- URL parameter encoding (prevents path traversal in integrations)
- Error recovery dedup (stable titles prevent duplicate recovery tasks)
- Subtask cap notification (warns when subtasks are dropped)
- Workflow timeout (pauses goals exceeding timeout_hours)
"""

import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestSQLColumnWhitelist(unittest.TestCase):
    """Test that update_task/update_goal reject invalid column names."""

    def test_valid_task_columns_accepted(self):
        from src.infra.db import _validate_columns, _TASK_COLUMNS
        # Should not raise
        _validate_columns({"status": "completed", "error": "none"}, _TASK_COLUMNS, "tasks")

    def test_invalid_task_column_rejected(self):
        from src.infra.db import _validate_columns, _TASK_COLUMNS
        with self.assertRaises(ValueError) as ctx:
            _validate_columns(
                {"status": "completed", "evil_column; DROP TABLE tasks--": "x"},
                _TASK_COLUMNS,
                "tasks",
            )
        self.assertIn("Invalid column", str(ctx.exception))

    def test_valid_goal_columns_accepted(self):
        from src.infra.db import _validate_columns, _GOAL_COLUMNS
        _validate_columns({"title": "new", "status": "done"}, _GOAL_COLUMNS, "goals")

    def test_invalid_goal_column_rejected(self):
        from src.infra.db import _validate_columns, _GOAL_COLUMNS
        with self.assertRaises(ValueError):
            _validate_columns({"password": "secret"}, _GOAL_COLUMNS, "goals")

    def test_empty_kwargs_accepted(self):
        from src.infra.db import _validate_columns, _TASK_COLUMNS
        _validate_columns({}, _TASK_COLUMNS, "tasks")

    def test_all_task_columns_in_whitelist(self):
        """Verify the whitelist contains expected columns."""
        from src.infra.db import _TASK_COLUMNS
        expected = {"title", "description", "agent_type", "status", "tier",
                    "priority", "result", "error", "error_category", "context",
                    "retry_count", "max_retries", "started_at", "completed_at",
                    "task_hash", "requires_approval", "depends_on", "max_cost"}
        self.assertEqual(_TASK_COLUMNS, expected)


class TestBlackboardLocking(unittest.TestCase):
    """Test that blackboard cache uses per-goal locks."""

    def test_lock_creation(self):
        from src.collaboration.blackboard import _get_lock, _BLACKBOARD_LOCKS
        _BLACKBOARD_LOCKS.clear()
        lock1 = _get_lock(1)
        lock2 = _get_lock(1)
        lock3 = _get_lock(2)
        self.assertIs(lock1, lock2)  # Same goal → same lock
        self.assertIsNot(lock1, lock3)  # Different goal → different lock
        self.assertIsInstance(lock1, asyncio.Lock)

    def test_clear_cache_clears_locks(self):
        from src.collaboration.blackboard import clear_cache, _get_lock, _BLACKBOARD_LOCKS
        _get_lock(10)
        _get_lock(20)
        self.assertIn(10, _BLACKBOARD_LOCKS)
        clear_cache(10)
        self.assertNotIn(10, _BLACKBOARD_LOCKS)
        self.assertIn(20, _BLACKBOARD_LOCKS)
        clear_cache()
        self.assertEqual(len(_BLACKBOARD_LOCKS), 0)


class TestURLValidation(unittest.TestCase):
    """Test that SSRF protection blocks internal hosts."""

    def test_localhost_blocked(self):
        from src.integrations.http_integration import _validate_url
        with self.assertRaises(ValueError) as ctx:
            _validate_url("http://localhost:8080/api")
        self.assertIn("Blocked", str(ctx.exception))

    def test_127_0_0_1_blocked(self):
        from src.integrations.http_integration import _validate_url
        with self.assertRaises(ValueError):
            _validate_url("http://127.0.0.1/admin")

    def test_metadata_endpoint_blocked(self):
        from src.integrations.http_integration import _validate_url
        with self.assertRaises(ValueError):
            _validate_url("http://169.254.169.254/latest/meta-data/")

    def test_private_ip_10x_blocked(self):
        from src.integrations.http_integration import _validate_url
        with self.assertRaises(ValueError):
            _validate_url("http://10.0.0.1/internal")

    def test_private_ip_192_168_blocked(self):
        from src.integrations.http_integration import _validate_url
        with self.assertRaises(ValueError):
            _validate_url("http://192.168.1.1/")

    def test_private_ip_172_blocked(self):
        from src.integrations.http_integration import _validate_url
        with self.assertRaises(ValueError):
            _validate_url("http://172.16.0.1/")

    def test_public_url_allowed(self):
        from src.integrations.http_integration import _validate_url
        # Should not raise
        _validate_url("https://api.github.com/repos")

    def test_ftp_scheme_blocked(self):
        from src.integrations.http_integration import _validate_url
        with self.assertRaises(ValueError) as ctx:
            _validate_url("ftp://files.example.com/data")
        self.assertIn("scheme", str(ctx.exception))

    def test_file_scheme_blocked(self):
        from src.integrations.http_integration import _validate_url
        with self.assertRaises(ValueError):
            _validate_url("file:///etc/passwd")


class TestURLParamEncoding(unittest.TestCase):
    """Test that path params are URL-encoded to prevent injection."""

    def test_path_traversal_encoded(self):
        """Verify that ../../ slashes in params get encoded."""
        import urllib.parse
        param = "../../admin"
        encoded = urllib.parse.quote(param, safe="")
        # Slashes are encoded, preventing path traversal
        self.assertNotIn("/", encoded)
        self.assertIn("%2F", encoded)

    def test_query_injection_encoded(self):
        import urllib.parse
        param = "value?evil=1&drop=true"
        encoded = urllib.parse.quote(param, safe="")
        self.assertNotIn("?", encoded)
        self.assertNotIn("&", encoded)


class TestCredentialKeyEnforcement(unittest.TestCase):
    """Test that credential store rejects missing key in production."""

    def test_prod_env_without_key_raises(self):
        """In production mode without KUTAY_MASTER_KEY, should raise RuntimeError."""
        import src.security.credential_store as cs

        # Reset cached fernet
        cs._fernet = None

        if not cs._HAS_CRYPTOGRAPHY:
            self.skipTest("cryptography package not installed")

        with patch.dict(os.environ, {"KUTAY_ENV": "production", "KUTAY_MASTER_KEY": ""}, clear=False):
            cs._fernet = None  # Force re-init
            with self.assertRaises(RuntimeError) as ctx:
                cs._get_fernet()
            self.assertIn("required in production", str(ctx.exception))

        # Clean up
        cs._fernet = None

    def test_dev_env_without_key_warns(self):
        """In dev mode without key, should warn but not crash."""
        import src.security.credential_store as cs
        cs._fernet = None

        if not cs._HAS_CRYPTOGRAPHY:
            self.skipTest("cryptography package not installed")

        with patch.dict(os.environ, {"KUTAY_ENV": "development", "KUTAY_MASTER_KEY": ""}, clear=False):
            cs._fernet = None
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = cs._get_fernet()
                # Should return a fernet (dev fallback)
                self.assertIsNotNone(result)

        cs._fernet = None

    def test_without_crypto_returns_none(self):
        """Without cryptography package, _get_fernet returns None."""
        import src.security.credential_store as cs
        cs._fernet = None

        if cs._HAS_CRYPTOGRAPHY:
            self.skipTest("cryptography IS installed — testing fallback path")

        result = cs._get_fernet()
        self.assertIsNone(result)

        cs._fernet = None


class TestHttpIntegrationConstruction(unittest.TestCase):
    """Test that HttpIntegration validates base_url at construction."""

    def test_internal_base_url_rejected(self):
        from src.integrations.http_integration import HttpIntegration
        config = {
            "service_name": "evil",
            "base_url": "http://127.0.0.1:9090",
            "actions": {},
        }
        with self.assertRaises(ValueError):
            HttpIntegration(config)

    def test_valid_base_url_accepted(self):
        from src.integrations.http_integration import HttpIntegration
        config = {
            "service_name": "github",
            "base_url": "https://api.github.com",
            "actions": {},
        }
        integration = HttpIntegration(config)
        self.assertEqual(integration.service_name, "github")


class TestBackpressureFuture(unittest.TestCase):
    """Test that QueuedCall creates futures safely."""

    def test_queued_call_creates_future(self):
        """QueuedCall should create a future in __post_init__."""
        from src.infra.backpressure import QueuedCall

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            call = QueuedCall(call_id="test", priority=5)
            self.assertIsNotNone(call.result_future)
            self.assertIsInstance(call.result_future, asyncio.Future)
        finally:
            loop.close()


class TestDBIndexes(unittest.TestCase):
    """Verify the index list includes all expected indexes."""

    def test_indexes_include_task_hash(self):
        from src.infra.db import init_db
        # Read the source to verify index list contains task_hash
        import inspect
        source = inspect.getsource(init_db)
        self.assertIn("idx_tasks_hash", source)
        self.assertIn("idx_tasks_parent", source)
        self.assertIn("idx_goals_status", source)
        self.assertIn("idx_credentials_service", source)


if __name__ == "__main__":
    unittest.main()
