# tests/test_integrations.py
"""
Tests for the external service integration layer (Gap 5).
  - BaseIntegration ABC
  - IntegrationRegistry register/get/list
  - HttpIntegration config loading
  - HttpIntegration execute with mocked HTTP
  - service_call tool function
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch, MagicMock

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


# ---------------------------------------------------------------------------
# Test BaseIntegration ABC
# ---------------------------------------------------------------------------

class TestBaseIntegration(unittest.TestCase):
    """Test that BaseIntegration is a proper ABC."""

    def test_cannot_instantiate_abc(self):
        """BaseIntegration cannot be instantiated directly."""
        from src.integrations.base import BaseIntegration

        with self.assertRaises(TypeError):
            BaseIntegration()

    def test_concrete_subclass(self):
        """A proper subclass can be instantiated."""
        from src.integrations.base import BaseIntegration

        class MyIntegration(BaseIntegration):
            service_name = "test"

            async def validate(self) -> bool:
                return True

            def capabilities(self) -> list[str]:
                return ["action_a"]

            async def execute(self, action: str, params: dict) -> dict:
                return {"status": "ok"}

        inst = MyIntegration()
        self.assertEqual(inst.service_name, "test")
        self.assertEqual(inst.capabilities(), ["action_a"])
        self.assertTrue(run_async(inst.validate()))
        self.assertEqual(
            run_async(inst.execute("action_a", {})),
            {"status": "ok"},
        )

    def test_incomplete_subclass_fails(self):
        """A subclass missing abstract methods cannot be instantiated."""
        from src.integrations.base import BaseIntegration

        class IncompleteIntegration(BaseIntegration):
            service_name = "incomplete"

            async def validate(self) -> bool:
                return True
            # Missing capabilities() and execute()

        with self.assertRaises(TypeError):
            IncompleteIntegration()


# ---------------------------------------------------------------------------
# Test IntegrationRegistry
# ---------------------------------------------------------------------------

class TestIntegrationRegistry(unittest.TestCase):
    """Test registry register/get/list."""

    def _make_integration(self, name: str):
        from src.integrations.base import BaseIntegration

        class DummyIntegration(BaseIntegration):
            service_name = name

            async def validate(self) -> bool:
                return True

            def capabilities(self) -> list[str]:
                return ["do_thing"]

            async def execute(self, action: str, params: dict) -> dict:
                return {"status": "ok"}

        return DummyIntegration()

    def test_register_and_get(self):
        """Register an integration and retrieve it."""
        from src.integrations.registry import IntegrationRegistry

        reg = IntegrationRegistry(auto_discover=False)
        dummy = self._make_integration("test_svc")
        reg.register(dummy)

        result = reg.get("test_svc")
        self.assertIs(result, dummy)

    def test_get_nonexistent(self):
        """Getting a non-registered service returns None."""
        from src.integrations.registry import IntegrationRegistry

        reg = IntegrationRegistry(auto_discover=False)
        self.assertIsNone(reg.get("nonexistent"))

    def test_list_services(self):
        """list_services returns sorted names."""
        from src.integrations.registry import IntegrationRegistry

        reg = IntegrationRegistry(auto_discover=False)
        reg.register(self._make_integration("zebra"))
        reg.register(self._make_integration("alpha"))
        reg.register(self._make_integration("middle"))

        self.assertEqual(reg.list_services(), ["alpha", "middle", "zebra"])

    def test_register_empty_name_fails(self):
        """Registering with empty service_name raises ValueError."""
        from src.integrations.registry import IntegrationRegistry
        from src.integrations.base import BaseIntegration

        class NoName(BaseIntegration):
            service_name = ""

            async def validate(self) -> bool:
                return True

            def capabilities(self) -> list[str]:
                return []

            async def execute(self, action: str, params: dict) -> dict:
                return {}

        reg = IntegrationRegistry(auto_discover=False)
        with self.assertRaises(ValueError):
            reg.register(NoName())

    def test_auto_discover_loads_configs(self):
        """Auto-discovery should find the built-in JSON configs."""
        from src.integrations.registry import IntegrationRegistry

        reg = IntegrationRegistry(auto_discover=True)
        services = reg.list_services()

        # We created github.json, vercel.json, railway.json
        self.assertIn("github", services)
        self.assertIn("vercel", services)
        self.assertIn("railway", services)


# ---------------------------------------------------------------------------
# Test HttpIntegration
# ---------------------------------------------------------------------------

class TestHttpIntegrationConfig(unittest.TestCase):
    """Test HttpIntegration config loading."""

    def test_load_from_dict(self):
        """Create HttpIntegration from a config dict."""
        from src.integrations.http_integration import HttpIntegration

        config = {
            "service_name": "testapi",
            "base_url": "https://api.test.com",
            "auth_type": "bearer",
            "auth_header": "Authorization",
            "actions": {
                "get_items": {
                    "method": "GET",
                    "path": "/items",
                    "required_params": [],
                },
                "create_item": {
                    "method": "POST",
                    "path": "/items",
                    "required_params": ["name"],
                },
            },
        }

        integration = HttpIntegration(config)
        self.assertEqual(integration.service_name, "testapi")
        self.assertEqual(integration.capabilities(), ["create_item", "get_items"])

    def test_load_from_service_name(self):
        """Load a built-in config by service name."""
        from src.integrations.http_integration import HttpIntegration

        integration = HttpIntegration.from_service_name("github")
        self.assertEqual(integration.service_name, "github")
        self.assertIn("list_repos", integration.capabilities())

    def test_load_nonexistent_service(self):
        """Loading a non-existent service raises FileNotFoundError."""
        from src.integrations.http_integration import HttpIntegration

        with self.assertRaises(FileNotFoundError):
            HttpIntegration.from_service_name("nonexistent_service_xyz")

    def test_unknown_action(self):
        """Executing an unknown action returns error."""
        from src.integrations.http_integration import HttpIntegration

        config = {
            "service_name": "testapi",
            "base_url": "https://api.test.com",
            "actions": {},
        }
        integration = HttpIntegration(config)

        result = run_async(integration.execute("nonexistent", {}))
        self.assertEqual(result["status"], "error")
        self.assertIn("Unknown action", result["error"])

    def test_missing_required_params(self):
        """Missing required params returns error."""
        from src.integrations.http_integration import HttpIntegration

        config = {
            "service_name": "testapi",
            "base_url": "https://api.test.com",
            "actions": {
                "create": {
                    "method": "POST",
                    "path": "/items",
                    "required_params": ["name", "type"],
                }
            },
        }
        integration = HttpIntegration(config)

        result = run_async(integration.execute("create", {"name": "foo"}))
        self.assertEqual(result["status"], "error")
        self.assertIn("Missing required params", result["error"])
        self.assertIn("type", result["error"])


class TestHttpIntegrationExecute(unittest.TestCase):
    """Test HttpIntegration.execute with mocked HTTP calls."""

    def _make_integration(self):
        from src.integrations.http_integration import HttpIntegration

        config = {
            "service_name": "mockapi",
            "base_url": "https://api.mock.com",
            "auth_type": "bearer",
            "auth_header": "Authorization",
            "actions": {
                "list_items": {
                    "method": "GET",
                    "path": "/v1/items",
                    "required_params": [],
                },
                "get_item": {
                    "method": "GET",
                    "path": "/v1/items/{item_id}",
                    "required_params": ["item_id"],
                },
                "create_item": {
                    "method": "POST",
                    "path": "/v1/items",
                    "required_params": ["name"],
                },
            },
        }
        return HttpIntegration(config)

    @patch("src.security.credential_store.get_credential")
    def test_execute_get_success(self, mock_get_cred):
        """Successful GET request."""
        mock_get_cred.return_value = {"token": "test_token_123"}
        integration = self._make_integration()

        mock_response = {
            "status_code": 200,
            "body": json.dumps({"items": [{"id": 1, "name": "test"}]}),
            "headers": {"content-type": "application/json"},
        }

        async def _test():
            with patch(
                "src.integrations.http_integration._get_http_func"
            ) as mock_http:
                mock_func = AsyncMock(return_value=mock_response)
                mock_http.return_value = mock_func

                result = await integration.execute("list_items", {})

                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["status_code"], 200)
                self.assertEqual(result["data"]["items"][0]["name"], "test")

                # Verify the HTTP call
                mock_func.assert_called_once()
                call_args = mock_func.call_args
                self.assertEqual(call_args[0][0], "GET")
                self.assertIn("api.mock.com/v1/items", call_args[0][1])
                # Check auth header
                headers = call_args[0][2]
                self.assertEqual(headers["Authorization"], "Bearer test_token_123")

        run_async(_test())

    @patch("src.security.credential_store.get_credential")
    def test_execute_post_with_body(self, mock_get_cred):
        """POST request sends body params."""
        mock_get_cred.return_value = {"token": "tok"}
        integration = self._make_integration()

        mock_response = {
            "status_code": 201,
            "body": json.dumps({"id": 42, "name": "new_item"}),
            "headers": {},
        }

        async def _test():
            with patch(
                "src.integrations.http_integration._get_http_func"
            ) as mock_http:
                mock_func = AsyncMock(return_value=mock_response)
                mock_http.return_value = mock_func

                result = await integration.execute(
                    "create_item", {"name": "new_item", "description": "hello"}
                )

                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["data"]["id"], 42)

                # Verify body was sent
                call_args = mock_func.call_args
                body = call_args[1].get("json_body") or call_args[0][3]
                self.assertEqual(body["name"], "new_item")

        run_async(_test())

    @patch("src.security.credential_store.get_credential")
    def test_execute_path_params(self, mock_get_cred):
        """Path parameters are substituted correctly."""
        mock_get_cred.return_value = {"token": "tok"}
        integration = self._make_integration()

        mock_response = {
            "status_code": 200,
            "body": json.dumps({"id": 99}),
            "headers": {},
        }

        async def _test():
            with patch(
                "src.integrations.http_integration._get_http_func"
            ) as mock_http:
                mock_func = AsyncMock(return_value=mock_response)
                mock_http.return_value = mock_func

                result = await integration.execute(
                    "get_item", {"item_id": "99"}
                )

                self.assertEqual(result["status"], "ok")

                # Verify URL contains substituted path
                call_args = mock_func.call_args
                url = call_args[0][1]
                self.assertIn("/v1/items/99", url)
                self.assertNotIn("{item_id}", url)

        run_async(_test())

    @patch("src.security.credential_store.get_credential")
    def test_execute_no_credentials(self, mock_get_cred):
        """Missing credentials returns error."""
        mock_get_cred.return_value = None
        integration = self._make_integration()

        async def _test():
            result = await integration.execute("list_items", {})
            self.assertEqual(result["status"], "error")
            self.assertIn("No credentials stored", result["error"])

        run_async(_test())

    @patch("src.security.credential_store.get_credential")
    def test_execute_http_error(self, mock_get_cred):
        """HTTP 4xx/5xx returns error status."""
        mock_get_cred.return_value = {"token": "tok"}
        integration = self._make_integration()

        mock_response = {
            "status_code": 403,
            "body": json.dumps({"message": "Forbidden"}),
            "headers": {},
        }

        async def _test():
            with patch(
                "src.integrations.http_integration._get_http_func"
            ) as mock_http:
                mock_func = AsyncMock(return_value=mock_response)
                mock_http.return_value = mock_func

                result = await integration.execute("list_items", {})

                self.assertEqual(result["status"], "error")
                self.assertIn("403", result["error"])

        run_async(_test())


# ---------------------------------------------------------------------------
# Test service_call tool
# ---------------------------------------------------------------------------

class _ToolDBTestBase(unittest.TestCase):
    """DB base for tool tests."""

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


class TestServiceCallTool(_ToolDBTestBase):
    """Test the service_call tool function."""

    def test_unknown_service(self):
        """Calling an unknown service returns helpful error."""
        from src.tools import TOOL_REGISTRY

        # service_call should be in registry
        self.assertIn("service_call", TOOL_REGISTRY)

        func = TOOL_REGISTRY["service_call"]["function"]

        async def _test():
            result = await func(
                service="nonexistent_xyz", action="do_thing"
            )
            self.assertIn("Unknown service", result)

        run_async(_test())

    @patch("src.security.credential_store.get_credential")
    def test_service_call_with_mock(self, mock_get_cred):
        """service_call tool invokes integration correctly."""
        mock_get_cred.return_value = {"token": "test_token"}

        from src.tools import TOOL_REGISTRY
        func = TOOL_REGISTRY["service_call"]["function"]

        mock_response = {
            "status_code": 200,
            "body": json.dumps({"repos": []}),
            "headers": {},
        }

        async def _test():
            with patch(
                "src.integrations.http_integration._get_http_func"
            ) as mock_http:
                mock_func = AsyncMock(return_value=mock_response)
                mock_http.return_value = mock_func

                result = await func(
                    service="github",
                    action="list_repos",
                    params="{}",
                )

                parsed = json.loads(result)
                self.assertEqual(parsed["status"], "ok")

        run_async(_test())


if __name__ == "__main__":
    unittest.main()
