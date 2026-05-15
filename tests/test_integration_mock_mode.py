# tests/test_integration_mock_mode.py
"""Z9 T1B — analytics adapter configs + IntegrationRegistry mock mode.

Covers:
  - posthog / intercom / zendesk configs load via the registry
  - mock_mode returns deterministic responses for a posthog read action
  - KUTAI_VENDOR_LIVE=1 flips mock off in a non-prod env
  - existing providers (github, stripe, sentry) still resolve — no regression
"""

import asyncio
import os
import sys
import unittest
from unittest.mock import patch

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_async(coro):
    """Run an async coroutine synchronously for tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# New analytics / support configs load
# ---------------------------------------------------------------------------

class TestAnalyticsConfigsLoad(unittest.TestCase):
    """The three new Z9 vendor configs auto-discover without error."""

    def test_new_configs_discovered(self):
        from src.integrations.registry import IntegrationRegistry

        reg = IntegrationRegistry(auto_discover=True, mock_mode=True)
        services = reg.list_services()
        for svc in ("posthog", "intercom", "zendesk"):
            self.assertIn(svc, services, f"{svc} not auto-discovered")

    def test_posthog_capabilities(self):
        from src.integrations.http_integration import HttpIntegration

        integration = HttpIntegration.from_service_name("posthog")
        self.assertEqual(integration.service_name, "posthog")
        self.assertIn("query_events", integration.capabilities())
        self.assertIn("query_funnel", integration.capabilities())

    def test_intercom_capabilities(self):
        from src.integrations.http_integration import HttpIntegration

        integration = HttpIntegration.from_service_name("intercom")
        self.assertEqual(integration.service_name, "intercom")
        self.assertIn("list_conversations", integration.capabilities())

    def test_zendesk_capabilities(self):
        from src.integrations.http_integration import HttpIntegration

        integration = HttpIntegration.from_service_name("zendesk")
        self.assertEqual(integration.service_name, "zendesk")
        self.assertIn("list_tickets", integration.capabilities())

    def test_existing_providers_still_resolve(self):
        """No regression — github / stripe / sentry still discovered."""
        from src.integrations.registry import IntegrationRegistry

        reg = IntegrationRegistry(auto_discover=True, mock_mode=False)
        services = reg.list_services()
        for svc in ("github", "stripe", "sentry"):
            self.assertIn(svc, services, f"{svc} regressed out of registry")
            self.assertIsNotNone(reg.get(svc))


# ---------------------------------------------------------------------------
# Mock mode default resolution
# ---------------------------------------------------------------------------

class TestMockModeDefault(unittest.TestCase):
    """KUTAI_ENV / KUTAI_VENDOR_LIVE drive the default mock_mode value."""

    def _resolve(self, env: dict):
        from src.integrations.registry import _resolve_mock_default

        with patch.dict(os.environ, env, clear=False):
            # Drop any inherited values not in `env` so the test is hermetic.
            for var in ("KUTAI_ENV", "KUTAI_VENDOR_LIVE"):
                if var not in env:
                    os.environ.pop(var, None)
            return _resolve_mock_default()

    def test_non_prod_defaults_to_mock(self):
        self.assertTrue(self._resolve({"KUTAI_ENV": "development"}))

    def test_unset_env_defaults_to_mock(self):
        self.assertTrue(self._resolve({}))

    def test_prod_defaults_to_live(self):
        self.assertFalse(self._resolve({"KUTAI_ENV": "prod"}))

    def test_vendor_live_forces_real_in_non_prod(self):
        """KUTAI_VENDOR_LIVE=1 flips mock OFF even in a non-prod env."""
        resolved = self._resolve(
            {"KUTAI_ENV": "development", "KUTAI_VENDOR_LIVE": "1"}
        )
        self.assertFalse(resolved)

    def test_registry_picks_up_env_default(self):
        from src.integrations.registry import IntegrationRegistry

        with patch.dict(os.environ, {"KUTAI_ENV": "staging"}, clear=False):
            os.environ.pop("KUTAI_VENDOR_LIVE", None)
            reg = IntegrationRegistry(auto_discover=False)
            self.assertTrue(reg.mock_mode)

    def test_registry_vendor_live_env(self):
        from src.integrations.registry import IntegrationRegistry

        with patch.dict(
            os.environ,
            {"KUTAI_ENV": "development", "KUTAI_VENDOR_LIVE": "1"},
            clear=False,
        ):
            reg = IntegrationRegistry(auto_discover=False)
            self.assertFalse(reg.mock_mode)


# ---------------------------------------------------------------------------
# Mock mode returns deterministic responses
# ---------------------------------------------------------------------------

class TestMockResponses(unittest.TestCase):
    """mock_mode serves deterministic payloads instead of hitting network."""

    def test_registry_mock_response_shape(self):
        from src.integrations.registry import IntegrationRegistry

        reg = IntegrationRegistry(auto_discover=True, mock_mode=True)
        resp = reg.mock_response("posthog", "query_events")
        self.assertIsNotNone(resp)
        self.assertEqual(resp["status"], "ok")
        self.assertTrue(resp["mocked"])
        self.assertIn("results", resp["data"])
        self.assertTrue(len(resp["data"]["results"]) > 0)

    def test_mock_response_is_deterministic(self):
        """Two calls return equal payloads (deterministic)."""
        from src.integrations.registry import IntegrationRegistry

        reg = IntegrationRegistry(auto_discover=True, mock_mode=True)
        a = reg.mock_response("posthog", "query_events")
        b = reg.mock_response("posthog", "query_events")
        self.assertEqual(a, b)

    def test_mock_response_is_a_copy(self):
        """Mutating one mock payload must not corrupt the next call."""
        from src.integrations.registry import IntegrationRegistry

        reg = IntegrationRegistry(auto_discover=True, mock_mode=True)
        a = reg.mock_response("posthog", "query_events")
        a["data"]["results"].clear()
        b = reg.mock_response("posthog", "query_events")
        self.assertTrue(len(b["data"]["results"]) > 0)

    def test_mock_response_unknown_action(self):
        from src.integrations.registry import IntegrationRegistry

        reg = IntegrationRegistry(auto_discover=True, mock_mode=True)
        self.assertIsNone(reg.mock_response("posthog", "no_such_action"))

    def test_execute_returns_mock_when_mock_mode_on(self):
        """HttpIntegration.execute short-circuits to the mock payload.

        No credential is stored — a real call would fail with 'No
        credentials'. A mock result proves the network/vault was skipped.
        """
        from src.integrations.registry import (
            IntegrationRegistry,
            get_integration_registry,
        )
        import src.integrations.registry as reg_mod

        # Install a mock-on registry as the global singleton.
        orig = reg_mod._registry
        reg_mod._registry = IntegrationRegistry(auto_discover=True, mock_mode=True)
        try:
            adapter = get_integration_registry().get("posthog")
            self.assertIsNotNone(adapter)
            result = run_async(
                adapter.execute("query_events", {"project_id": "123"})
            )
            self.assertEqual(result["status"], "ok")
            self.assertTrue(result.get("mocked"))
            self.assertIn("results", result["data"])
        finally:
            reg_mod._registry = orig

    def test_execute_skips_mock_when_mock_mode_off(self):
        """With mock_mode off, execute does NOT return a mocked payload.

        We don't make a real network call — instead we assert the result is
        the real-path error (no credentials stored), proving the mock
        short-circuit was bypassed.
        """
        from src.integrations.registry import IntegrationRegistry
        import src.integrations.registry as reg_mod

        orig = reg_mod._registry
        reg_mod._registry = IntegrationRegistry(
            auto_discover=True, mock_mode=False
        )
        try:
            adapter = reg_mod._registry.get("posthog")

            async def _no_cred(_service):
                return None

            with patch(
                "src.security.credential_store.get_credential", _no_cred
            ):
                result = run_async(
                    adapter.execute("query_events", {"project_id": "123"})
                )
            self.assertEqual(result["status"], "error")
            self.assertFalse(result.get("mocked", False))
            self.assertIn("No credentials", result["error"])
        finally:
            reg_mod._registry = orig

    def test_intercom_and_zendesk_mock(self):
        from src.integrations.registry import IntegrationRegistry

        reg = IntegrationRegistry(auto_discover=True, mock_mode=True)
        ic = reg.mock_response("intercom", "list_conversations")
        self.assertIsNotNone(ic)
        self.assertIn("conversations", ic["data"])

        zd = reg.mock_response("zendesk", "list_tickets")
        self.assertIsNotNone(zd)
        self.assertIn("tickets", zd["data"])


if __name__ == "__main__":
    unittest.main()
