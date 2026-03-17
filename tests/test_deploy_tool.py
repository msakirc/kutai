"""Tests for the deployment tool (Gap 6)."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.tools.deploy import deploy, _health_check, _extract_url, _check_quality_gate
from src.workflows.engine.artifacts import ArtifactStore


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_mock_registry(integration=None):
    """Build a mock IntegrationRegistry that returns the given integration."""
    mock_registry = MagicMock()
    mock_registry.get.return_value = integration
    return mock_registry


def _ok_health():
    return {"healthy": True, "status_code": 200, "attempts": 1, "error": None}


# Patch targets — deploy.py uses lazy imports, so we patch where they're imported from
_PATCH_GET_CRED = "src.security.credential_store.get_credential"
_PATCH_REGISTRY = "src.integrations.registry.get_integration_registry"
_PATCH_HEALTH = "src.tools.deploy._health_check"


# ---------------------------------------------------------------------------
# Pre-deploy validation
# ---------------------------------------------------------------------------

class TestPreDeployValidation(unittest.TestCase):
    """Tests for credential and quality gate checks before deploying."""

    @patch(_PATCH_GET_CRED, new_callable=AsyncMock, return_value=None)
    def test_missing_credentials_returns_error(self, mock_cred):
        result = run_async(deploy("vercel", "/app"))
        self.assertEqual(result["status"], "error")
        self.assertIn("No credentials", result["error"])

    def test_unsupported_target_returns_error(self):
        result = run_async(deploy("heroku", "/app"))
        self.assertEqual(result["status"], "error")
        self.assertIn("Unsupported target", result["error"])

    @patch(_PATCH_HEALTH, new_callable=AsyncMock, return_value=_ok_health())
    @patch(_PATCH_GET_CRED, new_callable=AsyncMock, return_value={"token": "tok"})
    def test_quality_gate_checked_when_goal_id_provided(self, mock_cred, mock_health):
        """When goal_id is given and gate artifact is missing, deploy should fail."""
        store = ArtifactStore(use_db=False)

        with patch("src.workflows.engine.artifacts.ArtifactStore", return_value=store):
            # No phase_13_gate_result artifact stored -> gate fails
            result = run_async(deploy("vercel", "/app", goal_id=1))
            self.assertEqual(result["status"], "error")
            self.assertIn("quality gate", result["error"].lower())

    @patch(_PATCH_HEALTH, new_callable=AsyncMock, return_value=_ok_health())
    @patch(_PATCH_GET_CRED, new_callable=AsyncMock, return_value={"token": "tok"})
    def test_quality_gate_passes_with_artifact(self, mock_cred, mock_health):
        """When gate artifact says PASSED, deployment proceeds."""
        store = ArtifactStore(use_db=False)
        run_async(store.store(1, "phase_13_gate_result", "PASSED"))

        mock_integration = MagicMock()
        mock_integration.execute = AsyncMock(return_value={
            "status": "ok",
            "data": {"url": "https://my-app.vercel.app"},
        })

        mock_registry = _make_mock_registry(mock_integration)

        with patch("src.workflows.engine.artifacts.ArtifactStore", return_value=store), \
             patch(_PATCH_REGISTRY, return_value=mock_registry):
            result = run_async(deploy("vercel", "/app", goal_id=1))
            self.assertEqual(result["status"], "ok")


# ---------------------------------------------------------------------------
# Quality gate helper
# ---------------------------------------------------------------------------

class TestCheckQualityGate(unittest.TestCase):

    def test_missing_artifact(self):
        store = ArtifactStore(use_db=False)
        passed, msg = run_async(_check_quality_gate(1, store))
        self.assertFalse(passed)
        self.assertIn("not found", msg)

    def test_passed_plain_text(self):
        store = ArtifactStore(use_db=False)
        run_async(store.store(1, "phase_13_gate_result", "PASSED"))
        passed, msg = run_async(_check_quality_gate(1, store))
        self.assertTrue(passed)

    def test_passed_json(self):
        store = ArtifactStore(use_db=False)
        run_async(store.store(1, "phase_13_gate_result", json.dumps({"passed": True})))
        passed, msg = run_async(_check_quality_gate(1, store))
        self.assertTrue(passed)

    def test_failed_json(self):
        store = ArtifactStore(use_db=False)
        run_async(store.store(1, "phase_13_gate_result", json.dumps({"passed": False})))
        passed, msg = run_async(_check_quality_gate(1, store))
        self.assertFalse(passed)


# ---------------------------------------------------------------------------
# Deployment execution (mocked integration)
# ---------------------------------------------------------------------------

class TestVercelDeployment(unittest.TestCase):

    @patch(_PATCH_HEALTH, new_callable=AsyncMock, return_value=_ok_health())
    @patch(_PATCH_GET_CRED, new_callable=AsyncMock, return_value={"token": "tok"})
    def test_successful_vercel_deploy(self, mock_cred, mock_health):
        mock_integration = MagicMock()
        mock_integration.execute = AsyncMock(return_value={
            "status": "ok",
            "data": {"url": "my-app-abc.vercel.app"},
        })

        mock_registry = _make_mock_registry(mock_integration)

        with patch(_PATCH_REGISTRY, return_value=mock_registry):
            result = run_async(deploy("vercel", "/my-app"))
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["url"], "https://my-app-abc.vercel.app")
            self.assertTrue(result["verification"]["healthy"])

            # Check integration was called with correct params
            call_args = mock_integration.execute.call_args
            self.assertEqual(call_args[0][0], "deploy")
            params = call_args[0][1]
            self.assertEqual(params["name"], "my-app")
            self.assertIn("gitSource", params)

    @patch(_PATCH_HEALTH, new_callable=AsyncMock, return_value=_ok_health())
    @patch(_PATCH_GET_CRED, new_callable=AsyncMock, return_value={"token": "tok"})
    def test_vercel_deploy_with_env_vars(self, mock_cred, mock_health):
        mock_integration = MagicMock()
        mock_integration.execute = AsyncMock(return_value={
            "status": "ok",
            "data": {"url": "https://app.vercel.app"},
        })

        mock_registry = _make_mock_registry(mock_integration)

        with patch(_PATCH_REGISTRY, return_value=mock_registry):
            result = run_async(deploy("vercel", "/app", env_vars={"API_KEY": "secret"}))
            self.assertEqual(result["status"], "ok")
            call_params = mock_integration.execute.call_args[0][1]
            self.assertEqual(call_params["env"], {"API_KEY": "secret"})


class TestRailwayDeployment(unittest.TestCase):

    @patch(_PATCH_HEALTH, new_callable=AsyncMock, return_value=_ok_health())
    @patch(_PATCH_GET_CRED, new_callable=AsyncMock, return_value={"token": "tok"})
    def test_successful_railway_deploy(self, mock_cred, mock_health):
        mock_integration = MagicMock()
        mock_integration.execute = AsyncMock(return_value={
            "status": "ok",
            "data": {"url": "my-app.up.railway.app"},
        })

        mock_registry = _make_mock_registry(mock_integration)

        with patch(_PATCH_REGISTRY, return_value=mock_registry):
            result = run_async(deploy("railway", "/my-app"))
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["url"], "https://my-app.up.railway.app")

            call_params = mock_integration.execute.call_args[0][1]
            self.assertIn("query", call_params)


# ---------------------------------------------------------------------------
# Post-deploy health check
# ---------------------------------------------------------------------------

class TestHealthCheck(unittest.TestCase):

    @patch("src.tools.deploy.asyncio.sleep", new_callable=AsyncMock)
    def test_health_check_success(self, mock_sleep):
        """Health check returns healthy on 200 response."""
        try:
            import httpx  # noqa: F401
            with patch("httpx.AsyncClient") as MockClient:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                instance = MagicMock()
                instance.get = AsyncMock(return_value=mock_resp)
                instance.__aenter__ = AsyncMock(return_value=instance)
                instance.__aexit__ = AsyncMock(return_value=False)
                MockClient.return_value = instance
                result = run_async(_health_check("https://example.com"))
                self.assertTrue(result["healthy"])
                self.assertEqual(result["status_code"], 200)
        except ImportError:
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.status = 200
                mock_response.__enter__ = MagicMock(return_value=mock_response)
                mock_response.__exit__ = MagicMock(return_value=False)
                mock_urlopen.return_value = mock_response
                result = run_async(_health_check("https://example.com"))
                self.assertTrue(result["healthy"])

    @patch("src.tools.deploy.asyncio.sleep", new_callable=AsyncMock)
    def test_health_check_failure_after_retries(self, mock_sleep):
        """Health check returns unhealthy after all retries fail."""
        try:
            import httpx  # noqa: F401
            with patch("httpx.AsyncClient") as MockClient:
                mock_resp = MagicMock()
                mock_resp.status_code = 500
                instance = MagicMock()
                instance.get = AsyncMock(return_value=mock_resp)
                instance.__aenter__ = AsyncMock(return_value=instance)
                instance.__aexit__ = AsyncMock(return_value=False)
                MockClient.return_value = instance
                result = run_async(_health_check("https://example.com"))
                self.assertFalse(result["healthy"])
                self.assertEqual(result["attempts"], 3)
                self.assertIn("500", result["error"])
        except ImportError:
            with patch("urllib.request.urlopen") as mock_urlopen:
                import urllib.error
                mock_urlopen.side_effect = urllib.error.HTTPError(
                    "https://example.com", 500, "Server Error", {}, None
                )
                result = run_async(_health_check("https://example.com"))
                self.assertFalse(result["healthy"])

    @patch("src.tools.deploy.asyncio.sleep", new_callable=AsyncMock)
    def test_health_check_connection_error(self, mock_sleep):
        """Health check handles connection errors gracefully."""
        try:
            import httpx  # noqa: F401
            with patch("httpx.AsyncClient") as MockClient:
                instance = MagicMock()
                instance.get = AsyncMock(side_effect=ConnectionError("refused"))
                instance.__aenter__ = AsyncMock(return_value=instance)
                instance.__aexit__ = AsyncMock(return_value=False)
                MockClient.return_value = instance
                result = run_async(_health_check("https://example.com"))
                self.assertFalse(result["healthy"])
                self.assertIsNotNone(result["error"])
        except ImportError:
            with patch("urllib.request.urlopen", side_effect=ConnectionError("refused")):
                result = run_async(_health_check("https://example.com"))
                self.assertFalse(result["healthy"])


# ---------------------------------------------------------------------------
# URL extraction
# ---------------------------------------------------------------------------

class TestExtractUrl(unittest.TestCase):

    def test_vercel_url(self):
        resp = {"status": "ok", "data": {"url": "my-app.vercel.app"}}
        self.assertEqual(_extract_url("vercel", resp), "https://my-app.vercel.app")

    def test_vercel_url_with_https(self):
        resp = {"status": "ok", "data": {"url": "https://my-app.vercel.app"}}
        self.assertEqual(_extract_url("vercel", resp), "https://my-app.vercel.app")

    def test_vercel_alias_fallback(self):
        resp = {"status": "ok", "data": {"alias": ["my-app.vercel.app"]}}
        self.assertEqual(_extract_url("vercel", resp), "https://my-app.vercel.app")

    def test_railway_url(self):
        resp = {"status": "ok", "data": {"url": "my-app.up.railway.app"}}
        self.assertEqual(_extract_url("railway", resp), "https://my-app.up.railway.app")

    def test_no_url_in_response(self):
        resp = {"status": "ok", "data": {"id": "deploy-123"}}
        self.assertIsNone(_extract_url("vercel", resp))

    def test_non_dict_data(self):
        resp = {"status": "ok", "data": "some string"}
        self.assertIsNone(_extract_url("vercel", resp))


# ---------------------------------------------------------------------------
# Artifact storage on deploy
# ---------------------------------------------------------------------------

class TestDeployStoresArtifact(unittest.TestCase):

    @patch(_PATCH_HEALTH, new_callable=AsyncMock, return_value=_ok_health())
    @patch(_PATCH_GET_CRED, new_callable=AsyncMock, return_value={"token": "tok"})
    def test_stores_deployment_result_artifact(self, mock_cred, mock_health):
        """When goal_id is given and deploy succeeds, result is stored as artifact."""
        store = ArtifactStore(use_db=False)
        run_async(store.store(1, "phase_13_gate_result", "PASSED"))

        mock_integration = MagicMock()
        mock_integration.execute = AsyncMock(return_value={
            "status": "ok",
            "data": {"url": "https://app.vercel.app"},
        })

        mock_registry = _make_mock_registry(mock_integration)

        with patch("src.workflows.engine.artifacts.ArtifactStore", return_value=store), \
             patch(_PATCH_REGISTRY, return_value=mock_registry):
            result = run_async(deploy("vercel", "/app", goal_id=1))
            self.assertEqual(result["status"], "ok")

            # Verify artifact was stored
            stored = run_async(store.retrieve(1, "deployment_result"))
            self.assertIsNotNone(stored)
            parsed = json.loads(stored)
            self.assertEqual(parsed["status"], "ok")
            self.assertEqual(parsed["url"], "https://app.vercel.app")


# ---------------------------------------------------------------------------
# Integration error handling
# ---------------------------------------------------------------------------

class TestDeployErrorHandling(unittest.TestCase):

    @patch(_PATCH_GET_CRED, new_callable=AsyncMock, return_value={"token": "tok"})
    def test_integration_error_response(self, mock_cred):
        """When integration returns error status, deploy returns error."""
        mock_integration = MagicMock()
        mock_integration.execute = AsyncMock(return_value={
            "status": "error",
            "error": "Rate limit exceeded",
        })

        mock_registry = _make_mock_registry(mock_integration)

        with patch(_PATCH_REGISTRY, return_value=mock_registry):
            result = run_async(deploy("vercel", "/app"))
            self.assertEqual(result["status"], "error")
            self.assertIn("Rate limit", result["error"])

    @patch(_PATCH_GET_CRED, new_callable=AsyncMock, return_value={"token": "tok"})
    def test_integration_exception(self, mock_cred):
        """When integration.execute raises, deploy catches and returns error."""
        mock_integration = MagicMock()
        mock_integration.execute = AsyncMock(side_effect=RuntimeError("connection lost"))

        mock_registry = _make_mock_registry(mock_integration)

        with patch(_PATCH_REGISTRY, return_value=mock_registry):
            result = run_async(deploy("vercel", "/app"))
            self.assertEqual(result["status"], "error")
            self.assertIn("connection lost", result["error"])

    @patch(_PATCH_GET_CRED, new_callable=AsyncMock, return_value={"token": "tok"})
    def test_no_integration_configured(self, mock_cred):
        """When registry has no integration for target, returns error."""
        mock_registry = _make_mock_registry(None)

        with patch(_PATCH_REGISTRY, return_value=mock_registry):
            result = run_async(deploy("vercel", "/app"))
            self.assertEqual(result["status"], "error")
            self.assertIn("No integration", result["error"])


if __name__ == "__main__":
    unittest.main()
