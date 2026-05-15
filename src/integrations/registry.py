# registry.py
"""
Integration registry — discover, register, and look up service integrations.

Mock mode (Z9 T1B)
------------------
The registry exposes a ``mock_mode`` flag. When active, ``HttpIntegration``
short-circuits ``execute()`` and returns the deterministic ``mock_responses``
shipped in each provider's config JSON instead of hitting the network — this
keeps the Z9 analytics tiers (T2 digest, T3 signal intake, T5 A/B) offline
testable.

Default policy (see docs/i2p-evolution/09-growth-v2.md secondary decisions):
  * mock is ON whenever ``KUTAI_ENV`` != ``prod``
  * ``KUTAI_VENDOR_LIVE=1`` forces real calls even in a non-prod env
Tests may override the resolved value by setting ``registry.mock_mode``
directly.
"""

import os
from typing import Optional

from src.infra.logging_config import get_logger
from .base import BaseIntegration

logger = get_logger("integrations.registry")


def _resolve_mock_default() -> bool:
    """Resolve the default mock_mode value from the environment.

    Mock when not in prod, unless ``KUTAI_VENDOR_LIVE=1`` forces real calls.
    """
    if os.getenv("KUTAI_VENDOR_LIVE", "").strip() == "1":
        return False
    env = (os.getenv("KUTAI_ENV") or "").strip().lower()
    return env != "prod"


class IntegrationRegistry:
    """Central registry for all service integrations."""

    def __init__(self, auto_discover: bool = True, mock_mode: Optional[bool] = None):
        self._integrations: dict[str, BaseIntegration] = {}
        # Per-provider deterministic mock payloads, populated during discovery
        # from each config's optional ``mock_responses`` block.
        self._mock_responses: dict[str, dict] = {}
        # When mock_mode is left as None, resolve it from the environment.
        self.mock_mode: bool = (
            _resolve_mock_default() if mock_mode is None else bool(mock_mode)
        )
        if auto_discover:
            self._auto_discover()

    def register(self, integration: BaseIntegration) -> None:
        """Register an integration instance."""
        name = integration.service_name
        if not name:
            raise ValueError("Integration must have a non-empty service_name")
        self._integrations[name] = integration
        # Capture any mock_responses block the integration carries so the
        # registry can serve deterministic payloads without the adapter.
        mock = getattr(integration, "mock_responses", None)
        if isinstance(mock, dict) and mock:
            self._mock_responses[name] = mock
        logger.debug(f"Registered integration: {name}")

    def get(self, service_name: str) -> Optional[BaseIntegration]:
        """Get an integration by service name, or None if not found."""
        return self._integrations.get(service_name)

    def list_services(self) -> list[str]:
        """Return sorted list of registered service names."""
        return sorted(self._integrations.keys())

    # -- mock mode ---------------------------------------------------------

    def mock_response(self, service_name: str, action: str) -> Optional[dict]:
        """Return the deterministic mock payload for *service_name*/*action*.

        Returns a fresh copy (so callers can mutate freely) shaped exactly
        like a real ``HttpIntegration.execute()`` success envelope, or None
        when no mock is registered for that service/action pair.
        """
        import copy

        provider = self._mock_responses.get(service_name)
        if not isinstance(provider, dict):
            return None
        if action not in provider:
            return None
        return {
            "status": "ok",
            "data": copy.deepcopy(provider[action]),
            "status_code": 200,
            "mocked": True,
        }

    def _auto_discover(self) -> None:
        """Auto-discover and register built-in HTTP integrations from configs/."""
        try:
            from .http_integration import HttpIntegration

            import os
            import json

            configs_dir = os.path.join(os.path.dirname(__file__), "configs")
            if not os.path.isdir(configs_dir):
                return

            for fname in sorted(os.listdir(configs_dir)):
                if not fname.endswith(".json"):
                    continue
                fpath = os.path.join(configs_dir, fname)
                try:
                    with open(fpath, "r") as f:
                        config = json.load(f)
                    integration = HttpIntegration(config)
                    self.register(integration)
                except Exception as e:
                    logger.warning(
                        f"Failed to load integration config {fname}: {e}"
                    )
        except Exception as e:
            logger.debug(f"Auto-discovery skipped: {e}")


# Module-level singleton (lazy)
_registry: Optional[IntegrationRegistry] = None


def get_integration_registry() -> IntegrationRegistry:
    """Return the global integration registry (creates on first call)."""
    global _registry
    if _registry is None:
        _registry = IntegrationRegistry()
    return _registry
