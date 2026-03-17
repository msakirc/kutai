# registry.py
"""
Integration registry — discover, register, and look up service integrations.
"""

from typing import Optional

from src.infra.logging_config import get_logger
from .base import BaseIntegration

logger = get_logger("integrations.registry")


class IntegrationRegistry:
    """Central registry for all service integrations."""

    def __init__(self, auto_discover: bool = True):
        self._integrations: dict[str, BaseIntegration] = {}
        if auto_discover:
            self._auto_discover()

    def register(self, integration: BaseIntegration) -> None:
        """Register an integration instance."""
        name = integration.service_name
        if not name:
            raise ValueError("Integration must have a non-empty service_name")
        self._integrations[name] = integration
        logger.debug(f"Registered integration: {name}")

    def get(self, service_name: str) -> Optional[BaseIntegration]:
        """Get an integration by service name, or None if not found."""
        return self._integrations.get(service_name)

    def list_services(self) -> list[str]:
        """Return sorted list of registered service names."""
        return sorted(self._integrations.keys())

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
