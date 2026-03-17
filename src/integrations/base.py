# base.py
"""Abstract base class for all external service integrations."""

from abc import ABC, abstractmethod

from src.infra.logging_config import get_logger

logger = get_logger("integrations.base")


class BaseIntegration(ABC):
    """
    Every integration must declare a service_name and implement:
    - validate()     — check that credentials / connectivity are OK
    - capabilities() — list of action names this integration supports
    - execute()      — run a named action with parameters
    """

    service_name: str = ""

    @abstractmethod
    async def validate(self) -> bool:
        """Return True if the integration is properly configured and reachable."""
        ...

    @abstractmethod
    def capabilities(self) -> list[str]:
        """Return the list of action names this integration supports."""
        ...

    @abstractmethod
    async def execute(self, action: str, params: dict) -> dict:
        """
        Execute a named action.

        Args:
            action: The action name (must be in capabilities()).
            params: Action-specific parameters.

        Returns:
            A dict with at least {"status": "ok"|"error", ...}.
        """
        ...
