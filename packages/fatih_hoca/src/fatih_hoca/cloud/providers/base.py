"""Adapter protocol shared by per-provider implementations."""
from __future__ import annotations

from typing import Protocol

from ..types import ProviderResult


class ProviderAdapter(Protocol):
    name: str

    async def fetch_models(self, api_key: str) -> ProviderResult:
        """Probe the provider's /models endpoint with the given key.

        Must NEVER raise. All errors map to ProviderResult.status +
        ProviderResult.error string.
        """
        ...
