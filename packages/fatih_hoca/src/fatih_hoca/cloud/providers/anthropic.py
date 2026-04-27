"""Anthropic /v1/models adapter.

Endpoint: GET https://api.anthropic.com/v1/models
Auth headers: x-api-key, anthropic-version
Response: {"data": [{"id": str, "display_name": str, "created_at": str,
                     "type": "model"}, ...]}
"""
from __future__ import annotations

import httpx

from ..types import DiscoveredModel, ProviderResult

_URL = "https://api.anthropic.com/v1/models"
_TIMEOUT = httpx.Timeout(10.0)
_API_VERSION = "2023-06-01"


class AnthropicAdapter:
    name = "anthropic"

    async def fetch_models(self, api_key: str) -> ProviderResult:
        headers = {"x-api-key": api_key, "anthropic-version": _API_VERSION}
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(_URL, headers=headers)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.NetworkError) as e:
            return ProviderResult(provider=self.name, status="network_error",
                                  auth_ok=False, error=str(e))
        if resp.status_code in (401, 403):
            return ProviderResult(provider=self.name, status="auth_fail",
                                  auth_ok=False, error=f"{resp.status_code}")
        if resp.status_code == 429:
            return ProviderResult(provider=self.name, status="rate_limited", auth_ok=True)
        if resp.status_code >= 500:
            return ProviderResult(provider=self.name, status="server_error", auth_ok=False)
        try:
            payload = resp.json()
        except Exception as e:  # noqa: BLE001
            return ProviderResult(provider=self.name, status="server_error",
                                  auth_ok=False, error=f"json parse: {e}")
        models: list[DiscoveredModel] = []
        for entry in payload.get("data", []):
            raw_id = entry.get("id", "")
            if not raw_id:
                continue
            models.append(DiscoveredModel(
                litellm_name=raw_id,  # litellm uses bare id for Anthropic
                raw_id=raw_id,
                extra={
                    "display_name": entry.get("display_name", ""),
                    "created_at": entry.get("created_at", ""),
                },
            ))
        return ProviderResult(provider=self.name, status="ok", auth_ok=True, models=models)
