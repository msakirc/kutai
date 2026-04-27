"""Groq /models adapter.

Endpoint: GET https://api.groq.com/openai/v1/models
Auth:     Bearer <api_key>
Response: {"data": [{"id": "...", "active": bool, "context_window": int,
                     "max_completion_tokens": int, "owned_by": str}, ...]}
"""
from __future__ import annotations

import httpx

from src.infra.logging_config import get_logger

from ..types import DiscoveredModel, ProviderResult

logger = get_logger("fatih_hoca.cloud.providers.groq")

_URL = "https://api.groq.com/openai/v1/models"
_TIMEOUT = httpx.Timeout(10.0)


class GroqAdapter:
    name = "groq"

    async def fetch_models(self, api_key: str) -> ProviderResult:
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(_URL, headers=headers)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.NetworkError) as e:
            return ProviderResult(provider=self.name, status="network_error",
                                  auth_ok=False, error=str(e))
        if resp.status_code in (401, 403):
            return ProviderResult(provider=self.name, status="auth_fail",
                                  auth_ok=False, error=f"{resp.status_code} {resp.text[:200]}")
        if resp.status_code == 429:
            return ProviderResult(provider=self.name, status="rate_limited",
                                  auth_ok=True, error="429 at /models probe")
        if resp.status_code >= 500:
            return ProviderResult(provider=self.name, status="server_error",
                                  auth_ok=False, error=f"{resp.status_code}")
        try:
            payload = resp.json()
        except Exception as e:  # noqa: BLE001
            return ProviderResult(provider=self.name, status="server_error",
                                  auth_ok=False, error=f"json parse: {e}")
        models: list[DiscoveredModel] = []
        for entry in payload.get("data", []):
            if not entry.get("active", True):
                continue
            raw_id = entry.get("id", "")
            if not raw_id:
                continue
            models.append(DiscoveredModel(
                litellm_name=f"groq/{raw_id}",
                raw_id=raw_id,
                active=True,
                context_length=entry.get("context_window"),
                max_output_tokens=entry.get("max_completion_tokens"),
                extra={"owned_by": entry.get("owned_by", "")},
            ))
        return ProviderResult(provider=self.name, status="ok", auth_ok=True, models=models)
