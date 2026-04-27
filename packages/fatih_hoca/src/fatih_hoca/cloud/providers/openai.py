"""OpenAI /v1/models adapter.

Endpoint: GET https://api.openai.com/v1/models
Auth:     Bearer <api_key>
Response: {"data": [{"id": str, "object": "model", "owned_by": str,
                     "created": int}, ...]}

OpenAI lumps every model into one list — chat, embedding, TTS, Whisper.
We filter to chat-completion-ish models by id-prefix allowlist.
"""
from __future__ import annotations

import httpx

from ..types import DiscoveredModel, ProviderResult

_URL = "https://api.openai.com/v1/models"
_TIMEOUT = httpx.Timeout(10.0)
_CHAT_PREFIXES = ("gpt-", "o1", "o3", "o4")
_EXCLUDE_SUBSTR = ("embedding", "whisper", "tts", "dall-e", "moderation")


class OpenAIAdapter:
    name = "openai"

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
            if not raw_id.startswith(_CHAT_PREFIXES):
                continue
            if any(s in raw_id for s in _EXCLUDE_SUBSTR):
                continue
            models.append(DiscoveredModel(
                litellm_name=raw_id,  # litellm uses bare id for OpenAI
                raw_id=raw_id,
                extra={"owned_by": entry.get("owned_by", ""), "created": entry.get("created")},
            ))
        return ProviderResult(provider=self.name, status="ok", auth_ok=True, models=models)
