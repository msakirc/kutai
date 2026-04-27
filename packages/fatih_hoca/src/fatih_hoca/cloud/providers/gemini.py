"""Google Gemini /v1beta/models adapter.

Endpoint: GET https://generativelanguage.googleapis.com/v1beta/models?key=<api_key>
Response: {"models": [{"name": "models/<id>", "inputTokenLimit": int,
                       "outputTokenLimit": int,
                       "supportedGenerationMethods": [...],
                       "temperature": float, "topP": float,
                       "topK": int, ...}, ...]}

Notes:
    - Gemini returns 400 with INVALID_ARGUMENT for bad key, not 401.
    - Filter to models that support ``generateContent``.
"""
from __future__ import annotations

import httpx

from ..types import DiscoveredModel, ProviderResult

_URL = "https://generativelanguage.googleapis.com/v1beta/models"
_TIMEOUT = httpx.Timeout(10.0)


class GeminiAdapter:
    name = "gemini"

    async def fetch_models(self, api_key: str) -> ProviderResult:
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(_URL, params={"key": api_key})
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.NetworkError) as e:
            return ProviderResult(provider=self.name, status="network_error",
                                  auth_ok=False, error=str(e))
        if resp.status_code in (401, 403):
            return ProviderResult(provider=self.name, status="auth_fail", auth_ok=False)
        if resp.status_code == 400:
            try:
                msg = resp.json().get("error", {}).get("message", "")
            except Exception:  # noqa: BLE001
                msg = ""
            if "API key" in msg or "INVALID_ARGUMENT" in resp.text.upper():
                return ProviderResult(provider=self.name, status="auth_fail",
                                      auth_ok=False, error=msg or "400")
            return ProviderResult(provider=self.name, status="server_error",
                                  auth_ok=False, error=f"400 {msg}")
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
        for entry in payload.get("models", []):
            full_name = entry.get("name", "")
            raw_id = full_name.split("/", 1)[-1] if "/" in full_name else full_name
            if not raw_id:
                continue
            methods = entry.get("supportedGenerationMethods", [])
            if "generateContent" not in methods:
                continue
            sampling: dict[str, float] = {}
            if entry.get("temperature") is not None:
                sampling["temperature"] = float(entry["temperature"])
            if entry.get("topP") is not None:
                sampling["top_p"] = float(entry["topP"])
            if entry.get("topK") is not None:
                sampling["top_k"] = float(entry["topK"])
            models.append(DiscoveredModel(
                litellm_name=f"gemini/{raw_id}",
                raw_id=raw_id,
                context_length=entry.get("inputTokenLimit"),
                max_output_tokens=entry.get("outputTokenLimit"),
                sampling_defaults=sampling,
                extra={"display_name": entry.get("displayName", "")},
            ))
        return ProviderResult(provider=self.name, status="ok", auth_ok=True, models=models)
