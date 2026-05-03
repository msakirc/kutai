"""Cerebras /v1/models adapter (OpenAI-compatible).

Endpoint: GET https://api.cerebras.ai/v1/models
Auth:     Bearer <api_key>
Response: {"object": "list",
           "data": [{"id": "...", "object": "model",
                     "created": int, "owned_by": str}, ...]}

Cerebras /models is the OpenAI-spec subset — it does NOT return
`context_window` or `max_completion_tokens`. Those stay None and the
registry falls back to litellm db / defaults. Rate limits are likewise
absent and come from `_FREE_TIER_RATE_LIMITS` below.
"""
from __future__ import annotations

import httpx

from src.infra.logging_config import get_logger

from ..types import DiscoveredModel, ProviderResult

logger = get_logger("fatih_hoca.cloud.providers.cerebras")

_URL = "https://api.cerebras.ai/v1/models"
_TIMEOUT = httpx.Timeout(10.0)

# Cerebras /models does not return rate limits; values below mirror the
# free-tier dashboard (https://inference-docs.cerebras.ai/support/rate-limits)
# as of 2026-04-30. Keys match raw_id (no "cerebras/" prefix). KDV reads these
# to gate admission. Update this table when the dashboard changes or the
# account moves tiers. Models not listed here fall through to a conservative
# baseline at registration time.
_FREE_TIER_RATE_LIMITS: dict[str, dict[str, int]] = {
    "llama3.1-8b":                       {"rpm": 30, "tpm": 60_000,  "rpd": 14_400, "tpd": 1_000_000},
    "llama-3.3-70b":                     {"rpm": 30, "tpm": 60_000,  "rpd": 14_400, "tpd": 1_000_000},
    "llama-4-scout-17b-16e-instruct":    {"rpm": 30, "tpm": 60_000,  "rpd": 14_400, "tpd": 1_000_000},
    "llama-4-maverick-17b-128e-instruct":{"rpm": 30, "tpm": 60_000,  "rpd": 14_400, "tpd": 1_000_000},
    "qwen-3-32b":                        {"rpm": 30, "tpm": 60_000,  "rpd": 14_400, "tpd": 1_000_000},
    "qwen-3-235b-a22b-instruct-2507":    {"rpm": 30, "tpm": 60_000,  "rpd": 14_400, "tpd": 1_000_000},
    "qwen-3-235b-a22b-thinking-2507":    {"rpm": 30, "tpm": 60_000,  "rpd": 14_400, "tpd": 1_000_000},
    "qwen-3-coder-480b":                 {"rpm": 10, "tpm": 150_000, "rpd": 100,    "tpd": 1_500_000},
    "gpt-oss-120b":                      {"rpm": 30, "tpm": 60_000,  "rpd": 14_400, "tpd": 1_000_000},
    "zai-glm-4.7":                       {"rpm": 30, "tpm": 60_000,  "rpd": 14_400, "tpd": 1_000_000},
}


class CerebrasAdapter:
    name = "cerebras"

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
            limits = _FREE_TIER_RATE_LIMITS.get(raw_id, {})
            models.append(DiscoveredModel(
                litellm_name=f"cerebras/{raw_id}",
                raw_id=raw_id,
                active=True,
                context_length=entry.get("context_window"),
                max_output_tokens=entry.get("max_completion_tokens"),
                rate_limit_rpm=limits.get("rpm"),
                rate_limit_tpm=limits.get("tpm"),
                rate_limit_rpd=limits.get("rpd"),
                rate_limit_tpd=limits.get("tpd"),
                extra={"owned_by": entry.get("owned_by", "")},
            ))
        return ProviderResult(provider=self.name, status="ok", auth_ok=True, models=models)
