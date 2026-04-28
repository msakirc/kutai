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

# Groq /models exposes audio (STT/TTS) entries that share the listing with
# chat-completion models but only work on /v1/audio/* endpoints. They cannot
# be called via the chat-completions path Hoca uses, so filter at discovery.
# Patterns are matched case-insensitively against the raw model id.
_NON_CHAT_ID_PREFIXES = (
    "whisper-",                 # Speech-to-text
    "canopylabs/orpheus-",      # Text-to-speech
)

# Groq /models does not return rate limits; values below mirror the free-tier
# dashboard (https://console.groq.com/settings/limits) as of 2026-04-28.
# Keys match raw_id (no "groq/" prefix). KDV reads these to gate admission.
# Update this table when the dashboard changes or the account moves tiers.
_FREE_TIER_RATE_LIMITS: dict[str, dict[str, int]] = {
    "allam-2-7b":                                {"rpm": 30, "tpm": 6_000,  "rpd": 7_000,   "tpd": 500_000},
    "groq/compound":                             {"rpm": 30, "tpm": 70_000, "rpd": 250},
    "groq/compound-mini":                        {"rpm": 30, "tpm": 70_000, "rpd": 250},
    "llama-3.1-8b-instant":                      {"rpm": 30, "tpm": 6_000,  "rpd": 14_400,  "tpd": 500_000},
    "llama-3.3-70b-versatile":                   {"rpm": 30, "tpm": 12_000, "rpd": 1_000,   "tpd": 100_000},
    "meta-llama/llama-4-scout-17b-16e-instruct": {"rpm": 30, "tpm": 30_000, "rpd": 1_000,   "tpd": 500_000},
    "meta-llama/llama-prompt-guard-2-22m":       {"rpm": 30, "tpm": 15_000, "rpd": 14_400,  "tpd": 500_000},
    "meta-llama/llama-prompt-guard-2-86m":       {"rpm": 30, "tpm": 15_000, "rpd": 14_400,  "tpd": 500_000},
    "openai/gpt-oss-120b":                       {"rpm": 30, "tpm": 8_000,  "rpd": 1_000,   "tpd": 200_000},
    "openai/gpt-oss-20b":                        {"rpm": 30, "tpm": 8_000,  "rpd": 1_000,   "tpd": 200_000},
    "openai/gpt-oss-safeguard-20b":              {"rpm": 30, "tpm": 8_000,  "rpd": 1_000,   "tpd": 200_000},
    "qwen/qwen3-32b":                            {"rpm": 60, "tpm": 6_000,  "rpd": 1_000,   "tpd": 500_000},
}


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
            lower_id = raw_id.lower()
            if any(lower_id.startswith(p) for p in _NON_CHAT_ID_PREFIXES):
                continue
            limits = _FREE_TIER_RATE_LIMITS.get(raw_id, {})
            models.append(DiscoveredModel(
                litellm_name=f"groq/{raw_id}",
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
