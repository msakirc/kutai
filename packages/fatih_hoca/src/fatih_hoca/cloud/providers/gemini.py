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
    - /v1beta/models does NOT expose per-model rate limits. The
      ``_FREE_TIER_QUOTAS`` table seeds rpm/tpm/rpd from the values
      published by Google AI Studio for free tier (2026-04-30 snapshot).
      Models with all-zero allocation are marked active=False so the
      registry skips them — selector never lists them as candidates and
      we don't waste a 429 round-trip just to learn they're tier-locked.
"""
from __future__ import annotations

import httpx

from ..types import DiscoveredModel, ProviderResult

_URL = "https://generativelanguage.googleapis.com/v1beta/models"
_TIMEOUT = httpx.Timeout(10.0)


# Free-tier quotas as published in Google AI Studio. Keys are matched by
# substring against the model id (longest match wins). Values are
# (rpm, tpm, rpd). 0 in any cell means the metric is disallowed on free
# tier — the entire model is skipped if all three are 0.
#
# Source: https://aistudio.google.com → Get API key → quota table.
# Snapshot date: 2026-04-30. Re-pull whenever Google changes pricing.
# When unsure, leaving a model out of this table means it falls through
# to _CONSERVATIVE_DEFAULT (low rpm/rpd) — better to under-utilize a
# new model than 429 it on first call.
_FREE_TIER_QUOTAS: dict[str, tuple[int, int, int]] = {
    # Text-out flagship models
    "gemini-3-flash-preview":          (5, 250_000, 20),
    "gemini-flash-latest":             (5, 250_000, 20),       # alias → 3-flash
    "gemini-pro-latest":               (0, 0, 0),                # 3.1 Pro = paid only
    "gemini-3.1-pro-preview":          (0, 0, 0),
    "gemini-3.1-flash-lite-preview":   (15, 250_000, 500),
    "gemini-flash-lite-latest":        (15, 250_000, 500),
    "gemini-2.5-pro":                  (0, 0, 0),
    "gemini-2.5-flash":                (5, 250_000, 20),
    "gemini-2.5-flash-lite":           (10, 250_000, 20),
    "gemini-2.5-flash-preview-05-20":  (5, 250_000, 20),         # legacy alias
    # 2.0-flash family is paid-only on this tier
    "gemini-2.0-flash":                (0, 0, 0),
    "gemini-2.0-flash-001":            (0, 0, 0),
    "gemini-2.0-flash-lite":           (0, 0, 0),
    "gemini-2.0-flash-lite-001":       (0, 0, 0),
    # Gemma family — Other models tier
    "gemma-3-1b-it":                   (30, 15_000, 14_400),
    "gemma-3-4b-it":                   (30, 15_000, 14_400),
    "gemma-3-12b-it":                  (30, 15_000, 14_400),
    "gemma-3-27b-it":                  (30, 15_000, 14_400),
    "gemma-3n-e2b-it":                 (30, 15_000, 14_400),
    "gemma-3n-e4b-it":                 (30, 15_000, 14_400),
    "gemma-4-26b-a4b-it":              (15, 1_000_000, 1_500),
    "gemma-4-31b-it":                  (15, 1_000_000, 1_500),
    # Robotics / agents
    "gemini-robotics-er-1.5-preview":  (10, 250_000, 20),
    "gemini-robotics-er-1.6-preview":  (5, 250_000, 20),
    "gemini-2.5-computer-use-preview": (0, 0, 0),
    "deep-research-preview":           (0, 0, 0),
    "deep-research-pro-preview":       (0, 0, 0),
    "deep-research-max-preview":       (0, 0, 0),
    # Image / TTS / video — handled by modality filter, listed for completeness
    "gemini-2.5-flash-preview-tts":    (3, 10_000, 10),
    "gemini-3.1-flash-tts-preview":    (3, 10_000, 10),
    "gemini-2.5-flash-image":          (0, 0, 0),
    "gemini-3-pro-image-preview":      (0, 0, 0),
    "gemini-3.1-flash-image-preview":  (0, 0, 0),
    "lyria-3-clip-preview":            (0, 0, 0),
    "lyria-3-pro-preview":             (0, 0, 0),
    "nano-banana":                     (0, 0, 0),
}

# Conservative default for unmatched models — assume some quota exists
# but throttle hard until headers/429 body teach the real values.
_CONSERVATIVE_DEFAULT: tuple[int, int, int] = (5, 100_000, 50)


def _quota_for(raw_id: str) -> tuple[int, int, int]:
    """Longest-substring lookup. Falls back to conservative default."""
    n = raw_id.lower()
    best_key = ""
    best_val = _CONSERVATIVE_DEFAULT
    for key, val in _FREE_TIER_QUOTAS.items():
        if key in n and len(key) > len(best_key):
            best_key = key
            best_val = val
    return best_val


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
            # Embedding models use embedContent (already excluded). Text /
            # image / audio models all use generateContent — modality is
            # inferred from the model name suffix below.
            if "generateContent" not in methods:
                continue
            modality = _infer_modality(raw_id)
            rpm, tpm, rpd = _quota_for(raw_id)
            # All-zero quota means the tier excludes this model entirely.
            # Mark inactive so register_cloud_from_discovered skips it —
            # selector never sees the model and we don't burn a 429
            # round-trip discovering tier-lock at runtime.
            tier_locked = (rpm == 0 and tpm == 0 and rpd == 0)
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
                active=not tier_locked,
                output_modality=modality,
                context_length=entry.get("inputTokenLimit"),
                max_output_tokens=entry.get("outputTokenLimit"),
                rate_limit_rpm=rpm if not tier_locked else None,
                rate_limit_tpm=tpm if not tier_locked else None,
                rate_limit_rpd=rpd if not tier_locked else None,
                sampling_defaults=sampling,
                extra={"display_name": entry.get("displayName", "")},
            ))
        return ProviderResult(provider=self.name, status="ok", auth_ok=True, models=models)


def _infer_modality(raw_id: str) -> str:
    """Map a Gemini model id to output modality.

    Gemini's /v1beta/models response doesn't expose modality as a structured
    field; it's encoded in the id. Image and TTS models share the same
    `generateContent` method as text models but produce a different output
    type. Without this distinction, registry treats them as generic text
    candidates and the selector picks them for code/reviewer tasks
    (witnessed in production: gemini-2.5-flash-image picked for a `coder`
    role, returning a 403/PERMISSION_DENIED on the chat endpoint).

    Patterns drawn from Gemini's published model family naming:
        *-image*       → image generation
        *-tts*         → text-to-speech
        *-audio*       → audio generation
        *-embedding*   → embedding (also filtered earlier by method check)
        *-video*       → video generation
    """
    n = raw_id.lower()
    if "embedding" in n:
        return "embedding"
    if "image" in n:
        return "image"
    if "tts" in n or "audio" in n:
        return "audio"
    if "video" in n:
        return "video"
    return "text"
