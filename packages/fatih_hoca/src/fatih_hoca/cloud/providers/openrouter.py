"""OpenRouter /api/v1/models adapter — rich field scrape.

Endpoint: GET https://openrouter.ai/api/v1/models
Response: {"data": [{"id": "<org>/<model>", "context_length": int,
                     "pricing": {"prompt": "<usd-per-token>", "completion": "<usd>"},
                     "top_provider": {"max_completion_tokens": int,
                                      "is_moderated": bool}}, ...]}

Pricing is per-token; we convert to per-1k to match ModelInfo.cost_per_1k_*.
"""
from __future__ import annotations

import httpx

from ..types import DiscoveredModel, ProviderResult

_URL = "https://openrouter.ai/api/v1/models"
_TIMEOUT = httpx.Timeout(15.0)


def _to_per_1k(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value) * 1000.0
    except (TypeError, ValueError):
        return None


class OpenRouterAdapter:
    name = "openrouter"

    async def fetch_models(self, api_key: str) -> ProviderResult:
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(_URL, headers=headers)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.NetworkError) as e:
            return ProviderResult(provider=self.name, status="network_error",
                                  auth_ok=False, error=str(e))
        if resp.status_code in (401, 403):
            return ProviderResult(provider=self.name, status="auth_fail", auth_ok=False)
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
            pricing = entry.get("pricing", {}) or {}
            top_prov = entry.get("top_provider", {}) or {}
            modality = _infer_modality(entry, raw_id)
            models.append(DiscoveredModel(
                litellm_name=f"openrouter/{raw_id}",
                raw_id=raw_id,
                output_modality=modality,
                context_length=entry.get("context_length"),
                max_output_tokens=top_prov.get("max_completion_tokens"),
                cost_per_1k_input=_to_per_1k(pricing.get("prompt")),
                cost_per_1k_output=_to_per_1k(pricing.get("completion")),
                extra={
                    "name": entry.get("name", ""),
                    "is_moderated": top_prov.get("is_moderated"),
                },
            ))
        return ProviderResult(provider=self.name, status="ok", auth_ok=True, models=models)


def _infer_modality(entry: dict, raw_id: str) -> str:
    """Read output modality from OpenRouter's `architecture.modality` field.

    Modality string format is `<input>-><output>` (e.g. "text->text",
    "text+image->text", "text->image"). Some entries use a list under
    `architecture.output_modalities`. Fallback to id-pattern detection
    (mirrors gemini adapter's `*-image*` / `*-tts*` heuristic) when
    architecture metadata is absent.
    """
    arch = entry.get("architecture") or {}
    out_mods = arch.get("output_modalities")
    if isinstance(out_mods, list) and out_mods:
        m = str(out_mods[0]).lower()
        if m in {"text", "image", "audio", "video", "embedding"}:
            return m
    modality_str = arch.get("modality")
    if isinstance(modality_str, str) and "->" in modality_str:
        out = modality_str.split("->", 1)[1].strip().lower()
        if "image" in out:
            return "image"
        if "audio" in out or "speech" in out:
            return "audio"
        if "video" in out:
            return "video"
        if "embedding" in out:
            return "embedding"
        if "text" in out:
            return "text"
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
