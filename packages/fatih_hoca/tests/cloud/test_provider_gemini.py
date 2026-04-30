from unittest.mock import AsyncMock, patch

import httpx
import pytest

from fatih_hoca.cloud.providers.gemini import GeminiAdapter


_OK = {
    "models": [
        {
            "name": "models/gemini-2.0-flash",
            "displayName": "Gemini 2.0 Flash",
            "inputTokenLimit": 1048576,
            "outputTokenLimit": 8192,
            "supportedGenerationMethods": ["generateContent", "countTokens"],
            "temperature": 1.0,
            "topP": 0.95,
            "topK": 64,
        },
        {
            "name": "models/embedding-001",
            "displayName": "Embedding",
            "supportedGenerationMethods": ["embedContent"],
        },
        {
            "name": "models/gemini-2.5-flash-image",
            "displayName": "Gemini 2.5 Flash Image",
            "supportedGenerationMethods": ["generateContent"],
        },
        {
            "name": "models/gemini-2.5-flash-preview-tts",
            "displayName": "Gemini 2.5 Flash TTS",
            "supportedGenerationMethods": ["generateContent"],
        },
    ],
}


def _resp(code, body):
    return httpx.Response(code, json=body, request=httpx.Request("GET", "https://generativelanguage.googleapis.com"))


@pytest.mark.asyncio
async def test_gemini_filters_non_generative():
    a = GeminiAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(200, _OK))):
        result = await a.fetch_models("k")
    assert result.status == "ok"
    names = [m.litellm_name for m in result.models]
    assert "gemini/gemini-2.0-flash" in names
    assert all("embedding" not in n for n in names)


@pytest.mark.asyncio
async def test_gemini_scrapes_token_limits_and_sampling():
    a = GeminiAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(200, _OK))):
        result = await a.fetch_models("k")
    m = next(x for x in result.models if x.raw_id == "gemini-2.0-flash")
    assert m.context_length == 1048576
    assert m.max_output_tokens == 8192
    assert m.sampling_defaults == {"temperature": 1.0, "top_p": 0.95, "top_k": 64.0}


@pytest.mark.asyncio
async def test_gemini_tags_modality_for_image_and_tts_models():
    """Image / TTS models share `generateContent` with text models so the
    method-based filter alone lets them through. Adapter must tag them with
    the right output_modality so registry skips them.
    """
    a = GeminiAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(200, _OK))):
        result = await a.fetch_models("k")
    by_id = {m.raw_id: m for m in result.models}
    assert by_id["gemini-2.0-flash"].output_modality == "text"
    assert by_id["gemini-2.5-flash-image"].output_modality == "image"
    assert by_id["gemini-2.5-flash-preview-tts"].output_modality == "audio"


@pytest.mark.asyncio
async def test_gemini_seeds_free_tier_quota_from_table():
    """Adapter populates rate_limit_rpm/tpm/rpd from the static
    free-tier table. Tier-locked models (all-zero allocation) are marked
    active=False so registry skips them — selector never sees them and
    we don't burn a 429 round-trip discovering the lock at runtime.
    """
    payload = {
        "models": [
            # Has free-tier allocation — should register active with quotas
            {
                "name": "models/gemini-2.5-flash",
                "supportedGenerationMethods": ["generateContent"],
                "inputTokenLimit": 1048576,
                "outputTokenLimit": 65536,
            },
            # Tier-locked (0/0/0) — should be active=False
            {
                "name": "models/gemini-2.5-pro",
                "supportedGenerationMethods": ["generateContent"],
                "inputTokenLimit": 1048576,
                "outputTokenLimit": 65536,
            },
            # Gemma family — has its own quota row
            {
                "name": "models/gemma-3-27b-it",
                "supportedGenerationMethods": ["generateContent"],
                "inputTokenLimit": 8192,
                "outputTokenLimit": 8192,
            },
        ],
    }
    a = GeminiAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(200, payload))):
        result = await a.fetch_models("k")
    by_id = {m.raw_id: m for m in result.models}

    # Free-tier flash → 5 RPM / 250K TPM / 20 RPD
    flash = by_id["gemini-2.5-flash"]
    assert flash.active is True
    assert flash.rate_limit_rpm == 5
    assert flash.rate_limit_tpm == 250_000
    assert flash.rate_limit_rpd == 20

    # Tier-locked pro → active=False, no quotas attached
    pro = by_id["gemini-2.5-pro"]
    assert pro.active is False

    # Gemma → 30 / 15K / 14400
    gemma = by_id["gemma-3-27b-it"]
    assert gemma.active is True
    assert gemma.rate_limit_rpm == 30
    assert gemma.rate_limit_tpm == 15_000
    assert gemma.rate_limit_rpd == 14_400


@pytest.mark.asyncio
async def test_gemini_400_invalid_key_treated_as_auth_fail():
    """Gemini returns 400 with key-invalid message instead of 401."""
    a = GeminiAdapter()
    body = {"error": {"code": 400, "message": "API key not valid", "status": "INVALID_ARGUMENT"}}
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(400, body))):
        result = await a.fetch_models("k")
    assert result.status == "auth_fail"
