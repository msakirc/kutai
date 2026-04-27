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
async def test_gemini_400_invalid_key_treated_as_auth_fail():
    """Gemini returns 400 with key-invalid message instead of 401."""
    a = GeminiAdapter()
    body = {"error": {"code": 400, "message": "API key not valid", "status": "INVALID_ARGUMENT"}}
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(400, body))):
        result = await a.fetch_models("k")
    assert result.status == "auth_fail"
