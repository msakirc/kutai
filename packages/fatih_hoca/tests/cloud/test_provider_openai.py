from unittest.mock import AsyncMock, patch

import httpx
import pytest

from fatih_hoca.cloud.providers.openai import OpenAIAdapter


_OPENAI_OK = {
    "object": "list",
    "data": [
        {"id": "gpt-4o", "object": "model", "owned_by": "openai", "created": 1700000000},
        {"id": "gpt-4o-mini", "object": "model", "owned_by": "openai", "created": 1710000000},
        {"id": "text-embedding-3-small", "object": "model", "owned_by": "openai", "created": 1690000000},
        {"id": "whisper-1", "object": "model", "owned_by": "openai", "created": 1680000000},
        {"id": "tts-1", "object": "model", "owned_by": "openai", "created": 1680000000},
    ],
}


def _resp(code, body):
    return httpx.Response(code, json=body, request=httpx.Request("GET", "https://api.openai.com"))


@pytest.mark.asyncio
async def test_openai_filters_non_chat_models():
    a = OpenAIAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(200, _OPENAI_OK))):
        result = await a.fetch_models("k")
    names = [m.raw_id for m in result.models]
    assert "gpt-4o" in names
    assert "gpt-4o-mini" in names
    assert "text-embedding-3-small" not in names
    assert "whisper-1" not in names
    assert "tts-1" not in names


@pytest.mark.asyncio
async def test_openai_litellm_name_has_no_provider_prefix():
    """litellm uses bare 'gpt-4o' for OpenAI, not 'openai/gpt-4o'."""
    a = OpenAIAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(200, _OPENAI_OK))):
        result = await a.fetch_models("k")
    assert all(not m.litellm_name.startswith("openai/") for m in result.models)


@pytest.mark.asyncio
async def test_openai_401():
    a = OpenAIAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(401, {"error": {"message": "bad"}}))):
        result = await a.fetch_models("k")
    assert result.status == "auth_fail"
