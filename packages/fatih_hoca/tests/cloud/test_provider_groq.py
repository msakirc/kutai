from unittest.mock import AsyncMock, patch

import httpx
import pytest

from fatih_hoca.cloud.providers.groq import GroqAdapter


_GROQ_OK = {
    "object": "list",
    "data": [
        {
            "id": "llama-3.3-70b-versatile",
            "object": "model",
            "owned_by": "Meta",
            "active": True,
            "context_window": 131072,
            "max_completion_tokens": 32768,
        },
        {
            "id": "deprecated-model",
            "object": "model",
            "owned_by": "Whoever",
            "active": False,
            "context_window": 8192,
        },
        {
            "id": "llama-3.1-8b-instant",
            "object": "model",
            "owned_by": "Meta",
            "active": True,
            "context_window": 131072,
            "max_completion_tokens": 8192,
        },
    ],
}


def _resp(status_code: int, body):
    if isinstance(body, dict):
        return httpx.Response(status_code, json=body, request=httpx.Request("GET", "https://api.groq.com"))
    return httpx.Response(status_code, text=body, request=httpx.Request("GET", "https://api.groq.com"))


@pytest.mark.asyncio
async def test_groq_ok_filters_inactive_and_prefixes():
    a = GroqAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(200, _GROQ_OK))):
        result = await a.fetch_models("fake-key")
    assert result.status == "ok"
    assert result.auth_ok is True
    names = [m.litellm_name for m in result.models]
    assert "groq/deprecated-model" not in names
    assert "groq/llama-3.3-70b-versatile" in names
    by_name = {m.litellm_name: m for m in result.models}
    assert by_name["groq/llama-3.3-70b-versatile"].context_length == 131072
    assert by_name["groq/llama-3.3-70b-versatile"].max_output_tokens == 32768


@pytest.mark.asyncio
async def test_groq_401_auth_fail():
    a = GroqAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(401, {"error": "Invalid API key"}))):
        result = await a.fetch_models("bad-key")
    assert result.status == "auth_fail"
    assert result.auth_ok is False
    assert result.models == []
    assert "401" in (result.error or "")


@pytest.mark.asyncio
async def test_groq_5xx_server_error():
    a = GroqAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(503, "down"))):
        result = await a.fetch_models("k")
    assert result.status == "server_error"
    assert result.auth_ok is False


@pytest.mark.asyncio
async def test_groq_network_error():
    a = GroqAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(side_effect=httpx.ConnectError("DNS"))):
        result = await a.fetch_models("k")
    assert result.status == "network_error"
    assert result.auth_ok is False


@pytest.mark.asyncio
async def test_groq_429_rate_limited_treated_as_ok_auth():
    a = GroqAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(429, "slow down"))):
        result = await a.fetch_models("k")
    assert result.status == "rate_limited"
    assert result.auth_ok is True


@pytest.mark.asyncio
async def test_groq_malformed_json():
    a = GroqAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(200, "not json {"))):
        result = await a.fetch_models("k")
    assert result.status == "server_error"
