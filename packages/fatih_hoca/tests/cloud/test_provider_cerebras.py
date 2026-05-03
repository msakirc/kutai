from unittest.mock import AsyncMock, patch

import httpx
import pytest

from fatih_hoca.cloud.providers.cerebras import CerebrasAdapter


_CEREBRAS_OK = {
    "object": "list",
    "data": [
        {"id": "llama-3.3-70b", "object": "model", "owned_by": "Meta"},
        {"id": "llama3.1-8b", "object": "model", "owned_by": "Meta"},
        {"id": "qwen-3-coder-480b", "object": "model", "owned_by": "Alibaba"},
    ],
}


def _resp(status_code: int, body):
    if isinstance(body, dict):
        return httpx.Response(status_code, json=body, request=httpx.Request("GET", "https://api.cerebras.ai"))
    return httpx.Response(status_code, text=body, request=httpx.Request("GET", "https://api.cerebras.ai"))


@pytest.mark.asyncio
async def test_cerebras_ok_prefixes_and_rate_limits():
    a = CerebrasAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(200, _CEREBRAS_OK))):
        result = await a.fetch_models("fake-key")
    assert result.status == "ok"
    assert result.auth_ok is True
    by_name = {m.litellm_name: m for m in result.models}
    assert "cerebras/llama-3.3-70b" in by_name
    assert "cerebras/llama3.1-8b" in by_name
    # known free-tier baseline applied
    assert by_name["cerebras/llama-3.3-70b"].rate_limit_rpm == 30
    assert by_name["cerebras/llama-3.3-70b"].rate_limit_tpd == 1_000_000
    # large-model override applied
    assert by_name["cerebras/qwen-3-coder-480b"].rate_limit_rpm == 10
    assert by_name["cerebras/qwen-3-coder-480b"].rate_limit_rpd == 100


@pytest.mark.asyncio
async def test_cerebras_401_auth_fail_carries_error():
    a = CerebrasAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(401, {"error": "Invalid API key"}))):
        result = await a.fetch_models("bad-key")
    assert result.status == "auth_fail"
    assert result.auth_ok is False
    assert result.models == []
    assert "401" in (result.error or "")


@pytest.mark.asyncio
async def test_cerebras_5xx_server_error():
    a = CerebrasAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(503, "down"))):
        result = await a.fetch_models("k")
    assert result.status == "server_error"
    assert result.auth_ok is False


@pytest.mark.asyncio
async def test_cerebras_429_rate_limited_treated_as_ok_auth():
    a = CerebrasAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(429, "slow down"))):
        result = await a.fetch_models("k")
    assert result.status == "rate_limited"
    assert result.auth_ok is True


@pytest.mark.asyncio
async def test_cerebras_network_error():
    a = CerebrasAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(side_effect=httpx.ConnectError("DNS"))):
        result = await a.fetch_models("k")
    assert result.status == "network_error"
    assert result.auth_ok is False


@pytest.mark.asyncio
async def test_cerebras_malformed_json():
    a = CerebrasAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(200, "not json {"))):
        result = await a.fetch_models("k")
    assert result.status == "server_error"


@pytest.mark.asyncio
async def test_cerebras_skips_inactive():
    body = {"data": [
        {"id": "active-one", "active": True, "owned_by": "x"},
        {"id": "dead-one", "active": False, "owned_by": "x"},
    ]}
    a = CerebrasAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(200, body))):
        result = await a.fetch_models("k")
    names = [m.litellm_name for m in result.models]
    assert "cerebras/active-one" in names
    assert "cerebras/dead-one" not in names
