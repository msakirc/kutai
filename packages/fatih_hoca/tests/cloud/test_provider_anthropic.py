from unittest.mock import AsyncMock, patch

import httpx
import pytest

from fatih_hoca.cloud.providers.anthropic import AnthropicAdapter


_OK = {
    "data": [
        {"id": "claude-sonnet-4-20250514", "display_name": "Claude Sonnet 4",
         "created_at": "2025-05-14T00:00:00Z", "type": "model"},
        {"id": "claude-3-5-sonnet-20241022", "display_name": "Claude 3.5 Sonnet",
         "created_at": "2024-10-22T00:00:00Z", "type": "model"},
    ],
    "has_more": False,
}


def _resp(code, body):
    return httpx.Response(code, json=body, request=httpx.Request("GET", "https://api.anthropic.com"))


@pytest.mark.asyncio
async def test_anthropic_ok():
    a = AnthropicAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(200, _OK))):
        result = await a.fetch_models("k")
    assert result.status == "ok"
    names = [m.litellm_name for m in result.models]
    assert "claude-sonnet-4-20250514" in names
    assert "claude-3-5-sonnet-20241022" in names


@pytest.mark.asyncio
async def test_anthropic_401():
    a = AnthropicAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(401, {"error": {"message": "bad"}}))):
        result = await a.fetch_models("k")
    assert result.status == "auth_fail"
