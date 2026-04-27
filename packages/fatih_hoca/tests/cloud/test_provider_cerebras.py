from unittest.mock import AsyncMock, patch

import httpx
import pytest

from fatih_hoca.cloud.providers.cerebras import CerebrasAdapter


_OK = {
    "data": [
        {"id": "llama3.3-70b", "object": "model", "owned_by": "Meta"},
        {"id": "llama-3.3-70b", "object": "model", "owned_by": "Meta"},
    ],
}


def _resp(code, body):
    return httpx.Response(code, json=body, request=httpx.Request("GET", "https://api.cerebras.ai"))


@pytest.mark.asyncio
async def test_cerebras_ok():
    a = CerebrasAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(200, _OK))):
        result = await a.fetch_models("k")
    names = [m.litellm_name for m in result.models]
    assert "cerebras/llama3.3-70b" in names


@pytest.mark.asyncio
async def test_cerebras_401():
    a = CerebrasAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(401, {}))):
        result = await a.fetch_models("k")
    assert result.status == "auth_fail"
