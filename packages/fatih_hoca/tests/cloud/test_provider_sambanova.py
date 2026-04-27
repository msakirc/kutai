from unittest.mock import AsyncMock, patch

import httpx
import pytest

from fatih_hoca.cloud.providers.sambanova import SambanovaAdapter


_OK = {
    "data": [
        {"id": "Qwen3-32B", "object": "model"},
        {"id": "Meta-Llama-3.3-70B-Instruct", "object": "model"},
    ],
}


def _resp(code, body):
    return httpx.Response(code, json=body, request=httpx.Request("GET", "https://api.sambanova.ai"))


@pytest.mark.asyncio
async def test_sambanova_ok():
    a = SambanovaAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(200, _OK))):
        result = await a.fetch_models("k")
    names = [m.litellm_name for m in result.models]
    assert "sambanova/Qwen3-32B" in names
    assert "sambanova/Meta-Llama-3.3-70B-Instruct" in names


@pytest.mark.asyncio
async def test_sambanova_401():
    a = SambanovaAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(401, {}))):
        result = await a.fetch_models("k")
    assert result.status == "auth_fail"
