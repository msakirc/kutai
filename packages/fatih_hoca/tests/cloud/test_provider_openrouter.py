from unittest.mock import AsyncMock, patch

import httpx
import pytest

from fatih_hoca.cloud.providers.openrouter import OpenRouterAdapter


_OK = {
    "data": [
        {
            "id": "meta-llama/llama-3.3-70b-instruct",
            "name": "Llama 3.3 70B Instruct",
            "context_length": 131072,
            "pricing": {"prompt": "0.0000005", "completion": "0.0000008"},
            "top_provider": {"max_completion_tokens": 32768, "is_moderated": False},
        },
    ],
}


def _resp(code, body):
    return httpx.Response(code, json=body, request=httpx.Request("GET", "https://openrouter.ai"))


@pytest.mark.asyncio
async def test_openrouter_scrapes_pricing_and_context():
    a = OpenRouterAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(200, _OK))):
        result = await a.fetch_models("k")
    assert result.status == "ok"
    m = result.models[0]
    assert m.litellm_name == "openrouter/meta-llama/llama-3.3-70b-instruct"
    assert m.context_length == 131072
    assert m.max_output_tokens == 32768
    # Pricing converted from per-token to per-1k.
    assert m.cost_per_1k_input == pytest.approx(0.0005, rel=1e-6)
    assert m.cost_per_1k_output == pytest.approx(0.0008, rel=1e-6)


@pytest.mark.asyncio
async def test_openrouter_401():
    a = OpenRouterAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(401, {}))):
        result = await a.fetch_models("k")
    assert result.status == "auth_fail"
