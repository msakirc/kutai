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


_MIXED_MODALITY = {
    "data": [
        {
            "id": "meta-llama/llama-3.3-70b-instruct",
            "architecture": {"modality": "text->text",
                              "output_modalities": ["text"]},
        },
        {
            "id": "stability-ai/stable-diffusion-3.5",
            "architecture": {"modality": "text->image",
                              "output_modalities": ["image"]},
        },
        {
            "id": "openai/gpt-4o-tts",
            "architecture": {"modality": "text->audio",
                              "output_modalities": ["audio"]},
        },
        {
            "id": "openai/embedding-3-large",
            "architecture": {"modality": "text->embedding"},
        },
        {
            # Architecture missing — fall back to id-pattern detection
            "id": "vendor/some-image-model",
        },
    ],
}


@pytest.mark.asyncio
async def test_openrouter_tags_modality_from_architecture_and_id():
    a = OpenRouterAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(200, _MIXED_MODALITY))):
        result = await a.fetch_models("k")
    by_id = {m.raw_id: m for m in result.models}
    assert by_id["meta-llama/llama-3.3-70b-instruct"].output_modality == "text"
    assert by_id["stability-ai/stable-diffusion-3.5"].output_modality == "image"
    assert by_id["openai/gpt-4o-tts"].output_modality == "audio"
    assert by_id["openai/embedding-3-large"].output_modality == "embedding"
    # ID-pattern fallback when architecture absent
    assert by_id["vendor/some-image-model"].output_modality == "image"
