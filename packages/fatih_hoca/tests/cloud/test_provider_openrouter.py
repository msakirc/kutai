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


@pytest.fixture(autouse=True)
def _clear_openrouter_free_only(monkeypatch):
    """OPENROUTER_FREE_ONLY can leak from .env or shell — clear by default
    so tests assume the legacy "all models" behavior. Tests that exercise
    the filter explicitly setenv it themselves."""
    monkeypatch.delenv("OPENROUTER_FREE_ONLY", raising=False)


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


_FREE_AND_PAID = {
    "data": [
        {
            "id": "qwen/qwen3-coder:free",
            "pricing": {"prompt": "0", "completion": "0"},
        },
        {
            "id": "google/gemma-3-27b-it:free",
            "pricing": {"prompt": "0", "completion": "0"},
        },
        {
            "id": "anthropic/claude-sonnet-4.6",
            "pricing": {"prompt": "0.000003", "completion": "0.000015"},
        },
        {
            "id": "openai/gpt-5",
            "pricing": {"prompt": "0.00001", "completion": "0.00003"},
        },
        {
            # Paid but no `:free` suffix — pricing tells the truth
            "id": "vendor/free-by-pricing",
            "pricing": {"prompt": "0", "completion": "0"},
        },
    ],
}


@pytest.mark.asyncio
async def test_openrouter_free_only_drops_paid_models(monkeypatch):
    """OPENROUTER_FREE_ONLY=1 restricts the adapter to free models only.
    Production triage 2026-04-30: user wanted to confine bot to OR free
    models without setting key spend cap to $0 (which blocks key
    altogether). Adapter filter implements 'free-only' at registration
    so selector never sees paid OR models."""
    monkeypatch.setenv("OPENROUTER_FREE_ONLY", "1")
    a = OpenRouterAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(200, _FREE_AND_PAID))):
        result = await a.fetch_models("k")
    assert result.status == "ok"
    ids = {m.raw_id for m in result.models}
    # Free survives (suffix OR pricing=0)
    assert "qwen/qwen3-coder:free" in ids
    assert "google/gemma-3-27b-it:free" in ids
    assert "vendor/free-by-pricing" in ids
    # Paid dropped
    assert "anthropic/claude-sonnet-4.6" not in ids
    assert "openai/gpt-5" not in ids
    assert len(result.models) == 3


@pytest.mark.asyncio
async def test_openrouter_default_keeps_paid_models(monkeypatch):
    """Without the env flag, every model survives — paid + free."""
    monkeypatch.delenv("OPENROUTER_FREE_ONLY", raising=False)
    a = OpenRouterAdapter()
    with patch("httpx.AsyncClient.get", AsyncMock(return_value=_resp(200, _FREE_AND_PAID))):
        result = await a.fetch_models("k")
    assert len(result.models) == 5  # all kept


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
