"""Live /models smoke tests for cloud provider adapters.

Skipped unless the corresponding API key is present in os.environ.
Run manually:

    pytest tests/integration/test_cloud_discovery_live.py -v

These hit real provider endpoints; do not run in CI without keys.
"""
import os

import pytest

pytestmark = pytest.mark.asyncio


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="no GROQ_API_KEY")
async def test_groq_live():
    from fatih_hoca.cloud.providers.groq import GroqAdapter
    result = await GroqAdapter().fetch_models(os.environ["GROQ_API_KEY"])
    assert result.status == "ok", f"unexpected: {result.status} / {result.error}"
    assert len(result.models) > 0
    m = result.models[0]
    assert m.litellm_name.startswith("groq/")
    assert m.context_length is not None and m.context_length > 0


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="no OPENAI_API_KEY")
async def test_openai_live():
    from fatih_hoca.cloud.providers.openai import OpenAIAdapter
    result = await OpenAIAdapter().fetch_models(os.environ["OPENAI_API_KEY"])
    assert result.status == "ok"
    assert any("gpt-4o" in m.raw_id for m in result.models)


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="no ANTHROPIC_API_KEY")
async def test_anthropic_live():
    from fatih_hoca.cloud.providers.anthropic import AnthropicAdapter
    result = await AnthropicAdapter().fetch_models(os.environ["ANTHROPIC_API_KEY"])
    assert result.status == "ok"
    assert any("claude" in m.raw_id for m in result.models)


@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="no GEMINI_API_KEY")
async def test_gemini_live():
    from fatih_hoca.cloud.providers.gemini import GeminiAdapter
    result = await GeminiAdapter().fetch_models(os.environ["GEMINI_API_KEY"])
    assert result.status == "ok"
    assert any("gemini" in m.raw_id for m in result.models)


@pytest.mark.skipif(not os.environ.get("CEREBRAS_API_KEY"), reason="no CEREBRAS_API_KEY")
async def test_cerebras_live():
    from fatih_hoca.cloud.providers.cerebras import CerebrasAdapter
    result = await CerebrasAdapter().fetch_models(os.environ["CEREBRAS_API_KEY"])
    assert result.status == "ok"


@pytest.mark.skipif(not os.environ.get("SAMBANOVA_API_KEY"), reason="no SAMBANOVA_API_KEY")
async def test_sambanova_live():
    from fatih_hoca.cloud.providers.sambanova import SambanovaAdapter
    result = await SambanovaAdapter().fetch_models(os.environ["SAMBANOVA_API_KEY"])
    assert result.status == "ok"


@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="no OPENROUTER_API_KEY")
async def test_openrouter_live():
    from fatih_hoca.cloud.providers.openrouter import OpenRouterAdapter
    result = await OpenRouterAdapter().fetch_models(os.environ["OPENROUTER_API_KEY"])
    assert result.status == "ok"
    assert len(result.models) > 50  # OpenRouter exposes hundreds
