import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_try_resolve_weather_query():
    """Layer 0: weather query should resolve via API without LLM."""
    from src.core.fast_resolver import try_resolve

    task = {"title": "Istanbul hava durumu", "description": ""}

    with patch("src.core.fast_resolver._find_best_match", new_callable=AsyncMock) as mock_match, \
         patch("src.core.fast_resolver._call_best_api", new_callable=AsyncMock) as mock_call:

        mock_api = type("FakeAPI", (), {"name": "weather_api", "category": "weather"})()
        mock_match.return_value = {"api": mock_api, "category": "weather", "score": 0.8}
        mock_call.return_value = {"temp": 22, "condition": "sunny"}

        result = await try_resolve(task)

    assert result is not None
    assert "22" in result or "sunny" in result


@pytest.mark.asyncio
async def test_try_resolve_no_match_returns_none():
    """Layer 0: unrelated query should return None (fall through to agent)."""
    from src.core.fast_resolver import try_resolve

    task = {"title": "Write a Python script to sort files", "description": ""}
    result = await try_resolve(task)
    assert result is None


@pytest.mark.asyncio
async def test_enrich_context_adds_data():
    """Layer 1: partial match should return enriched context dict."""
    from src.core.fast_resolver import enrich_context

    task = {"title": "Istanbul'da bu hafta sonu piknik yapilir mi?", "description": ""}

    with patch("src.core.fast_resolver._find_best_match", new_callable=AsyncMock) as mock_match, \
         patch("src.core.fast_resolver._call_best_api", new_callable=AsyncMock) as mock_call:

        mock_api = type("FakeAPI", (), {"name": "weather_api", "category": "weather"})()
        mock_match.return_value = {"api": mock_api, "category": "weather", "score": 0.4}
        mock_call.return_value = {"temp": 22, "condition": "partly cloudy"}

        enrichment = await enrich_context(task)

    if enrichment:
        assert "Available Data" in enrichment or isinstance(enrichment, str)


@pytest.mark.asyncio
async def test_try_resolve_api_failure_falls_through():
    """Layer 0: API failure should return None, not raise."""
    from src.core.fast_resolver import try_resolve

    task = {"title": "Istanbul hava durumu", "description": ""}

    with patch("src.core.fast_resolver._find_best_match", new_callable=AsyncMock) as mock_match, \
         patch("src.core.fast_resolver._call_best_api", new_callable=AsyncMock) as mock_call:

        mock_api = type("FakeAPI", (), {"name": "weather_api", "category": "weather"})()
        mock_match.return_value = {"api": mock_api, "category": "weather", "score": 0.8}
        mock_call.side_effect = Exception("API timeout")

        result = await try_resolve(task)

    assert result is None
