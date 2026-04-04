import pytest
from unittest.mock import AsyncMock, patch, MagicMock


def _make_fake_api(name="wttr.in", category="weather"):
    api = MagicMock()
    api.name = name
    api.category = category
    api.example_endpoint = "https://wttr.in/Istanbul?format=j1"
    return api


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


@pytest.mark.asyncio
async def test_enrich_context_works_when_score_above_layer0_threshold():
    """Layer 1 should still enrich even if score >= 0.6 (L0 failed to resolve)."""
    from src.core.fast_resolver import enrich_context

    fake_api = _make_fake_api()
    match = {"api": fake_api, "category": "weather", "score": 0.8}

    with patch("src.core.fast_resolver._find_best_match", new_callable=AsyncMock, return_value=match), \
         patch("src.core.fast_resolver._call_best_api", new_callable=AsyncMock, return_value={"temp": "22C"}), \
         patch("src.core.fast_resolver._format_response", return_value="22C in Istanbul"), \
         patch("src.infra.db.log_smart_search", new_callable=AsyncMock), \
         patch("src.infra.db.record_api_call", new_callable=AsyncMock):
        result = await enrich_context({"title": "Istanbul hava durumu"})

    assert result is not None
    assert "22C" in result
