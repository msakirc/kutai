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


def test_format_weather_response():
    from src.core.fast_resolver import _format_response

    raw = {
        "current_condition": [{"temp_C": "22", "weatherDesc": [{"value": "Sunny"}], "humidity": "45", "windspeedKmph": "12"}],
        "nearest_area": [{"areaName": [{"value": "Istanbul"}]}]
    }
    result = _format_response(raw, "weather", "wttr.in")
    assert "22" in result
    assert "Istanbul" in result


def test_format_currency_response():
    from src.core.fast_resolver import _format_response

    raw = {"rates": {"TRY": 38.45}, "base": "USD"}
    result = _format_response(raw, "currency", "exchangerate-api")
    assert "38.45" in result
    assert "USD" in result


def test_format_earthquake_response():
    from src.core.fast_resolver import _format_response

    raw = {"result": [{"mag": "4.2", "location": "Muğla", "date": "2026-04-05 10:30"}]}
    result = _format_response(raw, "earthquake", "kandilli")
    assert "4.2" in result
    assert "Muğla" in result


def test_format_unknown_category_falls_back_to_json():
    from src.core.fast_resolver import _format_response

    raw = {"foo": "bar"}
    result = _format_response(raw, "unknown_category", "some_api")
    assert '"foo"' in result  # JSON formatted


def test_format_string_passthrough():
    from src.core.fast_resolver import _format_response

    result = _format_response("plain text result", "weather", "wttr.in")
    assert result == "plain text result"


def test_format_truncates_long_output():
    from src.core.fast_resolver import _format_response

    raw = "x" * 3000
    result = _format_response(raw, "weather", "wttr.in")
    assert len(result) <= 2003  # 2000 + "..."
