import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_smart_search_routes_to_api_first():
    """smart_search should try API registry before web search."""
    from src.tools.smart_search import smart_search

    with patch("src.tools.smart_search._try_api_registry", new_callable=AsyncMock) as mock_api:
        mock_api.return_value = "22C, sunny in Istanbul"
        result = await smart_search("Istanbul weather")

    assert "22" in result
    assert "sunny" in result
    mock_api.assert_called_once()


@pytest.mark.asyncio
async def test_smart_search_falls_through_to_web():
    """If API registry has no match, fall through to web search."""
    from src.tools.smart_search import smart_search

    with patch("src.tools.smart_search._try_api_registry", new_callable=AsyncMock) as mock_api, \
         patch("src.tools.smart_search._try_web_search", new_callable=AsyncMock) as mock_web:
        mock_api.return_value = None
        mock_web.return_value = "Some web result about Python sorting"
        result = await smart_search("How to sort a list in Python")

    assert result is not None
    mock_web.assert_called_once()


@pytest.mark.asyncio
async def test_smart_search_includes_source_attribution():
    """Result should include source info."""
    from src.tools.smart_search import smart_search

    with patch("src.tools.smart_search._try_api_registry", new_callable=AsyncMock) as mock_api:
        mock_api.return_value = "22C in Istanbul (Source: wttr.in API)"
        result = await smart_search("Istanbul weather")

    assert "Source:" in result
