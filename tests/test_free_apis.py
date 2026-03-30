"""Tests for the free API registry and tools."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.free_apis import (
    API_REGISTRY,
    FreeAPI,
    call_api,
    find_apis,
    get_api,
)


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# find_apis
# ---------------------------------------------------------------------------


def test_find_apis_by_category():
    results = find_apis(category="weather")
    assert len(results) >= 2
    assert all(r.category == "weather" for r in results)


def test_find_apis_by_category_case_insensitive():
    results = find_apis(category="WEATHER")
    assert len(results) >= 2


def test_find_apis_by_query():
    results = find_apis(query="wikipedia")
    assert len(results) >= 1
    names = [r.name for r in results]
    assert any("Wikipedia" in n for n in names)


def test_find_apis_no_match():
    results = find_apis(query="nonexistent_api_xyz")
    assert results == []


def test_find_apis_no_filters_returns_all():
    results = find_apis()
    assert len(results) == len(API_REGISTRY)


def test_find_apis_category_and_query():
    results = find_apis(category="currency", query="frankfurter")
    assert len(results) == 1
    assert results[0].name == "Frankfurter"


# ---------------------------------------------------------------------------
# get_api
# ---------------------------------------------------------------------------


def test_get_api_by_name():
    api = get_api("wttr.in")
    assert api is not None
    assert api.name == "wttr.in"
    assert api.category == "weather"


def test_get_api_case_insensitive():
    api = get_api("WTTR.IN")
    assert api is not None


def test_get_api_unknown():
    api = get_api("does_not_exist")
    assert api is None


# ---------------------------------------------------------------------------
# call_api (mocked HTTP)
# ---------------------------------------------------------------------------


def _mock_session(status=200, text='{"ok": true}'):
    """Build a fully mocked aiohttp ClientSession."""
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.text = AsyncMock(return_value=text)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_sess = AsyncMock()
    mock_sess.get = MagicMock(return_value=mock_resp)
    mock_sess.__aenter__ = AsyncMock(return_value=mock_sess)
    mock_sess.__aexit__ = AsyncMock(return_value=False)
    return mock_sess


def test_call_api_success():
    api = get_api("wttr.in")
    assert api is not None

    session = _mock_session(200, '{"current": "sunny"}')
    with patch("src.tools.free_apis.aiohttp.ClientSession", return_value=session):
        result = _run(call_api(api))

    assert "sunny" in result


def test_call_api_http_error():
    api = get_api("wttr.in")
    assert api is not None

    session = _mock_session(500)
    with patch("src.tools.free_apis.aiohttp.ClientSession", return_value=session):
        result = _run(call_api(api))

    assert "HTTP 500" in result


def test_call_api_missing_key():
    api = get_api("GNews")
    assert api is not None

    with patch.dict("os.environ", {}, clear=True):
        result = _run(call_api(api))

    assert "GNEWS_API_KEY not set" in result


def test_call_api_truncates_large_response():
    api = get_api("wttr.in")
    assert api is not None

    session = _mock_session(200, "x" * 6000)
    with patch("src.tools.free_apis.aiohttp.ClientSession", return_value=session):
        result = _run(call_api(api))

    assert len(result) < 6000
    assert "truncated" in result


# ---------------------------------------------------------------------------
# Tool wrappers (from __init__.py)
# ---------------------------------------------------------------------------


def test_tool_api_lookup_returns_json():
    from src.tools import TOOL_REGISTRY

    fn = TOOL_REGISTRY["api_lookup"]["function"]
    result = _run(fn(category="weather"))
    parsed = json.loads(result)
    assert isinstance(parsed, list)
    assert len(parsed) >= 2
    assert parsed[0]["category"] == "weather"


def test_tool_api_lookup_no_args():
    from src.tools import TOOL_REGISTRY

    fn = TOOL_REGISTRY["api_lookup"]["function"]
    result = _run(fn())
    assert "Available API categories" in result


def test_tool_api_call_unknown():
    from src.tools import TOOL_REGISTRY

    fn = TOOL_REGISTRY["api_call"]["function"]
    result = _run(fn(api_name="nonexistent"))
    assert "Unknown API" in result


def test_tool_api_lookup_query():
    from src.tools import TOOL_REGISTRY

    fn = TOOL_REGISTRY["api_lookup"]["function"]
    result = _run(fn(query="exchange"))
    parsed = json.loads(result)
    assert isinstance(parsed, list)
    assert len(parsed) >= 1
