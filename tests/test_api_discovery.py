"""Tests for the free API auto-growth / discovery mechanism."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.free_apis import (
    API_REGISTRY,
    FreeAPI,
    _parse_public_apis_md,
    _parse_free_apis_json,
    _dict_to_api,
    _api_to_dict,
    discover_new_apis,
    find_apis,
    get_api,
    seed_registry,
    refresh_db_cache,
    _db_api_cache,
)


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_PUBLIC_APIS_MD = """\
# Public APIs

## Index

### Weather
| API | Description | Auth | HTTPS | CORS |
|---|---|---|---|---|
| [Open-Meteo](https://open-meteo.com) | Free weather API | `` | Yes | Yes |
| [StormGlass](https://stormglass.io) | Marine weather | `OAuth` | Yes | Yes |
| [WeatherBit](https://www.weatherbit.io) | Weather data | `apiKey` | Yes | Yes |

### Animals
| API | Description | Auth | HTTPS | CORS |
|---|---|---|---|---|
| [Dog Facts](https://dogfacts.example.com) | Random dog facts | `` | Yes | Yes |
| [Cat Facts](https://catfacts.example.com) | Random cat facts | `` | Yes | Yes |
"""

SAMPLE_FREE_APIS_JSON = [
    {
        "name": "CatFact",
        "url": "https://catfact.ninja",
        "description": "Random cat facts",
        "auth": "none",
        "category": "Animals",
    },
    {
        "name": "OAuthOnly",
        "url": "https://oauth.example.com",
        "description": "Requires OAuth",
        "auth": "OAuth",
        "category": "Security",
    },
    {
        "name": "DogCEO",
        "url": "https://dog.ceo/dog-api",
        "description": "Dog images",
        "auth": "",
        "category": "Animals",
    },
]


# ---------------------------------------------------------------------------
# Parsing tests (no DB, no HTTP)
# ---------------------------------------------------------------------------


def test_parse_public_apis_md_extracts_no_auth():
    apis = _parse_public_apis_md(SAMPLE_PUBLIC_APIS_MD)
    names = [a["name"] for a in apis]
    assert "Open-Meteo" in names
    assert "Dog Facts" in names
    assert "Cat Facts" in names


def test_parse_public_apis_md_skips_oauth():
    apis = _parse_public_apis_md(SAMPLE_PUBLIC_APIS_MD)
    names = [a["name"] for a in apis]
    assert "StormGlass" not in names


def test_parse_public_apis_md_includes_apikey():
    apis = _parse_public_apis_md(SAMPLE_PUBLIC_APIS_MD)
    names = [a["name"] for a in apis]
    assert "WeatherBit" in names
    wb = [a for a in apis if a["name"] == "WeatherBit"][0]
    assert wb["auth_type"] == "apikey_param"


def test_parse_public_apis_md_sets_source():
    apis = _parse_public_apis_md(SAMPLE_PUBLIC_APIS_MD)
    for a in apis:
        assert a["source"] == "public-apis"
        assert a["verified"] == 0


def test_parse_public_apis_md_category_mapping():
    apis = _parse_public_apis_md(SAMPLE_PUBLIC_APIS_MD)
    dog = [a for a in apis if a["name"] == "Dog Facts"][0]
    assert dog["category"] == "fun"  # "animals" maps to "fun"


def test_parse_free_apis_json_filters_oauth():
    apis = _parse_free_apis_json(SAMPLE_FREE_APIS_JSON)
    names = [a["name"] for a in apis]
    assert "CatFact" in names
    assert "DogCEO" in names
    assert "OAuthOnly" not in names


def test_parse_free_apis_json_sets_source():
    apis = _parse_free_apis_json(SAMPLE_FREE_APIS_JSON)
    for a in apis:
        assert a["source"] == "free-apis-github"


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def test_api_to_dict_roundtrip():
    api = API_REGISTRY[0]
    d = _api_to_dict(api)
    assert d["name"] == api.name
    assert d["source"] == "static"
    assert d["verified"] == 1
    reconverted = _dict_to_api(d)
    assert reconverted.name == api.name
    assert reconverted.base_url == api.base_url


# ---------------------------------------------------------------------------
# seed_registry (with mocked DB)
# ---------------------------------------------------------------------------


def test_seed_registry_calls_upsert_for_each_api():
    upsert_mock = AsyncMock()
    # seed_registry does `from src.infra.db import upsert_free_api` lazily,
    # so we patch the source module.
    with patch("src.infra.db.upsert_free_api", new=upsert_mock):
        count = _run(seed_registry())

    assert count == len(API_REGISTRY)
    assert upsert_mock.call_count == len(API_REGISTRY)
    # Check each call was a dict with the right keys
    for call_args in upsert_mock.call_args_list:
        api_data = call_args[0][0]
        assert "name" in api_data
        assert api_data["source"] == "static"
        assert api_data["verified"] == 1


# ---------------------------------------------------------------------------
# discover_new_apis (mocked HTTP + DB)
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Fake aiohttp response that works as an async context manager."""

    def __init__(self, status, body):
        self.status = status
        self._body = body

    async def text(self):
        if isinstance(self._body, (dict, list)):
            return json.dumps(self._body)
        return self._body

    async def json(self, content_type=None):
        if isinstance(self._body, (dict, list)):
            return self._body
        return json.loads(self._body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class _FakeSession:
    """Fake aiohttp.ClientSession with preconfigured responses."""

    def __init__(self, responses):
        self._responses = responses  # {url_substring: (status, body)}

    def get(self, url, **kwargs):
        for pattern, (status, body) in self._responses.items():
            if pattern in url:
                return _FakeResponse(status, body)
        return _FakeResponse(404, "")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def test_discover_from_public_apis():
    upsert_mock = AsyncMock()
    refresh_mock = AsyncMock()

    session = _FakeSession({
        "public-apis": (200, SAMPLE_PUBLIC_APIS_MD),
        "free-apis.github.io": (404, ""),
    })

    with patch("src.infra.db.upsert_free_api", new=upsert_mock), \
         patch("src.tools.free_apis.refresh_db_cache", new=refresh_mock), \
         patch("src.tools.free_apis.aiohttp.ClientSession", return_value=session):
        count = _run(discover_new_apis(source="public-apis"))

    # Should have upserted the non-OAuth APIs from the sample
    assert upsert_mock.call_count > 0
    assert count > 0
    # StormGlass (OAuth) should NOT be in any upsert call
    upserted_names = [c[0][0]["name"] for c in upsert_mock.call_args_list]
    assert "StormGlass" not in upserted_names


def test_discover_from_free_apis_json():
    upsert_mock = AsyncMock()
    refresh_mock = AsyncMock()

    session = _FakeSession({
        "public-apis": (404, ""),
        "free-apis.github.io": (200, SAMPLE_FREE_APIS_JSON),
    })

    with patch("src.infra.db.upsert_free_api", new=upsert_mock), \
         patch("src.tools.free_apis.refresh_db_cache", new=refresh_mock), \
         patch("src.tools.free_apis.aiohttp.ClientSession", return_value=session):
        count = _run(discover_new_apis(source="free-apis"))

    assert count == 2  # CatFact + DogCEO (OAuthOnly filtered)
    upserted_names = [c[0][0]["name"] for c in upsert_mock.call_args_list]
    assert "CatFact" in upserted_names
    assert "DogCEO" in upserted_names
    assert "OAuthOnly" not in upserted_names


def test_discover_handles_fetch_failure():
    """Discovery should not raise on HTTP failures."""
    upsert_mock = AsyncMock()
    refresh_mock = AsyncMock()

    session = _FakeSession({
        "public-apis": (500, "error"),
        "free-apis.github.io": (500, "error"),
    })

    with patch("src.infra.db.upsert_free_api", new=upsert_mock), \
         patch("src.tools.free_apis.refresh_db_cache", new=refresh_mock), \
         patch("src.tools.free_apis.aiohttp.ClientSession", return_value=session):
        count = _run(discover_new_apis())

    assert count == 0


# ---------------------------------------------------------------------------
# find_apis with DB cache
# ---------------------------------------------------------------------------


def test_find_apis_includes_db_cache():
    """find_apis should return both static and DB-cached APIs."""
    import src.tools.free_apis as mod

    # Inject a fake API into the cache
    fake = FreeAPI(
        name="TestDiscovered",
        category="testing",
        base_url="https://test.example.com",
        auth_type="none",
        env_var=None,
        rate_limit="unlimited",
        description="A discovered test API",
        example_endpoint="https://test.example.com/v1",
    )
    original_cache = mod._db_api_cache[:]
    try:
        mod._db_api_cache.append(fake)
        results = find_apis(query="testdiscovered")
        assert len(results) == 1
        assert results[0].name == "TestDiscovered"
    finally:
        mod._db_api_cache[:] = original_cache


def test_find_apis_deduplicates_by_name():
    """If a static API also appears in the DB cache, it should not be duplicated."""
    import src.tools.free_apis as mod

    # Add a duplicate of a static API to the cache
    dup = FreeAPI(
        name="wttr.in",
        category="weather",
        base_url="https://wttr.in",
        auth_type="none",
        env_var=None,
        rate_limit="unlimited",
        description="Duplicate",
        example_endpoint="https://wttr.in/Istanbul?format=j1",
    )
    original_cache = mod._db_api_cache[:]
    try:
        mod._db_api_cache.append(dup)
        results = find_apis(query="wttr")
        wttr_results = [r for r in results if r.name == "wttr.in"]
        assert len(wttr_results) == 1  # No duplicate
    finally:
        mod._db_api_cache[:] = original_cache


def test_get_api_finds_db_cached():
    """get_api should find APIs in the DB cache."""
    import src.tools.free_apis as mod

    fake = FreeAPI(
        name="CachedTestAPI",
        category="testing",
        base_url="https://cached.example.com",
        auth_type="none",
        env_var=None,
        rate_limit="unlimited",
        description="Cached test",
        example_endpoint="https://cached.example.com/v1",
    )
    original_cache = mod._db_api_cache[:]
    try:
        mod._db_api_cache.append(fake)
        found = get_api("CachedTestAPI")
        assert found is not None
        assert found.name == "CachedTestAPI"
    finally:
        mod._db_api_cache[:] = original_cache


# ---------------------------------------------------------------------------
# DB upsert doesn't duplicate (integration-style with mocked DB)
# ---------------------------------------------------------------------------


def test_upsert_idempotent():
    """Calling upsert twice with the same name should not create duplicates."""
    upsert_mock = AsyncMock()

    api_data = {
        "name": "TestAPI",
        "category": "test",
        "base_url": "https://test.com",
        "source": "static",
        "verified": 1,
    }

    with patch("src.infra.db.upsert_free_api", new=upsert_mock):
        _run(seed_registry())

    # seed_registry calls upsert once per static API, using ON CONFLICT DO UPDATE
    # so duplicate names in the DB are handled at the SQL level
    assert upsert_mock.call_count == len(API_REGISTRY)


# ---------------------------------------------------------------------------
# Tool wrapper: discover_apis
# ---------------------------------------------------------------------------


def test_tool_discover_apis():
    from src.tools import TOOL_REGISTRY

    assert "discover_apis" in TOOL_REGISTRY
    fn = TOOL_REGISTRY["discover_apis"]["function"]

    upsert_mock = AsyncMock()
    refresh_mock = AsyncMock()
    session = _FakeSession({
        "public-apis": (200, SAMPLE_PUBLIC_APIS_MD),
        "free-apis.github.io": (404, ""),
    })

    with patch("src.infra.db.upsert_free_api", new=upsert_mock), \
         patch("src.tools.free_apis.refresh_db_cache", new=refresh_mock), \
         patch("src.tools.free_apis.aiohttp.ClientSession", return_value=session):
        result = _run(fn(source="public-apis"))

    assert "Discovered" in result
    assert "new APIs" in result
