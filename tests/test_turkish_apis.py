"""Tests for Turkish API entries, routing skills, and MCP discovery."""

import asyncio
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.free_apis import (
    API_REGISTRY,
    FreeAPI,
    _discover_from_mcp_registry,
    find_apis,
    get_api,
)
from src.memory.seed_skills import SEED_SKILLS


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Part 1: All new Turkish APIs are in API_REGISTRY
# ---------------------------------------------------------------------------

EXPECTED_TURKISH_APIS = [
    "Nosyapi Pharmacy",
    "Kandilli Observatory",
    "Turkey Fuel Prices",
    "Diyanet Prayer Times",
    "Turkey Holidays",
    "Gold Price Turkey",
    "BIST Stock Data",
    "OSRM",
    "EnUygun Travel",
]


@pytest.mark.parametrize("api_name", EXPECTED_TURKISH_APIS)
def test_turkish_api_in_registry(api_name):
    """Each Turkish API must be present in API_REGISTRY."""
    api = get_api(api_name)
    assert api is not None, f"{api_name} not found in API_REGISTRY"


def test_nosyapi_requires_key():
    api = get_api("Nosyapi Pharmacy")
    assert api.auth_type == "apikey_param"
    assert api.env_var == "NOSYAPI_KEY"


def test_kandilli_no_auth():
    api = get_api("Kandilli Observatory")
    assert api.auth_type == "none"
    assert api.env_var is None


def test_collectapi_apis_share_key():
    """Turkey Fuel Prices, Gold Price Turkey, and BIST Stock Data all use COLLECTAPI_KEY."""
    for name in ["Turkey Fuel Prices", "Gold Price Turkey", "BIST Stock Data"]:
        api = get_api(name)
        assert api.env_var == "COLLECTAPI_KEY", f"{name} should use COLLECTAPI_KEY"
        assert api.auth_type == "apikey_header"


def test_osrm_is_geo_category():
    api = get_api("OSRM")
    assert api.category == "geo"


def test_enuygun_is_travel_category():
    api = get_api("EnUygun Travel")
    assert api.category == "travel"


def test_find_apis_health_includes_pharmacy():
    results = find_apis(category="health")
    names = [r.name for r in results]
    assert "Nosyapi Pharmacy" in names


def test_find_apis_earthquake_category():
    results = find_apis(category="earthquake")
    assert len(results) >= 1
    assert results[0].name == "Kandilli Observatory"


# ---------------------------------------------------------------------------
# Part 2: New routing skills are in SEED_SKILLS
# ---------------------------------------------------------------------------

EXPECTED_SKILLS = [
    "pharmacy_on_duty",
    "earthquake_data",
    "fuel_price_routing",
    "gold_price_routing",
    "map_directions_routing",
    "prayer_times_routing",
    "travel_ticket_routing",
    "turkish_holidays_routing",
]


@pytest.mark.parametrize("skill_name", EXPECTED_SKILLS)
def test_skill_in_seed_skills(skill_name):
    """Each new routing skill must be in SEED_SKILLS."""
    names = [s["name"] for s in SEED_SKILLS]
    assert skill_name in names, f"{skill_name} not found in SEED_SKILLS"


def test_pharmacy_skill_matches_nobetci_eczane():
    skill = next(s for s in SEED_SKILLS if s["name"] == "pharmacy_on_duty")
    pattern = skill["trigger_pattern"]
    assert re.search(pattern, "nöbetçi eczane", re.IGNORECASE)
    assert re.search(pattern, "pharmacy on duty istanbul", re.IGNORECASE)
    assert re.search(pattern, "en yakın eczane", re.IGNORECASE)


def test_earthquake_skill_matches_deprem():
    skill = next(s for s in SEED_SKILLS if s["name"] == "earthquake_data")
    pattern = skill["trigger_pattern"]
    assert re.search(pattern, "deprem oldu mu", re.IGNORECASE)
    assert re.search(pattern, "earthquake in Turkey", re.IGNORECASE)
    assert re.search(pattern, "kandilli son depremler", re.IGNORECASE)


def test_fuel_skill_matches_benzin():
    skill = next(s for s in SEED_SKILLS if s["name"] == "fuel_price_routing")
    pattern = skill["trigger_pattern"]
    assert re.search(pattern, "benzin fiyatı", re.IGNORECASE)
    assert re.search(pattern, "mazot ne kadar", re.IGNORECASE)


def test_gold_skill_matches_altin():
    skill = next(s for s in SEED_SKILLS if s["name"] == "gold_price_routing")
    pattern = skill["trigger_pattern"]
    assert re.search(pattern, "altın fiyatı", re.IGNORECASE)
    assert re.search(pattern, "çeyrek altın", re.IGNORECASE)
    assert re.search(pattern, "gram altın ne kadar", re.IGNORECASE)


def test_prayer_skill_matches_namaz():
    skill = next(s for s in SEED_SKILLS if s["name"] == "prayer_times_routing")
    pattern = skill["trigger_pattern"]
    assert re.search(pattern, "namaz vakitleri", re.IGNORECASE)
    assert re.search(pattern, "iftar saati", re.IGNORECASE)


def test_holiday_skill_matches_bayram():
    skill = next(s for s in SEED_SKILLS if s["name"] == "turkish_holidays_routing")
    pattern = skill["trigger_pattern"]
    assert re.search(pattern, "bayram ne zaman", re.IGNORECASE)
    assert re.search(pattern, "resmi tatil", re.IGNORECASE)


def test_travel_skill_matches_ucak():
    skill = next(s for s in SEED_SKILLS if s["name"] == "travel_ticket_routing")
    pattern = skill["trigger_pattern"]
    assert re.search(pattern, "uçak bileti", re.IGNORECASE)
    assert re.search(pattern, "enuygun", re.IGNORECASE)


def test_directions_skill_matches_yol_tarifi():
    skill = next(s for s in SEED_SKILLS if s["name"] == "map_directions_routing")
    pattern = skill["trigger_pattern"]
    assert re.search(pattern, "yol tarifi", re.IGNORECASE)
    assert re.search(pattern, "directions to airport", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Part 3: MCP registry parser with mocked markdown
# ---------------------------------------------------------------------------

SAMPLE_MCP_README = """\
# MCP Servers

## Implementations

### Reference Servers

- **Brave Search** - Web search using the Brave Search API [GitHub](https://github.com/example/brave-search)
- **PostgreSQL** - Read-only database access with schema inspection [GitHub](https://github.com/example/postgres)
- **Google Drive** - File access and search for Google Drive [GitHub](https://github.com/example/gdrive)
- **GitHub** - Repository management, file operations, and GitHub API integration [GitHub](https://github.com/example/github)
- **Weather Service** - Real-time weather data and forecasts [GitHub](https://github.com/example/weather)
"""


def test_mcp_registry_parser():
    """MCP discovery should parse markdown entries and categorize them."""
    from unittest.mock import MagicMock

    # Build mock response
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.text = AsyncMock(return_value=SAMPLE_MCP_README)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_sess = AsyncMock()
    mock_sess.get = MagicMock(return_value=mock_resp)
    mock_sess.__aenter__ = AsyncMock(return_value=mock_sess)
    mock_sess.__aexit__ = AsyncMock(return_value=False)

    upsert_mock = AsyncMock()

    with patch("src.tools.free_apis.aiohttp.ClientSession", return_value=mock_sess), \
         patch("src.infra.db.upsert_free_api", new=upsert_mock):
        count = _run(_discover_from_mcp_registry())

    assert count == 5
    upserted = {c[0][0]["name"]: c[0][0] for c in upsert_mock.call_args_list}

    assert "MCP: Brave Search" in upserted
    assert upserted["MCP: Brave Search"]["category"] == "search"

    assert "MCP: PostgreSQL" in upserted
    assert upserted["MCP: PostgreSQL"]["category"] == "database"

    assert "MCP: Google Drive" in upserted
    assert upserted["MCP: Google Drive"]["category"] == "storage"

    assert "MCP: GitHub" in upserted
    assert upserted["MCP: GitHub"]["category"] == "development"

    assert "MCP: Weather Service" in upserted
    assert upserted["MCP: Weather Service"]["category"] == "weather"

    # All should have source=mcp_registry
    for data in upserted.values():
        assert data["source"] == "mcp_registry"
        assert data["auth_type"] == "mcp"
        assert data["verified"] == 0


def test_mcp_registry_handles_failure():
    """MCP discovery should return 0 on HTTP failure, not raise."""
    mock_resp = AsyncMock()
    mock_resp.status = 500
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_sess = AsyncMock()
    mock_sess.get = MagicMock(return_value=mock_resp)
    mock_sess.__aenter__ = AsyncMock(return_value=mock_sess)
    mock_sess.__aexit__ = AsyncMock(return_value=False)

    with patch("src.tools.free_apis.aiohttp.ClientSession", return_value=mock_sess):
        count = _run(_discover_from_mcp_registry())

    assert count == 0
