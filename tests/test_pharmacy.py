"""Tests for src/tools/pharmacy.py — duty pharmacy finder."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.pharmacy import (
    Pharmacy,
    _haversine,
    _geocode_address,
    _get_osrm_distance,
    _get_user_location,
    find_nearest_pharmacy,
)


def _run(coro):
    """Helper to run async coroutines in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# _haversine
# ---------------------------------------------------------------------------

def test_haversine_kadikoy_to_taksim():
    """Kadikoy to Taksim should be roughly 6-8 km straight-line."""
    kadikoy = (40.9828, 29.0294)
    taksim = (41.0370, 28.9850)
    dist = _haversine(*kadikoy, *taksim)
    assert 5.0 < dist < 9.0, f"Expected 6-8 km, got {dist:.2f} km"


def test_haversine_same_point():
    """Same point should be 0 km."""
    dist = _haversine(41.0, 29.0, 41.0, 29.0)
    assert dist == 0.0


def test_haversine_known_distance():
    """London to Paris ~340 km."""
    london = (51.5074, -0.1278)
    paris = (48.8566, 2.3522)
    dist = _haversine(*london, *paris)
    assert 330 < dist < 360, f"Expected ~340 km, got {dist:.1f} km"


# ---------------------------------------------------------------------------
# _get_user_location
# ---------------------------------------------------------------------------

def test_get_user_location_with_env(monkeypatch):
    """Returns (lat, lon) when env vars are set."""
    monkeypatch.setenv("USER_LAT", "40.9828")
    monkeypatch.setenv("USER_LON", "29.0294")
    result = _get_user_location()
    assert result == (40.9828, 29.0294)


def test_get_user_location_not_set(monkeypatch):
    """Returns None when env vars are missing."""
    monkeypatch.delenv("USER_LAT", raising=False)
    monkeypatch.delenv("USER_LON", raising=False)
    result = _get_user_location()
    assert result is None


def test_get_user_location_invalid(monkeypatch):
    """Returns None when env vars are invalid."""
    monkeypatch.setenv("USER_LAT", "not_a_number")
    monkeypatch.setenv("USER_LON", "29.0")
    result = _get_user_location()
    assert result is None


# ---------------------------------------------------------------------------
# _geocode_address (mocked)
# ---------------------------------------------------------------------------

def _make_mock_session(status, json_data):
    """Create a mocked aiohttp session with given response."""
    mock_response = AsyncMock()
    mock_response.status = status
    mock_response.json = AsyncMock(return_value=json_data)
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    return mock_session


def test_geocode_address_success():
    """Should return (lat, lon) on successful geocode."""
    mock_session = _make_mock_session(200, [{"lat": "40.9828", "lon": "29.0294"}])

    with patch("src.tools.pharmacy.aiohttp.ClientSession", return_value=mock_session):
        result = _run(_geocode_address("Kadikoy, Istanbul"))

    assert result == (40.9828, 29.0294)


def test_geocode_address_no_results():
    """Should return None when no results found."""
    mock_session = _make_mock_session(200, [])

    with patch("src.tools.pharmacy.aiohttp.ClientSession", return_value=mock_session):
        result = _run(_geocode_address("Nonexistent Place"))

    assert result is None


# ---------------------------------------------------------------------------
# _get_osrm_distance (mocked)
# ---------------------------------------------------------------------------

def test_osrm_distance_success():
    """Should return (distance_km, duration_min) on success."""
    mock_session = _make_mock_session(200, {
        "routes": [{"distance": 5200.0, "duration": 3900.0}]
    })

    with patch("src.tools.pharmacy.aiohttp.ClientSession", return_value=mock_session):
        dist_km, dur_min = _run(_get_osrm_distance(40.98, 29.03, 41.04, 28.98, "foot"))

    assert dist_km == 5.2
    assert dur_min == 65.0


def test_osrm_distance_failure():
    """Should return (-1, -1) on HTTP error."""
    mock_session = _make_mock_session(500, {})

    with patch("src.tools.pharmacy.aiohttp.ClientSession", return_value=mock_session):
        dist_km, dur_min = _run(_get_osrm_distance(40.98, 29.03, 41.04, 28.98))

    assert dist_km == -1.0
    assert dur_min == -1.0


# ---------------------------------------------------------------------------
# find_nearest_pharmacy (mocked)
# ---------------------------------------------------------------------------

MOCK_NOSYAPI_RESPONSE = [
    {
        "pharmacyName": "Sifa Eczanesi",
        "address": "Caferaga Mah. Moda Cad. No:12",
        "district": "Kadikoy",
        "phone": "0216 345 6789",
        "lat": "40.9850",
        "lng": "29.0250",
    },
    {
        "pharmacyName": "Hayat Eczanesi",
        "address": "Osmanaga Mah. Sogutlu Cesme Cad. No:5",
        "district": "Kadikoy",
        "phone": "0216 345 1234",
        "lat": "40.9900",
        "lng": "29.0300",
    },
]


def test_find_nearest_pharmacy_with_location(monkeypatch):
    """Should return sorted pharmacies with distances when user location is set."""
    monkeypatch.setenv("USER_LAT", "40.9828")
    monkeypatch.setenv("USER_LON", "29.0294")
    monkeypatch.setenv("NOSYAPI_KEY", "test_key")

    with patch(
        "src.tools.pharmacy._fetch_duty_pharmacies_nosyapi",
        new_callable=AsyncMock,
        return_value=MOCK_NOSYAPI_RESPONSE,
    ), patch(
        "src.tools.pharmacy._get_osrm_distance",
        new_callable=AsyncMock,
        return_value=(1.5, 18.0),
    ):
        result = _run(find_nearest_pharmacy(city="istanbul", district="kadikoy"))

    assert "Nobetci Eczaneler" in result
    assert "Sifa Eczanesi" in result
    assert "Hayat Eczanesi" in result
    assert "km" in result


def test_find_nearest_pharmacy_without_location(monkeypatch):
    """Should list pharmacies without distances when no user location."""
    monkeypatch.delenv("USER_LAT", raising=False)
    monkeypatch.delenv("USER_LON", raising=False)
    monkeypatch.setenv("NOSYAPI_KEY", "test_key")

    with patch(
        "src.tools.pharmacy._fetch_duty_pharmacies_nosyapi",
        new_callable=AsyncMock,
        return_value=MOCK_NOSYAPI_RESPONSE,
    ):
        result = _run(find_nearest_pharmacy(city="istanbul", district="kadikoy"))

    assert "Sifa Eczanesi" in result
    assert "Hayat Eczanesi" in result
    assert "USER_LAT" in result  # hint to set env vars
    # No distance info
    assert "Yuruyus" not in result


def test_find_nearest_pharmacy_no_district(monkeypatch):
    """Should return helpful message when no district is set."""
    monkeypatch.delenv("USER_DISTRICT", raising=False)

    result = _run(find_nearest_pharmacy(city="istanbul", district=""))

    assert "District not specified" in result
    assert "USER_DISTRICT" in result


def test_find_nearest_pharmacy_no_results(monkeypatch):
    """Should return 'no pharmacies found' when API returns empty."""
    monkeypatch.setenv("NOSYAPI_KEY", "test_key")

    with patch(
        "src.tools.pharmacy._fetch_duty_pharmacies_nosyapi",
        new_callable=AsyncMock,
        return_value=[],
    ), patch(
        "src.tools.pharmacy._fetch_duty_pharmacies_web",
        new_callable=AsyncMock,
        return_value=[],
    ):
        result = _run(find_nearest_pharmacy(city="istanbul", district="kadikoy"))

    assert "No duty pharmacies found" in result
