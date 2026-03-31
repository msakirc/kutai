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
    _fetch_duty_pharmacies_eczaneler_gen_tr,
    _fetch_pharmacy_coordinates,
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
# _fetch_duty_pharmacies_eczaneler_gen_tr (mocked HTML)
# ---------------------------------------------------------------------------

# Realistic HTML snippet mimicking eczaneler.gen.tr structure
MOCK_ECZANELER_HTML = """
<html><body>
<table class="table table-striped mt-2">
<tr><td colspan="3">
<div class="row">
<div class="col-4 col-lg-3"><img class="ikon" src="/resimler/ikon/eczane.png"/> Eczane</div>
<div class="col-4 col-lg-6"><img class="ikon" src="/resimler/ikon/adres.png"/> Adres</div>
<div class="col-4 col-lg-3"><img class="ikon" src="/resimler/ikon/telefon.png"/> Telefon</div>
</div></td></tr>

<tr><td class="border-bottom" colspan="3"><div class="row" style="font-size:110%;">
<div class="col-lg-3"><a href="/eczane/ankara-cankaya-atakule-eczanesi"><span class="isim">Atakule Eczanesi</span></a></div>
<div class="col-lg-6">Hosdere Caddesi, Piyade Sokak No:26/A Cankaya / Ankara
<div class="py-2"><span class="text-success font-weight-bold">» </span><span class="font-italic">Atakule AVM yanı</span></div>
<div class="my-2"><span class="px-2 py-1 rounded bg-info text-white font-weight-bold">Çankaya</span></div>
</div>
<div class="col-lg-3 py-lg-2">0 (312) 441-56-59</div>
</div></td></tr>

<tr><td class="border-bottom" colspan="3"><div class="row" style="font-size:110%;">
<div class="col-lg-3"><a href="/eczane/ankara-kecioren-yildiz-eczanesi"><span class="isim">Yıldız Eczanesi</span></a></div>
<div class="col-lg-6">Etlik Caddesi No:12 Kecioren / Ankara
<div class="my-2"><span class="px-2 py-1 rounded bg-info text-white font-weight-bold">Keçiören</span></div>
</div>
<div class="col-lg-3 py-lg-2">0 (312) 555-12-34</div>
</div></td></tr>

<tr><td class="border-bottom" colspan="3"><div class="row" style="font-size:110%;">
<div class="col-lg-3"><a href="/eczane/ankara-cankaya-lalem-eczanesi"><span class="isim">Lalem Eczanesi</span></a></div>
<div class="col-lg-6">Huzur Mahallesi, 1066.Cadde No:35 Balgat / Çankaya / Ankara
<div class="my-2"><span class="px-2 py-1 rounded bg-info text-white font-weight-bold">Çankaya</span></div>
</div>
<div class="col-lg-3 py-lg-2">0 (312) 287-45-67</div>
</div></td></tr>
</table>
</body></html>
"""


def _make_scrape_result(html, status=200, ok=True):
    """Create a mock ScrapeResult."""
    mock = MagicMock()
    mock.html = html
    mock.status = status
    mock.ok = ok
    return mock


def test_eczaneler_gen_tr_parses_pharmacies():
    """Should parse pharmacy name, address, district, phone, detail_url from HTML."""
    with patch(
        "src.tools.scraper.scrape_url",
        new_callable=AsyncMock,
        return_value=_make_scrape_result(MOCK_ECZANELER_HTML),
    ):
        results = _run(_fetch_duty_pharmacies_eczaneler_gen_tr("ankara"))

    assert len(results) == 3
    # First pharmacy
    assert results[0]["pharmacyName"] == "Atakule Eczanesi"
    assert "Hosdere" in results[0]["address"] or "Piyade" in results[0]["address"]
    assert results[0]["district"] == "Çankaya"
    assert "441-56-59" in results[0]["phone"]
    assert results[0]["detail_url"] == "https://www.eczaneler.gen.tr/eczane/ankara-cankaya-atakule-eczanesi"
    # Second pharmacy
    assert results[1]["pharmacyName"] == "Yıldız Eczanesi"
    assert results[1]["district"] == "Keçiören"


def test_eczaneler_gen_tr_skips_header_rows():
    """Header rows (without span.isim) should be skipped."""
    with patch(
        "src.tools.scraper.scrape_url",
        new_callable=AsyncMock,
        return_value=_make_scrape_result(MOCK_ECZANELER_HTML),
    ):
        results = _run(_fetch_duty_pharmacies_eczaneler_gen_tr("ankara"))

    # Should not include "Eczane" header text as a pharmacy
    names = [r["pharmacyName"] for r in results]
    assert "Eczane" not in names


def test_eczaneler_gen_tr_date_in_url():
    """URL should include today's date as tarih parameter."""
    from datetime import datetime

    call_args = {}

    async def mock_scrape(url, **kwargs):
        call_args["url"] = url
        return _make_scrape_result("<html><body></body></html>")

    with patch("src.tools.scraper.scrape_url", side_effect=mock_scrape):
        _run(_fetch_duty_pharmacies_eczaneler_gen_tr("istanbul"))

    today = datetime.now().strftime("%Y-%m-%d")
    assert f"tarih={today}" in call_args["url"]
    assert "nobetci-istanbul" in call_args["url"]


def test_eczaneler_gen_tr_http_failure():
    """Should return empty list on HTTP failure."""
    with patch(
        "src.tools.scraper.scrape_url",
        new_callable=AsyncMock,
        return_value=_make_scrape_result("", status=403, ok=False),
    ):
        results = _run(_fetch_duty_pharmacies_eczaneler_gen_tr("ankara"))

    assert results == []


# HTML with three tab panes — each holding exactly one pharmacy
MOCK_ECZANELER_TABBED_HTML = """
<html><body>
<div class="tab-content" id="nav-tabContent">
  <div class="tab-pane" id="nav-dun">
    <table class="table table-striped">
      <tr><td colspan="3"><div class="row">
        <div class="col-lg-3"><a href="/eczane/test-dun-eczanesi"><span class="isim">Dun Eczanesi</span></a></div>
        <div class="col-lg-6">Dun Caddesi No:1
          <div class="my-2"><span class="px-2 py-1 rounded bg-info text-white font-weight-bold">Kadikoy</span></div>
        </div>
        <div class="col-lg-3 py-lg-2">0216 111-0000</div>
      </div></td></tr>
    </table>
  </div>
  <div class="tab-pane show active" id="nav-bugun">
    <table class="table table-striped">
      <tr><td colspan="3"><div class="row">
        <div class="col-lg-3"><a href="/eczane/test-bugun-eczanesi"><span class="isim">Bugun Eczanesi</span></a></div>
        <div class="col-lg-6">Bugun Caddesi No:2
          <div class="my-2"><span class="px-2 py-1 rounded bg-info text-white font-weight-bold">Besiktas</span></div>
        </div>
        <div class="col-lg-3 py-lg-2">0212 222-0000</div>
      </div></td></tr>
    </table>
  </div>
  <div class="tab-pane" id="nav-yarin">
    <table class="table table-striped">
      <tr><td colspan="3"><div class="row">
        <div class="col-lg-3"><a href="/eczane/test-yarin-eczanesi"><span class="isim">Yarin Eczanesi</span></a></div>
        <div class="col-lg-6">Yarin Caddesi No:3
          <div class="my-2"><span class="px-2 py-1 rounded bg-info text-white font-weight-bold">Sisli</span></div>
        </div>
        <div class="col-lg-3 py-lg-2">0212 333-0000</div>
      </div></td></tr>
    </table>
  </div>
</div>
</body></html>
"""


def test_eczaneler_scrapes_only_today_tab():
    """When tab='bugun', should return only today's pharmacy, not yesterday's or tomorrow's."""
    with patch(
        "src.tools.scraper.scrape_url",
        new_callable=AsyncMock,
        return_value=_make_scrape_result(MOCK_ECZANELER_TABBED_HTML),
    ):
        results = _run(_fetch_duty_pharmacies_eczaneler_gen_tr("istanbul", tab="bugun"))

    assert len(results) == 1
    assert results[0]["pharmacyName"] == "Bugun Eczanesi"
    assert results[0]["district"] == "Besiktas"
    names = [r["pharmacyName"] for r in results]
    assert "Dun Eczanesi" not in names
    assert "Yarin Eczanesi" not in names


def test_eczaneler_scrapes_tomorrow_tab():
    """When tab='yarin', should return only tomorrow's pharmacy."""
    with patch(
        "src.tools.scraper.scrape_url",
        new_callable=AsyncMock,
        return_value=_make_scrape_result(MOCK_ECZANELER_TABBED_HTML),
    ):
        results = _run(_fetch_duty_pharmacies_eczaneler_gen_tr("istanbul", tab="yarin"))

    assert len(results) == 1
    assert results[0]["pharmacyName"] == "Yarin Eczanesi"
    assert results[0]["district"] == "Sisli"
    names = [r["pharmacyName"] for r in results]
    assert "Dun Eczanesi" not in names
    assert "Bugun Eczanesi" not in names


# ---------------------------------------------------------------------------
# _fetch_pharmacy_coordinates (mocked)
# ---------------------------------------------------------------------------

MOCK_DETAIL_HTML_WITH_MAP = """
<html><body>
<h1>Atakule Eczanesi</h1>
<iframe src="https://maps.google.com/maps?q=39.8857,32.8543&output=embed"></iframe>
</body></html>
"""

MOCK_DETAIL_HTML_NO_MAP = """
<html><body>
<h1>Some Pharmacy</h1>
<p>No map here</p>
</body></html>
"""


def test_fetch_pharmacy_coordinates_from_maps_embed():
    """Should extract coordinates from Google Maps embed."""
    with patch(
        "src.tools.scraper.scrape_url",
        new_callable=AsyncMock,
        return_value=_make_scrape_result(MOCK_DETAIL_HTML_WITH_MAP),
    ):
        coords = _run(_fetch_pharmacy_coordinates("https://www.eczaneler.gen.tr/eczane/test"))

    assert coords is not None
    assert abs(coords[0] - 39.8857) < 0.001
    assert abs(coords[1] - 32.8543) < 0.001


def test_fetch_pharmacy_coordinates_no_map():
    """Should return None when no map embed found."""
    with patch(
        "src.tools.scraper.scrape_url",
        new_callable=AsyncMock,
        return_value=_make_scrape_result(MOCK_DETAIL_HTML_NO_MAP),
    ):
        coords = _run(_fetch_pharmacy_coordinates("https://www.eczaneler.gen.tr/eczane/test"))

    assert coords is None


def test_fetch_pharmacy_coordinates_empty_url():
    """Should return None for empty URL without making requests."""
    coords = _run(_fetch_pharmacy_coordinates(""))
    assert coords is None


def test_fetch_pharmacy_coordinates_data_attributes():
    """Should extract coordinates from data-lat/data-lng attributes."""
    html = '<html><body><div data-lat="40.123" data-lng="29.456"></div></body></html>'
    with patch(
        "src.tools.scraper.scrape_url",
        new_callable=AsyncMock,
        return_value=_make_scrape_result(html),
    ):
        coords = _run(_fetch_pharmacy_coordinates("https://www.eczaneler.gen.tr/eczane/test"))

    assert coords is not None
    assert abs(coords[0] - 40.123) < 0.001
    assert abs(coords[1] - 29.456) < 0.001


# ---------------------------------------------------------------------------
# Pharmacy dataclass
# ---------------------------------------------------------------------------

def test_pharmacy_dataclass_detail_url():
    """Pharmacy dataclass should support detail_url field."""
    p = Pharmacy(
        name="Test", address="Addr", district="Dist",
        detail_url="https://www.eczaneler.gen.tr/eczane/test"
    )
    assert p.detail_url == "https://www.eczaneler.gen.tr/eczane/test"
    assert p.distance_km == -1.0  # default


def test_pharmacy_dataclass_defaults():
    """Pharmacy should have sensible defaults for optional fields."""
    p = Pharmacy(name="Test", address="Addr", district="Dist")
    assert p.phone == ""
    assert p.lat == 0.0
    assert p.lon == 0.0
    assert p.distance_km == -1.0
    assert p.walking_min == -1.0
    assert p.driving_min == -1.0
    assert p.detail_url == ""


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


def test_find_nearest_pharmacy_no_district_uses_eczaneler_gen_tr(monkeypatch):
    """City-wide query (no district) should try eczaneler.gen.tr scraper."""
    monkeypatch.delenv("USER_DISTRICT", raising=False)
    monkeypatch.delenv("USER_LAT", raising=False)
    monkeypatch.delenv("USER_LON", raising=False)

    mock_pharmacies = [
        {"pharmacyName": "Merkez Eczanesi", "address": "Kizilay Mah.", "district": "Cankaya", "phone": ""},
        {"pharmacyName": "Yildiz Eczanesi", "address": "Ulus Mah.", "district": "Altindag", "phone": ""},
    ]

    with patch(
        "src.tools.pharmacy._fetch_duty_pharmacies_eczaneler_gen_tr",
        new_callable=AsyncMock,
        return_value=mock_pharmacies,
    ):
        result = _run(find_nearest_pharmacy(city="ankara", district=""))

    assert "Nobetci Eczaneler" in result
    assert "Ankara" in result
    assert "Merkez Eczanesi" in result
    assert "Yildiz Eczanesi" in result


def test_find_nearest_pharmacy_no_city(monkeypatch):
    """Should return helpful message when no city is given."""
    monkeypatch.delenv("USER_CITY", raising=False)
    monkeypatch.delenv("USER_DISTRICT", raising=False)

    with patch(
        "src.tools.pharmacy._get_user_city_district",
        new_callable=AsyncMock,
        return_value=("", ""),
    ):
        result = _run(find_nearest_pharmacy(city="", district=""))

    assert "Sehir" in result


def test_find_nearest_pharmacy_web_fallback(monkeypatch):
    """City-wide query with no scraper results should fall back to web search."""
    monkeypatch.delenv("USER_DISTRICT", raising=False)
    monkeypatch.delenv("USER_LAT", raising=False)
    monkeypatch.delenv("USER_LON", raising=False)

    with patch(
        "src.tools.pharmacy._fetch_duty_pharmacies_eczaneler_gen_tr",
        new_callable=AsyncMock,
        return_value=[],
    ), patch(
        "src.tools.pharmacy._fetch_duty_pharmacies_web",
        new_callable=AsyncMock,
        return_value=[{"_raw_search": True, "text": "Pharmacy X - Kizilay, Pharmacy Y - Ulus"}],
    ):
        result = _run(find_nearest_pharmacy(city="ankara", district=""))

    assert "Nobetci Eczaneler" in result
    assert "web aramasi" in result
    assert "Pharmacy X" in result


def test_find_nearest_pharmacy_no_results(monkeypatch):
    """Should return 'no pharmacies found' when API returns empty."""
    monkeypatch.setenv("NOSYAPI_KEY", "test_key")

    with patch(
        "src.tools.pharmacy._fetch_duty_pharmacies_nosyapi",
        new_callable=AsyncMock,
        return_value=[],
    ), patch(
        "src.tools.pharmacy._fetch_duty_pharmacies_eczaneler_gen_tr",
        new_callable=AsyncMock,
        return_value=[],
    ), patch(
        "src.tools.pharmacy._fetch_duty_pharmacies_web",
        new_callable=AsyncMock,
        return_value=[],
    ):
        result = _run(find_nearest_pharmacy(city="istanbul", district="kadikoy"))

    assert "No duty pharmacies found" in result


def test_find_nearest_pharmacy_detail_url_coordinate_fetch(monkeypatch):
    """Should try fetching coordinates from detail_url before geocoding."""
    monkeypatch.setenv("USER_LAT", "39.925")
    monkeypatch.setenv("USER_LON", "32.855")

    mock_pharmacies = [
        {
            "pharmacyName": "Test Eczanesi",
            "address": "Test Adres",
            "district": "Cankaya",
            "phone": "0312-111-2233",
            "detail_url": "https://www.eczaneler.gen.tr/eczane/test",
        },
    ]

    with patch(
        "src.tools.pharmacy._fetch_duty_pharmacies_eczaneler_gen_tr",
        new_callable=AsyncMock,
        return_value=mock_pharmacies,
    ), patch(
        "src.tools.pharmacy._fetch_pharmacy_coordinates",
        new_callable=AsyncMock,
        return_value=(39.886, 32.854),
    ) as mock_coord_fetch, patch(
        "src.tools.pharmacy._geocode_address",
        new_callable=AsyncMock,
        return_value=None,
    ) as mock_geocode, patch(
        "src.tools.pharmacy._get_osrm_distance",
        new_callable=AsyncMock,
        return_value=(-1.0, -1.0),
    ):
        result = _run(find_nearest_pharmacy(city="ankara", district=""))

    # Should have called coordinate fetch
    mock_coord_fetch.assert_called_once_with("https://www.eczaneler.gen.tr/eczane/test")
    # Should NOT have called geocode (coordinates already found)
    mock_geocode.assert_not_called()
    assert "Test Eczanesi" in result
    assert "km" in result
