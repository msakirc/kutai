"""Tests for src/tools/pharmacy.py — duty pharmacy finder."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.pharmacy import (
    Pharmacy,
    _haversine,
    _geocode_address,
    _get_osrm_distance,
    _get_user_coords,
    _get_user_city_district,
    _fetch_duty_pharmacies_eczaneler_gen_tr,
    _fetch_pharmacy_coordinates,
    find_nearest_pharmacy,
    find_pharmacies_structured,
    format_pharmacy_message,
    build_pharmacy_buttons,
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
# _get_user_coords — DB-only, no .env fallback
# ---------------------------------------------------------------------------

def test_get_user_coords_reads_correct_db_keys():
    """Returns (lat, lon) when DB has location_lat / location_lon."""
    async def mock_get_user_pref(key, default=""):
        return {"location_lat": "40.9828", "location_lon": "29.0294"}.get(key, default)

    with patch("src.infra.db.get_user_pref", side_effect=mock_get_user_pref):
        result = _run(_get_user_coords())

    assert result == (40.9828, 29.0294)


def test_get_user_coords_no_env_fallback(monkeypatch):
    """Returns None even when USER_LAT/USER_LON env vars are set — no .env fallback."""
    monkeypatch.setenv("USER_LAT", "40.9828")
    monkeypatch.setenv("USER_LON", "29.0294")

    async def mock_get_user_pref(key, default=""):
        return default  # DB has nothing

    with patch("src.infra.db.get_user_pref", side_effect=mock_get_user_pref):
        result = _run(_get_user_coords())

    assert result is None


def test_get_user_coords_returns_none_when_db_empty():
    """Returns None when DB has no location keys."""
    async def mock_get_user_pref(key, default=""):
        return default

    with patch("src.infra.db.get_user_pref", side_effect=mock_get_user_pref):
        result = _run(_get_user_coords())

    assert result is None


# ---------------------------------------------------------------------------
# _get_user_city_district — DB-only, no .env fallback
# ---------------------------------------------------------------------------

def test_get_user_city_district_reads_correct_db_keys():
    """Returns (city, district) when DB has location_city / location_district."""
    async def mock_get_user_pref(key, default=""):
        return {"location_city": "ankara", "location_district": "cankaya"}.get(key, default)

    with patch("src.infra.db.get_user_pref", side_effect=mock_get_user_pref):
        city, district = _run(_get_user_city_district())

    assert city == "ankara"
    assert district == "cankaya"


def test_get_user_city_district_no_env_fallback(monkeypatch):
    """Returns ('', '') even when USER_CITY env var is set — no .env fallback."""
    monkeypatch.setenv("USER_CITY", "istanbul")
    monkeypatch.setenv("USER_DISTRICT", "kadikoy")

    async def mock_get_user_pref(key, default=""):
        return default  # DB has nothing

    with patch("src.infra.db.get_user_pref", side_effect=mock_get_user_pref):
        city, district = _run(_get_user_city_district())

    assert city == ""
    assert district == ""


def test_get_user_city_district_city_only():
    """Returns (city, '') when only city is in DB."""
    async def mock_get_user_pref(key, default=""):
        return {"location_city": "izmir"}.get(key, default)

    with patch("src.infra.db.get_user_pref", side_effect=mock_get_user_pref):
        city, district = _run(_get_user_city_district())

    assert city == "izmir"
    assert district == ""


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

MOCK_PHARMACY_LIST = [
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


def test_find_nearest_pharmacy_with_location():
    """Should return sorted pharmacies with distances when user location is in DB."""
    async def mock_get_user_pref(key, default=""):
        return {"location_lat": "40.9828", "location_lon": "29.0294"}.get(key, default)

    with patch(
        "src.tools.pharmacy._fetch_duty_pharmacies_eczaneler_gen_tr",
        new_callable=AsyncMock,
        return_value=MOCK_PHARMACY_LIST,
    ), patch(
        "src.tools.pharmacy._get_osrm_distance",
        new_callable=AsyncMock,
        return_value=(1.5, 18.0),
    ), patch("src.infra.db.get_user_pref", side_effect=mock_get_user_pref):
        result = _run(find_nearest_pharmacy(city="istanbul", district="kadikoy"))

    assert "Nobetci Eczaneler" in result
    assert "Sifa Eczanesi" in result
    assert "Hayat Eczanesi" in result
    assert "km" in result


def test_find_nearest_pharmacy_without_location():
    """Should list pharmacies without distances when no user location in DB."""
    async def mock_get_user_pref(key, default=""):
        return default  # DB empty

    with patch(
        "src.tools.pharmacy._fetch_duty_pharmacies_eczaneler_gen_tr",
        new_callable=AsyncMock,
        return_value=MOCK_PHARMACY_LIST,
    ), patch("src.infra.db.get_user_pref", side_effect=mock_get_user_pref):
        result = _run(find_nearest_pharmacy(city="istanbul", district="kadikoy"))

    assert "Sifa Eczanesi" in result
    assert "Hayat Eczanesi" in result
    # Hint message — no env var reference
    assert "konum" in result.lower() or "Telegram" in result
    # No distance info
    assert "Yuruyus" not in result


def test_find_nearest_pharmacy_no_district_uses_eczaneler_gen_tr():
    """City-wide query (no district) should try eczaneler.gen.tr scraper."""
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


def test_find_nearest_pharmacy_no_city():
    """Should return helpful message when no city is given."""
    with patch(
        "src.tools.pharmacy._get_user_city_district",
        new_callable=AsyncMock,
        return_value=("", ""),
    ):
        result = _run(find_nearest_pharmacy(city="", district=""))

    assert "Sehir" in result


def test_find_nearest_pharmacy_web_fallback():
    """City-wide query with no scraper results should fall back to web search."""
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


def test_find_nearest_pharmacy_no_results():
    """Should return 'no pharmacies found' when all sources return empty."""
    with patch(
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


def test_find_nearest_pharmacy_detail_url_coordinate_fetch():
    """Should try fetching coordinates from detail_url before geocoding."""
    async def mock_get_user_pref(key, default=""):
        return {"location_lat": "39.925", "location_lon": "32.855"}.get(key, default)

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
    ), patch("src.infra.db.get_user_pref", side_effect=mock_get_user_pref):
        result = _run(find_nearest_pharmacy(city="ankara", district=""))

    # Should have called coordinate fetch
    mock_coord_fetch.assert_called_once_with("https://www.eczaneler.gen.tr/eczane/test")
    # Should NOT have called geocode (coordinates already found)
    mock_geocode.assert_not_called()
    assert "Test Eczanesi" in result
    assert "km" in result


# ---------------------------------------------------------------------------
# find_pharmacies_structured
# ---------------------------------------------------------------------------

def test_find_pharmacies_structured_returns_sorted_list():
    """Should return Pharmacy list sorted by distance (closest first)."""
    mock_pharmacies_raw = [
        {
            "pharmacyName": "Uzak Eczane",
            "address": "Uzak Mah. No:5",
            "district": "Kadikoy",
            "phone": "0216 100-0000",
            "lat": "41.0200",   # further from user
            "lng": "29.0400",
            "detail_url": "",
        },
        {
            "pharmacyName": "Yakin Eczane",
            "address": "Yakin Mah. No:1",
            "district": "Besiktas",
            "phone": "0212 200-0000",
            "lat": "40.9840",   # closer to user
            "lng": "29.0300",
            "detail_url": "",
        },
    ]

    async def mock_get_user_pref(key, default=""):
        return {"location_lat": "40.9828", "location_lon": "29.0294",
                "location_city": "istanbul", "location_district": ""}.get(key, default)

    with patch(
        "src.tools.pharmacy._fetch_duty_pharmacies_eczaneler_gen_tr",
        new_callable=AsyncMock,
        return_value=mock_pharmacies_raw,
    ), patch(
        "src.tools.pharmacy._get_osrm_distance",
        new_callable=AsyncMock,
        return_value=(-1.0, -1.0),  # disable OSRM so haversine distances hold
    ), patch("src.infra.db.get_user_pref", side_effect=mock_get_user_pref):
        result = _run(find_pharmacies_structured(city="istanbul"))

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(p, Pharmacy) for p in result)
    # Closest first
    assert result[0].name == "Yakin Eczane"
    assert result[1].name == "Uzak Eczane"
    # Distances are calculated
    assert result[0].distance_km >= 0
    assert result[1].distance_km >= result[0].distance_km


def test_find_pharmacies_structured_no_city_returns_empty():
    """Should return empty list when city cannot be determined."""
    async def mock_get_user_pref(key, default=""):
        return default  # DB has nothing

    with patch("src.infra.db.get_user_pref", side_effect=mock_get_user_pref):
        result = _run(find_pharmacies_structured(city=""))

    assert result == []


def test_find_pharmacies_structured_scraper_failure_returns_empty():
    """Should return empty list when scraper and web search both fail."""
    with patch(
        "src.tools.pharmacy._fetch_duty_pharmacies_eczaneler_gen_tr",
        new_callable=AsyncMock,
        return_value=[],
    ), patch(
        "src.tools.pharmacy._fetch_duty_pharmacies_web",
        new_callable=AsyncMock,
        return_value=[],
    ):
        result = _run(find_pharmacies_structured(city="istanbul"))

    assert result == []


def test_find_pharmacies_structured_web_fallback_returns_empty():
    """Web search fallback (_raw_search) cannot produce Pharmacy objects — returns empty list."""
    with patch(
        "src.tools.pharmacy._fetch_duty_pharmacies_eczaneler_gen_tr",
        new_callable=AsyncMock,
        return_value=[],
    ), patch(
        "src.tools.pharmacy._fetch_duty_pharmacies_web",
        new_callable=AsyncMock,
        return_value=[{"_raw_search": True, "text": "Eczane X - Kizilay"}],
    ):
        result = _run(find_pharmacies_structured(city="ankara"))

    assert result == []


# ---------------------------------------------------------------------------
# format_pharmacy_message
# ---------------------------------------------------------------------------

def _make_pharmacies(n: int, with_distance: bool = False) -> list[Pharmacy]:
    """Helper: build a list of n Pharmacy objects."""
    pharmacies = []
    for i in range(1, n + 1):
        p = Pharmacy(
            name=f"Eczane {i}",
            address=f"Cadde {i} No:{i}",
            district="Cankaya",
            phone=f"0312 {i:03d}-0000",
            lat=39.9 + i * 0.01,
            lon=32.8 + i * 0.01,
            distance_km=float(i) if with_distance else -1.0,
            walking_min=float(i * 10) if with_distance else -1.0,
            driving_min=float(i * 3) if with_distance else -1.0,
        )
        pharmacies.append(p)
    return pharmacies


def test_format_pharmacy_message_top3():
    """Default (show_all=False) shows only top 3 and appends total count."""
    pharmacies = _make_pharmacies(5, with_distance=True)
    msg = format_pharmacy_message(pharmacies, show_all=False)

    assert "Eczane 1" in msg
    assert "Eczane 2" in msg
    assert "Eczane 3" in msg
    assert "Eczane 4" not in msg
    assert "Eczane 5" not in msg
    assert "Toplam 5 eczane bulundu" in msg


def test_format_pharmacy_message_show_all():
    """show_all=True shows all pharmacies and no 'Toplam' footer."""
    pharmacies = _make_pharmacies(5, with_distance=True)
    msg = format_pharmacy_message(pharmacies, show_all=True)

    for i in range(1, 6):
        assert f"Eczane {i}" in msg
    # No "show all" footer when showing everything
    assert "Toplam" not in msg


def test_format_pharmacy_message_exactly_3():
    """Exactly 3 pharmacies: no 'Toplam' footer even when show_all=False."""
    pharmacies = _make_pharmacies(3, with_distance=True)
    msg = format_pharmacy_message(pharmacies, show_all=False)

    assert "Eczane 1" in msg
    assert "Eczane 2" in msg
    assert "Eczane 3" in msg
    assert "Toplam" not in msg


def test_format_pharmacy_message_distance_fields():
    """When distance is set, includes km, walking min, and driving min."""
    p = Pharmacy(
        name="Test Eczane", address="Test Adres", district="Test",
        distance_km=1.5, walking_min=18.0, driving_min=5.0,
    )
    msg = format_pharmacy_message([p], show_all=True)

    assert "1.5 km" in msg
    assert "18 dk" in msg
    assert "5 dk" in msg


def test_format_pharmacy_message_no_distance():
    """When distance_km is -1, no distance line in output."""
    p = Pharmacy(name="Test Eczane", address="Test Adres", district="Test")
    msg = format_pharmacy_message([p], show_all=True)

    assert "km" not in msg
    assert "dk" not in msg


def test_format_pharmacy_message_empty_list():
    """Empty list returns a sensible fallback message."""
    msg = format_pharmacy_message([], show_all=False)
    assert msg  # not empty string
    assert "bulunamadi" in msg.lower() or "eczane" in msg.lower()


# ---------------------------------------------------------------------------
# build_pharmacy_buttons
# ---------------------------------------------------------------------------

def test_build_pharmacy_buttons_map_links():
    """Pharmacies with coordinates get Google Maps URL buttons."""
    pharmacies = [
        Pharmacy(name="A Eczane", address="Addr", district="D",
                 lat=39.9255, lon=32.8660, distance_km=1.2),
        Pharmacy(name="B Eczane", address="Addr2", district="D",
                 lat=40.0000, lon=33.0000, distance_km=2.5),
    ]

    rows = build_pharmacy_buttons(pharmacies, total_count=2)

    assert len(rows) == 2  # 2 pharmacies, no "show all" (total == shown)
    btn_a = rows[0][0]
    btn_b = rows[1][0]

    # URL buttons for pharmacies with coords
    assert btn_a.url is not None
    assert "39.9255" in btn_a.url and "32.866" in btn_a.url
    assert "maps.google.com" in btn_a.url or "google.com/maps" in btn_a.url

    assert btn_b.url is not None
    assert "40.0" in btn_b.url

    # Labels include name and distance
    assert "A Eczane" in btn_a.text
    assert "1.2 km" in btn_a.text
    assert "B Eczane" in btn_b.text


def test_build_pharmacy_buttons_no_coords_callback():
    """Pharmacies without coordinates get callback_data buttons."""
    pharmacies = [
        Pharmacy(name="No Coord Eczane", address="Addr", district="D",
                 lat=0.0, lon=0.0, distance_km=-1.0),
    ]

    rows = build_pharmacy_buttons(pharmacies, total_count=1)

    assert len(rows) == 1
    btn = rows[0][0]
    assert btn.url is None
    assert btn.callback_data == "pharm:search:1"


def test_build_pharmacy_buttons_show_all_button():
    """When total_count > len(pharmacies), a 'show all' button is appended."""
    pharmacies = _make_pharmacies(3, with_distance=True)
    # Coordinates set to non-zero so they get URL buttons
    for i, p in enumerate(pharmacies):
        p.lat = 39.9 + i * 0.01
        p.lon = 32.8 + i * 0.01

    rows = build_pharmacy_buttons(pharmacies, total_count=10)

    # 3 pharmacy rows + 1 "show all" row
    assert len(rows) == 4
    last_btn = rows[-1][0]
    assert last_btn.callback_data == "pharm:all"
    assert "10" in last_btn.text


def test_build_pharmacy_buttons_no_show_all_when_few():
    """No 'show all' button when total_count <= len(pharmacies)."""
    pharmacies = _make_pharmacies(3, with_distance=True)
    for i, p in enumerate(pharmacies):
        p.lat = 39.9 + i * 0.01
        p.lon = 32.8 + i * 0.01

    rows = build_pharmacy_buttons(pharmacies, total_count=3)

    assert len(rows) == 3  # exactly 3, no extra row
    for row in rows:
        assert row[0].callback_data != "pharm:all"


def test_build_pharmacy_buttons_empty_list():
    """Empty pharmacy list returns empty rows (no crash)."""
    rows = build_pharmacy_buttons([], total_count=0)
    assert rows == []
