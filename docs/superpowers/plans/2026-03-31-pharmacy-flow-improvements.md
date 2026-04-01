# Pharmacy Flow Improvements

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the pharmacy scraper (currently returns 3 days instead of current day), consolidate location to DB-only, fetch city-wide and let distance sorting handle relevance (no district filter), add structured return for bot integration with map buttons, and add a "tomorrow" option.

**Architecture:** pharmacy.py becomes the single module for pharmacy data + formatting. Location reads exclusively from `user_preferences` DB (keys: `location_lat`, `location_lon`, `location_city`, `location_district`). The `.env` USER_LAT/USER_LON/USER_CITY/USER_DISTRICT fallbacks and the mismatched DB key names (`lat`/`city` vs `location_lat`/`location_city`) are removed. telegram_bot.py is NOT modified in this plan — it already writes the correct `location_*` keys. The scraper fetches city-wide (no district filter) — when the user lives near district borders, adjacent-district pharmacies may be closer. Distance sorting makes district filtering unnecessary.

**Tech Stack:** aiohttp, BeautifulSoup/lxml, python-telegram-bot (InlineKeyboardButton for button builder only)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/tools/pharmacy.py` | Modify | Fix scraper selector, consolidate location, structured return, button builder, tomorrow support |
| `tests/test_pharmacy.py` | Modify | Tests for all changes |

**NOT modified:** `src/app/telegram_bot.py` — bot wiring will be done separately.

---

### Task 1: Fix Scraper — Only Fetch Current Day

**Files:**
- Modify: `src/tools/pharmacy.py:156-279` (`_fetch_duty_pharmacies_eczaneler_gen_tr`)
- Test: `tests/test_pharmacy.py`

**Bug:** `soup.find_all("table")` at line 179 matches all 3 tab panes (`#nav-dun`, `#nav-bugun`, `#nav-yarin`). Fix: target only `#nav-bugun table` for today, or a specified tab.

- [ ] **Step 1: Write failing test for day-specific scraping**

Add to `tests/test_pharmacy.py`:

```python
@pytest.mark.asyncio
async def test_eczaneler_scrapes_only_today_tab():
    """Scraper should only parse #nav-bugun table, not yesterday/tomorrow."""
    from unittest.mock import AsyncMock, patch
    from src.tools.pharmacy import _fetch_duty_pharmacies_eczaneler_gen_tr

    # HTML with 3 tab panes, each containing a different pharmacy
    html = '''<html><body>
    <div class="tab-content" id="nav-tabContent">
      <div class="tab-pane" id="nav-dun">
        <table class="table table-striped">
          <tr><td colspan="3"><div class="row">
            <div class="col-lg-3"><a href="/eczane/yesterday"><span class="isim">Yesterday Eczane</span></a></div>
            <div class="col-lg-6"><span class="bg-info">Kadıköy</span> Yesterday addr is here for testing</div>
            <div class="col-lg-3">111-1111</div>
          </div></td></tr>
        </table>
      </div>
      <div class="tab-pane show active" id="nav-bugun">
        <table class="table table-striped">
          <tr><td colspan="3"><div class="row">
            <div class="col-lg-3"><a href="/eczane/today"><span class="isim">Today Eczane</span></a></div>
            <div class="col-lg-6"><span class="bg-info">Kadıköy</span> Today addr is here for testing</div>
            <div class="col-lg-3">222-2222</div>
          </div></td></tr>
        </table>
      </div>
      <div class="tab-pane" id="nav-yarin">
        <table class="table table-striped">
          <tr><td colspan="3"><div class="row">
            <div class="col-lg-3"><a href="/eczane/tomorrow"><span class="isim">Tomorrow Eczane</span></a></div>
            <div class="col-lg-6"><span class="bg-info">Kadıköy</span> Tomorrow addr is here for testing</div>
            <div class="col-lg-3">333-3333</div>
          </div></td></tr>
        </table>
      </div>
    </div>
    </body></html>'''

    mock_result = AsyncMock()
    mock_result.ok = True
    mock_result.html = html

    with patch("src.tools.pharmacy.scrape_url", new_callable=AsyncMock, return_value=mock_result):
        result = await _fetch_duty_pharmacies_eczaneler_gen_tr("istanbul")

    names = [p["pharmacyName"] for p in result]
    assert "Today Eczane" in names
    assert "Yesterday Eczane" not in names
    assert "Tomorrow Eczane" not in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pharmacy.py::test_eczaneler_scrapes_only_today_tab -v`
Expected: FAIL — returns all 3 pharmacies

- [ ] **Step 3: Fix the scraper selector**

In `src/tools/pharmacy.py`, modify `_fetch_duty_pharmacies_eczaneler_gen_tr`. Add a `tab` parameter for tomorrow support, and change the table selection:

Replace line 179 (`for table in soup.find_all("table"):`) and surrounding code with:

```python
async def _fetch_duty_pharmacies_eczaneler_gen_tr(
    city: str, tab: str = "bugun"
) -> list[dict]:
    """Scrape duty pharmacies from eczaneler.gen.tr (no API key needed).

    Always fetches city-wide — no district filter. Distance sorting handles
    relevance (adjacent-district pharmacies may be closer near borders).

    Args:
        tab: Which day tab to scrape — "bugun" (today) or "yarin" (tomorrow).
    """
```

Remove the `district` parameter entirely — the function no longer filters by district. Remove the district filtering block (lines 263-268 in the original: the `if district:` / `_normalize_turkish` comparison). Also remove the `_normalize_turkish` function if it's no longer used anywhere else.

And replace the table loop:

```python
        # Target the specific day tab — page has #nav-dun, #nav-bugun, #nav-yarin
        tab_id = f"nav-{tab}"
        tab_pane = soup.find("div", id=tab_id)
        tables = tab_pane.find_all("table") if tab_pane else []
        if not tables:
            # Fallback: try all tables (old page structure without tabs)
            tables = soup.find_all("table")

        pharmacies = []
        for table in tables:
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pharmacy.py::test_eczaneler_scrapes_only_today_tab -v`
Expected: PASS

- [ ] **Step 5: Write test for tomorrow tab**

```python
@pytest.mark.asyncio
async def test_eczaneler_scrapes_tomorrow_tab():
    """Passing tab='yarin' should scrape only the tomorrow pane."""
    from unittest.mock import AsyncMock, patch
    from src.tools.pharmacy import _fetch_duty_pharmacies_eczaneler_gen_tr

    # Same 3-tab HTML as above
    html = '''<html><body>
    <div class="tab-content" id="nav-tabContent">
      <div class="tab-pane" id="nav-dun">
        <table class="table table-striped">
          <tr><td colspan="3"><div class="row">
            <div class="col-lg-3"><a href="/eczane/y"><span class="isim">Yesterday Eczane</span></a></div>
            <div class="col-lg-6"><span class="bg-info">Kadıköy</span> Yesterday addr is here for testing</div>
            <div class="col-lg-3">111</div>
          </div></td></tr>
        </table>
      </div>
      <div class="tab-pane show active" id="nav-bugun">
        <table class="table table-striped">
          <tr><td colspan="3"><div class="row">
            <div class="col-lg-3"><a href="/eczane/t"><span class="isim">Today Eczane</span></a></div>
            <div class="col-lg-6"><span class="bg-info">Kadıköy</span> Today addr is here for testing</div>
            <div class="col-lg-3">222</div>
          </div></td></tr>
        </table>
      </div>
      <div class="tab-pane" id="nav-yarin">
        <table class="table table-striped">
          <tr><td colspan="3"><div class="row">
            <div class="col-lg-3"><a href="/eczane/tm"><span class="isim">Tomorrow Eczane</span></a></div>
            <div class="col-lg-6"><span class="bg-info">Kadıköy</span> Tomorrow addr is here for testing</div>
            <div class="col-lg-3">333</div>
          </div></td></tr>
        </table>
      </div>
    </div>
    </body></html>'''

    mock_result = AsyncMock()
    mock_result.ok = True
    mock_result.html = html

    with patch("src.tools.pharmacy.scrape_url", new_callable=AsyncMock, return_value=mock_result):
        result = await _fetch_duty_pharmacies_eczaneler_gen_tr("istanbul", tab="yarin")  # no district param

    names = [p["pharmacyName"] for p in result]
    assert names == ["Tomorrow Eczane"]
```

- [ ] **Step 6: Run test**

Run: `python -m pytest tests/test_pharmacy.py::test_eczaneler_scrapes_tomorrow_tab -v`
Expected: PASS

- [ ] **Step 7: Run all existing pharmacy tests to check for regressions**

Run: `python -m pytest tests/test_pharmacy.py -v`
Expected: All PASS (existing tests use HTML without tab-pane divs, so fallback to `soup.find_all("table")` kicks in)

- [ ] **Step 8: Commit**

```bash
git add src/tools/pharmacy.py tests/test_pharmacy.py
git commit -m "fix(pharmacy): scrape only today's tab, not all 3 days — add tomorrow support"
```

---

### Task 2: Consolidate Location + Kill Nosyapi + Kill .env Fallbacks

**Files:**
- Modify: `src/tools/pharmacy.py:97-153` (location functions + Nosyapi)
- Test: `tests/test_pharmacy.py`

**Bugs/cleanup:**
1. `telegram_bot.py` saves `location_lat`/`location_city` but `pharmacy.py` reads `lat`/`city` — they never match. The `.env` fallback masks this.
2. `_fetch_duty_pharmacies_nosyapi()` (lines 135-153) — delete entirely. We fetch from eczaneler.gen.tr for free.
3. `_get_user_location()` (lines 97-106) — delete entirely. Was the .env-only function.
4. Remove `NOSYAPI_KEY` from .env references in module docstring.
5. Remove `import os` if no longer needed after killing Nosyapi and .env fallbacks.
6. Delete `_normalize_turkish()` if no longer used after removing district filtering from the scraper (Task 1).
7. Delete any tests for `_fetch_duty_pharmacies_nosyapi` and `_normalize_turkish`.

- [ ] **Step 1: Write test for DB-only location lookup**

```python
@pytest.mark.asyncio
async def test_get_user_coords_reads_correct_db_keys():
    """_get_user_coords must read location_lat/location_lon, not lat/lon."""
    from unittest.mock import AsyncMock, patch
    from src.tools.pharmacy import _get_user_coords

    async def mock_get_pref(key, default=""):
        return {"location_lat": "41.0", "location_lon": "29.0"}.get(key, default)

    with patch("src.tools.pharmacy.get_user_pref", side_effect=mock_get_pref), \
         patch.dict("os.environ", {}, clear=True):
        result = await _get_user_coords()

    assert result == (41.0, 29.0)


@pytest.mark.asyncio
async def test_get_user_coords_no_env_fallback():
    """_get_user_coords should NOT fall back to .env when DB is empty."""
    from unittest.mock import AsyncMock, patch
    from src.tools.pharmacy import _get_user_coords

    async def mock_get_pref(key, default=""):
        return default  # DB has nothing

    with patch("src.tools.pharmacy.get_user_pref", side_effect=mock_get_pref), \
         patch.dict("os.environ", {"USER_LAT": "40.0", "USER_LON": "28.0"}):
        result = await _get_user_coords()

    assert result is None  # Should NOT fall back to .env


@pytest.mark.asyncio
async def test_get_user_city_district_reads_correct_db_keys():
    """_get_user_city_district must read location_city/location_district."""
    from unittest.mock import patch
    from src.tools.pharmacy import _get_user_city_district

    async def mock_get_pref(key, default=""):
        return {"location_city": "istanbul", "location_district": "kadıköy"}.get(key, default)

    with patch("src.tools.pharmacy.get_user_pref", side_effect=mock_get_pref), \
         patch.dict("os.environ", {}, clear=True):
        result = await _get_user_city_district()

    assert result == ("istanbul", "kadıköy")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pharmacy.py::test_get_user_coords_reads_correct_db_keys tests/test_pharmacy.py::test_get_user_coords_no_env_fallback tests/test_pharmacy.py::test_get_user_city_district_reads_correct_db_keys -v`
Expected: FAIL — wrong keys, .env fallback still present

- [ ] **Step 3: Rewrite location functions**

Replace `_get_user_location`, `_get_user_city_district`, and `_get_user_coords` in `src/tools/pharmacy.py`:

```python
# Remove _get_user_location() entirely — it was the .env-only function

# Top-level lazy import for DB access
_db_imported = False

async def _ensure_db():
    global get_user_pref, _db_imported
    if not _db_imported:
        from src.infra.db import get_user_pref as _gup
        get_user_pref = _gup
        _db_imported = True


async def _get_user_city_district() -> tuple[str, str]:
    """Get user's city and district from DB preferences."""
    try:
        await _ensure_db()
        city = await get_user_pref("location_city", "")
        district = await get_user_pref("location_district", "")
        return city, district
    except Exception:
        return "", ""


async def _get_user_coords() -> tuple[float, float] | None:
    """Get user's coordinates from DB preferences. Returns None if not set."""
    try:
        await _ensure_db()
        lat = await get_user_pref("location_lat", "")
        lon = await get_user_pref("location_lon", "")
        if lat and lon:
            return float(lat), float(lon)
    except Exception:
        pass
    return None
```

Also remove `import os` if no longer needed. Delete `_fetch_duty_pharmacies_nosyapi()` entirely — Nosyapi is being removed. Check if `os` is still used elsewhere in the file (it shouldn't be after removing Nosyapi and .env fallbacks).

Remove the `USER_LAT`, `USER_LON`, `USER_CITY`, `USER_DISTRICT` references from the module docstring at the top.

- [ ] **Step 4: Run new tests + existing tests**

Run: `python -m pytest tests/test_pharmacy.py -v`
Expected: New tests PASS, some existing tests that relied on .env fallback may need updating

- [ ] **Step 5: Fix any broken existing tests that used .env mocking**

Update tests that mock `USER_LAT`/`USER_LON` env vars to instead mock `get_user_pref` with the correct `location_*` keys. Grep the test file for `USER_LAT` to find them.

- [ ] **Step 6: Commit**

```bash
git add src/tools/pharmacy.py tests/test_pharmacy.py
git commit -m "fix(pharmacy): consolidate location to DB-only, fix key mismatch, remove .env fallback"
```

---

### Task 3: Structured Return + Formatter + Button Builder

**Files:**
- Modify: `src/tools/pharmacy.py`
- Test: `tests/test_pharmacy.py`

Extract pharmacy data fetching into `find_pharmacies_structured()` returning `list[Pharmacy]`. Add `format_pharmacy_message()` and `build_pharmacy_buttons()` for the bot to use. Keep `find_nearest_pharmacy()` as a text wrapper for agent use.

- [ ] **Step 1: Write test for structured return**

```python
@pytest.mark.asyncio
async def test_find_pharmacies_structured_returns_sorted_list():
    """find_pharmacies_structured returns Pharmacy list sorted by distance."""
    from unittest.mock import AsyncMock, patch
    from src.tools.pharmacy import find_pharmacies_structured, Pharmacy

    raw = [
        {"pharmacyName": "Far", "address": "A", "district": "K", "phone": "1",
         "detail_url": "", "lat": 41.1, "lng": 29.1},
        {"pharmacyName": "Close", "address": "B", "district": "K", "phone": "2",
         "detail_url": "", "lat": 41.001, "lng": 29.001},
    ]

    with patch("src.tools.pharmacy._fetch_duty_pharmacies_eczaneler_gen_tr",
               new_callable=AsyncMock, return_value=raw), \
         patch("src.tools.pharmacy._get_user_coords",
               new_callable=AsyncMock, return_value=(41.0, 29.0)), \
         patch("src.tools.pharmacy._get_user_city_district",
               new_callable=AsyncMock, return_value=("istanbul", "kadıköy")), \
         patch("src.tools.pharmacy._get_osrm_distance",
               new_callable=AsyncMock, return_value=(-1.0, -1.0)):
        result = await find_pharmacies_structured(city="istanbul", district="kadıköy")

    assert isinstance(result, list)
    assert all(isinstance(p, Pharmacy) for p in result)
    assert result[0].name == "Close"  # closer one first
    assert result[0].distance_km < result[1].distance_km
```

- [ ] **Step 2: Implement `find_pharmacies_structured`**

Add in `src/tools/pharmacy.py` — extract the core logic from `find_nearest_pharmacy` into a new function:

```python
async def find_pharmacies_structured(
    city: str = "",
    district: str = "",
    include_route: bool = True,
    tab: str = "bugun",
) -> list[Pharmacy]:
    """Find duty pharmacies sorted by distance.

    Args:
        tab: "bugun" for today, "yarin" for tomorrow.

    Returns list of Pharmacy dataclasses. Empty list on failure.
    """
    if not city or not district:
        pref_city, pref_district = await _get_user_city_district()
        if not city:
            city = pref_city
        if not district:
            district = pref_district

    if not city:
        return []

    # Fetch city-wide — no district filter, distance sorting handles relevance
    # (adjacent-district pharmacies may be closer near borders)
    pharmacies_raw = await _fetch_duty_pharmacies_eczaneler_gen_tr(city, tab=tab)
    if not pharmacies_raw:
        pharmacies_raw = await _fetch_duty_pharmacies_web(city)

    if not pharmacies_raw or pharmacies_raw[0].get("_raw_search"):
        return []

    # Parse into Pharmacy objects
    pharmacies = []
    for p in pharmacies_raw:
        pharmacies.append(Pharmacy(
            name=p.get("pharmacyName", p.get("name", "Unknown")),
            address=p.get("address", ""),
            district=p.get("district", district),
            phone=p.get("phone", ""),
            lat=float(p.get("lat", 0) or 0),
            lon=float(p.get("lng", p.get("lon", 0)) or 0),
            detail_url=p.get("detail_url", ""),
        ))

    # Calculate distances
    user_loc = await _get_user_coords()
    if user_loc and pharmacies:
        user_lat, user_lon = user_loc
        detail_fetch_budget = 10
        for pharmacy in pharmacies:
            if pharmacy.lat and pharmacy.lon:
                pharmacy.distance_km = round(_haversine(
                    user_lat, user_lon, pharmacy.lat, pharmacy.lon), 2)
            else:
                if pharmacy.detail_url and detail_fetch_budget > 0:
                    detail_fetch_budget -= 1
                    coords = await _fetch_pharmacy_coordinates(pharmacy.detail_url)
                    if coords:
                        pharmacy.lat, pharmacy.lon = coords
                if not (pharmacy.lat and pharmacy.lon) and pharmacy.address:
                    coords = await _geocode_address(f"{pharmacy.address}, {city}, Turkey")
                    if coords:
                        pharmacy.lat, pharmacy.lon = coords
                if pharmacy.lat and pharmacy.lon:
                    pharmacy.distance_km = round(_haversine(
                        user_lat, user_lon, pharmacy.lat, pharmacy.lon), 2)

        pharmacies.sort(key=lambda p: p.distance_km if p.distance_km >= 0 else 999)

        if include_route:
            for pharmacy in pharmacies[:3]:
                if pharmacy.lat and pharmacy.lon:
                    walk_km, walk_min = await _get_osrm_distance(
                        user_lat, user_lon, pharmacy.lat, pharmacy.lon, "foot")
                    drive_km, drive_min = await _get_osrm_distance(
                        user_lat, user_lon, pharmacy.lat, pharmacy.lon, "car")
                    if walk_km >= 0:
                        pharmacy.distance_km = walk_km
                        pharmacy.walking_min = walk_min
                    if drive_km >= 0:
                        pharmacy.driving_min = drive_min
            pharmacies.sort(key=lambda p: p.distance_km if p.distance_km >= 0 else 999)

    return pharmacies
```

Then simplify `find_nearest_pharmacy` to delegate:

```python
async def find_nearest_pharmacy(
    city: str = "",
    district: str = "",
    include_route: bool = True,
) -> str:
    """Find nearest duty pharmacy. Returns formatted text for agent use."""
    if not city and not district:
        city_d, dist_d = await _get_user_city_district()
        city = city or city_d
        district = district or dist_d

    if not city:
        return ("Sehir belirtilmedi. Lutfen sehir parametresi girin.\n"
                "Ornek: find_nearest_pharmacy(city='ankara')")

    pharmacies = await find_pharmacies_structured(city, district, include_route)
    if not pharmacies:
        return f"No duty pharmacies found for {city}" + (f"/{district}" if district else "") + "."

    return format_pharmacy_message(pharmacies, show_all=True)
```

- [ ] **Step 3: Write tests for formatter and button builder**

```python
def test_format_pharmacy_message_top3():
    from src.tools.pharmacy import Pharmacy, format_pharmacy_message
    pharmacies = [
        Pharmacy(name="A", address="Addr A", district="K", phone="111",
                 lat=40.99, lon=29.02, distance_km=0.5, walking_min=6.0, driving_min=2.0),
        Pharmacy(name="B", address="Addr B", district="K", phone="222",
                 lat=40.98, lon=29.03, distance_km=1.2, walking_min=15.0, driving_min=4.0),
    ]
    text = format_pharmacy_message(pharmacies)
    assert "A" in text
    assert "0.5 km" in text
    assert "111" in text


def test_format_pharmacy_message_show_all():
    from src.tools.pharmacy import Pharmacy, format_pharmacy_message
    pharmacies = [
        Pharmacy(name=f"E{i}", address=f"A{i}", district="K", phone=f"{i}11",
                 lat=40.99 - i * 0.01, lon=29.02)
        for i in range(5)
    ]
    text = format_pharmacy_message(pharmacies, show_all=True)
    assert "E4" in text  # last one included


def test_build_pharmacy_buttons_map_links():
    from src.tools.pharmacy import Pharmacy, build_pharmacy_buttons
    pharmacies = [
        Pharmacy(name="A", address="X", district="K", lat=40.99, lon=29.02, distance_km=0.5),
        Pharmacy(name="B", address="Y", district="K", lat=40.98, lon=29.03, distance_km=1.2),
    ]
    buttons = build_pharmacy_buttons(pharmacies, total_count=5)
    assert len(buttons) == 3  # 2 map rows + 1 show-all
    assert "google.com/maps" in buttons[0][0].url
    assert buttons[-1][0].callback_data == "pharm:all"


def test_build_pharmacy_buttons_no_show_all_when_few():
    from src.tools.pharmacy import Pharmacy, build_pharmacy_buttons
    pharmacies = [
        Pharmacy(name="A", address="X", district="K", lat=40.99, lon=29.02, distance_km=0.5),
    ]
    buttons = build_pharmacy_buttons(pharmacies, total_count=1)
    assert len(buttons) == 1
```

- [ ] **Step 4: Implement formatter and button builder**

Add at bottom of `src/tools/pharmacy.py`:

```python
def format_pharmacy_message(pharmacies: list[Pharmacy], show_all: bool = False) -> str:
    """Format pharmacies for display. Top 3 by default, all if show_all."""
    if not pharmacies:
        return "Nobetci eczane bulunamadi."

    subset = pharmacies if show_all else pharmacies[:3]
    lines = []
    for i, p in enumerate(subset, 1):
        line = f"{i}. {p.name}"
        if p.distance_km >= 0:
            line += f" — {p.distance_km} km"
            if p.walking_min >= 0:
                line += f" ({p.walking_min:.0f} dk yuruyus)"
            if p.driving_min >= 0:
                line += f" / {p.driving_min:.0f} dk arac"
        lines.append(line)
        if p.address:
            lines.append(f"   {p.address}")
        if p.phone:
            lines.append(f"   Tel: {p.phone}")
        lines.append("")

    if not show_all and len(pharmacies) > 3:
        lines.append(f"Toplam {len(pharmacies)} eczane bulundu")

    return "\n".join(lines)


def build_pharmacy_buttons(
    pharmacies: list[Pharmacy],
    total_count: int,
) -> list[list]:
    """Build inline keyboard rows: Google Maps link per pharmacy + optional show-all.

    Returns list of button rows for InlineKeyboardMarkup.
    """
    from telegram import InlineKeyboardButton

    rows = []
    for i, p in enumerate(pharmacies):
        if p.lat and p.lon:
            url = f"https://www.google.com/maps/search/?api=1&query={p.lat},{p.lon}"
            label = f"🗺 {i+1}. {p.name}"
            if p.distance_km >= 0:
                label += f" ({p.distance_km} km)"
            rows.append([InlineKeyboardButton(label, url=url)])
        else:
            rows.append([InlineKeyboardButton(
                f"📍 {i+1}. {p.name}", callback_data=f"pharm:search:{i}")])

    if total_count > len(pharmacies):
        rows.append([InlineKeyboardButton(
            f"📋 Tumunu Goster ({total_count})", callback_data="pharm:all")])

    return rows
```

- [ ] **Step 5: Run all tests**

Run: `python -m pytest tests/test_pharmacy.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/tools/pharmacy.py tests/test_pharmacy.py
git commit -m "feat(pharmacy): structured return, formatter, map button builder, tomorrow support"
```

---

## Public API Summary (for telegram_bot.py integration)

After this plan, `src/tools/pharmacy.py` exports:

| Function | Purpose | Used by |
|----------|---------|---------|
| `find_pharmacies_structured(city, include_route, tab)` | Returns `list[Pharmacy]` sorted by distance | Bot `_quick_pharmacy` |
| `find_nearest_pharmacy(city, include_route)` | Returns formatted text | Agents |
| `format_pharmacy_message(pharmacies, show_all)` | Text for Telegram message | Bot |
| `build_pharmacy_buttons(pharmacies, total_count)` | InlineKeyboardButton rows | Bot |
| `Pharmacy` dataclass | Data container | Bot cache |

**Bot integration notes (for separate agent):**
- `_quick_pharmacy`: call `find_pharmacies_structured()`, send `format_pharmacy_message(pharmacies[:3])` with `InlineKeyboardMarkup(build_pharmacy_buttons(pharmacies[:3], len(pharmacies)))`, stash full list in `self._pharmacy_cache[chat_id]`
- Add `pharm:all` callback in `handle_callback`: read cache, send `format_pharmacy_message(pharmacies, show_all=True)`
- Add `pharm:tomorrow` callback: call `find_pharmacies_structured(tab="yarin")`
- Add "📍 Farklı konum" inline button on results — triggers location setup flow, then re-runs pharmacy
- `_pharmacy_cache` dict on TelegramInterface `__init__`, entries expire after 1 hour

## Location Architecture

**Single source of truth:** `user_preferences` table in SQLite.

| Key | Written by | Read by |
|-----|-----------|---------|
| `location_lat` | telegram_bot.py `handle_location` / `_geocode_district` | pharmacy.py, weather, prayer |
| `location_lon` | telegram_bot.py | pharmacy.py, weather, prayer |
| `location_city` | telegram_bot.py | pharmacy.py |
| `location_district` | telegram_bot.py | pharmacy.py |

**No `.env` fallback.** If DB has no location, `_get_user_coords()` returns `None` and `_get_user_city_district()` returns `("", "")`. The bot's location setup flow handles prompting.

**Hybrid UX (bot-side, not in this plan):**
- Default: use saved location, no extra prompt
- Pharmacy results include "📍 Farklı konum" inline button
- Tapping it: same location setup flow (`_start_location_setup`), re-runs pharmacy after
