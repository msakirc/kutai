"""Pharmacy on duty (nöbetçi eczane) finder for Turkey.

Finds duty pharmacies, calculates distance from user's location,
returns sorted by proximity. Privacy-safe: user location from .env only.

Required .env vars for distance features:
    USER_LAT=         # latitude (e.g. 40.9876)
    USER_LON=         # longitude (e.g. 29.0250)
    USER_CITY=istanbul
    USER_DISTRICT=
"""

import asyncio
import math
import os
from dataclasses import dataclass

import aiohttp

from src.infra.logging_config import get_logger

logger = get_logger("tools.pharmacy")


@dataclass
class Pharmacy:
    name: str
    address: str
    district: str
    phone: str = ""
    lat: float = 0.0
    lon: float = 0.0
    distance_km: float = -1.0  # -1 = not calculated
    walking_min: float = -1.0
    driving_min: float = -1.0
    detail_url: str = ""


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate straight-line distance in km between two points."""
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


async def _geocode_address(address: str) -> tuple[float, float] | None:
    """Geocode an address using Nominatim (OpenStreetMap). Returns (lat, lon) or None."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": address, "format": "json", "limit": 1, "countrycodes": "tr"},
                headers={"User-Agent": "KutAI/1.0"},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                if data:
                    return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        logger.debug(f"Geocode failed for {address[:50]}: {e}")
    return None


async def _get_osrm_distance(
    from_lat: float, from_lon: float, to_lat: float, to_lon: float,
    profile: str = "foot"
) -> tuple[float, float]:
    """Get distance (km) and duration (minutes) via OSRM. Returns (-1, -1) on failure."""
    # OSRM uses lon,lat order (not lat,lon!)
    url = f"https://router.project-osrm.org/route/v1/{profile}/{from_lon},{from_lat};{to_lon},{to_lat}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                params={"overview": "false"},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    return -1.0, -1.0
                data = await resp.json()
                routes = data.get("routes", [])
                if routes:
                    dist_km = routes[0]["distance"] / 1000.0
                    dur_min = routes[0]["duration"] / 60.0
                    return round(dist_km, 2), round(dur_min, 1)
    except Exception as e:
        logger.debug(f"OSRM routing failed: {e}")
    return -1.0, -1.0


def _get_user_location() -> tuple[float, float] | None:
    """Get user's home location from .env. Returns (lat, lon) or None."""
    lat = os.getenv("USER_LAT", "")
    lon = os.getenv("USER_LON", "")
    if lat and lon:
        try:
            return float(lat), float(lon)
        except ValueError:
            pass
    return None


async def _get_user_city_district() -> tuple[str, str]:
    """Get user's city and district. DB prefs > .env > empty."""
    try:
        from src.infra.db import get_user_pref
        city = await get_user_pref("city", "")
        district = await get_user_pref("district", "")
        if city and district:
            return city, district
    except Exception:
        pass
    return os.getenv("USER_CITY", "istanbul"), os.getenv("USER_DISTRICT", "")


async def _get_user_coords() -> tuple[float, float] | None:
    """Get user's coordinates. DB prefs > .env > None."""
    try:
        from src.infra.db import get_user_pref
        lat = await get_user_pref("lat", "")
        lon = await get_user_pref("lon", "")
        if lat and lon:
            return float(lat), float(lon)
    except Exception:
        pass
    return _get_user_location()  # falls back to .env


async def _fetch_duty_pharmacies_nosyapi(city: str, district: str) -> list[dict]:
    """Fetch duty pharmacies from Nosyapi."""
    key = os.getenv("NOSYAPI_KEY", "")
    if not key:
        return []
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://nosyapi.com/apiv2/pharmacyOnDuty",
                params={"city": city, "district": district, "apikey": key},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                return data.get("data", [])
    except Exception as e:
        logger.debug(f"Nosyapi pharmacy fetch failed: {e}")
        return []


async def _fetch_duty_pharmacies_eczaneler_gen_tr(
    city: str, tab: str = "bugun"
) -> list[dict]:
    """Scrape duty pharmacies from eczaneler.gen.tr (no API key needed).

    Args:
        city: City name (e.g. "ankara", "istanbul").
        tab: Which Bootstrap tab to read — "dun" (yesterday), "bugun" (today),
             or "yarin" (tomorrow). Defaults to "bugun".
    """
    from datetime import datetime
    try:
        from src.tools.scraper import scrape_url, ScrapeTier

        today = datetime.now().strftime("%Y-%m-%d")
        url = f"https://www.eczaneler.gen.tr/nobetci-{city.lower()}?tarih={today}"
        result = await scrape_url(url, max_tier=ScrapeTier.TLS)
        if not result.ok:
            logger.debug(f"eczaneler.gen.tr: HTTP {result.status}")
            return []

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(result.html, "lxml")

        pharmacies = []
        # Target only the requested tab pane (e.g. #nav-bugun) so we don't
        # accidentally return yesterday's or tomorrow's on-duty pharmacies.
        tab_id = f"nav-{tab}"
        tab_pane = soup.find("div", id=tab_id)
        tables = tab_pane.find_all("table") if tab_pane else []
        if not tables:
            # Fallback: try all tables (old page structure without tabs)
            tables = soup.find_all("table")

        # Structure: Bootstrap table.table-striped tables
        # Each <tr> has one <td colspan=3> containing a div.row with:
        #   col-lg-3: <a href="/eczane/..."><span class="isim">Name</span></a>
        #   col-lg-6: address text, <span class="bg-info">District</span>
        #   col-lg-3: phone number
        # First row of each table is a header (contains "Eczane" / "Adres" icons)
        for table in tables:
            for row in table.find_all("tr"):
                # Skip header rows (no <span class="isim">)
                name_span = row.find("span", class_="isim")
                if not name_span:
                    continue

                name = name_span.get_text(strip=True)
                if not name or len(name) <= 2:
                    continue

                # Detail URL from parent <a>
                detail_url = ""
                link = name_span.find_parent("a")
                if link and link.get("href", "").startswith("/"):
                    detail_url = f"https://www.eczaneler.gen.tr{link['href']}"

                # Find the Bootstrap columns inside the row
                cols = row.select("div.col-lg-3, div.col-lg-6")

                # Address column (col-lg-6)
                address = ""
                found_district = ""
                for col in cols:
                    if "col-lg-6" in col.get("class", []):
                        # District badge: <span class="bg-info ...">District</span>
                        district_badge = col.find("span", class_="bg-info")
                        if district_badge:
                            found_district = district_badge.get_text(strip=True)
                        # Address is the direct text content (before sub-elements)
                        # Get all text, then remove district and landmark text
                        addr_parts = []
                        for child in col.children:
                            if isinstance(child, str):
                                t = child.strip()
                                if t:
                                    addr_parts.append(t)
                            elif child.name is None:
                                continue
                            elif "col-lg" not in " ".join(child.get("class", [])):
                                # Skip nested divs with district/landmark
                                pass
                        if not addr_parts:
                            # Fallback: get first text node from col-lg-6
                            full_text = col.get_text(separator="|", strip=True)
                            parts = [p.strip() for p in full_text.split("|") if p.strip()]
                            # First part that looks like an address
                            for p in parts:
                                if len(p) > 15 and p != found_district:
                                    address = p
                                    break
                        else:
                            address = " ".join(addr_parts)
                        break

                # If address still empty, try getting first substantial text from col-lg-6
                if not address:
                    addr_col = row.select_one("div.col-lg-6")
                    if addr_col:
                        # Get text of first text node or NavigableString
                        for item in addr_col.contents:
                            text = item if isinstance(item, str) else ""
                            if not text and hasattr(item, "string") and item.string:
                                text = item.string
                            text = text.strip() if text else ""
                            if len(text) > 10:
                                address = text
                                break

                # Phone: last col-lg-3 (first one is the name column)
                phone = ""
                phone_cols = row.select("div.col-lg-3")
                if len(phone_cols) >= 2:
                    phone = phone_cols[-1].get_text(strip=True)

                pharmacy = {
                    "pharmacyName": name,
                    "address": address,
                    "district": found_district,
                    "phone": phone,
                    "detail_url": detail_url,
                }

                pharmacies.append(pharmacy)

        logger.debug(
            f"eczaneler.gen.tr: found {len(pharmacies)} pharmacies for {city} (tab={tab})"
        )
        return pharmacies
    except Exception as e:
        logger.debug(f"eczaneler.gen.tr scrape failed: {e}")
        return []


async def _fetch_pharmacy_coordinates(detail_url: str) -> tuple[float, float] | None:
    """Fetch lat/lon from a pharmacy's detail page on eczaneler.gen.tr."""
    if not detail_url:
        return None
    try:
        from src.tools.scraper import scrape_url, ScrapeTier
        result = await scrape_url(detail_url, max_tier=ScrapeTier.HTTP, timeout=5.0)
        if not result.ok:
            return None

        import re
        # Look for Google Maps embed with coordinates
        coords = re.findall(r'maps[^"]*?[?&]q=(-?\d+\.?\d*),(-?\d+\.?\d*)', result.html)
        if coords:
            return float(coords[0][0]), float(coords[0][1])

        # Alternative: data-lat/data-lng attributes
        coords = re.findall(
            r'data-lat["\s=:]+(-?\d+\.?\d*).*?data-lng["\s=:]+(-?\d+\.?\d*)',
            result.html,
        )
        if coords:
            return float(coords[0][0]), float(coords[0][1])

    except Exception as e:
        logger.debug(f"Coordinate fetch failed for {detail_url[:50]}: {e}")
    return None


async def _fetch_duty_pharmacies_web(city: str, district: str = "") -> list[dict]:
    """Fallback: search for duty pharmacies via web search."""
    from src.tools.web_search import web_search
    query = f"nöbetçi eczane {city}"
    if district:
        query += f" {district}"
    query += " site:eczaneler.gen.tr OR site:nobetcieczane.com.tr"
    result = await web_search(query, max_results=3, _task_hints={"search_depth": "quick"})
    # Don't stuff raw text into a fake pharmacy dict — return it as a special marker
    return [{"_raw_search": True, "text": result[:3000]}]


async def find_nearest_pharmacy(
    city: str = "",
    district: str = "",
    include_route: bool = True,
) -> str:
    """Find nearest duty pharmacy. Returns formatted text.

    Uses .env USER_LAT/USER_LON for distance calculation.
    If no location set, returns pharmacy list without distances.
    """
    import json

    # Default city/district from DB prefs, then env
    if not city or not district:
        pref_city, pref_district = await _get_user_city_district()
        if not city:
            city = pref_city
        if not district:
            district = pref_district

    if not city:
        return ("Sehir belirtilmedi. Lutfen sehir parametresi girin.\n"
                "Ornek: find_nearest_pharmacy(city='ankara') veya "
                "find_nearest_pharmacy(city='istanbul', district='kadikoy')")

    # Fetch duty pharmacies — different fallback chains for district vs city-wide
    if district:
        # District specified: Nosyapi API -> eczaneler.gen.tr (filtered) -> web search
        pharmacies_raw = await _fetch_duty_pharmacies_nosyapi(city, district)
        if not pharmacies_raw:
            pharmacies_raw = await _fetch_duty_pharmacies_eczaneler_gen_tr(city)
            if not pharmacies_raw:
                # Try city-wide if district filter yields nothing
                all_city = await _fetch_duty_pharmacies_eczaneler_gen_tr(city)
                if all_city:
                    pharmacies_raw = all_city
        if not pharmacies_raw:
            pharmacies_raw = await _fetch_duty_pharmacies_web(city, district)
    else:
        # City-wide (no district): eczaneler.gen.tr -> web search
        pharmacies_raw = await _fetch_duty_pharmacies_eczaneler_gen_tr(city)
        if not pharmacies_raw:
            pharmacies_raw = await _fetch_duty_pharmacies_web(city)

    if not pharmacies_raw:
        return f"No duty pharmacies found for {city}" + (f"/{district}" if district else "") + "."

    # Handle web search fallback — return search results directly
    if pharmacies_raw and pharmacies_raw[0].get("_raw_search"):
        text = pharmacies_raw[0]["text"]
        header = f"Nobetci Eczaneler -- {city.title()}"
        if district:
            header += f" / {district.title()}"
        return header + "\n\n" + text + "\n\n(Kaynak: web aramasi)"

    # Parse into Pharmacy objects
    pharmacies = []
    for p in pharmacies_raw:
        pharmacy = Pharmacy(
            name=p.get("pharmacyName", p.get("name", "Unknown")),
            address=p.get("address", ""),
            district=p.get("district", district),
            phone=p.get("phone", ""),
            lat=float(p.get("lat", 0) or 0),
            lon=float(p.get("lng", p.get("lon", 0)) or 0),
            detail_url=p.get("detail_url", ""),
        )
        pharmacies.append(pharmacy)

    # Calculate distances if user location is available
    user_loc = await _get_user_coords()
    if user_loc and pharmacies:
        user_lat, user_lon = user_loc

        # Limit detail page fetches to avoid hammering the server
        detail_fetch_budget = 10
        for pharmacy in pharmacies:
            if pharmacy.lat and pharmacy.lon:
                # Haversine for quick straight-line distance
                pharmacy.distance_km = round(_haversine(
                    user_lat, user_lon, pharmacy.lat, pharmacy.lon
                ), 2)
            else:
                # Try detail page for coordinates (Google Maps embed)
                if pharmacy.detail_url and detail_fetch_budget > 0:
                    detail_fetch_budget -= 1
                    coords = await _fetch_pharmacy_coordinates(pharmacy.detail_url)
                    if coords:
                        pharmacy.lat, pharmacy.lon = coords
                # Fall back to geocoding
                if not (pharmacy.lat and pharmacy.lon) and pharmacy.address:
                    coords = await _geocode_address(f"{pharmacy.address}, {city}, Turkey")
                    if coords:
                        pharmacy.lat, pharmacy.lon = coords
                if pharmacy.lat and pharmacy.lon:
                    pharmacy.distance_km = round(_haversine(
                        user_lat, user_lon, pharmacy.lat, pharmacy.lon
                    ), 2)

        # Get OSRM walking distance for top 3 closest
        pharmacies.sort(key=lambda p: p.distance_km if p.distance_km >= 0 else 999)

        if include_route:
            for pharmacy in pharmacies[:3]:
                if pharmacy.lat and pharmacy.lon:
                    walk_km, walk_min = await _get_osrm_distance(
                        user_lat, user_lon, pharmacy.lat, pharmacy.lon, "foot"
                    )
                    drive_km, drive_min = await _get_osrm_distance(
                        user_lat, user_lon, pharmacy.lat, pharmacy.lon, "car"
                    )
                    if walk_km >= 0:
                        pharmacy.distance_km = walk_km  # Replace haversine with actual
                        pharmacy.walking_min = walk_min
                    if drive_km >= 0:
                        pharmacy.driving_min = drive_min

        # Re-sort after OSRM distances
        pharmacies.sort(key=lambda p: p.distance_km if p.distance_km >= 0 else 999)

    # Format output
    header = f"Nobetci Eczaneler -- {city.title()}"
    if district:
        header += f" / {district.title()}"
    lines = [header + "\n"]

    for i, p in enumerate(pharmacies, 1):
        lines.append(f"{i}. **{p.name}**")
        if p.address:
            lines.append(f"   Adres: {p.address}")
        if p.phone:
            lines.append(f"   Tel: {p.phone}")
        if p.distance_km >= 0:
            dist_str = f"   Yuruyus: {p.distance_km} km"
            if p.walking_min >= 0:
                dist_str += f" ({p.walking_min:.0f} dk)"
            if p.driving_min >= 0:
                dist_str += f" | Araba: {p.driving_min:.0f} dk"
            lines.append(dist_str)
        lines.append("")

    if not user_loc:
        lines.append("Not: Mesafe hesabi icin .env dosyasina USER_LAT ve USER_LON ekleyin.")

    return "\n".join(lines)
