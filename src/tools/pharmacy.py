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


async def _fetch_duty_pharmacies_eczaneler_gen_tr(city: str) -> list[dict]:
    """Scrape duty pharmacies from eczaneler.gen.tr (no API key needed)."""
    try:
        from src.tools.scraper import scrape_url, ScrapeTier
        url = f"https://www.eczaneler.gen.tr/nobetci-{city.lower()}"
        result = await scrape_url(url, max_tier=ScrapeTier.TLS)
        if not result.ok:
            return []

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(result.html, "lxml")

        pharmacies = []
        # eczaneler.gen.tr has pharmacy cards in table rows or div blocks
        rows = soup.select("tr.bg-white, tr.bg-light") or soup.select(".eczane-card") or soup.find_all("tr")

        for row in rows:
            cells = row.find_all("td")
            if len(cells) >= 2:
                name_cell = cells[0].get_text(strip=True) if cells else ""
                address_cell = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                # Try to extract district from address or separate cell
                district = ""
                phone = ""
                if len(cells) > 2:
                    for cell in cells[2:]:
                        text = cell.get_text(strip=True)
                        if text.startswith("0") or text.startswith("("):
                            phone = text
                        elif len(text) < 30:
                            district = text

                if name_cell and len(name_cell) > 2:
                    pharmacies.append({
                        "pharmacyName": name_cell,
                        "address": address_cell,
                        "district": district,
                        "phone": phone,
                    })

        logger.debug(f"eczaneler.gen.tr: found {len(pharmacies)} pharmacies for {city}")
        return pharmacies
    except Exception as e:
        logger.debug(f"eczaneler.gen.tr scrape failed: {e}")
        return []


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
            all_city = await _fetch_duty_pharmacies_eczaneler_gen_tr(city)
            if all_city:
                # Filter by district (case-insensitive partial match)
                dist_lower = district.lower()
                pharmacies_raw = [
                    p for p in all_city
                    if dist_lower in p.get("district", "").lower()
                    or dist_lower in p.get("address", "").lower()
                ]
                if not pharmacies_raw:
                    pharmacies_raw = all_city  # fallback: return all if filter yields nothing
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
        )
        pharmacies.append(pharmacy)

    # Calculate distances if user location is available
    user_loc = await _get_user_coords()
    if user_loc and pharmacies:
        user_lat, user_lon = user_loc

        for pharmacy in pharmacies:
            if pharmacy.lat and pharmacy.lon:
                # Haversine for quick straight-line distance
                pharmacy.distance_km = round(_haversine(
                    user_lat, user_lon, pharmacy.lat, pharmacy.lon
                ), 2)
            elif pharmacy.address:
                # Try geocoding the address
                coords = await _geocode_address(f"{pharmacy.address}, {city}, Turkey")
                if coords:
                    pharmacy.lat, pharmacy.lon = coords
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
