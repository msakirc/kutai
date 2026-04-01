"""Distance calculation utilities.

Provides haversine (straight-line), OSRM (walking/driving), and geocoding.
Used by pharmacy, shopping, and any feature needing location-based sorting.
"""

import math
from dataclasses import dataclass

import aiohttp

from src.infra.logging_config import get_logger

logger = get_logger("tools.distance")


@dataclass
class DistanceResult:
    """Distance between two points."""
    straight_km: float = -1.0   # haversine
    walking_km: float = -1.0    # OSRM foot
    walking_min: float = -1.0
    driving_km: float = -1.0    # OSRM car
    driving_min: float = -1.0


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate straight-line distance in km between two points."""
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


async def osrm_distance(
    from_lat: float, from_lon: float,
    to_lat: float, to_lon: float,
    profile: str = "foot",
) -> tuple[float, float]:
    """Get distance (km) and duration (minutes) via OSRM.

    Args:
        profile: "foot" or "car"

    Returns:
        (distance_km, duration_min) or (-1, -1) on failure.
    """
    # OSRM uses lon,lat order
    url = (f"https://router.project-osrm.org/route/v1/{profile}/"
           f"{from_lon},{from_lat};{to_lon},{to_lat}")
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
        logger.debug(f"OSRM routing failed ({profile}): {e}")
    return -1.0, -1.0


async def calculate_distance(
    from_lat: float, from_lon: float,
    to_lat: float, to_lon: float,
    include_route: bool = True,
) -> DistanceResult:
    """Calculate distance between two points using haversine + optional OSRM.

    Args:
        from_lat, from_lon: Origin (user location).
        to_lat, to_lon: Destination.
        include_route: If True, also fetch OSRM walking/driving distances.

    Returns:
        DistanceResult with all available distance metrics.
    """
    result = DistanceResult()
    result.straight_km = round(haversine(from_lat, from_lon, to_lat, to_lon), 2)

    if include_route:
        walk_km, walk_min = await osrm_distance(from_lat, from_lon, to_lat, to_lon, "foot")
        if walk_km >= 0:
            result.walking_km = walk_km
            result.walking_min = walk_min

        drive_km, drive_min = await osrm_distance(from_lat, from_lon, to_lat, to_lon, "car")
        if drive_km >= 0:
            result.driving_km = drive_km
            result.driving_min = drive_min

    return result


_geocode_cache: dict[str, tuple[float, float] | None] = {}


async def _geocode_here(address: str, session: aiohttp.ClientSession) -> tuple[float, float] | None:
    """Forward geocode via HERE Geocoding API."""
    from src.app.config import HERE_API_KEY
    if not HERE_API_KEY:
        raise RuntimeError("HERE_API_KEY not configured")
    async with session.get(
        "https://geocode.search.hereapi.com/v1/geocode",
        params={"q": address, "in": "countryCode:TUR", "limit": "1",
                "apiKey": HERE_API_KEY},
        timeout=aiohttp.ClientTimeout(total=8),
    ) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HERE geocode HTTP {resp.status}")
        data = await resp.json()
        items = data.get("items", [])
        if items:
            pos = items[0]["position"]
            return pos["lat"], pos["lng"]
    return None


async def _geocode_locationiq(address: str, session: aiohttp.ClientSession) -> tuple[float, float] | None:
    """Forward geocode via LocationIQ API."""
    from src.app.config import LOCATIONIQ_API_KEY
    if not LOCATIONIQ_API_KEY:
        raise RuntimeError("LOCATIONIQ_API_KEY not configured")
    async with session.get(
        "https://us1.locationiq.com/v1/search",
        params={"q": address, "format": "json", "limit": "1",
                "countrycodes": "tr", "key": LOCATIONIQ_API_KEY},
        timeout=aiohttp.ClientTimeout(total=8),
    ) as resp:
        if resp.status != 200:
            raise RuntimeError(f"LocationIQ geocode HTTP {resp.status}")
        data = await resp.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    return None


async def geocode_address(address: str) -> tuple[float, float] | None:
    """Geocode an address using HERE (primary) and LocationIQ (fallback).

    Provider order:
    1. In-memory cache
    2. HERE Geocoding API (best Turkish coverage, 5 req/s)
    3. LocationIQ (OSM-based fallback, 2 req/s)

    Returns (lat, lon) or None. Errors are logged, not silenced.
    """
    cache_key = address.strip().lower()
    if cache_key in _geocode_cache:
        return _geocode_cache[cache_key]

    result = None
    async with aiohttp.ClientSession() as session:
        # Primary: HERE
        try:
            result = await _geocode_here(address, session)
        except Exception as e:
            logger.warning(f"HERE geocode failed for '{address[:50]}': {e}")

        # Fallback: LocationIQ
        if not result:
            try:
                result = await _geocode_locationiq(address, session)
            except Exception as e:
                logger.warning(f"LocationIQ geocode failed for '{address[:50]}': {e}")

    _geocode_cache[cache_key] = result
    if result:
        logger.debug(f"Geocoded: {address[:40]} -> {result}")
    else:
        logger.warning(f"Geocode failed (all providers) for: {address[:60]}")
    return result


async def get_user_location() -> tuple[float, float] | None:
    """Get user's coordinates from DB. Returns (lat, lon) or None."""
    try:
        from src.infra.db import get_user_pref
        lat = await get_user_pref("location_lat", "")
        lon = await get_user_pref("location_lon", "")
        if lat and lon:
            return float(lat), float(lon)
    except Exception:
        pass
    return None
