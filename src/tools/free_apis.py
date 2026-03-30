"""Registry of free APIs for quick data lookups.

Agents can query this registry to find APIs that answer questions
faster than web search. All APIs have free tiers.

Supports both a static in-memory registry and a DB-backed dynamic
registry that grows over time via automated discovery.
"""

import asyncio
import os
import re
from dataclasses import dataclass

import aiohttp

from src.infra.logging_config import get_logger

logger = get_logger("tools.free_apis")


@dataclass
class FreeAPI:
    name: str
    category: str  # weather, currency, news, geo, time, etc.
    base_url: str
    auth_type: str  # "none", "apikey_header", "apikey_param"
    env_var: str | None  # env var name for API key
    rate_limit: str  # human-readable
    description: str
    example_endpoint: str  # ready-to-call example


# Registry of known free APIs
API_REGISTRY: list[FreeAPI] = [
    # --- Weather ---
    FreeAPI(
        name="wttr.in",
        category="weather",
        base_url="https://wttr.in",
        auth_type="none",
        env_var=None,
        rate_limit="unlimited",
        description="Weather forecasts in plain text or JSON. No API key needed.",
        example_endpoint="https://wttr.in/Istanbul?format=j1",
    ),
    FreeAPI(
        name="Open-Meteo",
        category="weather",
        base_url="https://api.open-meteo.com",
        auth_type="none",
        env_var=None,
        rate_limit="10000/day",
        description="Weather forecasts, historical data. No API key. Latitude/longitude based.",
        example_endpoint="https://api.open-meteo.com/v1/forecast?latitude=41.01&longitude=28.98&current_weather=true",
    ),

    # --- Currency / Exchange Rates ---
    FreeAPI(
        name="ExchangeRate-API",
        category="currency",
        base_url="https://open.er-api.com",
        auth_type="none",
        env_var=None,
        rate_limit="1500/month",
        description="Exchange rates for 161 currencies. No API key for open access.",
        example_endpoint="https://open.er-api.com/v6/latest/USD",
    ),
    FreeAPI(
        name="TCMB EVDS",
        category="currency",
        base_url="https://evds2.tcmb.gov.tr",
        auth_type="apikey_param",
        env_var="TCMB_EVDS_API_KEY",
        rate_limit="unlimited",
        description="Turkish Central Bank official exchange rates and economic data. Free API key from TCMB.",
        example_endpoint="https://evds2.tcmb.gov.tr/service/evds/series=TP.DK.USD.A-TP.DK.EUR.A&startDate=01-01-2026&endDate=31-12-2026&type=json&key={key}",
    ),
    FreeAPI(
        name="Frankfurter",
        category="currency",
        base_url="https://api.frankfurter.dev",
        auth_type="none",
        env_var=None,
        rate_limit="unlimited",
        description="ECB exchange rates. Historical and current. No API key.",
        example_endpoint="https://api.frankfurter.dev/latest?from=USD&to=TRY,EUR",
    ),

    # --- News ---
    FreeAPI(
        name="GNews",
        category="news",
        base_url="https://gnews.io/api/v4",
        auth_type="apikey_param",
        env_var="GNEWS_API_KEY",
        rate_limit="100/day free",
        description="News articles from 60k+ sources. Free tier: 100 requests/day.",
        example_endpoint="https://gnews.io/api/v4/top-headlines?lang=tr&token={key}",
    ),

    # --- Geocoding / Location ---
    FreeAPI(
        name="Nominatim (OpenStreetMap)",
        category="geo",
        base_url="https://nominatim.openstreetmap.org",
        auth_type="none",
        env_var=None,
        rate_limit="1/second",
        description="Geocoding and reverse geocoding. No API key. 1 req/sec.",
        example_endpoint="https://nominatim.openstreetmap.org/search?q=Istanbul&format=json",
    ),

    # --- Time ---
    FreeAPI(
        name="WorldTimeAPI",
        category="time",
        base_url="http://worldtimeapi.org/api",
        auth_type="none",
        env_var=None,
        rate_limit="unlimited",
        description="Current time for any timezone. No API key.",
        example_endpoint="http://worldtimeapi.org/api/timezone/Europe/Istanbul",
    ),

    # --- Wikipedia / Knowledge ---
    FreeAPI(
        name="Wikipedia API",
        category="knowledge",
        base_url="https://en.wikipedia.org/api/rest_v1",
        auth_type="none",
        env_var=None,
        rate_limit="200/second",
        description="Wikipedia summaries, pages, search. No API key.",
        example_endpoint="https://en.wikipedia.org/api/rest_v1/page/summary/Istanbul",
    ),
    FreeAPI(
        name="Wikipedia TR",
        category="knowledge",
        base_url="https://tr.wikipedia.org/api/rest_v1",
        auth_type="none",
        env_var=None,
        rate_limit="200/second",
        description="Turkish Wikipedia summaries and pages. No API key.",
        example_endpoint="https://tr.wikipedia.org/api/rest_v1/page/summary/İstanbul",
    ),

    # --- Translation ---
    FreeAPI(
        name="LibreTranslate",
        category="translation",
        base_url="https://libretranslate.com",
        auth_type="none",
        env_var=None,
        rate_limit="varies by instance",
        description="Free translation API. Self-hostable. Many public instances.",
        example_endpoint="https://libretranslate.com/translate",
    ),

    # --- IP / Network ---
    FreeAPI(
        name="ipapi",
        category="network",
        base_url="https://ipapi.co",
        auth_type="none",
        env_var=None,
        rate_limit="1000/day",
        description="IP geolocation. No API key for basic usage.",
        example_endpoint="https://ipapi.co/json/",
    ),

    # --- Jokes / Fun ---
    FreeAPI(
        name="JokeAPI",
        category="fun",
        base_url="https://v2.jokeapi.dev",
        auth_type="none",
        env_var=None,
        rate_limit="120/minute",
        description="Programming and general jokes. No API key.",
        example_endpoint="https://v2.jokeapi.dev/joke/Any",
    ),

    # --- Turkish Pharmacy on Duty ---
    FreeAPI(
        name="Nosyapi Pharmacy",
        category="health",
        base_url="https://nosyapi.com/apiv2",
        auth_type="apikey_param",
        env_var="NOSYAPI_KEY",
        rate_limit="100/day free",
        description="Turkey pharmacy on duty (nöbetçi eczane) by city/district. Free tier: 100 requests/day.",
        example_endpoint="https://nosyapi.com/apiv2/pharmacyOnDuty?city=istanbul&district=kadikoy&apikey={key}",
    ),

    # --- Earthquake/Disaster ---
    FreeAPI(
        name="Kandilli Observatory",
        category="earthquake",
        base_url="https://api.orhanaydogdu.com.tr",
        auth_type="none",
        env_var=None,
        rate_limit="unlimited",
        description="Turkey earthquake data from Kandilli Observatory (unofficial API wrapper). Real-time seismic data.",
        example_endpoint="https://api.orhanaydogdu.com.tr/deprem/kandilli/live",
    ),

    # --- Gas/Fuel Prices ---
    FreeAPI(
        name="Turkey Fuel Prices",
        category="fuel",
        base_url="https://api.collectapi.com/gasPrice",
        auth_type="apikey_header",
        env_var="COLLECTAPI_KEY",
        rate_limit="free tier available",
        description="Turkey fuel/gas prices by city and fuel type. CollectAPI free tier.",
        example_endpoint="https://api.collectapi.com/gasPrice/allUsa498",
    ),

    # --- Prayer Times ---
    FreeAPI(
        name="Diyanet Prayer Times",
        category="religion",
        base_url="https://ezanvakti.herokuapp.com",
        auth_type="none",
        env_var=None,
        rate_limit="unlimited",
        description="Turkey prayer times (namaz vakitleri) by city/district. Unofficial Diyanet API.",
        example_endpoint="https://ezanvakti.herokuapp.com/vakitler?ilce=9541",
    ),

    # --- Turkish Holidays ---
    FreeAPI(
        name="Turkey Holidays",
        category="calendar",
        base_url="https://date.nager.at/api/v3",
        auth_type="none",
        env_var=None,
        rate_limit="unlimited",
        description="Public holidays for Turkey (and 100+ countries). No API key needed.",
        example_endpoint="https://date.nager.at/api/v3/PublicHolidays/2026/TR",
    ),

    # --- Gold Prices ---
    FreeAPI(
        name="Gold Price Turkey",
        category="currency",
        base_url="https://api.collectapi.com/economy",
        auth_type="apikey_header",
        env_var="COLLECTAPI_KEY",
        rate_limit="free tier available",
        description="Live gold prices in Turkey (gram altın, çeyrek altın, etc.). CollectAPI.",
        example_endpoint="https://api.collectapi.com/economy/goldPrice",
    ),

    # --- BIST Stock Market ---
    FreeAPI(
        name="BIST Stock Data",
        category="finance",
        base_url="https://api.collectapi.com/economy",
        auth_type="apikey_header",
        env_var="COLLECTAPI_KEY",
        rate_limit="free tier available",
        description="BIST (Borsa Istanbul) stock data. CollectAPI.",
        example_endpoint="https://api.collectapi.com/economy/hpiIndex",
    ),

    # --- Routing/Directions ---
    FreeAPI(
        name="OSRM",
        category="geo",
        base_url="https://router.project-osrm.org",
        auth_type="none",
        env_var=None,
        rate_limit="unlimited (public demo server)",
        description="Open Source Routing Machine — driving/walking directions and distance. No API key. Privacy-safe (OpenStreetMap data).",
        example_endpoint="https://router.project-osrm.org/route/v1/driving/28.9784,41.0082;29.0291,41.0082?overview=false",
    ),

    # --- Travel/Tickets ---
    FreeAPI(
        name="EnUygun Travel",
        category="travel",
        base_url="https://mcp.enuygun.com",
        auth_type="none",
        env_var=None,
        rate_limit="unknown",
        description="EnUygun flight and bus ticket search (MCP endpoint). Turkish travel aggregator.",
        example_endpoint="https://mcp.enuygun.com/mcp",
    ),
    FreeAPI(
        name="Kiwi Tequila",
        category="travel",
        base_url="https://tequila-api.kiwi.com",
        auth_type="apikey_header",
        env_var="KIWI_API_KEY",
        rate_limit="free tier (generous)",
        description="Multi-modal travel search: flights + buses + trains + ferries. 750+ carriers. Free API key from tequila.kiwi.com.",
        example_endpoint="https://tequila-api.kiwi.com/v2/search?fly_from=IST&fly_to=ANK&date_from=01/04/2026&date_to=05/04/2026",
    ),
    FreeAPI(
        name="Rome2rio",
        category="travel",
        base_url="https://free.rome2rio.com",
        auth_type="apikey_param",
        env_var="ROME2RIO_API_KEY",
        rate_limit="100000/month free",
        description="Route planning across all transport modes. 160+ countries. Free tier 100K req/month. Returns routes and durations, estimated price ranges.",
        example_endpoint="https://free.rome2rio.com/api/1.4/json/Search?key={key}&oName=Istanbul&dName=Ankara",
    ),
]


def _api_to_dict(api: FreeAPI) -> dict:
    """Convert a FreeAPI dataclass to a dict matching the DB schema."""
    return {
        "name": api.name,
        "category": api.category,
        "base_url": api.base_url,
        "auth_type": api.auth_type,
        "env_var": api.env_var,
        "rate_limit": api.rate_limit,
        "description": api.description,
        "example_endpoint": api.example_endpoint,
        "source": "static",
        "verified": 1,
    }


def _dict_to_api(d: dict) -> FreeAPI:
    """Convert a DB row dict to a FreeAPI dataclass."""
    return FreeAPI(
        name=d["name"],
        category=d.get("category", "misc"),
        base_url=d.get("base_url", ""),
        auth_type=d.get("auth_type", "none"),
        env_var=d.get("env_var"),
        rate_limit=d.get("rate_limit", "unknown"),
        description=d.get("description", ""),
        example_endpoint=d.get("example_endpoint", ""),
    )


async def seed_registry() -> int:
    """Seed the DB with all static APIs if they don't exist yet.

    Returns the number of APIs seeded.
    """
    from src.infra.db import upsert_free_api

    count = 0
    for api in API_REGISTRY:
        await upsert_free_api(_api_to_dict(api))
        count += 1
    logger.info("Seeded %d static APIs into free_api_registry", count)
    return count


def find_apis(category: str | None = None, query: str | None = None) -> list[FreeAPI]:
    """Find APIs matching a category or keyword query.

    Searches both the static registry and any DB-discovered APIs
    (loaded via _load_db_apis_sync cache).
    """
    # Combine static + dynamic (deduplicate by name)
    seen_names = set()
    combined: list[FreeAPI] = []
    for api in API_REGISTRY:
        seen_names.add(api.name)
        combined.append(api)

    for api in _db_api_cache:
        if api.name not in seen_names:
            seen_names.add(api.name)
            combined.append(api)

    results = combined

    if category:
        cat_lower = category.lower()
        results = [a for a in results if cat_lower in a.category.lower()]

    if query:
        q_lower = query.lower()
        results = [
            a for a in results
            if q_lower in a.name.lower()
            or q_lower in a.description.lower()
            or q_lower in a.category.lower()
        ]

    return results


# In-memory cache of DB-discovered APIs (refreshed by discover/seed)
_db_api_cache: list[FreeAPI] = []


async def refresh_db_cache() -> None:
    """Reload discovered APIs from DB into the in-memory cache."""
    from src.infra.db import get_all_free_apis

    global _db_api_cache
    try:
        rows = await get_all_free_apis()
        _db_api_cache = [_dict_to_api(r) for r in rows]
        logger.debug("Refreshed DB API cache: %d entries", len(_db_api_cache))
    except Exception as e:
        logger.debug("refresh_db_cache failed: %s", e)


def get_api(name: str) -> FreeAPI | None:
    """Get a specific API by name (checks static + DB cache)."""
    for api in API_REGISTRY:
        if api.name.lower() == name.lower():
            return api
    for api in _db_api_cache:
        if api.name.lower() == name.lower():
            return api
    return None


async def call_api(
    api: "FreeAPI | dict",
    endpoint: str | None = None,
    params: dict | None = None,
) -> str:
    """Call a free API and return the response as text.

    Accepts either a FreeAPI dataclass or a dict (from DB).
    If endpoint is None, uses the example_endpoint.
    For APIs requiring keys, reads from environment.
    """
    # Normalize dict to FreeAPI
    if isinstance(api, dict):
        api = _dict_to_api(api)

    url = endpoint or api.example_endpoint

    # Substitute API key if needed
    if api.auth_type == "apikey_param" and api.env_var:
        key = os.getenv(api.env_var, "")
        if not key:
            return f"Error: {api.env_var} not set. Get a free key at {api.base_url}"
        url = url.replace("{key}", key)

    headers = {"User-Agent": "KutAI/1.0"}
    if api.auth_type == "apikey_header" and api.env_var:
        key = os.getenv(api.env_var, "")
        if not key:
            return f"Error: {api.env_var} not set."
        headers["Authorization"] = f"Bearer {key}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return f"API error: HTTP {resp.status}"
                text = await resp.text()
                # Truncate large responses
                if len(text) > 5000:
                    text = text[:5000] + "\n...(truncated)"
                return text
    except asyncio.TimeoutError:
        return f"API timeout: {api.name}"
    except Exception as e:
        return f"API error: {e}"


# ---------------------------------------------------------------------------
# Discovery: fetch new APIs from external sources
# ---------------------------------------------------------------------------

# Map public-apis Auth values to our auth_type
_AUTH_MAP = {
    "": "none",
    "no": "none",
    "apikey": "apikey_param",
    "apiKey": "apikey_param",
}

# Categories we care about from public-apis
_CATEGORY_MAP = {
    "weather": "weather",
    "currency": "currency",
    "currency exchange": "currency",
    "finance": "finance",
    "geocoding": "geo",
    "news": "news",
    "science & math": "science",
    "science": "science",
    "books": "knowledge",
    "dictionaries": "knowledge",
    "games & comics": "fun",
    "animals": "fun",
    "food & drink": "food",
    "health": "health",
    "music": "music",
    "sports & fitness": "sports",
    "transportation": "transport",
    "open data": "data",
}


def _parse_public_apis_md(md_text: str) -> list[dict]:
    """Parse the public-apis README markdown table into API dicts.

    Table format: | API | Description | Auth | HTTPS | CORS | Link |
    """
    apis: list[dict] = []
    current_category = "misc"

    for line in md_text.split("\n"):
        # Category headers: ### Category Name
        cat_match = re.match(r"^###\s+(.+)", line)
        if cat_match:
            raw_cat = cat_match.group(1).strip().lower()
            current_category = _CATEGORY_MAP.get(raw_cat, raw_cat)
            continue

        # Table rows: | Name | Description | Auth | HTTPS | CORS |
        if not line.startswith("|") or line.startswith("| API") or line.startswith("|---"):
            continue

        parts = [p.strip() for p in line.split("|")]
        # parts[0] is empty (before first |), then name, desc, auth, https, cors, link (optional)
        if len(parts) < 5:
            continue

        # Extract name (may be a markdown link)
        name_raw = parts[1]
        link_match = re.match(r"\[(.+?)\]\((.+?)\)", name_raw)
        if link_match:
            name = link_match.group(1)
            base_url = link_match.group(2)
        else:
            name = name_raw
            base_url = ""

        desc = parts[2] if len(parts) > 2 else ""
        auth_raw = parts[3].strip("`").strip() if len(parts) > 3 else ""

        # Only import no-auth or apiKey APIs
        auth_lower = auth_raw.lower()
        if auth_lower not in ("", "no", "apikey"):
            continue

        auth_type = _AUTH_MAP.get(auth_lower, "none")

        if not base_url or not name:
            continue

        apis.append({
            "name": name[:100],  # cap length
            "category": current_category,
            "base_url": base_url,
            "auth_type": auth_type,
            "description": desc[:300],
            "example_endpoint": base_url,
            "source": "public-apis",
            "verified": 0,
        })

    return apis


def _parse_free_apis_json(data: list) -> list[dict]:
    """Parse free-apis.github.io JSON format."""
    apis: list[dict] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue

        name = entry.get("name") or entry.get("API") or ""
        url = entry.get("url") or entry.get("Link") or ""
        if not name or not url:
            continue

        auth_raw = (entry.get("auth") or entry.get("Auth") or "").lower()
        if auth_raw not in ("", "none", "no", "apikey"):
            continue

        apis.append({
            "name": name[:100],
            "category": (entry.get("category") or entry.get("Category") or "misc").lower(),
            "base_url": url,
            "auth_type": _AUTH_MAP.get(auth_raw, "none"),
            "description": (entry.get("description") or entry.get("Description") or "")[:300],
            "example_endpoint": url,
            "source": "free-apis-github",
            "verified": 0,
        })

    return apis


async def _discover_from_mcp_registry() -> int:
    """Discover APIs from the MCP server registry."""
    count = 0
    # Fetch from the official MCP registry GitHub
    url = "https://raw.githubusercontent.com/modelcontextprotocol/servers/main/README.md"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    return 0
                text = await resp.text()

        # Parse MCP server entries from the markdown
        # Format: "- **Name** - Description [Link](url)"
        for match in re.finditer(r'\*\*([^*]+)\*\*\s*[-–]\s*([^\[]+)\[.*?\]\(([^)]+)\)', text):
            name = match.group(1).strip()
            description = match.group(2).strip()
            entry_url = match.group(3).strip()

            if not name or not description:
                continue

            # Categorize based on keywords in description
            # Order matters: more specific categories checked first
            desc_lower = description.lower()
            category = "other"
            if any(w in desc_lower for w in ["database", "sql", "postgres"]):
                category = "database"
            elif any(w in desc_lower for w in ["git", "github"]):
                category = "development"
            elif any(w in desc_lower for w in ["file", "storage", "drive"]):
                category = "storage"
            elif any(w in desc_lower for w in ["weather", "climate"]):
                category = "weather"
            elif any(w in desc_lower for w in ["search", "web", "browser"]):
                category = "search"

            from src.infra.db import upsert_free_api
            await upsert_free_api({
                "name": f"MCP: {name}",
                "category": category,
                "base_url": entry_url,
                "auth_type": "mcp",
                "description": description[:200],
                "example_endpoint": entry_url,
                "source": "mcp_registry",
                "verified": 0,
            })
            count += 1

    except Exception as e:
        logger.debug(f"MCP registry discovery failed: {e}")

    return count


async def discover_new_apis(source: str = "all") -> int:
    """Discover new free APIs from external registries.

    Sources:
    - "public-apis": GitHub public-apis README
    - "free-apis": free-apis.github.io JSON
    - "mcp": MCP server registry
    - "all": all sources

    Returns the number of newly discovered APIs.
    """
    from src.infra.db import upsert_free_api

    discovered = 0

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30)
    ) as session:

        # Source 1: public-apis GitHub
        if source in ("all", "public-apis"):
            try:
                url = "https://raw.githubusercontent.com/public-apis/public-apis/master/README.md"
                async with session.get(url, headers={"User-Agent": "KutAI/1.0"}) as resp:
                    if resp.status == 200:
                        md_text = await resp.text()
                        apis = _parse_public_apis_md(md_text)
                        for api_data in apis:
                            await upsert_free_api(api_data)
                            discovered += 1
                        logger.info("Discovered %d APIs from public-apis", len(apis))
                    else:
                        logger.warning("public-apis fetch failed: HTTP %d", resp.status)
            except Exception as e:
                logger.warning("public-apis discovery failed: %s", e)

        # Source 2: free-apis.github.io
        if source in ("all", "free-apis"):
            try:
                url = "https://free-apis.github.io/data/apis.json"
                async with session.get(url, headers={"User-Agent": "KutAI/1.0"}) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        if isinstance(data, list):
                            apis = _parse_free_apis_json(data)
                            for api_data in apis:
                                await upsert_free_api(api_data)
                                discovered += 1
                            logger.info("Discovered %d APIs from free-apis.github.io", len(apis))
                    else:
                        logger.debug("free-apis.github.io fetch returned HTTP %d", resp.status)
            except Exception as e:
                logger.debug("free-apis.github.io discovery failed: %s", e)

    # Source 3: MCP server registry
    if source in ("all", "mcp"):
        discovered += await _discover_from_mcp_registry()

    # Refresh the in-memory cache
    await refresh_db_cache()

    return discovered
