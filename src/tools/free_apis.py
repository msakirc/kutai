"""Registry of free APIs for quick data lookups.

Agents can query this registry to find APIs that answer questions
faster than web search. All APIs have free tiers.
"""

import asyncio
import os
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
]


def find_apis(category: str | None = None, query: str | None = None) -> list[FreeAPI]:
    """Find APIs matching a category or keyword query."""
    results = API_REGISTRY

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


def get_api(name: str) -> FreeAPI | None:
    """Get a specific API by name."""
    for api in API_REGISTRY:
        if api.name.lower() == name.lower():
            return api
    return None


async def call_api(api: FreeAPI, endpoint: str | None = None, params: dict | None = None) -> str:
    """Call a free API and return the response as text.

    If endpoint is None, uses the example_endpoint.
    For APIs requiring keys, reads from environment.
    """
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
