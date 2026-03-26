"""Currency impact on import prices.

Fetches USD/TRY exchange rate data and assesses how currency
fluctuations affect import-heavy product categories like electronics.
"""

from __future__ import annotations

import time
from typing import Any

import aiohttp

from src.infra.logging_config import get_logger

logger = get_logger("shopping.intelligence.special.exchange_rate")

# ─── API Configuration ──────────────────────────────────────────────────────
# Using exchangerate.host (free, no API key required for basic usage)

_BASE_URL = "https://api.exchangerate.host"
_CACHE_TTL = 3600  # Cache rate for 1 hour

# In-memory cache
_rate_cache: dict[str, Any] = {
    "rate": None,
    "fetched_at": 0.0,
}

# ─── Category Import Sensitivity ────────────────────────────────────────────
# How much a 1% USD/TRY change affects price (0-1 scale)

_IMPORT_SENSITIVITY: dict[str, float] = {
    "electronics": 0.85,
    "phone": 0.90,
    "laptop": 0.85,
    "tablet": 0.85,
    "tv": 0.80,
    "gaming": 0.80,
    "camera": 0.85,
    "audio": 0.75,
    "car_parts": 0.60,
    "cosmetics": 0.50,
    "clothing": 0.30,
    "food": 0.15,
    "home": 0.25,
    "furniture": 0.20,
    "book": 0.10,
    "stationery": 0.20,
    "toy": 0.50,
    "software": 0.70,
}


async def get_usd_try_rate() -> float:
    """Fetch the current USD/TRY exchange rate.

    Uses in-memory caching with a 1-hour TTL. Falls back to a hardcoded
    estimate if the API is unavailable.

    Returns
    -------
    Current USD/TRY rate as a float.
    """
    now = time.time()

    # Return cached if fresh
    if _rate_cache["rate"] is not None and (now - _rate_cache["fetched_at"]) < _CACHE_TTL:
        return _rate_cache["rate"]

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(
                f"{_BASE_URL}/live",
                params={"source": "USD", "currencies": "TRY"},
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("success") and "quotes" in data:
                        rate = data["quotes"].get("USDTRY")
                        if rate:
                            _rate_cache["rate"] = float(rate)
                            _rate_cache["fetched_at"] = now
                            logger.info("USD/TRY rate fetched: %.4f", rate)
                            return _rate_cache["rate"]

            # Try alternative endpoint
            async with session.get(
                f"{_BASE_URL}/convert",
                params={"from": "USD", "to": "TRY", "amount": 1},
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = data.get("result")
                    if result:
                        _rate_cache["rate"] = float(result)
                        _rate_cache["fetched_at"] = now
                        logger.info("USD/TRY rate fetched (convert): %.4f", result)
                        return _rate_cache["rate"]

    except Exception as exc:
        logger.warning("Failed to fetch USD/TRY rate: %s", exc)

    # Fallback: return cached (even if stale) or a hardcoded estimate
    if _rate_cache["rate"] is not None:
        logger.info("Using stale cached USD/TRY rate: %.4f", _rate_cache["rate"])
        return _rate_cache["rate"]

    fallback = 38.0  # Approximate rate as of early 2026
    logger.warning("Using hardcoded fallback USD/TRY rate: %.2f", fallback)
    return fallback


async def get_rate_trend(days: int = 30) -> dict:
    """Analyse USD/TRY rate trend over *days*.

    Parameters
    ----------
    days:
        Number of days of history to analyse.

    Returns
    -------
    Dict with ``current_rate``, ``change_pct``, ``trend``
    (``"rising"``, ``"falling"``, ``"stable"``), ``start_rate``,
    ``period_days``.
    """
    current = await get_usd_try_rate()

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(
                f"{_BASE_URL}/timeframe",
                params={
                    "source": "USD",
                    "currencies": "TRY",
                    "start_date": _days_ago_iso(days),
                    "end_date": _today_iso(),
                },
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    quotes = data.get("quotes", {})
                    if quotes:
                        rates = []
                        for day_data in quotes.values():
                            rate = day_data.get("USDTRY")
                            if rate:
                                rates.append(float(rate))

                        if rates:
                            start_rate = rates[0]
                            change_pct = ((current - start_rate) / start_rate) * 100

                            if change_pct > 2:
                                trend = "rising"
                            elif change_pct < -2:
                                trend = "falling"
                            else:
                                trend = "stable"

                            return {
                                "current_rate": round(current, 4),
                                "start_rate": round(start_rate, 4),
                                "change_pct": round(change_pct, 2),
                                "trend": trend,
                                "period_days": days,
                            }
    except Exception as exc:
        logger.warning("Failed to fetch rate trend: %s", exc)

    # Fallback: no trend data available
    return {
        "current_rate": round(current, 4),
        "start_rate": None,
        "change_pct": None,
        "trend": "unknown",
        "period_days": days,
    }


def assess_import_price_impact(
    product_category: str,
    rate_change_pct: float,
) -> dict:
    """Assess how a USD/TRY rate change impacts prices in a category.

    Parameters
    ----------
    product_category:
        Product category (e.g. ``"electronics"``, ``"phone"``).
    rate_change_pct:
        Percentage change in USD/TRY rate (positive = TRY weakened).

    Returns
    -------
    Dict with ``estimated_price_impact_pct``, ``sensitivity``,
    ``recommendation``.
    """
    category_lower = product_category.lower().replace(" ", "_")
    sensitivity = _IMPORT_SENSITIVITY.get(category_lower, 0.3)

    # Estimated price impact = rate change * category sensitivity
    impact_pct = rate_change_pct * sensitivity

    if impact_pct > 5:
        recommendation = (
            "Kur yukselisi bu kategoride fiyat artisina yol acabilir. "
            "Mumkunse simdi almak mantikli olabilir."
        )
    elif impact_pct > 2:
        recommendation = (
            "Kur degisimi bu kategoride kucuk bir fiyat artisina neden olabilir."
        )
    elif impact_pct < -5:
        recommendation = (
            "Kur dususu bu kategoride fiyat dususune yol acabilir. "
            "Acil degilse beklemek avantajli olabilir."
        )
    elif impact_pct < -2:
        recommendation = (
            "Kur dususu bu kategoride hafif fiyat dususune neden olabilir."
        )
    else:
        recommendation = "Kur degisimi bu kategoriyi onemli olcude etkilemiyor."

    return {
        "estimated_price_impact_pct": round(impact_pct, 1),
        "sensitivity": round(sensitivity, 2),
        "sensitivity_label": _sensitivity_label(sensitivity),
        "rate_change_pct": round(rate_change_pct, 2),
        "category": product_category,
        "recommendation": recommendation,
    }


def _sensitivity_label(sensitivity: float) -> str:
    """Human-readable import sensitivity label."""
    if sensitivity >= 0.8:
        return "Cok yuksek"
    if sensitivity >= 0.6:
        return "Yuksek"
    if sensitivity >= 0.4:
        return "Orta"
    if sensitivity >= 0.2:
        return "Dusuk"
    return "Cok dusuk"


def _today_iso() -> str:
    """Return today's date as ISO string."""
    from datetime import date
    return date.today().isoformat()


def _days_ago_iso(days: int) -> str:
    """Return the date *days* ago as ISO string."""
    from datetime import date, timedelta
    return (date.today() - timedelta(days=days)).isoformat()
