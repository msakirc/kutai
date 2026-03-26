"""Stale data detection for cached product information.

Monitors price volatility, detects flash sales, and warns users before
they make purchase decisions based on outdated cache entries.
"""

from __future__ import annotations

import re
import statistics

from src.infra.logging_config import get_logger

logger = get_logger("shopping.resilience.staleness")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TTL = 7200  # seconds (2 hours)
FRESH_THRESHOLD = 0.5   # fraction of TTL — below this the data is "fresh"
AGING_THRESHOLD = 0.8   # fraction of TTL — below this the data is "aging"
STALE_THRESHOLD = 1.0   # fraction of TTL — at or above this the data is "stale"
                         # beyond STALE_THRESHOLD it becomes "expired"

# Category-specific base TTL values (seconds)
_CATEGORY_TTL: dict[str, int] = {
    "electronics": 3600,    # prices change often
    "groceries": 1800,      # flash deals common
    "grocery": 1800,
    "food": 1800,
    "furniture": 86400,     # stable prices
    "home": 86400,
}

# Turkish flash-sale keyword patterns (case-insensitive)
_FLASH_PATTERNS: list[str] = [
    r"\bflash\b",
    r"\bfla[sş]\b",
    r"son\s+\d+\s+saat",
    r"s[iı]n[iı]rl[iı]\s+stok",
    r"bug[uü]ne\s+[oö]zel",
    r"[sş]imdi\s+al",
    r"ka[cç][iı]r[mı]a",
    r"indirim\s+biter",
    r"f[iı]rsat",
]
_FLASH_RE = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in _FLASH_PATTERNS]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_staleness(
    product: dict,
    cache_age_seconds: float,
    price_history: list[dict] | None = None,
) -> dict:
    """Assess how stale a cached product entry is.

    Parameters
    ----------
    product:
        Product dict; at minimum should contain ``category`` and ``price``.
    cache_age_seconds:
        How many seconds ago the entry was written to cache.
    price_history:
        Optional list of ``{"price": float, "observed_at": float}`` dicts,
        ordered oldest-first, used to compute price volatility.

    Returns
    -------
    Dict with keys:

    - ``is_stale`` -- ``True`` when staleness level is "stale" or "expired".
    - ``staleness_level`` -- one of "fresh" / "aging" / "stale" / "expired".
    - ``recommended_ttl_seconds`` -- suggested TTL for future caching.
    - ``warnings`` -- list of Turkish-friendly warning messages.
    - ``confidence`` -- 0.0–1.0 confidence in the staleness assessment.
    """
    category = product.get("category")
    volatility = _compute_volatility(price_history)
    recommended_ttl = get_recommended_ttl(category, volatility)

    age_ratio = cache_age_seconds / recommended_ttl if recommended_ttl > 0 else 1.0

    # Determine staleness level
    if age_ratio < FRESH_THRESHOLD:
        staleness_level = "fresh"
    elif age_ratio < AGING_THRESHOLD:
        staleness_level = "aging"
    elif age_ratio < STALE_THRESHOLD:
        staleness_level = "stale"
    else:
        staleness_level = "expired"

    is_stale = staleness_level in ("stale", "expired")

    warnings: list[str] = []

    if staleness_level == "aging":
        warnings.append(
            f"Ürün fiyatı {int(cache_age_seconds // 60)} dakika önce güncellendi; "
            "fiyat değişmiş olabilir."
        )
    elif staleness_level == "stale":
        warnings.append(
            f"Ürün fiyatı {_format_age(cache_age_seconds)} önce güncellendi. "
            "Satın almadan önce güncel fiyatı kontrol edin."
        )
    elif staleness_level == "expired":
        warnings.append(
            f"Fiyat bilgisi {_format_age(cache_age_seconds)} önce alındı ve süresi dolmuş. "
            "Satın alma kararı vermeden önce fiyatı yenileyin."
        )

    if volatility > 0.3 and staleness_level != "fresh":
        warnings.append(
            "Bu ürün fiyatı dalgalı bir geçmişe sahip; güncel fiyat önemli ölçüde farklı olabilir."
        )

    flash = detect_flash_sale(product)
    if flash["is_flash_sale"] and is_stale:
        warnings.append(
            "Flaş indirim tespit edildi ancak veri eskimiş olabilir; "
            "kampanya sona ermiş ya da fiyat değişmiş olabilir."
        )

    # Confidence: higher when we have price history and a clear age signal
    confidence = _compute_confidence(age_ratio, price_history)

    logger.debug(
        "Staleness for '%s': level=%s age=%.0fs ttl=%ds volatility=%.2f",
        product.get("name", "?"),
        staleness_level,
        cache_age_seconds,
        recommended_ttl,
        volatility,
    )

    return {
        "is_stale": is_stale,
        "staleness_level": staleness_level,
        "recommended_ttl_seconds": recommended_ttl,
        "warnings": warnings,
        "confidence": round(confidence, 2),
    }


def detect_flash_sale(product: dict) -> dict:
    """Check a product for flash-sale indicators.

    Parameters
    ----------
    product:
        Product dict; inspects ``price``, ``original_price``, ``title``,
        ``name``, and ``description`` fields.

    Returns
    -------
    Dict with keys:

    - ``is_flash_sale`` -- ``True`` if flash-sale signals are present.
    - ``indicators`` -- list of human-readable indicator strings.
    - ``urgency`` -- one of "none" / "low" / "medium" / "high".
    """
    indicators: list[str] = []

    # Signal 1: large discount
    price = product.get("price") or 0
    original_price = product.get("original_price") or 0
    if price and original_price and original_price > price:
        discount_pct = ((original_price - price) / original_price) * 100
        if discount_pct > 50:
            indicators.append(
                f"Yüksek indirim oranı: %{discount_pct:.0f} (flaş indirim olabilir)"
            )

    # Signal 2: time-limited / stock-limited keywords in text fields
    text_fields = [
        product.get("title", ""),
        product.get("name", ""),
        product.get("description", ""),
        product.get("badge", ""),
    ]
    combined_text = " ".join(str(f) for f in text_fields if f)

    keyword_hits: list[str] = []
    for pattern in _FLASH_RE:
        match = pattern.search(combined_text)
        if match:
            keyword_hits.append(match.group(0))

    if keyword_hits:
        unique_hits = list(dict.fromkeys(k.lower() for k in keyword_hits))
        indicators.append(
            "Flaş/kampanya anahtar kelimesi bulundu: " + ", ".join(f'"{k}"' for k in unique_hits)
        )

    is_flash_sale = len(indicators) > 0

    # Urgency scale
    if not is_flash_sale:
        urgency = "none"
    elif len(indicators) >= 2:
        urgency = "high"
    elif keyword_hits:
        urgency = "medium"
    else:
        urgency = "low"

    if is_flash_sale:
        logger.info(
            "Flash sale detected for '%s': urgency=%s indicators=%d",
            product.get("name", "?"),
            urgency,
            len(indicators),
        )

    return {
        "is_flash_sale": is_flash_sale,
        "indicators": indicators,
        "urgency": urgency,
    }


def get_recommended_ttl(category: str | None, volatility: float) -> int:
    """Return a cache TTL in seconds adjusted for category and price volatility.

    Parameters
    ----------
    category:
        Product category string (case-insensitive).  Unknown categories use
        ``DEFAULT_TTL``.
    volatility:
        Price volatility score in the range 0.0–1.0.  Higher values shorten
        the returned TTL.

    Returns
    -------
    Recommended TTL in seconds (always at least 60 seconds).
    """
    base_ttl = DEFAULT_TTL
    if category:
        base_ttl = _CATEGORY_TTL.get(category.lower().strip(), DEFAULT_TTL)

    # Volatility factor: 1.0 at zero volatility, 0.25 at full volatility
    volatility_clamped = max(0.0, min(1.0, volatility))
    factor = 1.0 - (volatility_clamped * 0.75)

    ttl = int(base_ttl * factor)
    return max(60, ttl)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_volatility(price_history: list[dict] | None) -> float:
    """Return a 0.0–1.0 volatility score from price history.

    Uses the coefficient of variation (std / mean) capped at 1.0.
    Returns 0.0 when there is insufficient history.
    """
    if not price_history or len(price_history) < 3:
        return 0.0

    prices = [float(p["price"]) for p in price_history if p.get("price")]
    if len(prices) < 3:
        return 0.0

    mean_price = statistics.mean(prices)
    if mean_price <= 0:
        return 0.0

    stdev = statistics.stdev(prices)
    cv = stdev / mean_price  # coefficient of variation
    return round(min(1.0, cv), 3)


def _compute_confidence(age_ratio: float, price_history: list[dict] | None) -> float:
    """Estimate confidence in the staleness assessment.

    Confidence is higher when we have price history and when the age ratio
    is far from the threshold boundaries.
    """
    # Base: how far from the nearest threshold boundary (0 = right on edge)
    thresholds = [FRESH_THRESHOLD, AGING_THRESHOLD, STALE_THRESHOLD]
    min_distance = min(abs(age_ratio - t) for t in thresholds)
    boundary_confidence = min(1.0, min_distance / 0.2)  # full confidence when 20% away

    has_history = price_history is not None and len(price_history) >= 3
    history_bonus = 0.2 if has_history else 0.0

    return min(1.0, boundary_confidence * 0.8 + history_bonus)


def _format_age(seconds: float) -> str:
    """Return a human-readable age string in Turkish."""
    if seconds < 3600:
        minutes = int(seconds // 60)
        return f"{minutes} dakika"
    elif seconds < 86400:
        hours = int(seconds // 3600)
        return f"{hours} saat"
    else:
        days = int(seconds // 86400)
        return f"{days} gün"
