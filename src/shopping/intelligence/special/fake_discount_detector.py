"""Detect fake discounts.

Identifies price manipulation patterns where sellers inflate the
"original" price before a campaign to show a larger discount percentage
than what customers actually receive.
"""

from __future__ import annotations

import statistics
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("shopping.intelligence.special.fake_discount")


async def detect_fake_discount(
    product: dict,
    price_history: list[dict],
) -> dict:
    """Determine whether a product's current discount is genuine.

    Parameters
    ----------
    product:
        Product dict with ``price``, ``original_price``, and optionally
        ``discount_pct``.
    price_history:
        List of ``{"price": float, "observed_at": float}`` dicts,
        ordered oldest-first.

    Returns
    -------
    Dict with keys:

    - ``is_fake`` -- ``True`` if the discount is likely inflated.
    - ``confidence`` -- 0.0 to 1.0 confidence in the assessment.
    - ``evidence`` -- list of human-readable evidence strings.
    - ``real_discount_pct`` -- estimated genuine discount percentage.
    """
    current_price = product.get("price", 0)
    original_price = product.get("original_price", 0)

    if not current_price or not original_price or original_price <= current_price:
        return {
            "is_fake": False,
            "confidence": 0.0,
            "evidence": ["Indirim bilgisi yetersiz veya indirim yok"],
            "real_discount_pct": 0.0,
        }

    claimed_discount = ((original_price - current_price) / original_price) * 100
    evidence: list[str] = []
    fake_signals = 0
    total_checks = 0

    # Check 1: Was the "original" price ever the actual selling price?
    if price_history and len(price_history) >= 3:
        total_checks += 1
        prices = [p["price"] for p in price_history]
        historical_avg = statistics.mean(prices)
        historical_max = max(prices)

        if original_price > historical_max * 1.1:
            fake_signals += 1
            evidence.append(
                f"Orijinal fiyat ({original_price:.0f} TL) gecmis en yuksek "
                f"fiyatin ({historical_max:.0f} TL) uzerinde"
            )

        # Check 2: Price inflation before discount
        inflation_result = check_price_inflation(original_price, current_price, historical_avg)
        total_checks += 1
        if inflation_result["is_inflated"]:
            fake_signals += 1
            evidence.append(inflation_result["reason"])

        # Calculate real discount based on historical average
        if historical_avg > 0:
            real_discount = max(0, ((historical_avg - current_price) / historical_avg) * 100)
        else:
            real_discount = 0.0
    else:
        total_checks += 1
        real_discount = claimed_discount * 0.5  # Without history, assume half is real
        evidence.append("Yeterli fiyat gecmisi yok; gercek indirim tahmin edilemedi")

    # Check 3: Suspiciously round original price
    total_checks += 1
    if original_price == round(original_price, -2) and original_price > 100:
        # Perfectly round original (e.g. 5000, 3000) can be a sign of made-up price
        fake_signals += 0.3
        evidence.append(f"Orijinal fiyat suspheli derecede yuvarlak: {original_price:.0f} TL")

    # Check 4: Discount percentage too good to be true
    total_checks += 1
    if claimed_discount > 70:
        fake_signals += 1
        evidence.append(f"Iddia edilen indirim orani cok yuksek: %{claimed_discount:.0f}")
    elif claimed_discount > 50:
        fake_signals += 0.5
        evidence.append(f"Indirim orani dikkat cekici derecede yuksek: %{claimed_discount:.0f}")

    confidence = min(1.0, fake_signals / max(total_checks, 1))
    is_fake = confidence >= 0.4

    if is_fake:
        logger.info(
            "Fake discount detected for '%s': claimed %.0f%%, real ~%.0f%%",
            product.get("name", "?"), claimed_discount, real_discount,
        )

    return {
        "is_fake": is_fake,
        "confidence": round(confidence, 2),
        "evidence": evidence,
        "real_discount_pct": round(real_discount, 1),
        "claimed_discount_pct": round(claimed_discount, 1),
    }


def check_price_inflation(
    original: float,
    current: float,
    history_avg: float,
) -> dict:
    """Check if the original price was artificially inflated.

    Parameters
    ----------
    original:
        The claimed original / list price.
    current:
        The current sale price.
    history_avg:
        The historical average selling price.

    Returns
    -------
    Dict with ``is_inflated`` (bool) and ``reason`` (str).
    """
    if history_avg <= 0:
        return {"is_inflated": False, "reason": "Yeterli veri yok"}

    # If "original" is much higher than historical average, it's inflated
    inflation_ratio = original / history_avg

    if inflation_ratio > 1.3:
        return {
            "is_inflated": True,
            "reason": (
                f"Orijinal fiyat ({original:.0f} TL) gecmis ortalamadan "
                f"({history_avg:.0f} TL) %{(inflation_ratio - 1) * 100:.0f} daha yuksek"
            ),
        }

    return {"is_inflated": False, "reason": "Orijinal fiyat gecmis ortalamayla tutarli"}


def check_cross_store_consistency(prices: dict[str, float]) -> dict:
    """Check price consistency across multiple stores.

    Parameters
    ----------
    prices:
        Mapping of store name to price (e.g. ``{"trendyol": 4999, "hepsiburada": 5200}``).

    Returns
    -------
    Dict with ``is_consistent`` (bool), ``spread_pct`` (float), ``cheapest``
    and ``most_expensive`` store names, and ``notes``.
    """
    if len(prices) < 2:
        return {
            "is_consistent": True,
            "spread_pct": 0.0,
            "cheapest": None,
            "most_expensive": None,
            "notes": ["Karsilastirma icin en az 2 magaza gerekli"],
        }

    values = list(prices.values())
    min_price = min(values)
    max_price = max(values)
    spread_pct = ((max_price - min_price) / min_price) * 100

    cheapest = min(prices, key=prices.get)
    most_expensive = max(prices, key=prices.get)

    notes = []
    if spread_pct > 30:
        notes.append(
            f"Fiyat farki cok yuksek (%{spread_pct:.0f}); "
            f"en ucuz magaza indirim yapmis olabilir veya en pahali magaza sisirmis olabilir"
        )
    elif spread_pct > 15:
        notes.append(f"Magazalar arasi makul fiyat farki: %{spread_pct:.0f}")

    return {
        "is_consistent": spread_pct <= 15,
        "spread_pct": round(spread_pct, 1),
        "cheapest": cheapest,
        "most_expensive": most_expensive,
        "notes": notes,
    }
