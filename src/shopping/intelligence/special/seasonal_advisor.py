"""Seasonal buying advice for the Turkish market.

Uses the Turkish shopping calendar (Black Friday, 11.11, Ramadan deals,
summer sales, back-to-school, etc.) to recommend optimal purchase timing.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("shopping.intelligence.special.seasonal")

# ─── Turkish Shopping Calendar ──────────────────────────────────────────────
# Each event: (name, month, day_start, day_end, categories, typical_discount_pct)

_SALES_EVENTS: list[dict] = [
    {
        "name": "Yilbasi Indirimleri",
        "month": 1, "day_start": 1, "day_end": 15,
        "categories": ["electronics", "clothing", "home", "gift"],
        "discount_pct": 20,
    },
    {
        "name": "Sevgililer Gunu",
        "month": 2, "day_start": 7, "day_end": 14,
        "categories": ["gift", "jewelry", "cosmetics", "clothing"],
        "discount_pct": 15,
    },
    {
        "name": "8 Mart Kadinlar Gunu",
        "month": 3, "day_start": 1, "day_end": 8,
        "categories": ["cosmetics", "clothing", "gift"],
        "discount_pct": 15,
    },
    {
        "name": "Bahar Temizligi / Ev Indirimleri",
        "month": 4, "day_start": 1, "day_end": 30,
        "categories": ["home", "cleaning", "garden", "home_improvement"],
        "discount_pct": 15,
    },
    {
        "name": "Anneler Gunu",
        "month": 5, "day_start": 5, "day_end": 12,
        "categories": ["gift", "electronics", "cosmetics", "home"],
        "discount_pct": 10,
    },
    {
        "name": "Babalar Gunu",
        "month": 6, "day_start": 14, "day_end": 21,
        "categories": ["electronics", "clothing", "gift"],
        "discount_pct": 10,
    },
    {
        "name": "Yaz Indirimleri",
        "month": 7, "day_start": 1, "day_end": 31,
        "categories": ["clothing", "shoes", "sports", "outdoor", "travel"],
        "discount_pct": 30,
    },
    {
        "name": "Yaz Sonu / Sezon Kapanisi",
        "month": 8, "day_start": 15, "day_end": 31,
        "categories": ["clothing", "shoes", "outdoor", "garden"],
        "discount_pct": 40,
    },
    {
        "name": "Okula Donus",
        "month": 9, "day_start": 1, "day_end": 20,
        "categories": ["electronics", "stationery", "backpack", "laptop", "tablet"],
        "discount_pct": 15,
    },
    {
        "name": "Ekim Elektronik Festivali",
        "month": 10, "day_start": 15, "day_end": 31,
        "categories": ["electronics", "phone", "laptop", "tv"],
        "discount_pct": 15,
    },
    {
        "name": "Bekarlar Gunu (11.11)",
        "month": 11, "day_start": 10, "day_end": 12,
        "categories": ["electronics", "clothing", "cosmetics", "home"],
        "discount_pct": 25,
    },
    {
        "name": "Black Friday / Efsane Cuma",
        "month": 11, "day_start": 20, "day_end": 30,
        "categories": ["electronics", "clothing", "home", "cosmetics", "everything"],
        "discount_pct": 30,
    },
    {
        "name": "Cyber Monday",
        "month": 12, "day_start": 1, "day_end": 3,
        "categories": ["electronics", "software", "gaming"],
        "discount_pct": 25,
    },
    {
        "name": "Yilbasi Alisverisi",
        "month": 12, "day_start": 15, "day_end": 31,
        "categories": ["gift", "food", "decoration", "clothing"],
        "discount_pct": 15,
    },
]

# Ramadan dates shift yearly; approximate for 2025-2027
_RAMADAN_EVENTS: list[dict] = [
    {"year": 2025, "month": 3, "day_start": 1, "day_end": 30},
    {"year": 2026, "month": 2, "day_start": 18, "day_end": 19},  # Eid approx
    {"year": 2027, "month": 2, "day_start": 8, "day_end": 9},
]

# Category -> best months to buy (0-indexed from most savings)
_BEST_MONTHS: dict[str, list[int]] = {
    "electronics": [11, 9, 1],          # Black Friday, back-to-school, new year
    "clothing": [7, 8, 11],             # summer sales, season end, Black Friday
    "home": [4, 11, 1],                 # spring cleaning, Black Friday, new year
    "phone": [11, 9, 10],              # Black Friday, school, October fest
    "laptop": [11, 9, 1],
    "tv": [11, 10, 1],
    "cosmetics": [11, 3, 2],
    "food": [12, 4],                    # Ramadan period, spring
    "garden": [4, 8],
    "sports": [7, 1],
}


def get_seasonal_advice(
    category: str,
    current_date: date | None = None,
) -> dict:
    """Get buying advice for *category* based on the current date.

    Parameters
    ----------
    category:
        Product category (e.g. ``"electronics"``, ``"clothing"``).
    current_date:
        Override for testing; defaults to today.

    Returns
    -------
    Dict with ``recommendation``, ``upcoming_events``, ``historical_discount_pct``,
    ``confidence``.
    """
    today = current_date or date.today()
    category_lower = category.lower().replace(" ", "_")

    upcoming = get_upcoming_sales(days_ahead=60, reference_date=today)
    relevant = [e for e in upcoming if category_lower in e["categories"] or "everything" in e["categories"]]

    best_months = _BEST_MONTHS.get(category_lower, [])
    is_best_month = today.month in best_months

    if relevant:
        nearest = relevant[0]
        days_until = nearest["days_until"]
        if days_until <= 7:
            recommendation = (
                f"Simdi alin! {nearest['name']} basladi/baslamak uzere. "
                f"Tahmini indirim: %{nearest['discount_pct']}"
            )
            confidence = 0.85
        elif days_until <= 30:
            recommendation = (
                f"{days_until} gun icinde {nearest['name']} var. "
                f"Acil degilse beklemeniz tavsiye edilir."
            )
            confidence = 0.75
        else:
            recommendation = (
                f"{nearest['name']} {days_until} gun sonra. "
                f"Aciliyet durumunuza gore karar verebilirsiniz."
            )
            confidence = 0.5
        historical_discount = nearest["discount_pct"]
    elif is_best_month:
        recommendation = "Bu ay bu kategori icin genellikle iyi firsatlar bulunur."
        confidence = 0.6
        historical_discount = 15
    else:
        recommendation = (
            "Yakin donemde bu kategoriye ozel buyuk bir indirim etkinligi yok. "
            "Ihtiyaciniz varsa alabilirsiniz."
        )
        confidence = 0.4
        historical_discount = 0

    return {
        "recommendation": recommendation,
        "upcoming_events": relevant[:3],
        "historical_discount_pct": historical_discount,
        "confidence": confidence,
        "category": category,
        "best_months": best_months,
    }


def get_upcoming_sales(
    days_ahead: int = 60,
    reference_date: date | None = None,
) -> list[dict]:
    """List upcoming Turkish sale events within *days_ahead* days.

    Parameters
    ----------
    days_ahead:
        How many days into the future to look.
    reference_date:
        Override for testing; defaults to today.

    Returns
    -------
    List of event dicts sorted by proximity, each with ``name``,
    ``start_date``, ``days_until``, ``categories``, ``discount_pct``.
    """
    today = reference_date or date.today()
    cutoff = today + timedelta(days=days_ahead)
    results: list[dict] = []

    for event in _SALES_EVENTS:
        try:
            start = date(today.year, event["month"], event["day_start"])
            end = date(today.year, event["month"], event["day_end"])
        except ValueError:
            continue

        # If the event already passed this year, try next year
        if end < today:
            try:
                start = date(today.year + 1, event["month"], event["day_start"])
                end = date(today.year + 1, event["month"], event["day_end"])
            except ValueError:
                continue

        if start <= cutoff:
            days_until = max(0, (start - today).days)
            results.append({
                "name": event["name"],
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "days_until": days_until,
                "categories": event["categories"],
                "discount_pct": event["discount_pct"],
                "active": start <= today <= end,
            })

    results.sort(key=lambda e: e["days_until"])
    return results


def is_good_time_to_buy(
    category: str,
    urgency: str = "normal",
    current_date: date | None = None,
) -> dict:
    """Simple yes/no recommendation on whether to buy now.

    Parameters
    ----------
    category:
        Product category.
    urgency:
        ``"low"`` (can wait months), ``"normal"`` (within weeks),
        ``"high"`` (need it now).
    current_date:
        Override for testing.

    Returns
    -------
    Dict with ``buy_now`` (bool), ``reason`` (str), ``wait_days`` (int or 0).
    """
    today = current_date or date.today()
    advice = get_seasonal_advice(category, today)

    if urgency == "high":
        return {
            "buy_now": True,
            "reason": "Acil ihtiyaciniz var; en iyi guncel fiyati arayiniz.",
            "wait_days": 0,
        }

    upcoming = advice.get("upcoming_events", [])
    if upcoming:
        nearest = upcoming[0]
        days = nearest["days_until"]

        if days == 0:
            return {
                "buy_now": True,
                "reason": f"{nearest['name']} aktif! Simdi almak mantikli.",
                "wait_days": 0,
            }

        if urgency == "low" and days <= 90:
            return {
                "buy_now": False,
                "reason": f"{days} gun sonra {nearest['name']} var; beklemeniz tavsiye edilir.",
                "wait_days": days,
            }

        if urgency == "normal" and days <= 30:
            return {
                "buy_now": False,
                "reason": f"{days} gun sonra {nearest['name']} var; mumkunse bekleyin.",
                "wait_days": days,
            }

    return {
        "buy_now": True,
        "reason": "Yakin donemde buyuk indirim etkinligi yok; ihtiyaciniz varsa alabilirsiniz.",
        "wait_days": 0,
    }
