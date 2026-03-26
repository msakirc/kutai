"""Seller trustworthiness scoring.

Evaluates sellers based on age, review patterns, and known signals
to produce a 0--100 trust score with badges and warnings.
"""

from __future__ import annotations

import statistics
from datetime import datetime, date
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("shopping.intelligence.special.seller_trust")

# ─── Badge Definitions ──────────────────────────────────────────────────────

_BADGES = {
    "established": {"label": "Koklu Satici", "min_months": 24},
    "highly_rated": {"label": "Yuksek Puanli", "min_rating": 4.5},
    "volume_seller": {"label": "Cok Satan", "min_sales": 1000},
    "verified": {"label": "Dogrulanmis", "requires": "is_verified"},
}


def score_seller(seller_info: dict) -> dict:
    """Compute a trust score for a seller.

    Parameters
    ----------
    seller_info:
        Dict with optional keys: ``name``, ``rating`` (0-5),
        ``review_count``, ``joined_date`` (ISO str), ``total_sales``,
        ``is_verified``, ``reviews`` (list of review dicts),
        ``return_rate``, ``response_time_hours``.

    Returns
    -------
    Dict with ``trust_score`` (0-100), ``badges`` (list of str),
    ``warnings`` (list of str), ``details`` (breakdown dict).
    """
    score = 50.0  # Start at neutral
    badges: list[str] = []
    warnings: list[str] = []

    # ── Rating component (max +20) ──────────────────────────────────────
    rating = seller_info.get("rating", 0)
    review_count = seller_info.get("review_count", 0)

    if rating >= 4.5 and review_count >= 50:
        score += 20
        badges.append(_BADGES["highly_rated"]["label"])
    elif rating >= 4.0 and review_count >= 20:
        score += 15
    elif rating >= 3.5:
        score += 5
    elif rating > 0 and rating < 3.0:
        score -= 15
        warnings.append(f"Dusuk puan: {rating:.1f}/5")

    # ── Age component (max +15) ─────────────────────────────────────────
    joined = seller_info.get("joined_date")
    if joined:
        age_result = check_seller_age(joined)
        months = age_result.get("months_active", 0)
        if months >= 24:
            score += 15
            badges.append(_BADGES["established"]["label"])
        elif months >= 12:
            score += 10
        elif months >= 6:
            score += 5
        elif months < 3:
            score -= 10
            warnings.append(age_result.get("warning", "Yeni satici"))

    # ── Volume component (max +10) ──────────────────────────────────────
    total_sales = seller_info.get("total_sales", 0)
    if total_sales >= 1000:
        score += 10
        badges.append(_BADGES["volume_seller"]["label"])
    elif total_sales >= 100:
        score += 5
    elif total_sales < 10 and total_sales > 0:
        warnings.append(f"Cok az satis: {total_sales}")

    # ── Verification bonus (+5) ─────────────────────────────────────────
    if seller_info.get("is_verified"):
        score += 5
        badges.append(_BADGES["verified"]["label"])

    # ── Return rate penalty ─────────────────────────────────────────────
    return_rate = seller_info.get("return_rate", 0)
    if return_rate > 0.15:
        score -= 15
        warnings.append(f"Yuksek iade orani: %{return_rate * 100:.0f}")
    elif return_rate > 0.10:
        score -= 5

    # ── Response time bonus ─────────────────────────────────────────────
    response_hours = seller_info.get("response_time_hours")
    if response_hours is not None:
        if response_hours <= 2:
            score += 5
        elif response_hours > 48:
            score -= 5
            warnings.append("Yavas yanit suresi")

    # ── Review authenticity check ───────────────────────────────────────
    reviews = seller_info.get("reviews", [])
    if reviews and len(reviews) >= 10:
        auth = check_review_authenticity(reviews)
        if auth["suspicious"]:
            score -= 15
            warnings.extend(auth.get("reasons", []))

    # Clamp score
    score = max(0, min(100, score))

    return {
        "trust_score": round(score),
        "badges": badges,
        "warnings": warnings,
        "details": {
            "rating": rating,
            "review_count": review_count,
            "total_sales": total_sales,
            "is_verified": seller_info.get("is_verified", False),
        },
    }


def check_seller_age(joined_date: str) -> dict:
    """Check how long a seller has been active and flag new sellers.

    Parameters
    ----------
    joined_date:
        ISO date string (e.g. ``"2023-05-15"``).

    Returns
    -------
    Dict with ``months_active``, ``is_new`` (< 3 months), and optional
    ``warning``.
    """
    try:
        joined = datetime.fromisoformat(joined_date).date()
    except (ValueError, TypeError):
        return {"months_active": 0, "is_new": True, "warning": "Katilim tarihi gecersiz"}

    today = date.today()
    delta = today - joined
    months = delta.days / 30.44  # Average month length

    result: dict[str, Any] = {
        "months_active": round(months, 1),
        "is_new": months < 3,
        "joined_date": joined.isoformat(),
    }

    if months < 1:
        result["warning"] = "Satici 1 aydan kisa suredir aktif; dikkatli olunuz"
    elif months < 3:
        result["warning"] = "Satici yeni (3 aydan az); referanslari sinirli olabilir"

    return result


def check_review_authenticity(reviews: list[dict]) -> dict:
    """Detect review manipulation patterns.

    Checks for:
    - Burst reviews (many reviews on the same day)
    - Suspiciously similar text
    - All-5-star pattern with no variance
    - Very short reviews

    Parameters
    ----------
    reviews:
        List of review dicts with optional keys: ``rating``, ``text``,
        ``date`` (ISO str), ``reviewer_name``.

    Returns
    -------
    Dict with ``suspicious`` (bool), ``confidence`` (0-1),
    ``reasons`` (list of str).
    """
    reasons: list[str] = []
    signals = 0
    total_checks = 0

    ratings = [r.get("rating", 0) for r in reviews if r.get("rating")]
    texts = [r.get("text", "") for r in reviews if r.get("text")]
    dates = [r.get("date", "") for r in reviews if r.get("date")]

    # Check 1: Rating variance
    if ratings and len(ratings) >= 5:
        total_checks += 1
        try:
            variance = statistics.variance(ratings)
        except statistics.StatisticsError:
            variance = 0
        if variance < 0.1 and statistics.mean(ratings) >= 4.8:
            signals += 1
            reasons.append("Tum degerlendirmeler neredeyse ayni puan (dusuk varyans)")

    # Check 2: Review burst (many on same day)
    if dates and len(dates) >= 5:
        total_checks += 1
        date_counts: dict[str, int] = {}
        for d in dates:
            day = d[:10]  # ISO date prefix
            date_counts[day] = date_counts.get(day, 0) + 1

        max_per_day = max(date_counts.values()) if date_counts else 0
        if max_per_day >= len(reviews) * 0.3:
            signals += 1
            reasons.append(
                f"Degerlendirmelerin %{max_per_day / len(reviews) * 100:.0f}'i ayni gunde yapilmis"
            )

    # Check 3: Very short reviews
    if texts and len(texts) >= 5:
        total_checks += 1
        short_count = sum(1 for t in texts if len(t.strip()) < 15)
        if short_count >= len(texts) * 0.6:
            signals += 1
            reasons.append("Degerlendirmelerin cogu cok kisa (< 15 karakter)")

    # Check 4: Duplicate / near-duplicate text
    if texts and len(texts) >= 5:
        total_checks += 1
        normalised = [t.lower().strip()[:50] for t in texts]
        unique_ratio = len(set(normalised)) / len(normalised)
        if unique_ratio < 0.5:
            signals += 1
            reasons.append("Benzer/ayni metinli degerlendirmeler tespit edildi")

    confidence = signals / max(total_checks, 1)

    return {
        "suspicious": confidence >= 0.4,
        "confidence": round(confidence, 2),
        "reasons": reasons,
        "total_reviews_checked": len(reviews),
    }
