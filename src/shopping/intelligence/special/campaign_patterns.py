"""Campaign Pattern Learner for the Turkish market.

Learns per-category discount patterns from observed price history and
predicts savings for upcoming campaign events. Storage is in-memory and
grows as new observations are recorded during runtime.
"""

from __future__ import annotations

from datetime import date

from src.infra.logging_config import get_logger

logger = get_logger("shopping.intelligence.special.campaign_patterns")

# ─── In-memory storage ──────────────────────────────────────────────────────
# Structure:
#   _observations[category] = [
#       {"event_name": str, "discount_pct": float, "observed_at": str}, ...
#   ]

_observations: dict[str, list[dict]] = {}

# ─── Static Turkish Shopping Calendar ────────────────────────────────────────
# Each entry describes one campaign window.
# Ramazan 2026: fasting ~Feb 18 – Mar 19; Eid al-Fitr (Ramazan Bayramı) ~Mar 20-22.
# Kurban Bayramı 2026: ~May 27-30 (approximate).

_CAMPAIGN_CALENDAR: list[dict] = [
    {
        "event": "Yılbaşı İndirimleri",
        "month": 1,
        "day_start": 1,
        "day_end": 15,
        "expected_discount_pct": 20.0,
        "categories_affected": ["electronics", "clothing", "home", "gift"],
        "advice": "Yılbaşı sonrası fiyatlar düşer; elektronik ve giyimde fırsat yakalayabilirsiniz.",
    },
    {
        "event": "Kış Sonu İndirimleri",
        "month": 1,
        "day_start": 16,
        "day_end": 31,
        "expected_discount_pct": 30.0,
        "categories_affected": ["clothing", "shoes", "outdoor", "home"],
        "advice": "Kış sonu sezon kapanışında tekstil ve ayakkabıda derin indirimler beklenir.",
    },
    {
        "event": "Sevgililer Günü",
        "month": 2,
        "day_start": 7,
        "day_end": 14,
        "expected_discount_pct": 15.0,
        "categories_affected": ["gift", "jewelry", "cosmetics", "clothing"],
        "advice": "Hediyelik, takı ve kozmetikte kampanyalar yoğunlaşır.",
    },
    {
        "event": "Ramazan Bayramı (Eid al-Fitr)",
        "month": 3,
        "day_start": 18,
        "day_end": 25,
        "expected_discount_pct": 20.0,
        "categories_affected": ["food", "clothing", "gift", "home"],
        "advice": "Ramazan döneminde gıda ve giyimde özel kampanyalar çıkar; erken alışveriş yapın.",
    },
    {
        "event": "8 Mart Dünya Kadınlar Günü",
        "month": 3,
        "day_start": 1,
        "day_end": 8,
        "expected_discount_pct": 15.0,
        "categories_affected": ["cosmetics", "clothing", "gift"],
        "advice": "Kozmetik ve kadın giyiminde 8 Mart kampanyaları başlar.",
    },
    {
        "event": "Nevruz",
        "month": 3,
        "day_start": 21,
        "day_end": 23,
        "expected_discount_pct": 10.0,
        "categories_affected": ["food", "home", "gift"],
        "advice": "Baharın gelişiyle birlikte ev ve gıda kategorisinde kısa süreli kampanyalar görülür.",
    },
    {
        "event": "23 Nisan Ulusal Egemenlik ve Çocuk Bayramı",
        "month": 4,
        "day_start": 22,
        "day_end": 24,
        "expected_discount_pct": 10.0,
        "categories_affected": ["toy", "stationery", "clothing", "gift"],
        "advice": "Çocuk ürünleri ve oyuncaklarda 23 Nisan indirimleri kaçırılmaz.",
    },
    {
        "event": "Anneler Günü",
        "month": 5,
        "day_start": 5,
        "day_end": 12,
        "expected_discount_pct": 15.0,
        "categories_affected": ["gift", "electronics", "cosmetics", "home", "jewelry"],
        "advice": "Anneler Günü öncesi hediyelik ve kozmetikte kampanya yoğunluğu artar.",
    },
    {
        "event": "Kurban Bayramı",
        "month": 5,
        "day_start": 27,
        "day_end": 30,
        "expected_discount_pct": 15.0,
        "categories_affected": ["food", "clothing", "home", "gift"],
        "advice": "Kurban Bayramı öncesinde gıda ve giyimde kampanyalar yoğunlaşır.",
    },
    {
        "event": "Babalar Günü",
        "month": 6,
        "day_start": 14,
        "day_end": 21,
        "expected_discount_pct": 10.0,
        "categories_affected": ["electronics", "clothing", "gift", "sports"],
        "advice": "Babalar Günü için elektronik ve spor ürünlerinde iyi fırsatlar çıkar.",
    },
    {
        "event": "Yaz İndirimleri",
        "month": 6,
        "day_start": 25,
        "day_end": 30,
        "expected_discount_pct": 25.0,
        "categories_affected": ["clothing", "shoes", "sports", "outdoor", "travel"],
        "advice": "Yaz indirimleri başlarken tekstil ve spor kategorilerinde büyük fırsatlar olur.",
    },
    {
        "event": "Prime Day / Trendyol Süper İndirim",
        "month": 7,
        "day_start": 8,
        "day_end": 15,
        "expected_discount_pct": 30.0,
        "categories_affected": ["electronics", "home", "clothing", "cosmetics"],
        "advice": "Temmuz'daki büyük platform kampanyaları için sepete ekleyip indirim gününü bekleyin.",
    },
    {
        "event": "Okula Dönüş",
        "month": 8,
        "day_start": 20,
        "day_end": 31,
        "expected_discount_pct": 20.0,
        "categories_affected": ["electronics", "stationery", "backpack", "laptop", "tablet", "clothing"],
        "advice": "Okul sezonu öncesi laptop, tablet ve kırtasiyede fiyatlar rekabetçi olur.",
    },
    {
        "event": "Okula Dönüş (Devam)",
        "month": 9,
        "day_start": 1,
        "day_end": 20,
        "expected_discount_pct": 15.0,
        "categories_affected": ["electronics", "stationery", "backpack", "laptop", "tablet"],
        "advice": "Eylül başında okul alışverişleri devam eder; fiyatlar hâlâ uygundur.",
    },
    {
        "event": "Cumhuriyet Bayramı (29 Ekim)",
        "month": 10,
        "day_start": 27,
        "day_end": 30,
        "expected_discount_pct": 20.0,
        "categories_affected": ["electronics", "home", "clothing"],
        "advice": "29 Ekim sürecinde büyük platformlar özel kampanya başlatır; elektronik için ideal.",
    },
    {
        "event": "11.11 Bekarlar Günü",
        "month": 11,
        "day_start": 10,
        "day_end": 12,
        "expected_discount_pct": 25.0,
        "categories_affected": ["electronics", "clothing", "cosmetics", "home"],
        "advice": "11.11 günü kısa ama derin indirimler sunar; hazırlıklı olun.",
    },
    {
        "event": "Black Friday / Efsane Cuma",
        "month": 11,
        "day_start": 20,
        "day_end": 30,
        "expected_discount_pct": 35.0,
        "categories_affected": [
            "electronics", "clothing", "home", "cosmetics", "shoes",
            "sports", "toy", "everything",
        ],
        "advice": "Yılın en büyük indirim dönemi; büyük alışverişleri bu döneme erteleyin.",
    },
    {
        "event": "Yılbaşı Alışverişi",
        "month": 12,
        "day_start": 15,
        "day_end": 31,
        "expected_discount_pct": 15.0,
        "categories_affected": ["gift", "food", "decoration", "clothing", "toy"],
        "advice": "Yılbaşı hediyelerini aralık ortasında almak hem seçenek hem de fiyat açısından avantajlıdır.",
    },
]


# ─── Public API ──────────────────────────────────────────────────────────────


def record_campaign(
    category: str,
    event_name: str,
    discount_pct: float,
    observed_at: str,
) -> None:
    """Record an observed campaign discount for a category.

    Parameters
    ----------
    category:
        Product category (e.g. ``"electronics"``).
    event_name:
        Name of the campaign or shopping event.
    discount_pct:
        Observed discount percentage (e.g. ``25.0`` for 25 %).
    observed_at:
        ISO-8601 date string when the discount was observed (e.g. ``"2026-11-28"``).
    """
    key = category.lower().strip()
    if key not in _observations:
        _observations[key] = []

    entry = {
        "event_name": event_name.strip(),
        "discount_pct": float(discount_pct),
        "observed_at": observed_at,
    }
    _observations[key].append(entry)
    logger.debug(
        "Recorded campaign: category=%s event=%s discount=%.1f%% at=%s",
        key,
        event_name,
        discount_pct,
        observed_at,
    )


def get_category_patterns(category: str) -> dict:
    """Return learned discount patterns for *category*.

    If no observations exist yet the function returns safe zero-value defaults
    with a Turkish note explaining that data is still being collected.

    Returns
    -------
    Dict with:
    - ``avg_discount_pct`` (float)
    - ``best_discount_pct`` (float)
    - ``typical_events`` (list[str]) — deduplicated event names seen
    - ``observations`` (int)
    - ``prediction`` (str) — Turkish advice based on collected data
    """
    key = category.lower().strip()
    records = _observations.get(key, [])

    if not records:
        logger.info("No observations yet for category '%s'", key)
        return {
            "avg_discount_pct": 0.0,
            "best_discount_pct": 0.0,
            "typical_events": [],
            "observations": 0,
            "prediction": (
                f"'{category}' kategorisi için henüz gözlem kaydı yok. "
                "Kampanyalar takip edildikçe tahminler otomatik olarak gelişecek."
            ),
        }

    discounts = [r["discount_pct"] for r in records]
    avg = round(sum(discounts) / len(discounts), 1)
    best = round(max(discounts), 1)
    seen_events: list[str] = []
    for r in records:
        if r["event_name"] not in seen_events:
            seen_events.append(r["event_name"])

    if best >= 40:
        quality = "çok yüksek"
        tip = "Büyük kampanyaları mutlaka bekleyin; tasarruf önemli olabilir."
    elif best >= 25:
        quality = "yüksek"
        tip = "Kampanya dönemlerinde alışveriş yapmanız tavsiye edilir."
    elif best >= 15:
        quality = "orta düzeyde"
        tip = "Kampanya dönemlerinde makul tasarruf sağlanabilir."
    else:
        quality = "düşük"
        tip = "İndirimler sınırlı; aciliyet varsa beklemeden alabilirsiniz."

    prediction = (
        f"'{category}' kategorisinde {len(records)} gözleme göre ortalama "
        f"%{avg} indirim, en iyi gözlenen indirim %{best}. "
        f"İndirim potansiyeli {quality}. {tip}"
    )

    logger.debug(
        "Patterns for '%s': obs=%d avg=%.1f best=%.1f",
        key,
        len(records),
        avg,
        best,
    )

    return {
        "avg_discount_pct": avg,
        "best_discount_pct": best,
        "typical_events": seen_events,
        "observations": len(records),
        "prediction": prediction,
    }


def predict_upcoming_campaigns(category: str | None = None) -> list[dict]:
    """Predict upcoming campaign windows from the Turkish shopping calendar.

    Parameters
    ----------
    category:
        When given, only returns campaigns that affect this category.
        ``None`` returns all upcoming campaigns.

    Returns
    -------
    List of dicts sorted by proximity, each with:
    - ``event`` (str)
    - ``expected_date_range`` (str)
    - ``expected_discount_pct`` (float) — blended from static calendar +
      any learned observations for the category
    - ``categories_affected`` (list[str])
    - ``advice`` (str) — Turkish advice
    """
    today = date.today()
    results: list[dict] = []

    filter_key = category.lower().strip() if category else None

    for item in _CAMPAIGN_CALENDAR:
        cats = item["categories_affected"]
        if filter_key and filter_key not in cats and "everything" not in cats:
            continue

        # Build start/end for this year and next if already passed
        try:
            start = date(today.year, item["month"], item["day_start"])
            end = date(today.year, item["month"], item["day_end"])
        except ValueError:
            logger.warning("Invalid date in calendar entry '%s'; skipping.", item["event"])
            continue

        if end < today:
            try:
                start = date(today.year + 1, item["month"], item["day_start"])
                end = date(today.year + 1, item["month"], item["day_end"])
            except ValueError:
                continue

        # Blend static discount with any learned observations
        static_discount = item["expected_discount_pct"]
        blended_discount = static_discount

        if filter_key:
            patterns = get_category_patterns(filter_key)
            if patterns["observations"] > 0 and patterns["avg_discount_pct"] > 0:
                # Weight: 60 % static calendar, 40 % learned average
                blended_discount = round(
                    0.6 * static_discount + 0.4 * patterns["avg_discount_pct"], 1
                )

        days_until = max(0, (start - today).days)
        date_range = f"{start.strftime('%d %B %Y')} – {end.strftime('%d %B %Y')}"

        results.append({
            "event": item["event"],
            "expected_date_range": date_range,
            "days_until": days_until,
            "expected_discount_pct": blended_discount,
            "categories_affected": cats,
            "advice": item["advice"],
        })

    results.sort(key=lambda x: x["days_until"])

    logger.info(
        "predict_upcoming_campaigns(category=%s) → %d events",
        filter_key or "all",
        len(results),
    )
    return results


def get_campaign_calendar() -> list[dict]:
    """Return the full static Turkish shopping campaign calendar.

    Returns
    -------
    List of dicts with ``event``, ``month``, ``day_start``, ``day_end``,
    ``expected_discount_pct``, ``categories_affected``, ``advice``.
    """
    logger.debug("Returning full campaign calendar (%d entries)", len(_CAMPAIGN_CALENDAR))
    return list(_CAMPAIGN_CALENDAR)
