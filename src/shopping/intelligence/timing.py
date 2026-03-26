"""Market Timing Advisor — recommends buy-now vs wait based on Turkish market calendar,
price history, and currency trends."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

from src.infra.logging_config import get_logger

logger = get_logger("shopping.intelligence.timing")

_KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"


# ─── LLM helper ─────────────────────────────────────────────────────────────

async def _llm_call(prompt: str, system: str = "", temperature: float = 0.3) -> str:
    try:
        import litellm
        response = await litellm.acompletion(
            model="openai/local",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=2048,
        )
        return response.choices[0].message.content
    except Exception:
        return ""


# ─── Knowledge loaders ──────────────────────────────────────────────────────

def _load_market_calendar() -> str:
    """Load Turkish market knowledge markdown."""
    path = _KNOWLEDGE_DIR / "turkish_market.md"
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        logger.warning("Could not load turkish_market.md")
        return ""


# ─── Heuristic helpers ──────────────────────────────────────────────────────

_SALE_EVENTS = [
    {"name": "11.11 Bekarlar Günü", "month": 11, "day_start": 10, "day_end": 11, "discount_pct": (20, 50)},
    {"name": "Efsane Cuma", "month": 11, "day_start": 20, "day_end": 30, "discount_pct": (15, 60)},
    {"name": "Yılbaşı İndirimleri", "month": 12, "day_start": 20, "day_end": 31, "discount_pct": (10, 40)},
    {"name": "Yılbaşı İndirimleri", "month": 1, "day_start": 1, "day_end": 5, "discount_pct": (10, 40)},
    {"name": "Sevgililer Günü", "month": 2, "day_start": 1, "day_end": 14, "discount_pct": (10, 30)},
    {"name": "Anneler Günü", "month": 5, "day_start": 1, "day_end": 14, "discount_pct": (10, 25)},
    {"name": "Babalar Günü", "month": 6, "day_start": 14, "day_end": 21, "discount_pct": (10, 25)},
    {"name": "Okula Dönüş", "month": 8, "day_start": 15, "day_end": 30, "discount_pct": (10, 35)},
    {"name": "Okula Dönüş", "month": 9, "day_start": 1, "day_end": 15, "discount_pct": (10, 35)},
]

_BEST_BUY_WINDOWS: dict[str, dict] = {
    "klima": {"months": [3, 4], "reason": "Pre-season discounts, installation availability"},
    "kombi": {"months": [5, 6], "reason": "Off-season clearance"},
    "beyaz eşya": {"months": [11], "reason": "Best sale prices at 11.11 and Efsane Cuma"},
    "laptop": {"months": [8, 11], "reason": "Okula Dönüş and Efsane Cuma"},
    "telefon": {"months": [11], "reason": "Efsane Cuma best electronics deals"},
    "kış lastiği": {"months": [8, 9], "reason": "Pre-season, before cold snap price surge"},
    "yaz lastiği": {"months": [2, 3], "reason": "Pre-season deals"},
    "bahçe": {"months": [2, 3], "reason": "Pre-season deals"},
    "electronics": {"months": [11], "reason": "Efsane Cuma best electronics deals"},
    "bilgisayar": {"months": [8, 11], "reason": "Okula Dönüş and Efsane Cuma"},
}


def _days_until_next_event(now: datetime) -> tuple[str, int] | None:
    """Return the nearest upcoming sale event name and days until it starts."""
    best: tuple[str, int] | None = None
    for evt in _SALE_EVENTS:
        try:
            start = datetime(now.year, evt["month"], evt["day_start"])
            if start < now:
                start = datetime(now.year + 1, evt["month"], evt["day_start"])
            delta = (start - now).days
            if best is None or delta < best[1]:
                best = (evt["name"], delta)
        except Exception:
            continue
    return best


def _check_best_window(category: str, now: datetime) -> dict | None:
    """Check if the current month falls in a known best-buy window for the category."""
    cat_lower = category.lower()
    for key, info in _BEST_BUY_WINDOWS.items():
        if key in cat_lower or cat_lower in key:
            in_window = now.month in info["months"]
            return {"category_match": key, "in_window": in_window, "best_months": info["months"], "reason": info["reason"]}
    return None


def _analyze_price_trend(history: list[dict]) -> dict:
    """Simple price trend analysis from price history entries."""
    if len(history) < 2:
        return {"trend": "insufficient_data", "change_pct": 0.0}

    recent = history[-1]["price"]
    oldest = history[0]["price"]
    change_pct = ((recent - oldest) / oldest) * 100 if oldest else 0.0

    # Check last 30 days trend
    thirty_days_ago = time.time() - (30 * 86400)
    recent_entries = [h for h in history if h.get("observed_at", 0) > thirty_days_ago]
    if len(recent_entries) >= 2:
        short_change = ((recent_entries[-1]["price"] - recent_entries[0]["price"]) / recent_entries[0]["price"]) * 100
    else:
        short_change = 0.0

    if short_change < -5:
        trend = "dropping"
    elif short_change > 5:
        trend = "rising"
    else:
        trend = "stable"

    return {
        "trend": trend,
        "change_pct": round(change_pct, 1),
        "short_term_change_pct": round(short_change, 1),
        "data_points": len(history),
    }


# ─── Main entry point ───────────────────────────────────────────────────────

async def advise_timing(category: str, product_name: str = "") -> dict:
    """Produce a buy/wait/neutral recommendation for the given category and product.

    Returns:
        dict with keys: recommendation, reason, confidence, details
    """
    logger.info("Advising timing", category=category, product_name=product_name)

    now = datetime.now()
    signals: list[dict] = []

    # --- Signal 1: upcoming sale events ---
    next_event = _days_until_next_event(now)
    if next_event:
        event_name, days_away = next_event
        if days_away <= 14:
            signals.append({
                "source": "sale_calendar",
                "direction": "wait",
                "weight": 0.8,
                "detail": f"{event_name} in {days_away} days — discounts likely",
            })
        elif days_away <= 30:
            signals.append({
                "source": "sale_calendar",
                "direction": "wait",
                "weight": 0.4,
                "detail": f"{event_name} in {days_away} days — may be worth waiting",
            })
        else:
            signals.append({
                "source": "sale_calendar",
                "direction": "neutral",
                "weight": 0.1,
                "detail": f"Next major sale ({event_name}) is {days_away} days away",
            })

    # --- Signal 2: best-buy window for category ---
    window = _check_best_window(category, now)
    if window:
        if window["in_window"]:
            signals.append({
                "source": "seasonal_window",
                "direction": "buy_now",
                "weight": 0.7,
                "detail": f"Currently in best-buy window for {window['category_match']}: {window['reason']}",
            })
        else:
            months_str = ", ".join(str(m) for m in window["best_months"])
            signals.append({
                "source": "seasonal_window",
                "direction": "wait",
                "weight": 0.3,
                "detail": f"Best months for {window['category_match']} are {months_str}: {window['reason']}",
            })

    # --- Signal 3: price history from cache (if available) ---
    price_trend = {"trend": "insufficient_data"}
    try:
        from src.shopping.cache import get_price_history
        if product_name:
            history = await get_price_history(product_name)
            if history:
                price_trend = _analyze_price_trend(history)
                if price_trend["trend"] == "dropping":
                    signals.append({
                        "source": "price_trend",
                        "direction": "wait",
                        "weight": 0.6,
                        "detail": f"Price trending down ({price_trend['short_term_change_pct']}% in 30d)",
                    })
                elif price_trend["trend"] == "rising":
                    signals.append({
                        "source": "price_trend",
                        "direction": "buy_now",
                        "weight": 0.5,
                        "detail": f"Price trending up ({price_trend['short_term_change_pct']}% in 30d)",
                    })
    except Exception as exc:
        logger.debug("Price history lookup skipped", error=str(exc))

    # --- Signal 4: LLM-based analysis (for nuanced cases) ---
    if product_name and category:
        market_knowledge = _load_market_calendar()
        llm_prompt = (
            f"Product: {product_name}\nCategory: {category}\n"
            f"Current date: {now.strftime('%Y-%m-%d')}\n"
            f"Price trend: {price_trend.get('trend', 'unknown')}\n\n"
            f"Based on the Turkish market calendar below, should the buyer wait or buy now?\n"
            f"Answer with JSON: {{\"recommendation\": \"wait\"|\"buy_now\"|\"neutral\", "
            f"\"reason\": \"...\", \"confidence\": 0.0-1.0}}\n\n"
            f"--- Market Knowledge ---\n{market_knowledge[:3000]}"
        )
        llm_response = await _llm_call(
            llm_prompt,
            system="You are a Turkish market timing advisor. Return only valid JSON.",
        )
        if llm_response:
            try:
                llm_data = json.loads(llm_response)
                llm_rec = llm_data.get("recommendation", "neutral")
                signals.append({
                    "source": "llm_analysis",
                    "direction": llm_rec,
                    "weight": 0.5,
                    "detail": llm_data.get("reason", ""),
                })
            except (json.JSONDecodeError, TypeError):
                logger.debug("LLM response was not valid JSON", response=llm_response[:200])

    # --- Aggregate signals ---
    if not signals:
        logger.info("No timing signals available, returning neutral")
        return {
            "recommendation": "neutral",
            "reason": "Insufficient data for a timing recommendation.",
            "confidence": 0.2,
            "details": {"signals": [], "price_trend": price_trend},
        }

    wait_score = sum(s["weight"] for s in signals if s["direction"] == "wait")
    buy_score = sum(s["weight"] for s in signals if s["direction"] == "buy_now")
    total_weight = sum(s["weight"] for s in signals)

    if wait_score > buy_score and wait_score > 0.4:
        recommendation = "wait"
        primary_reasons = [s["detail"] for s in signals if s["direction"] == "wait"]
    elif buy_score > wait_score and buy_score > 0.4:
        recommendation = "buy_now"
        primary_reasons = [s["detail"] for s in signals if s["direction"] == "buy_now"]
    else:
        recommendation = "neutral"
        primary_reasons = [s["detail"] for s in signals]

    confidence = min(max(total_weight / 2.0, 0.1), 1.0)
    reason = primary_reasons[0] if primary_reasons else "Mixed signals."

    result = {
        "recommendation": recommendation,
        "reason": reason,
        "confidence": round(confidence, 2),
        "details": {
            "signals": signals,
            "price_trend": price_trend,
            "next_sale_event": next_event,
            "seasonal_window": window,
        },
    }
    logger.info("Timing advice generated", recommendation=recommendation, confidence=result["confidence"])
    return result
