"""Review Synthesizer — aggregates reviews from multiple sources with temporal
weighting, volume-aware confidence, cross-source sentiment comparison, and
defect pattern extraction."""

from __future__ import annotations

import json
import time
from datetime import datetime

from src.infra.logging_config import get_logger

logger = get_logger("shopping.intelligence.review_synthesizer")


from ._llm import _llm_call

# ─── Temporal weighting ─────────────────────────────────────────────────────

_RECENT_THRESHOLD_DAYS = 180  # 6 months


def _compute_temporal_weight(review_date: str | None) -> float:
    """Return 2.0 for recent reviews (< 6 months), 1.0 for older ones."""
    if not review_date:
        return 1.0
    try:
        dt = datetime.fromisoformat(review_date.replace("Z", "+00:00"))
        age_days = (datetime.now(dt.tzinfo) - dt).days if dt.tzinfo else (datetime.now() - dt).days
        if age_days < _RECENT_THRESHOLD_DAYS:
            return 2.0
        return 1.0
    except Exception:
        return 1.0


# ─── Volume-aware confidence ────────────────────────────────────────────────

def _volume_confidence(total_reviews: int) -> float:
    """Map review volume to a confidence multiplier (0.3 to 1.0).

    - < 5 reviews: low confidence (0.3)
    - 5-20: moderate (0.5-0.7)
    - 20-100: good (0.7-0.9)
    - 100+: high (1.0)
    """
    if total_reviews < 5:
        return 0.3
    if total_reviews < 20:
        return 0.5 + (total_reviews - 5) * 0.013  # 0.5 -> 0.7
    if total_reviews < 100:
        return 0.7 + (total_reviews - 20) * 0.0025  # 0.7 -> 0.9
    return 1.0


# ─── Sentiment helpers ──────────────────────────────────────────────────────

def _rating_to_sentiment(rating: float | None) -> str:
    if rating is None:
        return "neutral"
    if rating >= 4.0:
        return "positive"
    if rating >= 3.0:
        return "neutral"
    return "negative"


def _compute_source_breakdown(reviews: list[dict]) -> dict[str, dict]:
    """Group reviews by source and compute per-source stats."""
    sources: dict[str, list[dict]] = {}
    for r in reviews:
        src = r.get("source", "unknown")
        sources.setdefault(src, []).append(r)

    breakdown: dict[str, dict] = {}
    for src, src_reviews in sources.items():
        ratings = [r["rating"] for r in src_reviews if r.get("rating") is not None]
        avg = sum(ratings) / len(ratings) if ratings else None
        breakdown[src] = {
            "count": len(src_reviews),
            "avg_rating": round(avg, 2) if avg is not None else None,
            "sentiment": _rating_to_sentiment(avg),
        }
    return breakdown


# ─── Review quality assessment ──────────────────────────────────────────────

def _assess_review_quality(reviews: list[dict]) -> dict:
    """Produce a quality assessment of the review pool."""
    total = len(reviews)
    if total == 0:
        return {"quality": "none", "verified_pct": 0.0, "avg_length": 0}

    verified = sum(1 for r in reviews if r.get("verified_purchase"))
    verified_pct = (verified / total) * 100

    text_lengths = [len(r.get("text", "")) for r in reviews]
    avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0

    # Quality heuristic
    if verified_pct > 60 and avg_length > 50 and total >= 10:
        quality = "high"
    elif verified_pct > 30 and total >= 5:
        quality = "moderate"
    else:
        quality = "low"

    return {
        "quality": quality,
        "total_reviews": total,
        "verified_count": verified,
        "verified_pct": round(verified_pct, 1),
        "avg_text_length": round(avg_length),
    }


# ─── Main entry point ───────────────────────────────────────────────────────

async def synthesize_reviews(reviews: list[dict], product_name: str) -> dict:
    """Synthesize reviews from multiple sources into a unified analysis.

    Args:
        reviews: list of review dicts (text, source, rating, date, author,
                 verified_purchase, helpful_count, language)
        product_name: canonical product name

    Returns:
        dict with: overall_sentiment, confidence_adjusted_rating, positive_themes,
                   negative_themes, defect_patterns, warnings, turkey_specific,
                   review_quality
    """
    logger.info("Synthesizing reviews", product=product_name, review_count=len(reviews))

    if not reviews:
        return {
            "overall_sentiment": "unknown",
            "confidence_adjusted_rating": None,
            "positive_themes": [],
            "negative_themes": [],
            "defect_patterns": [],
            "warnings": ["No reviews available"],
            "turkey_specific": [],
            "review_quality": _assess_review_quality([]),
        }

    # --- Weighted average rating ---
    weighted_sum = 0.0
    weight_total = 0.0
    for r in reviews:
        rating = r.get("rating")
        if rating is None:
            continue
        w = _compute_temporal_weight(r.get("date"))
        # Boost verified purchases
        if r.get("verified_purchase"):
            w *= 1.3
        # Boost helpful reviews
        helpful = r.get("helpful_count", 0)
        if helpful > 5:
            w *= 1.2
        weighted_sum += rating * w
        weight_total += w

    raw_weighted_avg = (weighted_sum / weight_total) if weight_total else None

    # --- Volume-adjusted confidence ---
    vol_conf = _volume_confidence(len(reviews))
    if raw_weighted_avg is not None:
        # Pull toward 3.0 (neutral) when confidence is low
        confidence_adjusted = raw_weighted_avg * vol_conf + 3.0 * (1 - vol_conf)
    else:
        confidence_adjusted = None

    # --- Source breakdown ---
    source_breakdown = _compute_source_breakdown(reviews)

    # --- Cross-source divergence warning ---
    warnings: list[str] = []
    source_avgs = [s["avg_rating"] for s in source_breakdown.values() if s["avg_rating"] is not None]
    if len(source_avgs) >= 2:
        divergence = max(source_avgs) - min(source_avgs)
        if divergence > 1.5:
            warnings.append(
                f"Rating divergence across sources: {round(divergence, 1)} stars — "
                "reviews may not reflect consistent experience"
            )

    # --- Review quality ---
    review_quality = _assess_review_quality(reviews)
    if review_quality["quality"] == "low":
        warnings.append("Low review quality: few verified purchases or very short texts")

    # --- LLM-based theme/defect extraction ---
    positive_themes: list[str] = []
    negative_themes: list[str] = []
    defect_patterns: list[str] = []
    turkey_specific: list[str] = []

    # Build review sample for LLM (cap at ~30 reviews to stay within token limits)
    sample_reviews = sorted(
        reviews,
        key=lambda r: (_compute_temporal_weight(r.get("date")), r.get("helpful_count", 0)),
        reverse=True,
    )[:30]

    review_texts = []
    for r in sample_reviews:
        prefix = f"[{r.get('source', '?')} | {r.get('rating', '?')}★]"
        review_texts.append(f"{prefix} {r.get('text', '')[:300]}")

    llm_prompt = (
        f"Product: {product_name}\n"
        f"Reviews ({len(reviews)} total, showing {len(sample_reviews)} most relevant):\n\n"
        + "\n".join(review_texts)
        + "\n\nAnalyze these reviews and return JSON with:\n"
        "{\n"
        '  "positive_themes": ["theme1", "theme2", ...],\n'
        '  "negative_themes": ["theme1", "theme2", ...],\n'
        '  "defect_patterns": ["pattern1", ...],  // recurring hardware/software defects\n'
        '  "turkey_specific": ["note1", ...],  // Turkey-specific issues (service, customs, voltage, etc.)\n'
        '  "overall_sentiment": "positive"|"mixed"|"negative"\n'
        "}\n"
        "Keep each list to max 5 items. Be specific and concise."
    )

    llm_response = await _llm_call(
        llm_prompt,
        system="You are a product review analyst. Extract patterns from Turkish and English reviews. Return only valid JSON.",
    )

    if llm_response:
        try:
            llm_data = json.loads(llm_response)
            positive_themes = llm_data.get("positive_themes", [])[:5]
            negative_themes = llm_data.get("negative_themes", [])[:5]
            defect_patterns = llm_data.get("defect_patterns", [])[:5]
            turkey_specific = llm_data.get("turkey_specific", [])[:5]
            llm_sentiment = llm_data.get("overall_sentiment")
        except (json.JSONDecodeError, TypeError):
            logger.debug("LLM review synthesis response was not valid JSON")
            llm_sentiment = None
    else:
        llm_sentiment = None

    # --- Determine overall sentiment ---
    if confidence_adjusted is not None:
        if confidence_adjusted >= 4.0:
            overall_sentiment = "positive"
        elif confidence_adjusted >= 3.0:
            overall_sentiment = "mixed"
        else:
            overall_sentiment = "negative"
    elif llm_sentiment:
        overall_sentiment = llm_sentiment
    else:
        overall_sentiment = "unknown"

    result = {
        "overall_sentiment": overall_sentiment,
        "confidence_adjusted_rating": round(confidence_adjusted, 2) if confidence_adjusted is not None else None,
        "raw_weighted_rating": round(raw_weighted_avg, 2) if raw_weighted_avg is not None else None,
        "volume_confidence": round(vol_conf, 2),
        "positive_themes": positive_themes,
        "negative_themes": negative_themes,
        "defect_patterns": defect_patterns,
        "warnings": warnings,
        "turkey_specific": turkey_specific,
        "review_quality": review_quality,
        "source_breakdown": source_breakdown,
    }
    logger.info(
        "Review synthesis complete",
        product=product_name,
        sentiment=overall_sentiment,
        adjusted_rating=result["confidence_adjusted_rating"],
    )

    # Phase C: Embed the synthesized review into vector store for shopping RAG
    try:
        from src.shopping.intelligence.vector_bridge import embed_review_synthesis
        await embed_review_synthesis(result, product_name)
    except Exception as e:
        logger.debug("Review synthesis embedding skipped: %s", e)

    return result
