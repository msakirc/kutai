"""Value scoring module for the shopping intelligence system.

Calculates a normalised 0-100 value score for products by considering price,
seller reputation, shipping cost, warranty, and total cost of ownership.
Provides multiple scoring perspectives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.infra.logging_config import get_logger
from src.shopping.models import Product

logger = get_logger("shopping.intelligence.value_scorer")

# ── Scoring weights by category ──────────────────────────────────────────────

@dataclass
class _Weights:
    price: float = 0.35
    seller: float = 0.15
    shipping: float = 0.10
    warranty: float = 0.10
    rating: float = 0.15
    availability: float = 0.05
    review_volume: float = 0.10


_CATEGORY_WEIGHTS: dict[str, _Weights] = {
    "electronics": _Weights(
        price=0.30, seller=0.15, shipping=0.05, warranty=0.15,
        rating=0.20, availability=0.05, review_volume=0.10,
    ),
    "appliances": _Weights(
        price=0.25, seller=0.15, shipping=0.10, warranty=0.20,
        rating=0.15, availability=0.05, review_volume=0.10,
    ),
    "furniture": _Weights(
        price=0.30, seller=0.20, shipping=0.15, warranty=0.10,
        rating=0.10, availability=0.05, review_volume=0.10,
    ),
    "grocery": _Weights(
        price=0.50, seller=0.10, shipping=0.15, warranty=0.0,
        rating=0.10, availability=0.10, review_volume=0.05,
    ),
    "clothing": _Weights(
        price=0.35, seller=0.15, shipping=0.10, warranty=0.05,
        rating=0.20, availability=0.05, review_volume=0.10,
    ),
}

_DEFAULT_WEIGHTS = _Weights()

# ── Helper functions ─────────────────────────────────────────────────────────


def _effective_price(product: Product) -> float:
    """Return the effective purchase price."""
    return product.discounted_price or product.original_price or 0.0


def _total_cost(product: Product) -> float:
    """Purchase price + shipping."""
    return _effective_price(product) + (product.shipping_cost or 0.0)


def _normalize(value: float, min_val: float, max_val: float) -> float:
    """Normalize *value* to 0-100 within [min_val, max_val].

    Lower values score higher for cost-like metrics (inverted).
    """
    if max_val == min_val:
        return 50.0
    return max(0.0, min(100.0, (1.0 - (value - min_val) / (max_val - min_val)) * 100))


def _normalize_positive(value: float, max_val: float) -> float:
    """Normalize where higher is better."""
    if max_val <= 0:
        return 50.0
    return max(0.0, min(100.0, (value / max_val) * 100))


def _seller_score(product: Product) -> float:
    """Score seller reputation 0-100."""
    score = 50.0  # default for unknown sellers
    if product.seller_rating is not None:
        # Seller rating typically 0-5 or 0-10
        max_rating = 10.0 if product.seller_rating > 5 else 5.0
        score = (product.seller_rating / max_rating) * 80

        # Boost for high review count
        if product.seller_review_count and product.seller_review_count > 100:
            score += min(20, math.log10(product.seller_review_count) * 5)

    return min(100.0, score)


def _warranty_score(product: Product) -> float:
    """Score warranty coverage 0-100."""
    months = product.warranty_months
    if months is None:
        return 30.0  # unknown
    if months == 0:
        return 0.0
    if months <= 12:
        return 40.0 + (months / 12.0) * 20.0
    if months <= 24:
        return 60.0 + ((months - 12) / 12.0) * 20.0
    return min(100.0, 80.0 + ((months - 24) / 12.0) * 20.0)


def _rating_score(product: Product) -> float:
    """Score user rating 0-100 (Bayesian adjustment for review count)."""
    if product.rating is None:
        return 40.0  # neutral for unrated

    max_rating = 10.0 if product.rating > 5 else 5.0
    normalized_rating = product.rating / max_rating

    # Bayesian prior: 3.5/5 with weight of 10 reviews
    prior_rating = 0.7
    prior_weight = 10
    count = product.review_count or 0

    adjusted = (
        (normalized_rating * count + prior_rating * prior_weight)
        / (count + prior_weight)
    )
    return adjusted * 100


def _availability_score(product: Product) -> float:
    """Score product availability 0-100."""
    mapping = {
        "in_stock": 100.0,
        "low_stock": 60.0,
        "preorder": 30.0,
        "out_of_stock": 0.0,
    }
    return mapping.get(product.availability, 50.0)


def _review_volume_score(product: Product, max_reviews: int) -> float:
    """Score based on review volume (more reviews = more trusted)."""
    count = product.review_count or 0
    if max_reviews <= 0:
        return 50.0
    # Logarithmic scale to avoid domination by mega-sellers
    if count == 0:
        return 10.0
    return min(100.0, (math.log10(count + 1) / math.log10(max_reviews + 1)) * 100)


# ── Scoring perspectives ────────────────────────────────────────────────────

def _score_best_price(products: list[Product]) -> list[tuple[int, float]]:
    """Rank purely by lowest effective price."""
    prices = [(_effective_price(p), i) for i, p in enumerate(products)]
    if not prices:
        return []
    min_p = min(p for p, _ in prices) or 1.0
    max_p = max(p for p, _ in prices) or 1.0
    return [(i, _normalize(p, min_p, max_p)) for p, i in prices]


def _score_best_tco(products: list[Product]) -> list[tuple[int, float]]:
    """Rank by total cost of ownership (price + shipping)."""
    costs = [(_total_cost(p), i) for i, p in enumerate(products)]
    if not costs:
        return []
    min_c = min(c for c, _ in costs) or 1.0
    max_c = max(c for c, _ in costs) or 1.0
    return [(i, _normalize(c, min_c, max_c)) for c, i in costs]


def _score_best_installment(products: list[Product]) -> list[tuple[int, float]]:
    """Rank by best installment deal (lowest monthly payment)."""
    results: list[tuple[int, float]] = []
    for i, p in enumerate(products):
        if p.installment_info and isinstance(p.installment_info, dict):
            monthly = p.installment_info.get("monthly_payment", 0)
            months = p.installment_info.get("months", 1)
            if monthly and months:
                # Prefer longer terms with lower monthly
                score = 100.0 - min(100.0, (monthly / 1000.0) * 50)
                score += min(20.0, months * 2)
                results.append((i, min(100.0, score)))
            else:
                results.append((i, 30.0))
        else:
            results.append((i, 20.0))  # no installment info
    return results


# ── Public API ───────────────────────────────────────────────────────────────

async def score_products(
    products: list[Product],
    category: str = "",
) -> list[dict]:
    """Calculate value scores for a list of products.

    Parameters
    ----------
    products:
        Products to score.
    category:
        Product category for weight selection.

    Returns
    -------
    List of dicts (one per product), each with:
    - product_name: str
    - value_score: float 0-100 (weighted composite)
    - breakdown: dict of component scores
    - perspectives: dict with best_price, best_tco, best_installment scores
    - rank: int (1 = best)
    """
    if not products:
        return []

    weights = _CATEGORY_WEIGHTS.get(category, _DEFAULT_WEIGHTS)

    # Pre-compute range values for normalization
    prices = [_effective_price(p) for p in products]
    min_price = min(prices) if prices else 0
    max_price = max(prices) if prices else 0
    total_costs = [_total_cost(p) for p in products]
    min_tc = min(total_costs) if total_costs else 0
    max_tc = max(total_costs) if total_costs else 0
    max_reviews = max((p.review_count or 0) for p in products) if products else 0

    # Perspective scores
    price_scores = dict(_score_best_price(products))
    tco_scores = dict(_score_best_tco(products))
    installment_scores = dict(_score_best_installment(products))

    scored: list[dict] = []

    for i, product in enumerate(products):
        price = _effective_price(product)
        price_sc = _normalize(price, min_price, max_price)
        tc = _total_cost(product)
        shipping_sc = _normalize(tc, min_tc, max_tc)
        seller_sc = _seller_score(product)
        warranty_sc = _warranty_score(product)
        rating_sc = _rating_score(product)
        avail_sc = _availability_score(product)
        review_sc = _review_volume_score(product, max_reviews)

        # Weighted composite
        composite = (
            weights.price * price_sc
            + weights.seller * seller_sc
            + weights.shipping * shipping_sc
            + weights.warranty * warranty_sc
            + weights.rating * rating_sc
            + weights.availability * avail_sc
            + weights.review_volume * review_sc
        )
        composite = max(0.0, min(100.0, composite))

        scored.append({
            "product_name": product.name,
            "value_score": round(composite, 1),
            "breakdown": {
                "price": round(price_sc, 1),
                "seller": round(seller_sc, 1),
                "shipping": round(shipping_sc, 1),
                "warranty": round(warranty_sc, 1),
                "rating": round(rating_sc, 1),
                "availability": round(avail_sc, 1),
                "review_volume": round(review_sc, 1),
            },
            "perspectives": {
                "best_price": round(price_scores.get(i, 50.0), 1),
                "best_tco": round(tco_scores.get(i, 50.0), 1),
                "best_installment": round(installment_scores.get(i, 20.0), 1),
            },
            "rank": 0,  # filled below
        })

    # Assign ranks
    scored.sort(key=lambda s: s["value_score"], reverse=True)
    for rank, item in enumerate(scored, start=1):
        item["rank"] = rank

    logger.info(
        "Scored %d products (category=%s), top=%s (%.1f)",
        len(scored),
        category or "general",
        scored[0]["product_name"] if scored else "N/A",
        scored[0]["value_score"] if scored else 0,
    )
    return scored
