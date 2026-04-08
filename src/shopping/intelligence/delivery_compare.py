"""Delivery Compare — compares delivery options across products, flags
international shipping, and calculates effective price including shipping."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from src.infra.logging_config import get_logger
from src.infra.times import utc_now

logger = get_logger("shopping.intelligence.delivery_compare")

from ._llm import _llm_call

# ─── Store delivery defaults ────────────────────────────────────────────────

_STORE_DELIVERY_DEFAULTS: dict[str, dict] = {
    "trendyol": {
        "carrier": "Trendyol Express / Yurtiçi Kargo",
        "typical_days": (2, 5),
        "free_threshold_try": 250,
        "express_available": True,
        "express_days": (0, 1),
    },
    "hepsiburada": {
        "carrier": "HepsiJet / MNG Kargo",
        "typical_days": (1, 3),
        "free_threshold_try": 150,
        "express_available": True,
        "express_days": (0, 1),
    },
    "amazon_tr": {
        "carrier": "Amazon Lojistik / Aras Kargo",
        "typical_days": (1, 3),
        "free_threshold_try": 200,
        "express_available": True,
        "express_days": (0, 0),
    },
    "n11": {
        "carrier": "Seller-dependent",
        "typical_days": (2, 5),
        "free_threshold_try": 0,  # seller-dependent
        "express_available": False,
        "express_days": None,
    },
    "mediamarkt": {
        "carrier": "MediaMarkt Lojistik / Kargo",
        "typical_days": (2, 4),
        "free_threshold_try": 300,
        "express_available": True,
        "express_days": (0, 0),  # click & collect same day
    },
    "vatanbilgisayar": {
        "carrier": "Vatan Kargo / Yurtiçi",
        "typical_days": (1, 3),
        "free_threshold_try": 250,
        "express_available": True,
        "express_days": (0, 0),  # store pickup
    },
    "koctas": {
        "carrier": "Koçtaş Lojistik / Kargo",
        "typical_days": (2, 5),
        "free_threshold_try": 300,
        "express_available": False,
        "express_days": None,
    },
    "ikea_tr": {
        "carrier": "IKEA Teslimat",
        "typical_days": (3, 7),
        "free_threshold_try": 0,  # delivery always has a fee
        "express_available": False,
        "express_days": None,
    },
    "getir": {
        "carrier": "Getir Kurye",
        "typical_days": (0, 0),
        "free_threshold_try": 175,
        "express_available": True,
        "express_days": (0, 0),
    },
    "migros": {
        "carrier": "Migros Teslimat",
        "typical_days": (0, 1),
        "free_threshold_try": 400,
        "express_available": True,
        "express_days": (0, 0),
    },
}

_INTERNATIONAL_MARKERS = [
    "yurt dışı", "abroad", "ithal", "import", "china", "çin",
    "aliexpress", "amazon.com", "amazon.de", "banggood", "global",
    "uluslararası", "international",
]

_STORE_ALIASES: dict[str, str] = {
    "trendyol": "trendyol",
    "trendyol.com": "trendyol",
    "hepsiburada": "hepsiburada",
    "hepsiburada.com": "hepsiburada",
    "amazon": "amazon_tr",
    "amazon.com.tr": "amazon_tr",
    "amazon_tr": "amazon_tr",
    "n11": "n11",
    "n11.com": "n11",
    "mediamarkt": "mediamarkt",
    "mediamarkt.com.tr": "mediamarkt",
    "vatanbilgisayar": "vatanbilgisayar",
    "vatanbilgisayar.com": "vatanbilgisayar",
    "vatan": "vatanbilgisayar",
    "koctas": "koctas",
    "koçtaş": "koctas",
    "koctas.com.tr": "koctas",
    "ikea": "ikea_tr",
    "ikea.com.tr": "ikea_tr",
    "ikea_tr": "ikea_tr",
    "getir": "getir",
    "migros": "migros",
    "migros.com.tr": "migros",
}


def _normalize_store(store: str) -> str:
    return _STORE_ALIASES.get(store.lower().strip(), store.lower().strip())


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _is_international(product: dict) -> bool:
    """Check if a product likely ships from abroad."""
    searchable = " ".join([
        product.get("seller_name", ""),
        product.get("name", ""),
        product.get("url", ""),
        str(product.get("specs", {})),
    ]).lower()
    return any(marker in searchable for marker in _INTERNATIONAL_MARKERS)


def _estimate_delivery(product: dict, store_key: str) -> dict:
    """Estimate delivery details for a single product."""
    defaults = _STORE_DELIVERY_DEFAULTS.get(store_key, {})
    is_abroad = _is_international(product)

    # Use product-level data if available, fall back to store defaults
    shipping_cost = product.get("shipping_cost")
    shipping_days = product.get("shipping_time_days")
    free_shipping = product.get("free_shipping", False)

    price = product.get("discounted_price") or product.get("original_price") or 0.0

    if shipping_cost is None:
        if free_shipping:
            shipping_cost = 0.0
        elif defaults:
            threshold = defaults.get("free_threshold_try", 0)
            if threshold > 0 and price >= threshold:
                shipping_cost = 0.0
            else:
                shipping_cost = 29.90  # typical default kargo fee
        else:
            shipping_cost = 29.90

    if shipping_days is None:
        if is_abroad:
            shipping_days = 15  # international average
        elif defaults:
            typical = defaults.get("typical_days", (2, 5))
            shipping_days = typical[1]  # use upper bound
        else:
            shipping_days = 5

    now = utc_now()
    estimated_date = (now + timedelta(days=shipping_days)).strftime("%Y-%m-%d")

    carrier = defaults.get("carrier", "Unknown")
    express_available = defaults.get("express_available", False) and not is_abroad
    express_days = defaults.get("express_days")

    express_date = None
    if express_available and express_days:
        express_date = (now + timedelta(days=express_days[1])).strftime("%Y-%m-%d")

    effective_price = round(price + shipping_cost, 2)

    return {
        "product_name": product.get("name", ""),
        "source": product.get("source", ""),
        "product_price": round(price, 2),
        "shipping_cost": round(shipping_cost, 2),
        "effective_price": effective_price,
        "estimated_delivery_date": estimated_date,
        "estimated_days": shipping_days,
        "carrier": carrier,
        "express_available": express_available,
        "express_delivery_date": express_date,
        "ships_from_abroad": is_abroad,
        "free_shipping": shipping_cost == 0.0,
    }


# ─── Main entry point ───────────────────────────────────────────────────────

async def compare_delivery(products: list[dict]) -> list[dict]:
    """Compare delivery options across a list of products.

    Args:
        products: list of product dicts (from models.Product as dict)

    Returns:
        list of delivery comparison dicts, sorted by effective_price ascending.
        Each entry includes warnings for international shipping.
    """
    logger.info("Comparing delivery options", product_count=len(products))

    if not products:
        return []

    results: list[dict] = []

    for product in products:
        try:
            source = product.get("source", "")
            store_key = _normalize_store(source)
            delivery = _estimate_delivery(product, store_key)

            # Add warnings
            warnings: list[str] = []
            if delivery["ships_from_abroad"]:
                warnings.append(
                    "Ships from abroad — expect 10-20 day delivery, "
                    "possible customs duty above 150 EUR"
                )
            if delivery["shipping_cost"] > 0 and delivery["product_price"] > 0:
                shipping_pct = (delivery["shipping_cost"] / delivery["product_price"]) * 100
                if shipping_pct > 10:
                    warnings.append(
                        f"Shipping is {round(shipping_pct, 1)}% of product price"
                    )

            delivery["warnings"] = warnings
            delivery["url"] = product.get("url", "")
            results.append(delivery)

        except Exception as exc:
            logger.warning(
                "Error estimating delivery for product",
                product_name=product.get("name", "?"),
                error=str(exc),
            )
            continue

    # Sort by effective price
    results.sort(key=lambda r: r["effective_price"])

    # Add ranking
    for i, r in enumerate(results):
        r["rank"] = i + 1
        if i == 0:
            r["badge"] = "Best effective price"
        elif r.get("express_available") and not any(
            prev.get("express_available") for prev in results[:i]
        ):
            r["badge"] = "Fastest with express"
        else:
            r["badge"] = ""

    # LLM summary if multiple products
    if len(results) >= 2:
        summary_input = json.dumps(
            [
                {
                    "name": r["product_name"][:60],
                    "source": r["source"],
                    "effective_price": r["effective_price"],
                    "days": r["estimated_days"],
                    "abroad": r["ships_from_abroad"],
                }
                for r in results[:5]
            ],
            ensure_ascii=False,
        )
        llm_resp = await _llm_call(
            f"Compare these delivery options and write 1-2 sentence Turkish summary:\n{summary_input}\n\n"
            'Return JSON: {{"summary": "..."}}',
            system="You are a Turkish shopping delivery advisor. Be concise.",
        )
        if llm_resp:
            try:
                summary = json.loads(llm_resp).get("summary", "")
                if summary:
                    for r in results:
                        r["comparison_summary"] = summary
            except (json.JSONDecodeError, TypeError):
                pass

    logger.info("Delivery comparison complete", results_count=len(results))
    return results
