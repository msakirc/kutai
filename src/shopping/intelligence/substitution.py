"""Product substitution module for the shopping intelligence system.

Suggests substitute products that solve the same underlying need, using a
knowledge base of known substitutions and LLM reasoning.  Includes a
price-triggered mode for when search results exceed the user's budget.
"""

from __future__ import annotations

import json
import re

from src.infra.logging_config import get_logger

logger = get_logger("shopping.intelligence.substitution")

from ._llm import _llm_call

# ── Built-in substitution knowledge base ─────────────────────────────────────
# In a full system this would be loaded from substitutions.json; we embed
# the core data here to keep the module self-contained.

_SUBSTITUTIONS: dict[str, list[dict]] = {
    # intent -> need -> substitutes
    "kahve makinesi": [
        {
            "substitute": "French press",
            "reasoning": "Elektriksiz, uygun fiyat, iyi kahve kalitesi",
            "price_range": "low",
        },
        {
            "substitute": "Moka pot (cezve)",
            "reasoning": "Klasik İtalyan kahvesi, düşük maliyet",
            "price_range": "low",
        },
        {
            "substitute": "Pour-over dripper",
            "reasoning": "Filtre kahve severler için minimal ekipman",
            "price_range": "low",
        },
    ],
    "kurutma makinesi": [
        {
            "substitute": "Çamaşır makinesi (kurutmalı)",
            "reasoning": "Tek cihazda yıkama ve kurutma, alan tasarrufu",
            "price_range": "mid",
        },
        {
            "substitute": "Portatif kurutma askısı + vantilatör",
            "reasoning": "Düşük bütçeli çözüm, elektrik tasarrufu",
            "price_range": "low",
        },
    ],
    "bulaşık makinesi": [
        {
            "substitute": "Tezgah üstü mini bulaşık makinesi",
            "reasoning": "Küçük mutfaklar için, tesisat gerektirmez",
            "price_range": "mid",
        },
    ],
    "robot süpürge": [
        {
            "substitute": "Kablosuz dikey süpürge",
            "reasoning": "Daha güçlü emme, manuel kontrol",
            "price_range": "mid",
        },
        {
            "substitute": "Klasik elektrikli süpürge",
            "reasoning": "En güçlü emme, en düşük fiyat",
            "price_range": "low",
        },
    ],
    "tablet": [
        {
            "substitute": "Büyük ekranlı telefon (phablet)",
            "reasoning": "Tek cihaz, her zaman yanınızda",
            "price_range": "mid",
        },
        {
            "substitute": "Chromebook / ucuz laptop",
            "reasoning": "Tam klavye, daha verimli çalışma",
            "price_range": "mid",
        },
    ],
    "klima": [
        {
            "substitute": "Portatif klima",
            "reasoning": "Montaj gerektirmez, taşınabilir",
            "price_range": "mid",
        },
        {
            "substitute": "Tower fan (kule vantilatör)",
            "reasoning": "Çok düşük enerji tüketimi, uygun fiyat",
            "price_range": "low",
        },
    ],
    "monitor": [
        {
            "substitute": "TV (HDMI girişli)",
            "reasoning": "Büyük ekran, uygun fiyat, çoklu kullanım",
            "price_range": "mid",
        },
        {
            "substitute": "Portatif monitör",
            "reasoning": "Taşınabilir, USB-C ile çalışır",
            "price_range": "mid",
        },
    ],
}

# Price tiers for the price-triggered mode
_PRICE_TIERS = {"low": 0.3, "mid": 0.6, "high": 1.0}


def _kb_substitutions(product: str) -> list[dict]:
    """Look up substitutions from the built-in knowledge base."""
    lower = product.lower()
    results: list[dict] = []
    for keyword, subs in _SUBSTITUTIONS.items():
        if keyword in lower:
            for sub in subs:
                results.append({
                    **sub,
                    "source": "knowledge_base",
                    "original_product": product,
                })
            break
    return results


# ── LLM-based substitution ──────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a Turkish shopping advisor specialising in product substitutions.
Given a product, the user's intent, and category, suggest 2-4 substitute
products that solve the same UNDERLYING NEED differently.

A substitution is NOT just a cheaper version of the same product -- it is
a fundamentally different product or approach.

Return ONLY a JSON array.  Each element:
- substitute: product name (Turkish preferred)
- reasoning: why this substitution works (Turkish, one line)
- price_range: "low" | "mid" | "high" relative to original

Example: Instead of a bread machine -> suggest a Dutch oven for artisan bread.
"""


async def _llm_substitutions(product: str, intent: str, category: str) -> list[dict]:
    """Ask the LLM for substitution ideas."""
    prompt = json.dumps(
        {"product": product, "intent": intent, "category": category},
        ensure_ascii=False,
    )
    try:
        response = await _llm_call(prompt=prompt, system=_SYSTEM_PROMPT, temperature=0.5)
        if not response:
            return []

        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        items = json.loads(cleaned)
        if not isinstance(items, list):
            return []

        valid: list[dict] = []
        for item in items:
            if isinstance(item, dict) and "substitute" in item:
                item.setdefault("reasoning", "")
                item.setdefault("price_range", "mid")
                item["source"] = "llm"
                item["original_product"] = product
                valid.append(item)
        return valid

    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("LLM substitution parsing failed: %s", exc)
        return []


# ── Public API ───────────────────────────────────────────────────────────────

async def suggest_substitutions(
    product: str,
    intent: str = "explore",
    category: str = "",
    *,
    budget: float | None = None,
    found_min_price: float | None = None,
) -> list[dict]:
    """Suggest substitute products that address the same underlying need.

    Parameters
    ----------
    product:
        The product the user originally asked about.
    intent:
        Query intent (compare, find_cheapest, find_best, etc.).
    category:
        Product category.
    budget:
        Optional user budget in TRY.
    found_min_price:
        Lowest price found during search.  When this exceeds *budget*,
        the module enters price-triggered mode and prioritises low-cost
        substitutions.

    Returns
    -------
    List of dicts with keys: substitute, reasoning, price_range, source,
    original_product.  Optionally includes ``price_triggered: True`` when
    the suggestion was boosted due to budget constraints.
    """
    if not product:
        return []

    results: list[dict] = []

    # Knowledge-base lookup
    kb_results = _kb_substitutions(product)
    results.extend(kb_results)

    # LLM-based reasoning
    try:
        llm_results = await _llm_substitutions(product, intent, category)
        if llm_results:
            existing = {r["substitute"].lower() for r in results}
            for lr in llm_results:
                if lr["substitute"].lower() not in existing:
                    results.append(lr)
                    existing.add(lr["substitute"].lower())
    except Exception as exc:
        logger.warning("LLM substitution generation failed: %s", exc)

    # Price-triggered mode: boost low-cost alternatives when over budget
    price_triggered = (
        budget is not None
        and found_min_price is not None
        and found_min_price > budget
    )

    if price_triggered:
        for r in results:
            r["price_triggered"] = True
            if r.get("price_range") == "low":
                # Move low-cost substitutions to the top
                r["_sort_priority"] = 0
            elif r.get("price_range") == "mid":
                r["_sort_priority"] = 1
            else:
                r["_sort_priority"] = 2
        results.sort(key=lambda r: r.get("_sort_priority", 9))
        # Clean up internal key
        for r in results:
            r.pop("_sort_priority", None)

    logger.info(
        "Generated %d substitutions for '%s' (kb=%d, llm=%d, price_triggered=%s)",
        len(results),
        product,
        sum(1 for r in results if r.get("source") == "knowledge_base"),
        sum(1 for r in results if r.get("source") == "llm"),
        price_triggered,
    )
    return results
