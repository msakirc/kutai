"""Alternative product suggestion module for the shopping intelligence system.

Generates alternative product suggestions using rule-based patterns for known
categories and LLM reasoning for novel queries.
"""

from __future__ import annotations

import json
import re

from src.infra.logging_config import get_logger

logger = get_logger("shopping.intelligence.alternatives")

# ── LLM helper ───────────────────────────────────────────────────────────────

async def _llm_call(prompt: str, system: str = "", temperature: float = 0.3) -> str:
    """Call local LLM.  Uses litellm if available, otherwise returns empty."""
    try:
        import litellm  # type: ignore[import-untyped]

        response = await litellm.acompletion(
            model="openai/local",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=2048,
        )
        return response.choices[0].message.content
    except Exception:
        return ""


# ── Rule-based alternative patterns ─────────────────────────────────────────

# Maps a product keyword to a list of alternative suggestions with reasoning.
_KNOWN_ALTERNATIVES: dict[str, list[dict]] = {
    # Electronics
    "iphone": [
        {"product": "Samsung Galaxy S serisi", "reasoning": "Android flagship, benzer fiyat segmenti"},
        {"product": "Google Pixel", "reasoning": "Saf Android deneyimi, iyi kamera"},
        {"product": "Xiaomi 14 serisi", "reasoning": "Uygun fiyatlı flagship, iyi donanım"},
    ],
    "macbook": [
        {"product": "Lenovo ThinkPad", "reasoning": "İş odaklı, dayanıklı, iyi klavye"},
        {"product": "Dell XPS", "reasoning": "Premium tasarım, güçlü performans"},
        {"product": "ASUS ZenBook", "reasoning": "Hafif, uygun fiyatlı ultrabook"},
    ],
    "airpods": [
        {"product": "Samsung Galaxy Buds", "reasoning": "Android uyumlu, benzer özellikler"},
        {"product": "Sony WF-1000XM5", "reasoning": "En iyi gürültü engelleme"},
        {"product": "JBL Tune Buds", "reasoning": "Bütçe dostu, iyi ses kalitesi"},
    ],
    # Appliances
    "dyson": [
        {"product": "Xiaomi süpürge", "reasoning": "Bütçe dostu kablosuz alternaltif"},
        {"product": "Bosch kablosuz süpürge", "reasoning": "Avrupa kalitesi, güçlü emme"},
        {"product": "Philips SpeedPro", "reasoning": "Orta segment, iyi performans"},
    ],
    "buzdolabı": [
        {"product": "Arçelik buzdolabı", "reasoning": "Yerli üretim, yaygın servis ağı"},
        {"product": "Beko buzdolabı", "reasoning": "Uygun fiyat, iyi enerji verimliliği"},
        {"product": "Bosch buzdolabı", "reasoning": "Premium kalite, sessiz çalışma"},
    ],
    "çamaşır makinesi": [
        {"product": "Arçelik çamaşır makinesi", "reasoning": "Yerli üretim, kolay servis"},
        {"product": "Bosch çamaşır makinesi", "reasoning": "Alman kalitesi, uzun ömür"},
        {"product": "Samsung çamaşır makinesi", "reasoning": "Akıllı özellikler, enerji tasarrufu"},
    ],
}

# Category-level alternatives when no specific product pattern matches.
_CATEGORY_ALTERNATIVES: dict[str, list[dict]] = {
    "electronics": [
        {"product": "Bir üst model", "reasoning": "Biraz daha fazla bütçeyle daha iyi özellikler"},
        {"product": "Bir alt model", "reasoning": "Bütçe tasarrufu, temel ihtiyaçları karşılar"},
        {"product": "Yenilenmiş (refurbished)", "reasoning": "Garantili, %30-50 tasarruf"},
    ],
    "appliances": [
        {"product": "A+++ enerji sınıfı modeller", "reasoning": "Uzun vadede elektrik tasarrufu"},
        {"product": "Yerli marka", "reasoning": "Kolay servis, uygun yedek parça"},
    ],
    "furniture": [
        {"product": "Modüler mobilya", "reasoning": "Esnek kullanım, taşınabilir"},
        {"product": "İkinci el / vintage", "reasoning": "Uygun fiyat, benzersiz tasarım"},
    ],
}


def _rule_based_alternatives(
    product_query: str, category: str, constraints: list[str]
) -> list[dict]:
    """Generate alternatives using known patterns."""
    results: list[dict] = []
    lower = product_query.lower()

    # Check specific product patterns
    for keyword, alts in _KNOWN_ALTERNATIVES.items():
        if keyword in lower:
            for alt in alts:
                result = {**alt, "source": "rule_based", "confidence": 0.7}
                results.append(result)
            break  # first match only

    # Add category-level suggestions
    if category in _CATEGORY_ALTERNATIVES:
        for alt in _CATEGORY_ALTERNATIVES[category]:
            result = {**alt, "source": "rule_based", "confidence": 0.4}
            results.append(result)

    # Constraint-aware filtering
    if "budget" in constraints:
        # Boost budget-friendly alternatives
        for r in results:
            if any(kw in r.get("reasoning", "").lower()
                   for kw in ["bütçe", "uygun fiyat", "tasarruf"]):
                r["confidence"] = min(r["confidence"] + 0.2, 1.0)

    return results


# ── LLM-based alternatives ──────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a Turkish shopping advisor.  Given a product query, its category,
and constraints, suggest 3-5 alternative products.

Return ONLY a JSON array.  Each element must have:
- product: product name (Turkish preferred)
- reasoning: one-line explanation of why this is a good alternative (Turkish)
- confidence: float 0-1

Consider:
- Price alternatives (cheaper and premium)
- Brand alternatives
- Category alternatives (different product solving the same need)
- Turkish market availability
"""


async def _llm_alternatives(
    product_query: str, category: str, constraints: list[str]
) -> list[dict]:
    """Ask the LLM for alternative suggestions."""
    prompt = json.dumps(
        {
            "product": product_query,
            "category": category,
            "constraints": constraints,
        },
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
            if isinstance(item, dict) and "product" in item:
                item.setdefault("reasoning", "")
                item.setdefault("confidence", 0.5)
                item["source"] = "llm"
                valid.append(item)
        return valid

    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("LLM alternatives parsing failed: %s", exc)
        return []


# ── Public API ───────────────────────────────────────────────────────────────

async def generate_alternatives(
    product_query: str,
    category: str = "",
    constraints: list | None = None,
) -> list[dict]:
    """Generate alternative product suggestions.

    Uses two strategies:
    1. Rule-based: known product/category patterns for instant results.
    2. LLM-based: reasoning about the underlying need for novel queries.

    Parameters
    ----------
    product_query:
        The product the user is looking for.
    category:
        Product category (electronics, appliances, furniture, etc.).
    constraints:
        List of constraint strings from the query analysis.

    Returns
    -------
    List of dicts with keys: product, reasoning, source, confidence.
    Sorted by confidence descending.
    """
    if not product_query:
        return []

    _constraints = constraints or []
    results: list[dict] = []

    # Always include rule-based results (instant, no network)
    rule_results = _rule_based_alternatives(product_query, category, _constraints)
    results.extend(rule_results)

    # Try LLM for additional/better suggestions
    try:
        llm_results = await _llm_alternatives(product_query, category, _constraints)
        if llm_results:
            # Merge: avoid duplicates by product name
            existing_names = {r["product"].lower() for r in results}
            for lr in llm_results:
                if lr["product"].lower() not in existing_names:
                    results.append(lr)
                    existing_names.add(lr["product"].lower())
    except Exception as exc:
        logger.warning("LLM alternative generation failed: %s", exc)

    # Sort by confidence
    results.sort(key=lambda r: r.get("confidence", 0), reverse=True)

    logger.info(
        "Generated %d alternatives for '%s' (rule=%d, llm=%d)",
        len(results),
        product_query,
        sum(1 for r in results if r.get("source") == "rule_based"),
        sum(1 for r in results if r.get("source") == "llm"),
    )
    return results
