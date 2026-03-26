"""Complementary product suggester.

Suggests accessories and consumables that buyers typically need alongside a
product — e.g. a phone case when buying a phone, or printer cartridges when
buying a printer.  Also provides economic intelligence for consumable-heavy
products (like printers) so the user understands the true ongoing cost.
"""

from __future__ import annotations

import json
import re

from src.infra.logging_config import get_logger
from src.shopping.intelligence._llm import _llm_call

logger = get_logger("shopping.intelligence.special.complementary")

# ─── Static complement map (Turkish market) ──────────────────────────────────
# Each entry: {"product": str, "reason": str, "priority": str,
#              "is_consumable": bool, "recurring_cost_note": str | None}

_COMPLEMENT_MAP: dict[str, list[dict]] = {
    "telefon": [
        {
            "product": "kılıf",
            "reason": "Telefonu düşme ve çizilmelere karşı korur",
            "priority": "yüksek",
            "is_consumable": False,
            "recurring_cost_note": None,
        },
        {
            "product": "ekran koruyucu",
            "reason": "Ekranı çizik ve kırılmaya karşı korur",
            "priority": "yüksek",
            "is_consumable": False,
            "recurring_cost_note": None,
        },
        {
            "product": "şarj kablosu",
            "reason": "Yedek/ek şarj kablosu pratikte çok işe yarar",
            "priority": "orta",
            "is_consumable": False,
            "recurring_cost_note": None,
        },
        {
            "product": "kablosuz şarj",
            "reason": "Kablo bağlamadan hızlı şarj imkânı sağlar",
            "priority": "düşük",
            "is_consumable": False,
            "recurring_cost_note": None,
        },
    ],
    "yazıcı": [
        {
            "product": "kartuş / toner",
            "reason": "Yazıcı kartuşsuz çalışmaz; başlangıç kartuşu genellikle yarı dolu gelir",
            "priority": "yüksek",
            "is_consumable": True,
            "recurring_cost_note": "Kartuş maliyeti yıllık 500–2.000 TL arasında değişebilir",
        },
        {
            "product": "kağıt",
            "reason": "Baskı için A4 kağıt gereklidir",
            "priority": "yüksek",
            "is_consumable": True,
            "recurring_cost_note": "500 sayfalık top yaklaşık 100–150 TL",
        },
    ],
    "laptop": [
        {
            "product": "çanta / kılıf",
            "reason": "Taşıma sırasında laptopı darbelerden korur",
            "priority": "yüksek",
            "is_consumable": False,
            "recurring_cost_note": None,
        },
        {
            "product": "mouse",
            "reason": "Daha rahat kullanım için harici mouse önerilir",
            "priority": "orta",
            "is_consumable": False,
            "recurring_cost_note": None,
        },
        {
            "product": "USB hub",
            "reason": "Modern laptoplardaki az sayıda port için port genişletici",
            "priority": "orta",
            "is_consumable": False,
            "recurring_cost_note": None,
        },
        {
            "product": "ekran koruyucu",
            "reason": "Laptop ekranını çiziklerden korur",
            "priority": "düşük",
            "is_consumable": False,
            "recurring_cost_note": None,
        },
    ],
    "tablet": [
        {
            "product": "kılıf",
            "reason": "Tableti düşme ve çizilmelerden korur",
            "priority": "yüksek",
            "is_consumable": False,
            "recurring_cost_note": None,
        },
        {
            "product": "kalem (stylus)",
            "reason": "Not alma ve çizim için stylus kullanımı tabletlerde oldukça yaygın",
            "priority": "orta",
            "is_consumable": False,
            "recurring_cost_note": None,
        },
        {
            "product": "klavye",
            "reason": "Tablet klavyeyle mini laptopa dönüşür",
            "priority": "düşük",
            "is_consumable": False,
            "recurring_cost_note": None,
        },
    ],
    "kamera": [
        {
            "product": "hafıza kartı",
            "reason": "Çoğu kamera hafıza kartı olmadan kayıt yapamaz",
            "priority": "yüksek",
            "is_consumable": False,
            "recurring_cost_note": None,
        },
        {
            "product": "çanta",
            "reason": "Kamera ve aksesuarları için koruyucu taşıma çantası",
            "priority": "yüksek",
            "is_consumable": False,
            "recurring_cost_note": None,
        },
        {
            "product": "tripod",
            "reason": "Uzun pozlama, video çekimi ve selfie için şart",
            "priority": "orta",
            "is_consumable": False,
            "recurring_cost_note": None,
        },
        {
            "product": "ekstra pil",
            "reason": "Pil ömrü kısaldıkça yedek pil hayat kurtarır",
            "priority": "orta",
            "is_consumable": False,
            "recurring_cost_note": None,
        },
    ],
    "buzdolabı": [],
    "çamaşır makinesi": [
        {
            "product": "deterjan",
            "reason": "Makine için uygun toz veya sıvı deterjan gereklidir",
            "priority": "yüksek",
            "is_consumable": True,
            "recurring_cost_note": "Aylık yaklaşık 100–300 TL",
        },
        {
            "product": "yumuşatıcı",
            "reason": "Çamaşırları yumuşatır ve taze koku verir",
            "priority": "orta",
            "is_consumable": True,
            "recurring_cost_note": "Aylık yaklaşık 50–150 TL",
        },
    ],
    "bulaşık makinesi": [
        {
            "product": "tablet / tuz / parlatıcı",
            "reason": "Bulaşık makinesi deterjanı, tuz ve parlatıcısı olmadan verimli çalışmaz",
            "priority": "yüksek",
            "is_consumable": True,
            "recurring_cost_note": "Aylık yaklaşık 150–400 TL (tablet + tuz + parlatıcı)",
        },
    ],
    "klima": [
        {
            "product": "montaj hizmeti",
            "reason": "Klima kurulumu mutlaka yetkili servis tarafından yapılmalıdır",
            "priority": "yüksek",
            "is_consumable": False,
            "recurring_cost_note": "Montaj ücreti ayrı; 500–1.500 TL arasında değişebilir",
        },
    ],
    "kahve makinesi": [
        {
            "product": "filtre",
            "reason": "Makine tipine göre kağıt filtre veya yıkanabilir filtre gerekebilir",
            "priority": "yüksek",
            "is_consumable": True,
            "recurring_cost_note": "Aylık yaklaşık 30–100 TL",
        },
        {
            "product": "çekirdek / kapsül",
            "reason": "Makinenin çalışması için kahve çekirdeği veya kapsül şart",
            "priority": "yüksek",
            "is_consumable": True,
            "recurring_cost_note": "Aylık yaklaşık 200–800 TL (kapsül makinelerde daha pahalı)",
        },
    ],
}

# Normalisation aliases so "cep telefonu", "akıllı telefon" etc. all match "telefon"
_CATEGORY_ALIASES: dict[str, str] = {
    "cep telefonu": "telefon",
    "akıllı telefon": "telefon",
    "smartphone": "telefon",
    "phone": "telefon",
    "printer": "yazıcı",
    "notebook": "laptop",
    "dizüstü": "laptop",
    "dizüstü bilgisayar": "laptop",
    "bilgisayar": "laptop",
    "kamera": "kamera",
    "fotoğraf makinesi": "kamera",
    "camera": "kamera",
    "fridge": "buzdolabı",
    "refrigerator": "buzdolabı",
    "washing machine": "çamaşır makinesi",
    "dishwasher": "bulaşık makinesi",
    "air conditioner": "klima",
    "coffee machine": "kahve makinesi",
    "espresso": "kahve makinesi",
}

# Consumable warning categories (printer ink is notably expensive)
_CONSUMABLE_WARNINGS: dict[str, str] = {
    "yazıcı": (
        "Dikkat: kartuş/toner maliyeti yüksek olabilir. "
        "Satın almadan önce yıllık sarf malzemesi bütçesini hesaplayın."
    ),
    "klima": (
        "Dikkat: montaj ücreti cihaz fiyatına dahil değildir. "
        "Yetkili servis ücretini önceden öğrenin."
    ),
    "kahve makinesi": (
        "Kapsüllü makinelerde kapsül maliyeti yüksek olabilir; "
        "kapsül başı maliyet ile aylık bütçenizi karşılaştırın."
    ),
}

# Rough annual consumable cost strings (Turkish text, for assess_consumable_cost)
_ANNUAL_COST_ESTIMATES: dict[str, tuple[list[str], str]] = {
    "yazıcı": (
        ["kartuş / toner", "kağıt"],
        "Yıllık tahmini 500–2.000 TL (kullanım yoğunluğuna göre değişir)",
    ),
    "çamaşır makinesi": (
        ["deterjan", "yumuşatıcı"],
        "Yıllık tahmini 1.500–5.000 TL",
    ),
    "bulaşık makinesi": (
        ["tablet", "tuz", "parlatıcı"],
        "Yıllık tahmini 1.800–5.000 TL",
    ),
    "kahve makinesi": (
        ["filtre", "çekirdek / kapsül"],
        "Yıllık tahmini 2.500–10.000 TL (kapsüllü makinelerde daha fazla)",
    ),
    "klima": (
        ["yıllık bakım / gaz dolumu"],
        "Yıllık tahmini 500–1.500 TL (bakım + olası servis)",
    ),
}


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _ascii_fold(text: str) -> str:
    """Replace common Turkish chars with ASCII equivalents for fuzzy matching."""
    return (
        text.replace("ı", "i")
        .replace("İ", "I")
        .replace("ş", "s")
        .replace("Ş", "S")
        .replace("ğ", "g")
        .replace("Ğ", "G")
        .replace("ç", "c")
        .replace("Ç", "C")
        .replace("ö", "o")
        .replace("Ö", "O")
        .replace("ü", "u")
        .replace("Ü", "U")
    )


def _normalise_category(category: str) -> str:
    """Lower-case and apply alias map, with Turkish ASCII-fold fallback."""
    cat = category.strip().lower()
    # Direct match (Turkish chars present)
    if cat in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[cat]
    if cat in _COMPLEMENT_MAP:
        return cat
    # Alias map with ASCII-folded keys
    cat_ascii = _ascii_fold(cat)
    for alias, canonical in _CATEGORY_ALIASES.items():
        if _ascii_fold(alias) == cat_ascii:
            return canonical
    # Complement map keys with ASCII-fold
    for key in _COMPLEMENT_MAP:
        if _ascii_fold(key) == cat_ascii:
            return key
    return cat


def _infer_category_from_name(product_name: str) -> str | None:
    """Try to infer a canonical category from the product name."""
    name_lower = product_name.lower()
    # Simple keyword scan over category keys + aliases
    candidates: list[tuple[str, str]] = []
    for alias, canonical in _CATEGORY_ALIASES.items():
        if alias in name_lower:
            candidates.append((alias, canonical))
    for key in _COMPLEMENT_MAP:
        if key in name_lower:
            candidates.append((key, key))
    if not candidates:
        return None
    # Pick the longest match (most specific)
    candidates.sort(key=lambda t: len(t[0]), reverse=True)
    return candidates[0][1]


def _extract_json_list(text: str) -> list[dict]:
    """Extract first JSON array from LLM response text."""
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group())
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return []


# ─── Public API ──────────────────────────────────────────────────────────────


def get_complement_map(category: str) -> list[dict]:
    """Return the static complement list for *category*.

    Parameters
    ----------
    category:
        Product category in Turkish or English (e.g. ``"telefon"``,
        ``"laptop"``, ``"printer"``).

    Returns
    -------
    List of complement dicts.  Returns an empty list if the category is
    not in the static map (use :func:`suggest_complements` for LLM fallback).
    """
    canonical = _normalise_category(category)
    complements = _COMPLEMENT_MAP.get(canonical, [])
    logger.debug(
        "get_complement_map: category=%r -> canonical=%r -> %d items",
        category,
        canonical,
        len(complements),
    )
    return complements


async def suggest_complements(
    product_name: str,
    category: str | None = None,
) -> list[dict]:
    """Suggest complementary products for *product_name*.

    Tries the static map first.  Falls back to the LLM for unknown
    categories.

    Parameters
    ----------
    product_name:
        Human-readable product name (e.g. ``"iPhone 15"``).
    category:
        Optional category hint.  If omitted, the function attempts to
        infer the category from *product_name*.

    Returns
    -------
    List of dicts, each with keys:

    - ``product`` — complement product name (Turkish)
    - ``reason`` — short explanation
    - ``priority`` — ``"yüksek"``, ``"orta"``, or ``"düşük"``
    - ``is_consumable`` — whether the item needs regular repurchasing
    - ``recurring_cost_note`` — rough cost note or ``None``
    """
    # 1. Resolve canonical category
    canonical: str | None = None
    if category:
        canonical = _normalise_category(category)
    else:
        canonical = _infer_category_from_name(product_name)

    logger.info(
        "suggest_complements: product=%r, category=%r -> canonical=%r",
        product_name,
        category,
        canonical,
    )

    # 2. Static map lookup
    if canonical and canonical in _COMPLEMENT_MAP:
        result = _COMPLEMENT_MAP[canonical]
        logger.debug("suggest_complements: returning %d static items", len(result))
        return result

    # 3. LLM fallback for unknown categories
    logger.info(
        "suggest_complements: no static entry for %r, querying LLM", canonical
    )
    prompt = (
        f"Ürün adı: {product_name}\n"
        + (f"Kategori: {category}\n" if category else "")
        + """
Türkiye pazarı için bu ürünle birlikte alınması gereken tamamlayıcı ürünleri öner.
En fazla 5 ürün öner. Cevabı aşağıdaki JSON dizisi formatında ver:

[
  {
    "product": "ürün adı",
    "reason": "kısa gerekçe",
    "priority": "yüksek|orta|düşük",
    "is_consumable": true|false,
    "recurring_cost_note": "maliyet notu veya null"
  }
]

Sadece JSON döndür, başka açıklama ekleme."""
    )
    system = (
        "Sen Türkiye pazarı için çalışan bir alışveriş asistanısın. "
        "Verilen ürün için pratik tamamlayıcı ürün önerileri sunuyorsun."
    )
    raw = await _llm_call(prompt, system=system, temperature=0.3)
    if not raw:
        logger.warning("suggest_complements: LLM returned empty response")
        return []

    items = _extract_json_list(raw)
    if not items:
        logger.warning(
            "suggest_complements: could not parse LLM response: %.200s", raw
        )
        return []

    # Validate / normalise fields
    validated: list[dict] = []
    for item in items:
        if not isinstance(item, dict) or "product" not in item:
            continue
        validated.append(
            {
                "product": str(item.get("product", "")),
                "reason": str(item.get("reason", "")),
                "priority": str(item.get("priority", "orta")),
                "is_consumable": bool(item.get("is_consumable", False)),
                "recurring_cost_note": item.get("recurring_cost_note") or None,
            }
        )

    logger.info("suggest_complements: LLM returned %d items", len(validated))
    return validated


def assess_consumable_cost(product: dict) -> dict:
    """Assess ongoing consumable costs for *product*.

    Parameters
    ----------
    product:
        Product dict with at least a ``category`` key (and optionally
        ``name``).

    Returns
    -------
    Dict with keys:

    - ``has_consumables`` — ``True`` if the product category has known
      ongoing consumable costs
    - ``consumable_items`` — list of consumable item names
    - ``estimated_annual_cost`` — rough annual cost string in Turkish
    - ``warning`` — warning string for expensive consumables, or ``None``
    """
    raw_category = product.get("category", "")
    name = product.get("name", "")

    # Try to resolve category; fall back to name-based inference
    canonical: str | None = None
    if raw_category:
        canonical = _normalise_category(raw_category)
    if not canonical or canonical not in _ANNUAL_COST_ESTIMATES:
        inferred = _infer_category_from_name(name)
        if inferred and inferred in _ANNUAL_COST_ESTIMATES:
            canonical = inferred

    logger.debug(
        "assess_consumable_cost: product=%r, raw_category=%r -> canonical=%r",
        name or product,
        raw_category,
        canonical,
    )

    if canonical and canonical in _ANNUAL_COST_ESTIMATES:
        items, cost_str = _ANNUAL_COST_ESTIMATES[canonical]
        warning = _CONSUMABLE_WARNINGS.get(canonical)
        return {
            "has_consumables": True,
            "consumable_items": items,
            "estimated_annual_cost": cost_str,
            "warning": warning,
        }

    return {
        "has_consumables": False,
        "consumable_items": [],
        "estimated_annual_cost": "Bilinen sürekli sarf maliyeti yok",
        "warning": None,
    }
