"""Bulk/wholesale deal detector.

Analyses products with quantity variants to identify genuine bulk savings,
expose fake bulk deals (where per-unit price is actually higher), and advise
whether buying in bulk makes practical sense for a given household.
"""

from __future__ import annotations

import re
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("shopping.intelligence.special.bulk_detector")

# ─── Turkish quantity pattern parsing ────────────────────────────────────────

# Patterns recognised:
#   "6'lı paket", "12li", "6'li", "3lü", "8 adet", "8adet",
#   "toptan", "koli", "düzine" (=12), "çift" (=2)
_BULK_PATTERNS: list[tuple[re.Pattern, int | None]] = [
    # "Xli paket", "X'li", "Xli", "Xlü", "Xlı", "Xlu"  (explicit count)
    (re.compile(r"(\d+)['\u2019]?\s*l[iı\u00fc u]\b", re.IGNORECASE), None),
    # "X adet" / "Xadet"
    (re.compile(r"(\d+)\s*adet\b", re.IGNORECASE), None),
    # "düzine" → 12
    (re.compile(r"d\u00fcz[iı]ne\b", re.IGNORECASE), 12),
    # "çift" → 2
    (re.compile(r"\u00e7ift\b", re.IGNORECASE), 2),
    # "koli" alone (no count found elsewhere) → treat as bulk flag
    (re.compile(r"\bkoli\b", re.IGNORECASE), 0),
    # "toptan" → bulk flag without known count
    (re.compile(r"\btoptan\b", re.IGNORECASE), 0),
]

# Perishable category keywords (Turkish product category names)
_PERISHABLE_KEYWORDS = frozenset([
    "gıda", "gida", "food",
    "süt", "sut", "dairy",
    "meyve", "sebze",
    "ekmek", "bread",
    "yoğurt", "yogurt",
    "peynir", "cheese",
])

# Meat keywords need whole-word matching to avoid false positives
# (e.g. "deterjan" contains "et", "tablet" contains "et")
_MEAT_PATTERN = re.compile(r"\b(et|meat|tavuk|chicken|balık|balik|hindi)\b", re.IGNORECASE)

# ─── Helpers ─────────────────────────────────────────────────────────────────


def _parse_quantity_from_name(name: str) -> int | None:
    """Extract numeric quantity from a Turkish product name.

    Returns the count as an integer, or ``None`` if no pattern matches.
    """
    if not name:
        return None
    for pattern, fixed_count in _BULK_PATTERNS:
        match = pattern.search(name)
        if match:
            if fixed_count is None:
                # Explicit number captured in group 1
                try:
                    return int(match.group(1))
                except (IndexError, ValueError):
                    return None
            elif fixed_count == 0:
                # Bulk keyword found but no numeric count
                return None
            else:
                return fixed_count
    return None


def _is_perishable(product: dict) -> bool:
    """Return True if the product belongs to a perishable category."""
    category = (product.get("category") or "").lower()
    name = (product.get("name") or "").lower()
    combined = f"{category} {name}"
    if _MEAT_PATTERN.search(combined):
        return True
    return any(kw in combined for kw in _PERISHABLE_KEYWORDS)


# ─── Public API ──────────────────────────────────────────────────────────────


def analyze_bulk_pricing(products: list[dict]) -> list[dict]:
    """Annotate products with per-unit pricing and bulk-deal quality.

    For each product the function resolves the quantity from either an
    explicit ``quantity`` field or by parsing the product ``name`` using
    Turkish quantity patterns.  Products with ``quantity == 1`` (or no
    detectable quantity) are treated as the single-unit reference.

    Parameters
    ----------
    products:
        List of product dicts.  Each dict should contain at minimum:
        ``price`` (float), ``name`` (str).  Optional fields:
        ``quantity`` (int), ``unit_price`` (float).

    Returns
    -------
    List of the same products enriched with:

    - ``unit_price`` -- price per single unit (float, rounded to 2 dp).
    - ``quantity`` -- resolved quantity (int).
    - ``is_bulk`` -- ``True`` when quantity > 1.
    - ``bulk_savings_pct`` -- percentage saved vs cheapest single-unit
      reference (0.0 if no reference exists or not bulk).
    - ``is_fake_bulk`` -- ``True`` when per-unit price is *higher* than
      the cheapest single-unit option.
    """
    if not products:
        return []

    annotated: list[dict] = []

    for product in products:
        result = dict(product)
        price: float = float(product.get("price") or 0)

        # Resolve quantity
        qty: int | None = product.get("quantity")
        if qty is None:
            qty = _parse_quantity_from_name(product.get("name", ""))
        if qty is None or qty <= 0:
            qty = 1

        unit_price = round(price / qty, 2) if price > 0 else 0.0

        result["quantity"] = qty
        result["unit_price"] = unit_price
        result["is_bulk"] = qty > 1
        result["bulk_savings_pct"] = 0.0
        result["is_fake_bulk"] = False
        annotated.append(result)

    # Find the cheapest single-unit reference price
    single_unit_prices = [
        p["unit_price"] for p in annotated
        if not p["is_bulk"] and p["unit_price"] > 0
    ]
    reference_unit_price: float | None = min(single_unit_prices) if single_unit_prices else None

    # Annotate bulk entries relative to reference
    for product in annotated:
        if not product["is_bulk"] or product["unit_price"] <= 0:
            continue
        if reference_unit_price is not None:
            savings = (reference_unit_price - product["unit_price"]) / reference_unit_price * 100
            product["bulk_savings_pct"] = round(savings, 1)
            product["is_fake_bulk"] = product["unit_price"] > reference_unit_price
        else:
            # No single-unit baseline — compare within bulk options
            product["bulk_savings_pct"] = 0.0

    # Sort: cheapest unit price first, unknowns last
    annotated.sort(key=lambda p: (p["unit_price"] == 0, p["unit_price"]))

    logger.debug(
        "analyze_bulk_pricing: %d products, reference_unit_price=%s",
        len(annotated), reference_unit_price,
    )
    return annotated


def detect_fake_bulk_deal(
    single_price: float,
    bulk_price: float,
    bulk_quantity: int,
) -> dict:
    """Check whether a bulk offer is actually cheaper per unit.

    Parameters
    ----------
    single_price:
        Price for one unit (TL).
    bulk_price:
        Total price for the bulk pack (TL).
    bulk_quantity:
        Number of units in the bulk pack.

    Returns
    -------
    Dict with:

    - ``is_fake`` -- ``True`` when bulk per-unit cost exceeds single unit cost.
    - ``single_unit_price`` -- cost of one unit when bought individually.
    - ``bulk_unit_price`` -- cost of one unit inside the bulk pack.
    - ``difference_pct`` -- signed percentage difference
      ``(bulk_unit_price - single_unit_price) / single_unit_price * 100``.
      Negative means bulk is cheaper; positive means bulk is more expensive.
    """
    if bulk_quantity <= 0:
        logger.warning("detect_fake_bulk_deal: bulk_quantity must be > 0")
        return {
            "is_fake": False,
            "single_unit_price": single_price,
            "bulk_unit_price": 0.0,
            "difference_pct": 0.0,
        }

    bulk_unit_price = round(bulk_price / bulk_quantity, 2)
    single_unit_price = round(single_price, 2)

    if single_unit_price <= 0:
        return {
            "is_fake": False,
            "single_unit_price": 0.0,
            "bulk_unit_price": bulk_unit_price,
            "difference_pct": 0.0,
        }

    difference_pct = round(
        (bulk_unit_price - single_unit_price) / single_unit_price * 100, 2
    )
    is_fake = bulk_unit_price > single_unit_price

    if is_fake:
        logger.info(
            "Fake bulk deal detected: single=%.2f TL, bulk_unit=%.2f TL (+%.1f%%)",
            single_unit_price, bulk_unit_price, difference_pct,
        )

    return {
        "is_fake": is_fake,
        "single_unit_price": single_unit_price,
        "bulk_unit_price": bulk_unit_price,
        "difference_pct": difference_pct,
    }


def assess_bulk_value(product: dict, household_size: int = 2) -> dict:
    """Decide whether buying in bulk is practical for the household.

    The assessment weighs:

    - **Shelf life** — perishable items (gıda, süt, et, …) carry a higher
      waste risk.
    - **Storage needs** — large quantities need space.
    - **Consumption rate** — estimated monthly usage based on household size
      and category.
    - **Break-even** — how many months until the bulk purchase pays off vs
      buying one at a time.

    Parameters
    ----------
    product:
        Product dict with at minimum ``price`` (float) and ``quantity`` (int).
        Optional useful fields: ``name``, ``category``, ``single_price``
        (price of the non-bulk equivalent per unit).
    household_size:
        Number of people in the household (default 2).

    Returns
    -------
    Dict with:

    - ``recommended`` -- ``True`` if bulk purchase is advisable.
    - ``reason`` -- human-readable Turkish explanation.
    - ``waste_risk`` -- ``"low"``, ``"medium"``, or ``"high"``.
    - ``break_even_months`` -- estimated months to consume the bulk pack
      (float, rounded to 1 dp).
    """
    if household_size <= 0:
        household_size = 1

    price: float = float(product.get("price") or 0)
    quantity: int = int(product.get("quantity") or 1)
    single_price: float = float(product.get("single_price") or 0)
    category: str = (product.get("category") or "").lower()
    name: str = (product.get("name") or "").lower()

    perishable = _is_perishable(product)

    # ── Shelf-life estimate ───────────────────────────────────────────────────
    # Express as approximate months before spoilage
    if _MEAT_PATTERN.search(f"{category} {name}"):
        shelf_life_months = 0.5          # ~2 weeks frozen, less fresh
    elif any(kw in f"{category} {name}" for kw in ("süt", "sut", "dairy", "yoğurt", "yogurt")):
        shelf_life_months = 1.0          # ~1 month
    elif any(kw in f"{category} {name}" for kw in ("ekmek", "bread", "meyve", "sebze")):
        shelf_life_months = 0.25         # ~1 week
    elif perishable:
        shelf_life_months = 2.0          # generic food
    else:
        shelf_life_months = 24.0         # non-perishable (detergent, batteries, etc.)

    # ── Consumption rate ─────────────────────────────────────────────────────
    # Units consumed per month scales roughly with household size.
    # Base assumption: a 2-person household consumes ~1 unit/month for most
    # single-serve products.  Adjust by category.
    if any(kw in f"{category} {name}" for kw in ("süt", "sut", "dairy")):
        base_consumption = 4.0           # ~4 litres/packs per month for 2 people
    elif any(kw in f"{category} {name}" for kw in ("ekmek", "bread")):
        base_consumption = 8.0
    elif any(kw in f"{category} {name}" for kw in ("deterjan", "detergent", "temizlik")):
        base_consumption = 0.5
    else:
        base_consumption = 1.0

    household_factor = max(0.5, household_size / 2.0)
    monthly_consumption = base_consumption * household_factor

    # ── Break-even months ────────────────────────────────────────────────────
    break_even_months = round(quantity / monthly_consumption, 1) if monthly_consumption > 0 else 99.0

    # ── Waste risk ───────────────────────────────────────────────────────────
    if not perishable:
        waste_risk = "low"
    elif break_even_months <= shelf_life_months * 0.75:
        waste_risk = "low"
    elif break_even_months <= shelf_life_months:
        waste_risk = "medium"
    else:
        waste_risk = "high"

    # ── Storage assessment ───────────────────────────────────────────────────
    storage_issue = quantity >= 24 and household_size < 3

    # ── Savings check ────────────────────────────────────────────────────────
    if single_price > 0 and price > 0:
        bulk_unit_price = price / quantity
        is_actually_cheaper = bulk_unit_price < single_price
        savings_pct = (single_price - bulk_unit_price) / single_price * 100
    else:
        is_actually_cheaper = quantity > 1  # assume bulk is cheaper if no single price
        savings_pct = 0.0

    # ── Decision logic ───────────────────────────────────────────────────────
    recommended: bool
    reason: str

    if waste_risk == "high":
        recommended = False
        reason = (
            f"Ürün bozulmadan tüketime yetişilemeyebilir. "
            f"Tahmini tüketim süresi ({break_even_months:.1f} ay), "
            f"raf ömrünü ({shelf_life_months:.1f} ay) aşıyor. İsraf riski yüksek."
        )
    elif not is_actually_cheaper:
        recommended = False
        reason = "Bu toplu paket birim başına daha pahalı — gerçek bir toplu alım avantajı yok."
    elif storage_issue:
        recommended = False
        reason = (
            f"{quantity} adetlik paket küçük hane için fazla büyük olabilir "
            f"ve depolama sorunu yaratabilir."
        )
    elif waste_risk == "medium":
        recommended = True
        reason = (
            f"Orta düzeyde israf riski mevcut (tahmini tüketim {break_even_months:.1f} ay). "
            f"Raf ömrüne dikkat edin."
            + (f" Yaklaşık %{savings_pct:.0f} tasarruf sağlayabilirsiniz." if savings_pct > 0 else "")
        )
    else:
        recommended = True
        reason = (
            f"Toplu alım mantıklı. Tahmini {break_even_months:.1f} ayda tüketilir, raf ömrü sorun değil."
            + (f" Yaklaşık %{savings_pct:.0f} tasarruf sağlar." if savings_pct > 0 else "")
        )

    logger.debug(
        "assess_bulk_value: qty=%d, household=%d, break_even=%.1f mo, waste_risk=%s, recommended=%s",
        quantity, household_size, break_even_months, waste_risk, recommended,
    )

    return {
        "recommended": recommended,
        "reason": reason,
        "waste_risk": waste_risk,
        "break_even_months": break_even_months,
    }
