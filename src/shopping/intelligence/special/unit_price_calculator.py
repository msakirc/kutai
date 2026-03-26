"""Per-unit price calculator for fair comparison.

Extracts quantity information from product names and normalises prices
to a common unit so that different package sizes can be compared fairly.
Supports Turkish units: kg, g, L, mL, adet, m, m2, tablet, kapsul.
"""

from __future__ import annotations

import re
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("shopping.intelligence.special.unit_price")

# ─── Unit Normalisation ─────────────────────────────────────────────────────
# Maps raw unit strings to (canonical_unit, multiplier_to_base)
# Base units: kg, L, adet, m, m2, tablet, kapsul

_UNIT_MAP: dict[str, tuple[str, float]] = {
    # Weight
    "kg": ("kg", 1.0),
    "kilo": ("kg", 1.0),
    "g": ("kg", 0.001),
    "gr": ("kg", 0.001),
    "gram": ("kg", 0.001),
    "mg": ("kg", 0.000001),
    # Volume
    "l": ("L", 1.0),
    "lt": ("L", 1.0),
    "litre": ("L", 1.0),
    "ml": ("L", 0.001),
    "cl": ("L", 0.01),
    # Count
    "adet": ("adet", 1.0),
    "pcs": ("adet", 1.0),
    "piece": ("adet", 1.0),
    "pack": ("adet", 1.0),
    "paket": ("adet", 1.0),
    "li": ("adet", 1.0),    # e.g. "12'li"
    # Length / area
    "m": ("m", 1.0),
    "cm": ("m", 0.01),
    "mm": ("m", 0.001),
    "m2": ("m\u00b2", 1.0),
    "m\u00b2": ("m\u00b2", 1.0),
    # Supplements
    "tablet": ("tablet", 1.0),
    "kapsul": ("kaps\u00fcl", 1.0),
    "kaps\u00fcl": ("kaps\u00fcl", 1.0),
    "capsule": ("kaps\u00fcl", 1.0),
}

# Regex to find quantity patterns in product names
_QTY_PATTERNS = [
    # "500ml", "2.5kg", "1L", "750 ml"
    re.compile(r"(\d+[.,]?\d*)\s*(kg|kilo|g|gr|gram|mg|l|lt|litre|ml|cl|m2|m\u00b2|m|cm|mm)\b", re.IGNORECASE),
    # "12'li", "6 li", "24-pack"
    re.compile(r"(\d+)['\u2019]?\s*(?:li|lu|l\u00fc|l\u0131|pack|paket|adet|pcs)\b", re.IGNORECASE),
    # "x12", "x 24"
    re.compile(r"x\s*(\d+)\b", re.IGNORECASE),
    # "100 tablet", "60 kapsul"
    re.compile(r"(\d+)\s*(tablet|kaps[u\u00fc]l|capsule)\b", re.IGNORECASE),
]


def calculate_unit_price(
    price: float,
    quantity: float,
    unit: str,
) -> dict:
    """Calculate price per normalised unit.

    Parameters
    ----------
    price:
        Total price in TL.
    quantity:
        Quantity in the given unit.
    unit:
        Unit string (e.g. ``"g"``, ``"ml"``, ``"adet"``).

    Returns
    -------
    Dict with ``unit_price``, ``normalized_unit``, ``display_string``.
    """
    unit_lower = unit.lower().strip()
    canonical, multiplier = _UNIT_MAP.get(unit_lower, (unit_lower, 1.0))
    base_quantity = quantity * multiplier

    if base_quantity <= 0:
        return {
            "unit_price": 0.0,
            "normalized_unit": canonical,
            "display_string": "Hesaplanamadi",
        }

    unit_price = price / base_quantity

    # Format display string
    display = f"{unit_price:,.2f} TL/{canonical}"

    return {
        "unit_price": round(unit_price, 2),
        "normalized_unit": canonical,
        "display_string": display,
    }


def compare_unit_prices(products: list[dict]) -> list[dict]:
    """Side-by-side unit price comparison with best-value flag.

    Parameters
    ----------
    products:
        List of product dicts, each with ``price``, ``name``, and
        optionally ``quantity`` and ``unit``.

    Returns
    -------
    Sorted list (cheapest first) with unit price info and ``best_value`` flag.
    """
    results = []

    for product in products:
        price = product.get("price", 0)
        name = product.get("name", "")

        # Try explicit quantity/unit first, then detect from name
        quantity = product.get("quantity")
        unit = product.get("unit")

        if quantity is None or unit is None:
            detected = detect_quantity(name)
            if detected:
                quantity = quantity or detected["quantity"]
                unit = unit or detected["unit"]

        if quantity and unit and price > 0:
            up = calculate_unit_price(price, quantity, unit)
            results.append({
                "product_name": name,
                "price": price,
                "quantity": quantity,
                "unit": unit,
                "unit_price": up["unit_price"],
                "normalized_unit": up["normalized_unit"],
                "display_string": up["display_string"],
                "best_value": False,
            })
        else:
            results.append({
                "product_name": name,
                "price": price,
                "quantity": None,
                "unit": None,
                "unit_price": None,
                "normalized_unit": None,
                "display_string": "Birim fiyat hesaplanamadi",
                "best_value": False,
            })

    # Sort by unit price (None values go to end)
    results.sort(key=lambda r: (r["unit_price"] is None, r["unit_price"] or float("inf")))

    # Mark best value
    if results and results[0]["unit_price"] is not None:
        results[0]["best_value"] = True

    return results


def detect_quantity(product_name: str) -> dict | None:
    """Extract quantity and unit from a product name string.

    Parameters
    ----------
    product_name:
        Product name, e.g. ``"Pinar Sut 1L"``, ``"Colgate 75ml 3'lu"``

    Returns
    -------
    Dict with ``quantity`` (float) and ``unit`` (str), or ``None`` if
    no quantity pattern is found.
    """
    if not product_name:
        return None

    name = product_name.strip()

    for pattern in _QTY_PATTERNS:
        match = pattern.search(name)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                qty_str, unit = groups
                qty_str = qty_str.replace(",", ".")
                try:
                    quantity = float(qty_str)
                except ValueError:
                    continue
                return {"quantity": quantity, "unit": unit.lower()}
            elif len(groups) == 1:
                # Count-based pattern (12'li, x24)
                try:
                    quantity = float(groups[0])
                except ValueError:
                    continue
                return {"quantity": quantity, "unit": "adet"}

    return None
