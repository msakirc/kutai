"""Cross-category bundle deal detector for the Turkish market.

Scans product names and descriptions for Turkish bundle patterns such as
"al X öde Y", "set fiyatı", "çeyiz paketi", and store cart discounts.
Also suggests item combinations that reach free-shipping thresholds.
"""

from __future__ import annotations

import re

from src.infra.logging_config import get_logger

logger = get_logger("shopping.intelligence.special.bundle_detector")

# ─── Bundle Keyword Patterns ────────────────────────────────────────────────

# "al X öde Y" — buy X pay Y promotions (e.g. "3 al 2 öde", "2 al 1 öde")
_AL_ODE_RE = re.compile(
    r"(\d+)\s*al[,\s]+(\d+)\s*[oö]de",
    re.IGNORECASE,
)

# Set / kit price signals
_SET_KEYWORDS = re.compile(
    r"\b(set|paket|takım|takim|kit|kombin|kombinasyon|çeyiz|ceyiz|trousseau)\b",
    re.IGNORECASE,
)

# Campaign / freebie signals
_CAMPAIGN_KEYWORDS = re.compile(
    r"\b(kampanya|hediyeli|yanında|yaninda|birlikte\s+al|bundle|ücretsiz\s+kargo|ucretsiz\s+kargo)\b",
    re.IGNORECASE,
)

# "set fiyatı" or "paket fiyatı" — explicit set-pricing label
_SET_PRICE_RE = re.compile(
    r"(set|paket|takım|takim|kit)\s+fiyat[ıi]",
    re.IGNORECASE,
)

# Quantity multiplier in name: "2'li set", "3'lü paket", "5li takım"
_MULTI_QTY_RE = re.compile(
    r"(\d+)['\u2019]?\s*(?:li|lu|lü|lı|li)\s+(?:set|paket|takım|takim|kit)",
    re.IGNORECASE,
)

# "X adet" in name indicating multi-pack
_ADET_RE = re.compile(r"(\d+)\s*adet\b", re.IGNORECASE)


def _searchable_text(product: dict) -> str:
    """Combine name and description into a single searchable string."""
    parts = [
        product.get("name", ""),
        product.get("description", ""),
        product.get("title", ""),
    ]
    return " ".join(p for p in parts if p).lower()


# ─── Public Functions ────────────────────────────────────────────────────────


def detect_bundle_deals(products: list[dict]) -> list[dict]:
    """Scan product names/descriptions for Turkish bundle patterns.

    Parameters
    ----------
    products:
        List of product dicts. Each should have at least ``name`` and
        optionally ``description``, ``price``, ``original_price``.

    Returns
    -------
    List of detected bundle dicts with keys:

    - ``type`` -- bundle type label (e.g. ``"al_ode"``, ``"set"``, ``"campaign"``).
    - ``description`` -- human-readable Turkish explanation.
    - ``savings_estimate`` -- estimated savings in TL (0.0 if unknown).
    - ``products_involved`` -- list of matching product names.
    """
    bundles: list[dict] = []

    for product in products:
        name = product.get("name", "")
        text = _searchable_text(product)
        price = product.get("price", 0.0) or 0.0
        original = product.get("original_price", 0.0) or 0.0

        # ── "X al Y öde" pattern ────────────────────────────────────────────
        al_ode_match = _AL_ODE_RE.search(text)
        if al_ode_match:
            buy_qty = int(al_ode_match.group(1))
            pay_qty = int(al_ode_match.group(2))
            if buy_qty > pay_qty > 0:
                free_items = buy_qty - pay_qty
                savings = (price / buy_qty) * free_items if price > 0 else 0.0
                bundles.append({
                    "type": "al_ode",
                    "description": (
                        f"{buy_qty} al {pay_qty} öde kampanyası — "
                        f"{free_items} ürün bedava"
                    ),
                    "savings_estimate": round(savings, 2),
                    "products_involved": [name],
                })
                logger.debug("al_ode bundle detected: %s", name)
                continue  # already categorised, skip further checks

        # ── Explicit "set fiyatı" label ─────────────────────────────────────
        if _SET_PRICE_RE.search(text):
            savings = round(original - price, 2) if original > price else 0.0
            bundles.append({
                "type": "set_price",
                "description": "Set/paket fiyatıyla satılıyor — bireysel alıma göre avantajlı",
                "savings_estimate": savings,
                "products_involved": [name],
            })
            logger.debug("set_price bundle detected: %s", name)
            continue

        # ── Generic set / kit keyword ────────────────────────────────────────
        if _SET_KEYWORDS.search(text):
            # Try to infer quantity from name
            multi_match = _MULTI_QTY_RE.search(text)
            adet_match = _ADET_RE.search(text)
            qty = None
            if multi_match:
                qty = int(multi_match.group(1))
            elif adet_match:
                qty = int(adet_match.group(1))

            if qty and price > 0:
                unit_price = price / qty
                description = (
                    f"{qty} parçalı set — birim fiyat yaklaşık {unit_price:,.2f} TL"
                )
            else:
                description = "Set/takım/paket ürünü — birden fazla parça içeriyor olabilir"

            savings = round(original - price, 2) if original > price else 0.0
            bundles.append({
                "type": "set",
                "description": description,
                "savings_estimate": savings,
                "products_involved": [name],
            })
            logger.debug("set bundle detected: %s", name)
            continue

        # ── Campaign / freebie keyword ───────────────────────────────────────
        if _CAMPAIGN_KEYWORDS.search(text):
            savings = round(original - price, 2) if original > price else 0.0
            bundles.append({
                "type": "campaign",
                "description": "Kampanyalı/hediyeli ürün — ek avantaj içeriyor",
                "savings_estimate": savings,
                "products_involved": [name],
            })
            logger.debug("campaign bundle detected: %s", name)

    logger.info(
        "detect_bundle_deals: %d products scanned, %d bundles found",
        len(products), len(bundles),
    )
    return bundles


def suggest_shipping_combos(
    products: list[dict],
    free_shipping_threshold: float = 150.0,
) -> list[dict]:
    """Suggest adding items to reach the free-shipping threshold.

    Parameters
    ----------
    products:
        Products currently in the cart. Each must have ``price`` (float).
    free_shipping_threshold:
        Cart total above which shipping is free. Default 150 TL (common on
        Trendyol / Hepsiburada for standard orders).

    Returns
    -------
    List of suggestion dicts with keys:

    - ``current_total`` -- sum of prices in ``products``.
    - ``threshold`` -- the free-shipping threshold used.
    - ``gap`` -- how many TL away from free shipping (0 if already reached).
    - ``suggestion`` -- human-readable Turkish advice string.
    """
    if not products:
        return []

    current_total = sum(p.get("price", 0.0) or 0.0 for p in products)
    gap = max(0.0, free_shipping_threshold - current_total)

    if gap == 0.0:
        suggestion = (
            f"Sepetiniz zaten ücretsiz kargo limitini ({free_shipping_threshold:.0f} TL) "
            f"geçiyor. Ek ürün eklemenize gerek yok."
        )
    elif gap <= 10.0:
        suggestion = (
            f"Ücretsiz kargoya sadece {gap:.2f} TL kaldı! "
            f"Küçük bir ürün ekleyerek kargo ücretinden tasarruf edebilirsiniz."
        )
    elif gap <= 30.0:
        suggestion = (
            f"Sepetinize {gap:.2f} TL değerinde ürün eklersen ücretsiz kargo "
            f"({free_shipping_threshold:.0f} TL) kazanırsınız — kargo ücreti genellikle "
            f"bu miktarın üzerindedir."
        )
    else:
        suggestion = (
            f"Ücretsiz kargo için {gap:.2f} TL daha eklemeniz gerekiyor "
            f"(hedef: {free_shipping_threshold:.0f} TL). "
            f"İhtiyacınız olan başka ürünler varsa birlikte sipariş verin."
        )

    logger.debug(
        "suggest_shipping_combos: total=%.2f, threshold=%.2f, gap=%.2f",
        current_total, free_shipping_threshold, gap,
    )

    return [{
        "current_total": round(current_total, 2),
        "threshold": free_shipping_threshold,
        "gap": round(gap, 2),
        "suggestion": suggestion,
    }]


def detect_set_pricing(products: list[dict]) -> list[dict]:
    """Find products sold as sets where per-unit price is better than individual.

    Compares ``price`` against ``individual_price`` (if provided) or estimates
    per-unit cost from quantity hints in the product name.  Products without
    any quantity signal are still flagged if they carry explicit set keywords,
    but without a numeric savings estimate.

    Parameters
    ----------
    products:
        List of product dicts. Useful optional keys:

        - ``price`` -- set/pack selling price.
        - ``original_price`` -- crossed-out price (used as proxy for individual cost).
        - ``individual_price`` -- explicit single-unit price for comparison.
        - ``name`` / ``description`` -- scanned for quantity signals.

    Returns
    -------
    List of dicts with keys:

    - ``product_name`` -- name of the product.
    - ``set_price`` -- the price paid for the set.
    - ``unit_count`` -- inferred number of units (``None`` if unknown).
    - ``price_per_unit`` -- ``set_price / unit_count`` (``None`` if unknown).
    - ``individual_price`` -- reference single-unit price (``None`` if unknown).
    - ``savings_per_unit`` -- how much cheaper per unit vs individual (``None`` if unknown).
    - ``savings_pct`` -- percentage savings vs individual (``None`` if unknown).
    - ``verdict`` -- human-readable Turkish assessment string.
    """
    results: list[dict] = []

    for product in products:
        name = product.get("name", "")
        text = _searchable_text(product)
        set_price = product.get("price", 0.0) or 0.0

        if set_price <= 0:
            continue  # can't analyse without a price

        # ── Detect unit count ────────────────────────────────────────────────
        unit_count: int | None = None

        multi_match = _MULTI_QTY_RE.search(text)
        adet_match = _ADET_RE.search(text)
        al_ode_match = _AL_ODE_RE.search(text)

        if multi_match:
            unit_count = int(multi_match.group(1))
        elif adet_match:
            candidate = int(adet_match.group(1))
            if candidate > 1:  # single-adet products not interesting
                unit_count = candidate
        elif al_ode_match:
            # For "3 al 2 öde", the set contains 3 units but you pay for 2
            unit_count = int(al_ode_match.group(1))

        # ── Determine individual reference price ─────────────────────────────
        individual_price: float | None = product.get("individual_price")
        if individual_price is None:
            # Fall back to original_price as a proxy (often the per-unit list price)
            op = product.get("original_price", 0.0) or 0.0
            if op > set_price:
                individual_price = op

        # ── Skip if neither set keyword nor qty found ────────────────────────
        has_set_signal = bool(
            _SET_KEYWORDS.search(text)
            or _SET_PRICE_RE.search(text)
            or al_ode_match
        )
        if not has_set_signal and unit_count is None:
            continue

        # ── Compute savings ──────────────────────────────────────────────────
        price_per_unit: float | None = None
        savings_per_unit: float | None = None
        savings_pct: float | None = None

        if unit_count and unit_count > 0:
            price_per_unit = round(set_price / unit_count, 2)

        if individual_price and price_per_unit is not None:
            savings_per_unit = round(individual_price - price_per_unit, 2)
            if individual_price > 0:
                savings_pct = round((savings_per_unit / individual_price) * 100, 1)
        elif individual_price and unit_count:
            total_individual = individual_price * unit_count
            if total_individual > set_price:
                savings_per_unit = round((total_individual - set_price) / unit_count, 2)
                savings_pct = round(((total_individual - set_price) / total_individual) * 100, 1)

        # ── Build verdict string ─────────────────────────────────────────────
        if savings_pct is not None and savings_pct > 0:
            verdict = (
                f"Set alımı, tekil fiyata göre birim başına %{savings_pct:.1f} daha ucuz "
                f"({savings_per_unit:.2f} TL/adet tasarruf)."
            )
        elif unit_count and price_per_unit:
            verdict = (
                f"{unit_count} adetlik set; birim fiyat yaklaşık {price_per_unit:.2f} TL. "
                f"Tekil fiyatla karşılaştırın."
            )
        elif has_set_signal:
            verdict = "Set/paket ürünü tespit edildi; tekil fiyatla karşılaştırmanız önerilir."
        else:
            verdict = "Çoklu paket — birim fiyat avantajı doğrulanamadı."

        results.append({
            "product_name": name,
            "set_price": set_price,
            "unit_count": unit_count,
            "price_per_unit": price_per_unit,
            "individual_price": individual_price,
            "savings_per_unit": savings_per_unit,
            "savings_pct": savings_pct,
            "verdict": verdict,
        })

        logger.debug(
            "detect_set_pricing: '%s' — %s units @ %.2f TL/unit, savings_pct=%s",
            name, unit_count, price_per_unit or 0.0, savings_pct,
        )

    logger.info(
        "detect_set_pricing: %d products scanned, %d set-price products found",
        len(products), len(results),
    )
    return results
