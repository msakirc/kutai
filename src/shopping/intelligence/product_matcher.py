"""Product matching module for the shopping intelligence system.

Matches products from different sources to identify the same item, handling
variants (colour, size, bundle).  Uses a matching hierarchy:
EAN/UPC > MPN > fuzzy name > spec fingerprint.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from src.infra.logging_config import get_logger
from src.shopping.models import Product, ProductMatch
from src.shopping.text_utils import normalize_product_name, normalize_turkish

logger = get_logger("shopping.intelligence.product_matcher")

# ── Constants ────────────────────────────────────────────────────────────────

# Minimum confidence thresholds for different matching strategies.
_THRESHOLD_EAN = 0.99
_THRESHOLD_MPN = 0.95
_THRESHOLD_FUZZY_NAME = 0.70
_THRESHOLD_SPEC_FINGERPRINT = 0.60
_THRESHOLD_OVERALL = 0.55  # minimum to consider products matched

# Spec keys that are strong identifiers
_IDENTITY_SPECS = [
    "ean", "upc", "barcode", "barkod",
    "mpn", "model_no", "model_number", "model_numarası",
    "sku",
]

# Spec keys used for fingerprinting
_FINGERPRINT_SPECS = [
    "brand", "marka",
    "model", "model_no",
    "capacity", "kapasite",
    "color", "renk",
    "size", "boyut", "beden",
    "storage", "depolama",
    "ram",
    "processor", "işlemci",
    "screen_size", "ekran_boyutu",
    "resolution", "çözünürlük",
    "weight", "ağırlık",
    "wattage", "güç",
]

# Known variant attributes (colour, size, bundle type)
_VARIANT_KEYS = ["color", "renk", "size", "boyut", "beden", "bundle", "paket"]


# ── Matching helpers ─────────────────────────────────────────────────────────

def _get_spec(product: Product, keys: list[str]) -> str | None:
    """Return the first non-empty spec value matching any of *keys*."""
    for key in keys:
        val = product.specs.get(key)
        if val:
            return str(val).strip()
    return None


def _ean_match(a: Product, b: Product) -> float:
    """Match by EAN / UPC / barcode.  Returns 0.0 or ~1.0."""
    ean_keys = ["ean", "upc", "barcode", "barkod"]
    ean_a = _get_spec(a, ean_keys)
    ean_b = _get_spec(b, ean_keys)
    if ean_a and ean_b:
        # Normalise: strip leading zeros, compare digits only
        norm_a = re.sub(r"[^0-9]", "", ean_a).lstrip("0")
        norm_b = re.sub(r"[^0-9]", "", ean_b).lstrip("0")
        if norm_a and norm_b and norm_a == norm_b:
            return _THRESHOLD_EAN
    return 0.0


def _mpn_match(a: Product, b: Product) -> float:
    """Match by manufacturer part number.  Returns 0.0 or ~0.95."""
    mpn_keys = ["mpn", "model_no", "model_number", "model_numarası"]
    mpn_a = _get_spec(a, mpn_keys)
    mpn_b = _get_spec(b, mpn_keys)
    if mpn_a and mpn_b:
        norm_a = re.sub(r"[\s\-_]", "", mpn_a).upper()
        norm_b = re.sub(r"[\s\-_]", "", mpn_b).upper()
        if norm_a and norm_b and norm_a == norm_b:
            return _THRESHOLD_MPN
    return 0.0


def _fuzzy_name_match(a: Product, b: Product) -> float:
    """Fuzzy match on normalised product names.  Returns 0.0 - 1.0."""
    name_a = normalize_turkish(normalize_product_name(a.name))
    name_b = normalize_turkish(normalize_product_name(b.name))
    if not name_a or not name_b:
        return 0.0
    ratio = SequenceMatcher(None, name_a, name_b).ratio()
    return ratio if ratio >= _THRESHOLD_FUZZY_NAME else 0.0


def _spec_fingerprint_match(a: Product, b: Product) -> float:
    """Compare products by their spec fingerprint.

    Counts matching non-empty spec values across fingerprint keys.
    """
    matches = 0
    compared = 0

    for key in _FINGERPRINT_SPECS:
        val_a = a.specs.get(key)
        val_b = b.specs.get(key)
        if val_a and val_b:
            compared += 1
            norm_a = normalize_turkish(str(val_a).strip())
            norm_b = normalize_turkish(str(val_b).strip())
            if norm_a == norm_b:
                matches += 1
            elif SequenceMatcher(None, norm_a, norm_b).ratio() > 0.85:
                matches += 0.7

    if compared == 0:
        return 0.0

    score = matches / compared
    return score if score >= _THRESHOLD_SPEC_FINGERPRINT else 0.0


def _detect_variant(a: Product, b: Product) -> dict | None:
    """Detect if two products are variants (same base, different colour/size)."""
    diffs: dict[str, tuple[str, str]] = {}
    for key in _VARIANT_KEYS:
        val_a = a.specs.get(key)
        val_b = b.specs.get(key)
        if val_a and val_b:
            if normalize_turkish(str(val_a)) != normalize_turkish(str(val_b)):
                diffs[key] = (str(val_a), str(val_b))

    if diffs:
        return {"variant_type": list(diffs.keys()), "differences": diffs}
    return None


def _compute_confidence(a: Product, b: Product) -> tuple[float, str]:
    """Compute match confidence using the hierarchy.

    Returns (confidence, method) where method is the strategy that produced
    the score.
    """
    # Tier 1: EAN/UPC
    score = _ean_match(a, b)
    if score > 0:
        return score, "ean"

    # Tier 2: MPN
    score = _mpn_match(a, b)
    if score > 0:
        return score, "mpn"

    # Tier 3: Fuzzy name
    name_score = _fuzzy_name_match(a, b)

    # Tier 4: Spec fingerprint
    spec_score = _spec_fingerprint_match(a, b)

    # Combine tiers 3+4 with more weight on specs when available
    if name_score > 0 and spec_score > 0:
        combined = name_score * 0.4 + spec_score * 0.6
        return combined, "name+specs"
    if name_score > 0:
        return name_score * 0.8, "fuzzy_name"
    if spec_score > 0:
        return spec_score * 0.7, "spec_fingerprint"

    return 0.0, "none"


def _canonical_name(products: list[Product]) -> str:
    """Pick the best canonical name from matched products."""
    if not products:
        return ""
    # Prefer the longest cleaned name (usually most complete)
    cleaned = [
        (normalize_product_name(p.name), p.name) for p in products
    ]
    cleaned.sort(key=lambda c: len(c[0]), reverse=True)
    return cleaned[0][1]


def _canonical_specs(products: list[Product]) -> dict:
    """Merge specs from all matched products, preferring non-empty values."""
    merged: dict = {}
    for p in products:
        for key, val in p.specs.items():
            if val and (key not in merged or not merged[key]):
                merged[key] = val
    return merged


# ── Public API ───────────────────────────────────────────────────────────────

async def match_products(products: list[Product]) -> list[dict]:
    """Match products from different sources to find identical/variant items.

    Uses a hierarchical matching approach:
    1. EAN / UPC barcode (exact)
    2. Manufacturer Part Number (exact)
    3. Fuzzy product name similarity
    4. Spec fingerprint comparison

    Parameters
    ----------
    products:
        Products collected from various sources.

    Returns
    -------
    List of match group dicts, each with:
    - canonical_name: str
    - canonical_specs: dict
    - confidence_score: float (0-1)
    - match_method: str
    - products: list of product dicts (name, source, price, url)
    - variant_info: dict or None
    - product_count: int
    """
    if not products:
        return []

    if len(products) == 1:
        p = products[0]
        return [{
            "canonical_name": p.name,
            "canonical_specs": dict(p.specs),
            "confidence_score": 1.0,
            "match_method": "single",
            "products": [{
                "name": p.name,
                "source": p.source,
                "price": p.discounted_price or p.original_price,
                "url": p.url,
            }],
            "variant_info": None,
            "product_count": 1,
        }]

    # Greedy matching: assign each product to a group
    groups: list[list[int]] = []  # list of product index lists
    group_confidence: list[float] = []
    group_method: list[str] = []
    assigned: set[int] = set()

    for i in range(len(products)):
        if i in assigned:
            continue

        current_group = [i]
        assigned.add(i)
        best_conf = 0.0
        best_method = "single"

        for j in range(i + 1, len(products)):
            if j in assigned:
                continue

            conf, method = _compute_confidence(products[i], products[j])
            if conf >= _THRESHOLD_OVERALL:
                current_group.append(j)
                assigned.add(j)
                if conf > best_conf:
                    best_conf = conf
                    best_method = method

        groups.append(current_group)
        group_confidence.append(best_conf if len(current_group) > 1 else 1.0)
        group_method.append(best_method)

    # Build output
    results: list[dict] = []
    for g_idx, indices in enumerate(groups):
        group_products = [products[i] for i in indices]
        c_name = _canonical_name(group_products)
        c_specs = _canonical_specs(group_products)

        # Detect variants within the group
        variant_info = None
        if len(group_products) >= 2:
            variant_info = _detect_variant(group_products[0], group_products[1])

        results.append({
            "canonical_name": c_name,
            "canonical_specs": c_specs,
            "confidence_score": round(group_confidence[g_idx], 3),
            "match_method": group_method[g_idx],
            "products": [
                {
                    "name": p.name,
                    "source": p.source,
                    "price": p.discounted_price or p.original_price,
                    "url": p.url,
                }
                for p in group_products
            ],
            "variant_info": variant_info,
            "product_count": len(group_products),
        })

    # Sort: multi-source matches first, then by confidence
    results.sort(
        key=lambda r: (r["product_count"], r["confidence_score"]),
        reverse=True,
    )

    multi = sum(1 for r in results if r["product_count"] > 1)
    logger.info(
        "Matched %d products into %d groups (%d multi-source matches)",
        len(products),
        len(results),
        multi,
    )
    return results
