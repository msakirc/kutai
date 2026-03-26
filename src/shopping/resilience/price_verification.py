"""Cross-source price verification for the shopping system.

When the same product is found on multiple sources, prices deviating more than
a configurable threshold from the group median are flagged as suspicious.
Suspicious prices are never silently dropped — they are annotated and returned
alongside trustworthy ones so the caller can decide how to handle them.
"""

from __future__ import annotations

import re
from statistics import median
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("shopping.resilience.price_verification")

try:
    from src.shopping.text_utils import normalize_turkish
except ImportError:  # pragma: no cover — fallback when text_utils not available
    def normalize_turkish(text: str) -> str:  # type: ignore[misc]
        """Minimal Turkish-aware lowercase fallback."""
        _MAP = {
            "İ": "i", "I": "ı", "Ş": "ş",
            "Ç": "ç", "Ö": "ö", "Ü": "ü", "Ğ": "ğ",
        }
        return "".join(_MAP.get(ch, ch.lower()) for ch in text)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Common suffixes that don't help distinguish products, e.g. brand/model noise
_STRIP_SUFFIXES_RE = re.compile(
    r"\s*(adet|paket|kutu|set|fiyat[ıi]|fiyat|fiyatı|indirimli)\s*$",
    re.IGNORECASE,
)

# Characters that are not alphanumeric or whitespace
_NON_ALNUM_RE = re.compile(r"[^\w\s]", re.UNICODE)

# Runs of whitespace
_WHITESPACE_RE = re.compile(r"\s+")


def _normalise_name(name: str) -> str:
    """Return a cleaned, lowercased, stripped name suitable for comparison.

    Steps:
    1. Turkish-aware lowercase.
    2. Remove punctuation.
    3. Strip common marketing/unit suffixes.
    4. Collapse whitespace.
    """
    text = normalize_turkish(name)
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _STRIP_SUFFIXES_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _token_overlap(a: str, b: str) -> float:
    """Return Jaccard similarity of word token sets for *a* and *b*."""
    tokens_a = set(_normalise_name(a).split())
    tokens_b = set(_normalise_name(b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


# Minimum Jaccard similarity to consider two names the same product
_SIMILARITY_THRESHOLD = 0.6


def _fuzzy_group_products(products: list[dict]) -> dict[str, list[int]]:
    """Group product indices by similar names using token-based fuzzy matching.

    Parameters
    ----------
    products:
        List of product dicts, each with at least a ``name`` field.

    Returns
    -------
    Dict mapping a *canonical group key* (the normalised name of the first
    member that formed the group) to a list of indices into *products*.

    Algorithm
    ---------
    Single-pass greedy grouping: for each product we check whether its name is
    sufficiently similar (Jaccard >= 0.6) to any existing group representative.
    If yes it joins that group; if no it starts a new group.  This is O(n²) in
    the number of distinct groups but practical for the expected input sizes
    (tens to low hundreds of products per shopping session).
    """
    # group_key -> (representative_normalised_name, [indices])
    groups: dict[str, tuple[str, list[int]]] = {}

    for idx, product in enumerate(products):
        raw_name = product.get("name", "")
        norm = _normalise_name(raw_name)

        matched_key: str | None = None
        best_sim = 0.0

        for key, (rep_norm, _indices) in groups.items():
            sim = _token_overlap(norm, rep_norm)
            if sim >= _SIMILARITY_THRESHOLD and sim > best_sim:
                best_sim = sim
                matched_key = key

        if matched_key is not None:
            groups[matched_key][1].append(idx)
        else:
            groups[norm] = (norm, [idx])

    return {key: indices for key, (_rep, indices) in groups.items()}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def flag_outliers(
    prices: list[float],
    threshold_pct: float = 40.0,
) -> list[dict]:
    """Identify price outliers relative to the group median.

    Parameters
    ----------
    prices:
        List of numeric prices for what is considered the same product.
    threshold_pct:
        Percentage deviation from the median above which a price is flagged.
        Default is 40 %.

    Returns
    -------
    List of dicts, one per input price, each containing:

    - ``price`` -- the original price value.
    - ``is_outlier`` -- ``True`` if the price deviates more than
      *threshold_pct* % from the median.
    - ``deviation_from_median_pct`` -- signed percentage deviation
      ``(price - median) / median * 100``.  Positive means above median.

    Edge cases
    ----------
    If *prices* is empty or contains only a single entry the deviation is
    0.0 and ``is_outlier`` is always ``False`` (no reference point).
    """
    if not prices:
        return []

    med = median(prices)

    results: list[dict] = []
    for price in prices:
        if med == 0.0:
            deviation_pct = 0.0
        else:
            deviation_pct = (price - med) / med * 100.0

        is_outlier = abs(deviation_pct) > threshold_pct

        results.append(
            {
                "price": price,
                "is_outlier": is_outlier,
                "deviation_from_median_pct": round(deviation_pct, 2),
            }
        )

    return results


def verify_prices(
    products: list[dict],
    threshold_pct: float = 40.0,
) -> list[dict]:
    """Annotate products with cross-source price verification metadata.

    Products are grouped by similar name using fuzzy token matching.  Within
    each group the median price is computed, and any product whose price
    deviates from that median by more than *threshold_pct* % is marked
    suspicious.

    Parameters
    ----------
    products:
        List of product dicts.  Each dict must have at minimum:

        - ``name`` (str) -- product name.
        - ``price`` (float | int | None) -- numeric price.
        - ``source`` (str) -- originating scraper/source name.

        Additional fields are preserved untouched.
    threshold_pct:
        Deviation threshold in percent.  Default is 40 %.

    Returns
    -------
    A new list of product dicts (shallow copies of inputs) with four extra
    fields added to every item:

    - ``price_verified`` (bool) -- ``True`` if the price was checked against
      at least one other source; ``False`` when the product is the only member
      of its name group (no cross-source comparison was possible).
    - ``price_suspicious`` (bool) -- ``True`` if the price deviates more than
      *threshold_pct* % from the group median.
    - ``price_deviation_pct`` (float | None) -- signed deviation from median,
      ``None`` when the product has no valid price or no comparison group.
    - ``verification_note`` (str) -- human-readable explanation, empty string
      when everything is fine.

    Suspicious prices are never removed — they are returned alongside normal
    ones so callers can decide how to present or handle them.
    """
    if not products:
        return []

    annotated: list[dict] = [dict(p) for p in products]

    # Default annotations
    for p in annotated:
        p["price_verified"] = False
        p["price_suspicious"] = False
        p["price_deviation_pct"] = None
        p["verification_note"] = ""

    groups = _fuzzy_group_products(products)

    for group_key, indices in groups.items():
        if len(indices) < 2:
            # Single product in group — cannot cross-verify
            idx = indices[0]
            price = annotated[idx].get("price")
            if price is not None:
                annotated[idx]["verification_note"] = (
                    "Only one source for this product; cross-verification not possible."
                )
            logger.debug(
                "Group '%s': single product (source=%s), skipping cross-verification",
                group_key,
                annotated[indices[0]].get("source", "unknown"),
            )
            continue

        # Collect valid prices and their source for logging
        valid_indices: list[int] = []
        valid_prices: list[float] = []
        for idx in indices:
            price = annotated[idx].get("price")
            if price is not None:
                try:
                    valid_prices.append(float(price))
                    valid_indices.append(idx)
                except (TypeError, ValueError):
                    logger.warning(
                        "Non-numeric price for product '%s' from source '%s': %r",
                        annotated[idx].get("name"),
                        annotated[idx].get("source"),
                        price,
                    )

        if not valid_prices:
            logger.debug("Group '%s': no valid prices, skipping", group_key)
            continue

        # Mark all products in the group as cross-verified
        for idx in indices:
            annotated[idx]["price_verified"] = True

        if len(valid_prices) < 2:
            # Only one member has a price — still flag as verified but note it
            idx = valid_indices[0]
            annotated[idx]["verification_note"] = (
                "Other products in group lack prices; full cross-verification not possible."
            )
            continue

        outlier_results = flag_outliers(valid_prices, threshold_pct)
        group_median = median(valid_prices)

        sources_in_group = [
            annotated[i].get("source", "unknown") for i in valid_indices
        ]
        logger.debug(
            "Group '%s': %d products, median=%.2f, sources=%s",
            group_key,
            len(valid_prices),
            group_median,
            sources_in_group,
        )

        for list_pos, (idx, outlier_info) in enumerate(
            zip(valid_indices, outlier_results)
        ):
            product = annotated[idx]
            deviation = outlier_info["deviation_from_median_pct"]
            is_outlier = outlier_info["is_outlier"]

            product["price_deviation_pct"] = deviation
            product["price_suspicious"] = is_outlier

            if is_outlier:
                direction = "above" if deviation > 0 else "below"
                product["verification_note"] = (
                    f"Price {abs(deviation):.1f}% {direction} group median "
                    f"({group_median:.2f}); flagged as suspicious. "
                    f"Compared against {len(valid_prices) - 1} other source(s): "
                    f"{', '.join(s for i, s in enumerate(sources_in_group) if i != list_pos)}."
                )
                logger.info(
                    "Suspicious price for '%s' from '%s': %.2f "
                    "(%.1f%% %s median %.2f)",
                    product.get("name"),
                    product.get("source"),
                    float(product.get("price", 0)),
                    abs(deviation),
                    direction,
                    group_median,
                )
            else:
                product["verification_note"] = (
                    f"Price within {threshold_pct:.0f}% of group median ({group_median:.2f}); "
                    f"verified across {len(valid_prices)} source(s)."
                )

    return annotated
