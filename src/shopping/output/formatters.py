"""Comparison table formatter for shopping results.

Formats product comparison data for Telegram, terminal, and JSON output.
All functions are pure — no I/O side effects.
"""

from __future__ import annotations

import json
import math


# ---------------------------------------------------------------------------
# Price formatting
# ---------------------------------------------------------------------------

def format_price(price: float, currency: str = "TL") -> str:
    """Format price in Turkish locale: 1.299,99 TL."""
    if price is None:
        return ""
    negative = price < 0
    price = abs(price)
    integer_part = int(price)
    decimal_part = round((price - integer_part) * 100)
    # Thousands separator with dots
    int_str = f"{integer_part:,}".replace(",", ".")
    formatted = f"{int_str},{decimal_part:02d} {currency}"
    if negative:
        formatted = "-" + formatted
    return formatted


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_len: int) -> str:
    """Truncate *text* to *max_len* characters, adding ellipsis if needed."""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "\u2026"


def _seller_badge(rating: float) -> str:
    """Return a star badge string for the seller rating."""
    if rating is None:
        return ""
    if rating >= 4.5:
        return "\u2b50 Top Seller"
    if rating >= 4.0:
        return "\u2b50 Trusted"
    if rating >= 3.0:
        return "\u2b50 Average"
    return "\u26a0\ufe0f Low Rating"


def _value_badge(product: dict) -> str:
    """Return a value badge based on product attributes.

    Expected keys: ``is_best_value``, ``is_deal``, ``has_warning``,
    ``is_suggestion``, ``discount_percentage``.
    """
    if product.get("is_best_value"):
        return "\U0001f3c6 Best Value"
    if product.get("has_warning"):
        return "\u26a0\ufe0f Warning"
    pct = product.get("discount_percentage") or 0
    if pct >= 20 or product.get("is_deal"):
        return "\U0001f525 Deal"
    if product.get("is_suggestion"):
        return "\U0001f4a1 Suggestion"
    return ""


def _price_trend_badge(history: list) -> str:
    """Return a trend badge from a price history list.

    Each entry is expected to have a ``price`` key.  The badge reflects
    whether the current price is near a 3-month low, rising, or stable.
    """
    if not history or len(history) < 2:
        return ""
    prices = [h.get("price", 0) for h in history if h.get("price")]
    if not prices:
        return ""
    current = prices[-1]
    min_price = min(prices)
    max_price = max(prices)
    if max_price == min_price:
        return ""
    # Near 3-month low (within 5% of minimum)
    if current <= min_price * 1.05:
        return "\U0001f4c9 3-Month Low"
    # Rising trend — current is in the top 15% of the range
    if current >= max_price * 0.85:
        return "\U0001f4c8 Price Rising"
    return ""


# ---------------------------------------------------------------------------
# Installment formatting
# ---------------------------------------------------------------------------

def format_installment_options(options: list[dict]) -> str:
    """Format installment options as ``'12 ay x 833 TL'`` lines.

    Each dict is expected to have ``months`` and ``monthly_amount`` keys,
    and optionally ``bank`` and ``interest_rate``.
    """
    if not options:
        return ""
    lines: list[str] = []
    for opt in options:
        months = opt.get("months", 0)
        amount = opt.get("monthly_amount", 0)
        price_str = format_price(amount)
        line = f"{months} ay \u00d7 {price_str}"
        bank = opt.get("bank")
        rate = opt.get("interest_rate")
        extras: list[str] = []
        if bank:
            extras.append(bank)
        if rate is not None:
            extras.append(f"%{rate}")
        if extras:
            line += f" ({', '.join(extras)})"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Price comparison (same product, multiple sources)
# ---------------------------------------------------------------------------

def format_price_comparison(product_matches: list[dict]) -> str:
    """Format a comparison of the same product across different sources.

    Each dict should contain ``source``, ``price``, ``currency``,
    ``seller_name``, ``seller_rating``, ``shipping_cost``,
    ``shipping_time_days``, ``url``.
    """
    if not product_matches:
        return ""
    lines: list[str] = []
    # Sort by effective price (price + shipping)
    sorted_matches = sorted(
        product_matches,
        key=lambda m: (m.get("price") or math.inf)
        + (m.get("shipping_cost") or 0),
    )
    for i, m in enumerate(sorted_matches):
        price = format_price(
            m.get("price", 0), m.get("currency", "TL")
        )
        source = m.get("source", "")
        seller = m.get("seller_name", "")
        badge = _seller_badge(m.get("seller_rating"))
        shipping = m.get("shipping_cost")
        days = m.get("shipping_time_days")

        marker = "\U0001f947" if i == 0 else f" {i + 1}."
        parts = [f"{marker} **{source}**"]
        if seller:
            parts.append(f"({seller})")
        if badge:
            parts.append(badge)
        parts.append(f"  {price}")
        if shipping is not None:
            if shipping == 0:
                parts.append("+ Free Shipping")
            else:
                parts.append(f"+ {format_price(shipping)} kargo")
        if days is not None:
            parts.append(f"({days} g\u00fcn)")

        lines.append(" ".join(parts))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comparison table — main entry point
# ---------------------------------------------------------------------------

def _build_telegram_table(products: list[dict]) -> str:
    """Build a Telegram-friendly comparison string."""
    if not products:
        return "No products to compare."
    lines: list[str] = []
    for i, p in enumerate(products, 1):
        name = _truncate(p.get("name", ""), 50)
        price_val = p.get("discounted_price") or p.get("price") or p.get("original_price")
        price = format_price(price_val, p.get("currency", "TL")) if price_val else "N/A"
        source = p.get("source", "")
        rating = p.get("rating")
        review_count = p.get("review_count")
        badge = _value_badge(p)
        trend = _price_trend_badge(p.get("price_history", []))

        header = f"**{i}. {name}**"
        if badge:
            header += f"  {badge}"
        if trend:
            header += f"  {trend}"
        lines.append(header)
        lines.append(f"   {price}  \u2014  {source}")

        if rating is not None:
            rating_str = f"\u2b50 {rating}"
            if review_count:
                rating_str += f" ({review_count} yorum)"
            lines.append(f"   {rating_str}")

        seller = p.get("seller_name")
        seller_r = p.get("seller_rating")
        if seller:
            seller_line = f"   Seller: {seller}"
            if seller_r is not None:
                seller_line += f" {_seller_badge(seller_r)}"
            lines.append(seller_line)

        shipping = p.get("shipping_cost")
        days = p.get("shipping_time_days")
        if shipping is not None or days is not None:
            ship_parts: list[str] = []
            if shipping == 0 or p.get("free_shipping"):
                ship_parts.append("Free Shipping")
            elif shipping is not None:
                ship_parts.append(f"{format_price(shipping)} kargo")
            if days is not None:
                ship_parts.append(f"{days} g\u00fcn")
            lines.append(f"   Kargo: {', '.join(ship_parts)}")

        installments = p.get("installment_options") or p.get("installment_info")
        if isinstance(installments, list) and installments:
            best = installments[0]
            lines.append(
                f"   Taksit: {best.get('months', '?')} ay \u00d7 "
                f"{format_price(best.get('monthly_amount', 0))}"
            )

        lines.append("")  # blank separator
    return "\n".join(lines).rstrip()


def _build_terminal_table(products: list[dict]) -> str:
    """Build a plain-text table suitable for terminal display."""
    if not products:
        return "No products to compare."

    # Column definitions: (header, width, key-extractor)
    def _price_cell(p: dict) -> str:
        pv = p.get("discounted_price") or p.get("price") or p.get("original_price")
        return format_price(pv, p.get("currency", "TL")) if pv else "N/A"

    def _rating_cell(p: dict) -> str:
        r = p.get("rating")
        if r is None:
            return "-"
        rc = p.get("review_count")
        return f"{r}" + (f" ({rc})" if rc else "")

    def _shipping_cell(p: dict) -> str:
        sc = p.get("shipping_cost")
        if sc == 0 or p.get("free_shipping"):
            return "Free"
        if sc is not None:
            return format_price(sc)
        return "-"

    cols = [
        ("Product", 40, lambda p: _truncate(p.get("name", ""), 38)),
        ("Price", 16, _price_cell),
        ("Source", 14, lambda p: _truncate(p.get("source", ""), 12)),
        ("Rating", 12, _rating_cell),
        ("Shipping", 14, _shipping_cell),
    ]

    header = " | ".join(h.ljust(w) for h, w, _ in cols)
    separator = "-+-".join("-" * w for _, w, _ in cols)
    rows: list[str] = [header, separator]
    for p in products:
        row = " | ".join(fn(p).ljust(w) for _, w, fn in cols)
        rows.append(row)
    return "\n".join(rows)


def _build_json_table(products: list[dict]) -> str:
    """Return a structured JSON string of the comparison."""
    items: list[dict] = []
    for p in products:
        price_val = p.get("discounted_price") or p.get("price") or p.get("original_price")
        items.append(
            {
                "name": p.get("name", ""),
                "price": price_val,
                "price_formatted": format_price(price_val, p.get("currency", "TL")) if price_val else None,
                "currency": p.get("currency", "TL"),
                "source": p.get("source", ""),
                "url": p.get("url", ""),
                "rating": p.get("rating"),
                "review_count": p.get("review_count"),
                "seller_name": p.get("seller_name"),
                "seller_rating": p.get("seller_rating"),
                "shipping_cost": p.get("shipping_cost"),
                "shipping_time_days": p.get("shipping_time_days"),
                "free_shipping": p.get("free_shipping", False),
                "value_badge": _value_badge(p),
                "price_trend": _price_trend_badge(p.get("price_history", [])),
            }
        )
    return json.dumps({"products": items, "count": len(items)}, ensure_ascii=False, indent=2)


def format_comparison_table(
    products: list[dict],
    format: str = "telegram",
) -> str:
    """Format a list of products into a comparison view.

    Parameters
    ----------
    products:
        Each dict mirrors :class:`Product` fields, passed as plain dicts
        so callers are not forced to construct dataclass instances.
    format:
        ``"telegram"`` — Telegram-safe markdown.
        ``"terminal"`` — Plain-text table for CLI.
        ``"json"``     — Structured JSON string.
    """
    builders = {
        "telegram": _build_telegram_table,
        "terminal": _build_terminal_table,
        "json": _build_json_table,
    }
    builder = builders.get(format, _build_telegram_table)
    return builder(products)
