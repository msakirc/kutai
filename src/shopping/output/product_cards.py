"""Visual product cards for Telegram.

Each ``format_*`` function returns a dict with ``text`` (str) and
``reply_markup`` (list of button rows) ready to be passed to
Telegram's ``sendMessage`` API.  All functions are pure — no I/O.
"""

from __future__ import annotations

from .formatters import format_price, _truncate


# ---------------------------------------------------------------------------
# Inline-button helper
# ---------------------------------------------------------------------------

def _inline_button(
    text: str,
    url: str | None = None,
    callback_data: str | None = None,
) -> dict:
    """Build a single Telegram InlineKeyboardButton dict."""
    btn: dict = {"text": text}
    if url:
        btn["url"] = url
    elif callback_data:
        btn["callback_data"] = callback_data
    return btn


# ---------------------------------------------------------------------------
# Rating / discount helpers
# ---------------------------------------------------------------------------

def _format_rating_stars(rating: float) -> str:
    """Return a visual star string like ``'\u2605\u2605\u2605\u2605\u2606 4.2'``."""
    if rating is None:
        return ""
    clamped = max(0.0, min(5.0, rating))
    full = int(clamped)
    empty = 5 - full
    stars = "\u2605" * full + "\u2606" * empty
    return f"{stars} {rating}"


def _format_discount(original: float, current: float) -> str:
    """Format a discount line: ``'~3.999 TL~ \u2192 **2.799 TL** (-30%)'``.

    Uses Telegram markdown: ``~text~`` for strikethrough, ``**text**``
    for bold.
    """
    if original is None or current is None or original <= 0:
        return format_price(current) if current else ""
    if current >= original:
        return f"**{format_price(current)}**"
    pct = round((1 - current / original) * 100)
    return f"~{format_price(original)}~ \u2192 **{format_price(current)}** (-{pct}%)"


# ---------------------------------------------------------------------------
# Product card
# ---------------------------------------------------------------------------

def format_product_card(product: dict) -> dict:
    """Build a Telegram product card.

    Returns ``{"text": str, "reply_markup": [[button, ...], ...]}``.
    """
    name = product.get("name", "Unknown Product")
    original = product.get("original_price")
    discounted = product.get("discounted_price")
    current_price = discounted or product.get("price") or original
    currency = product.get("currency", "TL")
    rating = product.get("rating")
    review_count = product.get("review_count")
    source = product.get("source", "")
    seller = product.get("seller_name", "")
    url = product.get("url", "")
    review_summary = product.get("review_summary", "")

    # --- text ---
    lines: list[str] = [f"**{_truncate(name, 60)}**"]

    # Price line
    if original and discounted and discounted < original:
        lines.append(_format_discount(original, discounted))
    elif current_price is not None:
        lines.append(f"**{format_price(current_price, currency)}**")

    # Rating
    if rating is not None:
        r_line = _format_rating_stars(rating)
        if review_count:
            r_line += f" ({review_count} yorum)"
        lines.append(r_line)

    # Source / seller
    src_line = source
    if seller and seller != source:
        src_line += f" \u2014 {seller}"
    if src_line:
        lines.append(src_line)

    # One-line review summary
    if review_summary:
        lines.append(f"\U0001f4ac {_truncate(review_summary, 80)}")

    text = "\n".join(lines)

    # --- buttons ---
    row1: list[dict] = []
    if url:
        row1.append(_inline_button("\U0001f517 Open Link", url=url))
    row1.append(
        _inline_button(
            "\U0001f440 Watch Price",
            callback_data=f"watch:{product.get('id', '')}",
        )
    )
    row2 = [
        _inline_button(
            "\U0001f4ca Compare",
            callback_data=f"compare:{product.get('id', '')}",
        )
    ]

    reply_markup: list[list[dict]] = [row1, row2]

    return {"text": text, "reply_markup": reply_markup}


# ---------------------------------------------------------------------------
# Batch cards
# ---------------------------------------------------------------------------

def format_product_cards_batch(products: list[dict]) -> list[dict]:
    """Return a list of product cards for multiple products."""
    return [format_product_card(p) for p in products]


# ---------------------------------------------------------------------------
# Deal card (highlighted)
# ---------------------------------------------------------------------------

def format_deal_card(product: dict, discount_pct: float) -> dict:
    """Build a highlighted deal card.

    Same structure as :func:`format_product_card` but with a deal
    banner and prominent discount display.
    """
    name = product.get("name", "Unknown Product")
    original = product.get("original_price")
    discounted = product.get("discounted_price")
    current_price = discounted or product.get("price") or original
    currency = product.get("currency", "TL")
    url = product.get("url", "")
    source = product.get("source", "")

    lines: list[str] = [
        f"\U0001f525\U0001f525 **DEAL — %{round(discount_pct)} OFF** \U0001f525\U0001f525",
        f"**{_truncate(name, 55)}**",
    ]

    if original and current_price and current_price < original:
        lines.append(_format_discount(original, current_price))
    elif current_price is not None:
        lines.append(f"**{format_price(current_price, currency)}**")

    if source:
        lines.append(source)

    text = "\n".join(lines)

    row1: list[dict] = []
    if url:
        row1.append(_inline_button("\U0001f6d2 Buy Now", url=url))
    row1.append(
        _inline_button(
            "\U0001f440 Watch Price",
            callback_data=f"watch:{product.get('id', '')}",
        )
    )

    return {"text": text, "reply_markup": [row1]}


# ---------------------------------------------------------------------------
# Combo card (multi-product bundle)
# ---------------------------------------------------------------------------

def format_combo_card(combo: dict) -> dict:
    """Build a combo card for a multi-product bundle.

    Expected keys: ``products`` (list[dict]), ``total_price`` (float),
    ``compatibility_notes`` (list[str]), ``value_score`` (float).
    """
    products = combo.get("products", [])
    total = combo.get("total_price", 0)
    notes = combo.get("compatibility_notes", [])
    currency = combo.get("currency", "TL")

    lines: list[str] = ["\U0001f4e6 **Combo Deal**", ""]

    for i, p in enumerate(products, 1):
        name = _truncate(p.get("name", ""), 45)
        price_val = (
            p.get("discounted_price")
            or p.get("price")
            or p.get("original_price")
        )
        price_str = format_price(price_val, p.get("currency", currency)) if price_val else ""
        lines.append(f"{i}. {name}  —  {price_str}")

    lines.append("")
    lines.append(f"**Total: {format_price(total, currency)}**")

    if notes:
        lines.append("")
        for note in notes:
            lines.append(f"\u2714\ufe0f {note}")

    text = "\n".join(lines)

    buttons: list[dict] = [
        _inline_button(
            "\U0001f4ca Compare Individually",
            callback_data=f"combo_compare:{combo.get('id', '')}",
        ),
    ]
    # Add buy links for each product
    for p in products:
        url = p.get("url")
        if url:
            short_name = _truncate(p.get("name", ""), 20)
            buttons.append(_inline_button(f"\U0001f517 {short_name}", url=url))

    # Arrange buttons in rows of 2
    rows: list[list[dict]] = []
    for j in range(0, len(buttons), 2):
        rows.append(buttons[j : j + 2])

    return {"text": text, "reply_markup": rows}
