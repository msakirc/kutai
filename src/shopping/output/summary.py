"""Recommendation summary formatter.

Produces a structured recommendation message with top pick, budget option,
alternatives, warnings, timing advice, and action buttons.
All functions are pure — no I/O side effects.
"""

from __future__ import annotations

from .formatters import format_price, _truncate


# ---------------------------------------------------------------------------
# Individual section formatters
# ---------------------------------------------------------------------------

def format_top_pick(product: dict) -> str:
    """Format the top-pick product with bold name, price, and reasoning."""
    if not product:
        return ""
    name = product.get("name", "")
    price_val = product.get("discounted_price") or product.get("price") or product.get("original_price")
    price = format_price(price_val, product.get("currency", "TL")) if price_val else ""
    reason = product.get("reason", "")
    source = product.get("source", "")

    lines = [f"\U0001f3c6 **{name}**"]
    if price:
        lines[0] += f"  —  **{price}**"
    if source:
        lines.append(f"   {source}")
    if reason:
        lines.append(f"   {reason}")
    return "\n".join(lines)


def format_budget_option(product: dict) -> str:
    """Format the budget-friendly alternative."""
    if not product:
        return ""
    name = product.get("name", "")
    price_val = product.get("discounted_price") or product.get("price") or product.get("original_price")
    price = format_price(price_val, product.get("currency", "TL")) if price_val else ""
    reason = product.get("reason", "")

    line = f"\U0001f4b0 **{name}**"
    if price:
        line += f"  —  {price}"
    lines = [line]
    if reason:
        lines.append(f"   {reason}")
    return "\n".join(lines)


def format_alternatives(products: list[dict]) -> str:
    """Format a brief list of alternatives with why they were not chosen."""
    if not products:
        return ""
    lines: list[str] = []
    for p in products:
        name = _truncate(p.get("name", ""), 45)
        price_val = p.get("discounted_price") or p.get("price") or p.get("original_price")
        price = format_price(price_val, p.get("currency", "TL")) if price_val else ""
        why_not = p.get("why_not", "")
        entry = f"\u2022 {name}"
        if price:
            entry += f" ({price})"
        if why_not:
            entry += f" \u2014 {why_not}"
        lines.append(entry)
    return "\n".join(lines)


def format_warnings(warnings: list[str]) -> str:
    """Format warnings from Sikayetvar, bad reviews, etc."""
    if not warnings:
        return ""
    lines = ["\u26a0\ufe0f **Warnings:**"]
    for w in warnings:
        lines.append(f"\u2022 {w}")
    return "\n".join(lines)


def format_timing_advice(timing: dict) -> str:
    """Format buy-now / wait / neutral timing guidance.

    Expected keys: ``action`` (``"buy"`` | ``"wait"`` | ``"neutral"``),
    ``reason``, ``until`` (optional date string).
    """
    if not timing:
        return ""
    action = timing.get("action", "neutral")
    reason = timing.get("reason", "")
    until = timing.get("until")

    icons = {"buy": "\u2705", "wait": "\u23f0", "neutral": "\U0001f914"}
    labels = {"buy": "Buy Now", "wait": "Wait", "neutral": "No Rush"}
    icon = icons.get(action, "\U0001f914")
    label = labels.get(action, "Neutral")

    line = f"{icon} **{label}**"
    if reason:
        line += f": {reason}"
    if until:
        line += f" (hedef: {until})"
    return line


def format_action_buttons(results: dict) -> str:
    """Return Telegram-style action-button text lines.

    These are rendered as text hints — actual inline buttons are handled
    by the product-card layer.
    """
    lines: list[str] = []
    top = results.get("top_pick", {})
    url = top.get("url")
    if url:
        lines.append(f"\U0001f6d2 Buy now: {url}")

    timing = results.get("timing", {})
    if timing.get("action") == "wait":
        reason = timing.get("reason", "")
        lines.append(f"\u23f0 Wait: {reason}")

    lines.append("\U0001f440 Watch price")
    lines.append("\U0001f50d Compare more")
    lines.append("\u2753 Ask me")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Confidence & complexity helpers
# ---------------------------------------------------------------------------

def _confidence_indicator(sources: int) -> str:
    """Describe confidence based on the number of data sources."""
    if sources >= 4:
        return f"\u2705 Found consistent data across {sources} sources"
    if sources >= 2:
        return f"\U0001f7e1 Based on {sources} sources"
    if sources == 1:
        return "\U0001f7e0 Limited data (1 source)"
    return "\u26a0\ufe0f No source data available"


def _adapt_to_complexity(results: dict) -> str:
    """Return either a compact or full summary depending on complexity.

    Simple queries (single product, high confidence, few alternatives)
    get a 3-line answer.  Complex queries get full sections.
    """
    alternatives = results.get("alternatives") or []
    warnings = results.get("warnings") or []
    confidence = results.get("confidence", 0)
    top = results.get("top_pick")

    is_simple = (
        len(alternatives) <= 1
        and len(warnings) == 0
        and confidence >= 0.8
        and top is not None
    )

    if is_simple:
        # Compact 3-line answer
        lines: list[str] = []
        lines.append(format_top_pick(top))
        timing = results.get("timing")
        if timing:
            lines.append(format_timing_advice(timing))
        url = (top or {}).get("url")
        if url:
            lines.append(f"\U0001f6d2 {url}")
        return "\n".join(lines)

    # Full sections — handled by the main formatter
    return ""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def format_recommendation_summary(
    results: dict,
    format: str = "telegram",
) -> str:
    """Build a full recommendation summary.

    Parameters
    ----------
    results:
        Expected keys: ``top_pick``, ``budget_option``, ``alternatives``,
        ``lateral_suggestions``, ``warnings``, ``timing``,
        ``where_to_buy``, ``confidence``, ``sources`` (int).
    format:
        ``"telegram"`` or ``"terminal"`` (currently both produce
        similar markdown; terminal omits emoji).
    """
    # Try compact form first
    compact = _adapt_to_complexity(results)
    if compact:
        return compact

    use_emoji = format != "terminal"
    sections: list[str] = []

    # Top pick
    top = results.get("top_pick")
    if top:
        sections.append(format_top_pick(top))

    # Budget option
    budget = results.get("budget_option")
    if budget:
        sections.append(format_budget_option(budget))

    # Alternatives
    alts = results.get("alternatives")
    if alts:
        header = "\U0001f504 **Alternatives:**" if use_emoji else "Alternatives:"
        sections.append(header + "\n" + format_alternatives(alts))

    # Lateral suggestions
    lateral = results.get("lateral_suggestions")
    if lateral:
        header = "\U0001f4a1 **Also consider:**" if use_emoji else "Also consider:"
        sections.append(header + "\n" + format_alternatives(lateral))

    # Warnings
    warns = results.get("warnings")
    if warns:
        sections.append(format_warnings(warns))

    # Timing
    timing = results.get("timing")
    if timing:
        sections.append(format_timing_advice(timing))

    # Where to buy
    where = results.get("where_to_buy")
    if where:
        header = "\U0001f6d2 **Where to buy:**" if use_emoji else "Where to buy:"
        where_lines = [header]
        for w in where:
            name = w.get("source", "")
            url = w.get("url", "")
            price_val = w.get("price")
            price = format_price(price_val, w.get("currency", "TL")) if price_val else ""
            entry = f"\u2022 {name}"
            if price:
                entry += f" — {price}"
            if url:
                entry += f"  {url}"
            where_lines.append(entry)
        sections.append("\n".join(where_lines))

    # Confidence
    sources = results.get("sources", 0)
    if sources:
        sections.append(_confidence_indicator(sources))

    # Action buttons
    if format == "telegram":
        sections.append(format_action_buttons(results))

    return "\n\n".join(sections)
