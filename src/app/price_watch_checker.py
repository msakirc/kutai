"""Price watch scheduler — re-scrape watched products and notify on drops."""

from __future__ import annotations

import asyncio
from urllib.parse import urlparse

from src.infra.logging_config import get_logger

logger = get_logger("app.price_watch_checker")


def _domain_from_source(source: str | None) -> str | None:
    """Normalise a source string (e.g. 'Trendyol', 'trendyol.com') to a
    registry-compatible domain key like 'trendyol'."""
    if not source:
        return None
    s = source.strip().lower()
    # Strip common suffixes
    for suffix in (".com", ".com.tr", ".tr"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    # Strip www.
    if s.startswith("www."):
        s = s[4:]
    return s or None


def _domain_from_url(url: str | None) -> str | None:
    """Extract the scraper domain key from a product URL."""
    if not url:
        return None
    try:
        host = urlparse(url).hostname or ""
        return _domain_from_source(host)
    except Exception:
        return None


def format_price_drop_message(
    product_name: str,
    old_price: float,
    new_price: float,
    source: str | None,
    url: str | None,
    target_price: float | None = None,
) -> str:
    """Build a Telegram notification message for a price drop."""
    drop = old_price - new_price
    pct = (drop / old_price * 100) if old_price > 0 else 0.0

    lines = [
        "\U0001f4b0 Fiyat D\u00fc\u015ft\u00fc!",
        "",
        f"*{product_name}*",
        f"  Eski fiyat: {old_price:,.0f} TL",
        f"  Yeni fiyat: {new_price:,.0f} TL",
        f"  D\u00fc\u015f\u00fc\u015f: -{drop:,.0f} TL (-{pct:.1f}%)",
    ]
    if target_price is not None:
        if new_price <= target_price:
            lines.append(f"  \u2705 Hedef fiyat ({target_price:,.0f} TL) alt\u0131na d\u00fc\u015ft\u00fc!")
        else:
            remaining = new_price - target_price
            lines.append(f"  \U0001f3af Hedefe {remaining:,.0f} TL kald\u0131")
    if source:
        lines.append(f"\n  Kaynak: {source}")
    if url:
        lines.append(f"  \U0001f517 [Link]({url})")

    return "\n".join(lines)


async def _try_get_price_by_url(url: str) -> tuple[float | None, str | None]:
    """Attempt to scrape a product page by URL and return (price, source)."""
    from src.shopping.scrapers import get_scraper

    domain = _domain_from_url(url)
    if not domain:
        return None, None

    scraper_cls = get_scraper(domain)
    if not scraper_cls:
        return None, None

    try:
        scraper = scraper_cls(domain)
        product = await scraper.get_product(url)
        if product:
            price = product.discounted_price or product.original_price
            return price, product.source
    except Exception as exc:
        logger.warning("scrape by URL failed", url=url, error=str(exc))

    return None, None


async def _try_get_price_by_search(
    product_name: str, source: str | None
) -> tuple[float | None, str | None, str | None]:
    """Search for a product by name on the source domain.

    Returns (price, source, url) or (None, None, None).
    """
    from src.shopping.scrapers import get_scraper, list_scrapers

    domain = _domain_from_source(source)

    # If we have a specific source, try that scraper first
    domains_to_try: list[str] = []
    if domain:
        domains_to_try.append(domain)
    else:
        # Fallback: try the major price-comparison scrapers
        for d in ("akakce", "trendyol", "hepsiburada"):
            domains_to_try.append(d)

    for d in domains_to_try:
        scraper_cls = get_scraper(d)
        if not scraper_cls:
            continue
        try:
            scraper = scraper_cls(d)
            results = await scraper.search(product_name, max_results=3)
            if results:
                best = results[0]
                price = best.discounted_price or best.original_price
                return price, best.source, best.url
        except Exception as exc:
            logger.warning(
                "search scrape failed",
                domain=d,
                product=product_name,
                error=str(exc),
            )

    return None, None, None


async def check_price_watches(telegram=None) -> dict:
    """Main entry point: check all active price watches.

    Returns a summary dict with counts.
    """
    from src.shopping.memory.price_watch import (
        get_all_active_watches,
        update_watch_price,
        trigger_watch,
        expire_old_watches,
    )

    # Expire stale watches first
    await expire_old_watches(days=90)

    watches = await get_all_active_watches()
    if not watches:
        logger.info("No active price watches to check")
        return {"checked": 0, "drops": 0, "errors": 0}

    logger.info("Checking %d active price watches", len(watches))

    checked = 0
    drops = 0
    errors = 0

    for watch in watches:
        watch_id = watch["id"]
        product_name = watch["product_name"]
        old_price = watch["current_price"]
        target_price = watch.get("target_price")
        source = watch.get("source")
        product_url = watch.get("product_url")

        try:
            new_price = None
            resolved_source = source
            resolved_url = product_url

            # Strategy 1: scrape by URL if available
            if product_url:
                new_price, resolved_source = await _try_get_price_by_url(
                    product_url
                )

            # Strategy 2: search by product name
            if new_price is None:
                new_price, resolved_source, found_url = (
                    await _try_get_price_by_search(product_name, source)
                )
                if found_url and not product_url:
                    resolved_url = found_url

            if new_price is None:
                logger.warning(
                    "Could not fetch price for watch #%d: %s",
                    watch_id,
                    product_name,
                )
                errors += 1
                continue

            checked += 1

            # Update the stored price
            await update_watch_price(
                watch_id, new_price, resolved_source or source or ""
            )

            # Detect significant price drop (> 1% or below target)
            is_drop = old_price > 0 and new_price < old_price * 0.99
            hit_target = (
                target_price is not None
                and new_price <= target_price
                and old_price > target_price
            )

            if is_drop or hit_target:
                drops += 1
                msg = format_price_drop_message(
                    product_name=product_name,
                    old_price=old_price,
                    new_price=new_price,
                    source=resolved_source or source,
                    url=resolved_url,
                    target_price=target_price,
                )
                await _send_notification(telegram, watch, msg)

                # If target was hit, mark as triggered
                if hit_target:
                    await trigger_watch(watch_id)
                    logger.info(
                        "Watch #%d triggered — target %s reached (now %s)",
                        watch_id,
                        target_price,
                        new_price,
                    )
            else:
                logger.debug(
                    "Watch #%d: %s — price %s -> %s (no drop)",
                    watch_id,
                    product_name,
                    old_price,
                    new_price,
                )

        except Exception as exc:
            errors += 1
            logger.error(
                "Error checking watch #%d: %s",
                watch_id,
                str(exc),
                exc_info=True,
            )

        # Small delay between watches to avoid hammering scrapers
        await asyncio.sleep(2)

    summary = {"checked": checked, "drops": drops, "errors": errors}
    logger.info("Price watch check complete: %s", summary)
    return summary


async def _send_notification(telegram, watch: dict, message: str) -> None:
    """Send a price drop notification via Telegram."""
    try:
        if telegram and hasattr(telegram, "app"):
            from src.app.config import TELEGRAM_ADMIN_CHAT_ID

            chat_id = watch.get("user_id") or TELEGRAM_ADMIN_CHAT_ID
            await telegram.app.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode="Markdown",
                disable_web_page_preview=True,
            )
            logger.info(
                "Price drop notification sent for watch #%d",
                watch["id"],
            )
        else:
            # Fallback: log the notification
            logger.warning(
                "Telegram not available — logging price drop:\n%s",
                message,
            )
    except Exception as exc:
        logger.error(
            "Failed to send price drop notification for watch #%d: %s",
            watch["id"],
            str(exc),
        )
