"""Price aggregator redirect resolver.

Akakçe, Cimri, and Epey show prices from multiple stores.  When you
click "go to store" they redirect through a tracking URL to the actual
product page.  This module extracts those destination URLs so that
store-specific scrapers can be invoked directly.

Research findings (2026-03-31):
- **Akakçe**: Direct store URLs are already embedded in the product page
  HTML as plain ``href`` / ``data-link`` attributes -- no redirect needed.
  The page contains attributes like:
    ``<a class="git" data-link="https%3A%2F%2Fwww.trendyol.com/..." ...>``
- **Epey**: Same pattern -- ``a.git[data-link]`` with URL-encoded store URLs.
- **Cimri**: Similar HTML embedding pattern.
"""

from __future__ import annotations

import urllib.parse
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("shopping.resilience.redirect_resolver")

# Domains that are final store destinations (no further resolution needed)
_STORE_DOMAINS = {
    "trendyol.com",
    "hepsiburada.com",
    "amazon.com.tr",
    "n11.com",
    "gittigidiyor.com",
    "mediamarkt.com.tr",
    "teknosa.com",
    "vatan.com.tr",
    "migros.com.tr",
    "a101.com.tr",
    "bim.com.tr",
    "decathlon.com.tr",
    "pttavm.com",
    "morhipo.com",
    "ciceksepeti.com",
    "gratis.com",
}

# Aggregator domains whose pages we can parse for embedded store URLs
_AGGREGATOR_DOMAINS = {"akakce.com", "epey.com", "cimri.com"}


def is_store_url(url: str) -> bool:
    """Return True if the URL is a known end-store (not an aggregator)."""
    try:
        host = urllib.parse.urlparse(url).netloc.lower().lstrip("www.")
        return any(host == d or host.endswith("." + d) for d in _STORE_DOMAINS)
    except Exception:
        return False


def is_aggregator_url(url: str) -> bool:
    """Return True if the URL points to a price aggregator."""
    try:
        host = urllib.parse.urlparse(url).netloc.lower().lstrip("www.")
        return any(host == d or host.endswith("." + d) for d in _AGGREGATOR_DOMAINS)
    except Exception:
        return False


async def extract_store_urls_from_aggregator(aggregator_url: str) -> list[dict[str, Any]]:
    """Fetch an aggregator product page and extract all embedded store URLs.

    Supports Akakçe and Epey (both embed store URLs directly in HTML).

    Args:
        aggregator_url: URL of a product page on akakce.com or epey.com.

    Returns:
        List of dicts: ``{url, store, price, currency}``.
        ``url`` is the fully decoded direct store URL.
    """
    if not is_aggregator_url(aggregator_url):
        logger.debug("not an aggregator URL, skipping", url=aggregator_url)
        return []

    try:
        from src.tools.scraper import scrape_url, ScrapeTier
        from bs4 import BeautifulSoup
    except ImportError as exc:
        logger.error("scraper or bs4 not available", error=str(exc))
        return []

    try:
        result = await scrape_url(aggregator_url, max_tier=ScrapeTier.TLS, timeout=12.0)
    except Exception as exc:
        logger.error("aggregator fetch failed", url=aggregator_url, error=str(exc))
        return []

    if not result.ok:
        logger.warning("aggregator page non-OK", url=aggregator_url, status=result.status)
        return []

    host = urllib.parse.urlparse(aggregator_url).netloc.lower()

    if "akakce.com" in host:
        return _extract_akakce_store_urls(result.html)
    elif "epey.com" in host:
        return _extract_epey_store_urls(result.html)
    elif "cimri.com" in host:
        return _extract_cimri_store_urls(result.html)
    else:
        logger.warning("unsupported aggregator", host=host)
        return []


def _extract_akakce_store_urls(html: str) -> list[dict[str, Any]]:
    """Extract store URLs from an Akakçe product page.

    Akakçe embeds store URLs in two places:
    1. **JSON-LD structured data** (primary): ``<script type="application/ld+json">``
       contains an AggregateOffer with individual ``Offer`` items, each with
       ``url`` and ``price``.
    2. **Raw HTML anchors** (fallback): direct ``href`` links to store product pages.

    Store entry JSON-LD example::

        {"@type": "Offer", "price": "47240.00", "url": "https://www.trendyol.com/..."}
    """
    import json as _json
    import re as _re

    try:
        from ..text_utils import parse_turkish_price
    except ImportError:
        parse_turkish_price = None  # type: ignore[assignment]

    stores: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    # --- Strategy 1: JSON-LD structured data ---
    try:
        ld_blocks = _re.findall(
            r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
            html,
            _re.DOTALL | _re.IGNORECASE,
        )
        for block in ld_blocks:
            try:
                data = _json.loads(block)
            except Exception:
                continue

            # Handle both single object and list
            objects = data if isinstance(data, list) else [data]
            for obj in objects:
                offers = obj.get("offers", {})
                if isinstance(offers, dict):
                    # AggregateOffer with nested Offer list
                    offer_list = offers.get("offers", [])
                    if not offer_list and offers.get("@type") == "Offer":
                        offer_list = [offers]
                elif isinstance(offers, list):
                    offer_list = offers
                else:
                    continue

                for offer in offer_list:
                    if not isinstance(offer, dict):
                        continue
                    offer_url = offer.get("url", "")
                    if not offer_url or not offer_url.startswith("http"):
                        continue
                    if "akakce.com" in offer_url:
                        continue
                    if offer_url in seen_urls:
                        continue
                    seen_urls.add(offer_url)

                    price: float | None = None
                    try:
                        price_raw = offer.get("price")
                        if price_raw is not None:
                            price = float(str(price_raw).replace(",", "."))
                    except (ValueError, TypeError):
                        pass

                    seller = offer.get("seller", {})
                    store_name = ""
                    if isinstance(seller, dict):
                        store_name = seller.get("name", "")
                    elif isinstance(seller, str):
                        store_name = seller

                    if not store_name:
                        store_name = _infer_store_from_url(offer_url)

                    stores.append({
                        "url": offer_url,
                        "store": store_name,
                        "price": price,
                        "currency": offer.get("priceCurrency", "TRY"),
                        "source": "akakce",
                    })

    except Exception as exc:
        logger.debug("akakce JSON-LD parse error", error=str(exc))

    # --- Strategy 2: raw HTML anchor scan (fallback) ---
    if not stores:
        try:
            from bs4 import BeautifulSoup
            try:
                soup = BeautifulSoup(html, "lxml")
            except Exception:
                soup = BeautifulSoup(html, "html.parser")

            for a in soup.find_all("a", href=True):
                href = a.get("href", "")
                data_link = a.get("data-link", "")
                raw = urllib.parse.unquote(data_link or href)
                if not raw.startswith("http"):
                    continue
                if "akakce.com" in raw:
                    continue
                if not is_store_url(raw):
                    continue
                if raw in seen_urls:
                    continue
                seen_urls.add(raw)

                stores.append({
                    "url": raw,
                    "store": _infer_store_from_url(raw),
                    "price": None,
                    "currency": "TRY",
                    "source": "akakce",
                })
        except Exception as exc:
            logger.debug("akakce anchor scan error", error=str(exc))

    logger.info("akakce store URLs extracted", count=len(stores))
    return stores


def _extract_epey_store_urls(html: str) -> list[dict[str, Any]]:
    """Extract store URLs from an Epey product page.

    Epey embeds store URLs in ``a.git[data-link]`` (URL-encoded) within the
    ``#fiyatlar`` section.  Each entry also has price and store logo.

    Store entry HTML example::

        <a class="git c<id>" data-id="<id>" data-link="https%3A%2F%2F..."
           data-pos="1" ...>
          <span class="site_logo"><img alt="StoreName fiyatı" ...></span>
          <span class="urun_fiyat">286.200,00 TL</span>
        </a>
    """
    try:
        from bs4 import BeautifulSoup
        from ..text_utils import parse_turkish_price
    except ImportError:
        return []

    stores: list[dict[str, Any]] = []

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    # Epey price entries are a.git elements with data-link
    git_links = soup.select("a.git[data-link]")

    seen_urls: set[str] = set()

    for a in git_links:
        try:
            encoded = a.get("data-link", "")
            if not encoded:
                continue

            store_url = urllib.parse.unquote(encoded)
            if not store_url.startswith("http"):
                continue

            if store_url in seen_urls:
                continue
            seen_urls.add(store_url)

            # Store name from logo image alt text
            logo_img = a.select_one(".site_logo img")
            store_name = ""
            if logo_img:
                alt = logo_img.get("alt", "")
                # Alt text format: "ProductName StoreName fiyatı"
                # Strip suffix
                if " fiyatı" in alt:
                    # Take word before " fiyatı"
                    parts = alt.replace(" fiyatı", "").split()
                    store_name = parts[-1] if parts else alt
                else:
                    store_name = alt

            # Price: prefer the hidden sort value (integer, e.g. 28620000 = 28620.00 TRY)
            # or fall back to parsing the visible text.
            price: float | None = None
            sort_el = a.select_one(".urun_fiyat_sort")
            if sort_el:
                try:
                    sort_val = sort_el.get_text(strip=True).replace(".", "").replace(",", "")
                    price = float(sort_val) / 100.0
                except (ValueError, TypeError):
                    pass
            if price is None:
                price_el = a.select_one(".urun_fiyat")
                if price_el:
                    try:
                        from ..text_utils import parse_turkish_price
                        # Use only the first text node, not the nested kupon span
                        first_text = next(
                            (t.strip() for t in price_el.strings if t.strip() and "TL" in t),
                            "",
                        )
                        price = parse_turkish_price(first_text)
                    except Exception:
                        pass

            stores.append({
                "url": store_url,
                "store": store_name,
                "price": price,
                "currency": "TRY",
                "source": "epey",
            })

        except Exception as exc:
            logger.debug("epey link parse error", error=str(exc))
            continue

    logger.info("epey store URLs extracted", count=len(stores))
    return stores


def _extract_cimri_store_urls(html: str) -> list[dict[str, Any]]:
    """Extract store URLs from a Cimri product page.

    Cimri uses a similar pattern to Akakçe/Epey.  This implementation
    falls back to scanning all external anchor hrefs since Cimri's exact
    HTML structure may vary.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    stores: list[dict[str, Any]] = []

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    seen_urls: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        data_link = a.get("data-link", "")
        raw = urllib.parse.unquote(data_link or href)

        if not raw.startswith("http"):
            continue
        if "cimri.com" in raw:
            continue
        if not is_store_url(raw):
            continue
        if raw in seen_urls:
            continue

        seen_urls.add(raw)

        img = a.find("img", alt=True)
        store_name = img.get("alt", "") if img else a.get_text(strip=True)[:40]

        stores.append({
            "url": raw,
            "store": store_name,
            "price": None,
            "currency": "TRY",
            "source": "cimri",
        })

    logger.info("cimri store URLs extracted", count=len(stores))
    return stores


async def find_product_across_stores(
    product_name: str,
    aggregator_url: str = "",
) -> dict[str, dict[str, Any]]:
    """Find a product across multiple stores using aggregator redirects.

    Strategy:
    1. If ``aggregator_url`` provided, fetch the aggregator page and extract
       embedded store URLs (Akakçe / Epey).
    2. For each store URL found, identify the scraper and optionally fetch
       live price data.

    Args:
        product_name: Human-readable product name (used for labelling).
        aggregator_url: Optional URL of a product page on akakce.com or epey.com.

    Returns:
        Dict mapping store name → ``{url, price, currency, source}``.
    """
    result: dict[str, dict[str, Any]] = {}

    if aggregator_url and is_aggregator_url(aggregator_url):
        store_entries = await extract_store_urls_from_aggregator(aggregator_url)
        for entry in store_entries:
            store = entry.get("store") or _infer_store_from_url(entry["url"])
            if store not in result:
                result[store] = {
                    "url": entry["url"],
                    "price": entry.get("price"),
                    "currency": entry.get("currency", "TRY"),
                    "source": entry.get("source", "aggregator"),
                }

    logger.info(
        "find_product_across_stores done",
        product=product_name,
        store_count=len(result),
    )
    return result


def _infer_store_from_url(url: str) -> str:
    """Infer a human-readable store name from a URL."""
    try:
        host = urllib.parse.urlparse(url).netloc.lower().lstrip("www.")
        # Strip TLD
        parts = host.split(".")
        return parts[0].capitalize() if parts else host
    except Exception:
        return url[:30]
