"""Batch-1 scraper SKU extraction tests.

Verifies that trendyol, hepsiburada, and amazon_tr populate Product.sku
from search-result HTML without extra HTTP fetches.
"""

from pathlib import Path

import pytest

FIX = Path(__file__).parent / "fixtures"


def test_trendyol_search_populates_sku():
    from src.shopping.scrapers.trendyol import TrendyolScraper

    html = (FIX / "trendyol_search.html").read_text(encoding="utf-8")
    s = TrendyolScraper()
    products = s._parse_search_html(html, max_results=10)
    assert len(products) >= 1
    skus = [p.sku for p in products if p.sku]
    assert len(skus) >= 1, f"No SKUs extracted; products={products}"
    assert all(sku.startswith("ty-") for sku in skus), f"Unexpected SKU format: {skus}"
    # Verify digit portion matches URL pattern
    assert all(sku[3:].isdigit() for sku in skus), f"SKU digit part invalid: {skus}"


def test_hepsiburada_search_populates_sku():
    from src.shopping.scrapers.hepsiburada import HepsiburadaScraper

    html = (FIX / "hepsiburada_search.html").read_text(encoding="utf-8")
    s = HepsiburadaScraper()
    products = s._parse_search_html(html, max_results=10)
    assert len(products) >= 1
    skus = [p.sku for p in products if p.sku]
    assert len(skus) >= 1, f"No SKUs extracted; products={products}"
    assert all(
        sku.startswith(("HBV", "HBC", "HBCV")) for sku in skus
    ), f"Unexpected SKU format: {skus}"


def test_amazon_tr_search_populates_sku():
    from src.shopping.scrapers.amazon_tr import AmazonTrScraper

    html = (FIX / "amazon_tr_search.html").read_text(encoding="utf-8")
    s = AmazonTrScraper()
    products = s._parse_search_html(html, max_results=10)
    assert len(products) >= 1
    skus = [p.sku for p in products if p.sku]
    assert len(skus) >= 1, f"No SKUs extracted; products={products}"
    assert all(len(sku) == 10 for sku in skus), f"ASIN must be 10 chars: {skus}"
    assert all(sku.isalnum() for sku in skus), f"ASIN must be alphanumeric: {skus}"
