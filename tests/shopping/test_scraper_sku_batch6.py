"""Batch 6 SKU audit: google_cse and forum scrapers.

google_cse: sku extracted from destination URL with gc- prefix (or None).
forums: sku is always None; category_path is set to mark community content.
"""
import json
from pathlib import Path

from src.shopping.scrapers.google_cse import GoogleCSEScraper
from src.shopping.scrapers.forums import TechnopatScraper, DonanimHaberScraper

FIX = Path(__file__).parent / "fixtures"


def test_google_cse_parses_without_error():
    raw = json.loads((FIX / "google_cse_search.json").read_text(encoding="utf-8"))
    s = GoogleCSEScraper()
    products = s._parse_search_response(raw)
    assert len(products) >= 1
    # If any sku is set it must carry the gc- prefix
    for p in products:
        if p.sku:
            assert p.sku.startswith("gc-"), f"Expected gc- prefix, got: {p.sku!r}"


def test_google_cse_trendyol_sku():
    """Trendyol -p-<id> pattern must be recognised."""
    raw = json.loads((FIX / "google_cse_search.json").read_text(encoding="utf-8"))
    s = GoogleCSEScraper()
    products = s._parse_search_response(raw)
    trendyol = [p for p in products if "trendyol" in p.url]
    assert trendyol, "Expected at least one trendyol result"
    assert trendyol[0].sku == "gc-123456789"


def test_google_cse_amazon_asin():
    """Amazon /dp/<ASIN> pattern must be recognised."""
    raw = json.loads((FIX / "google_cse_search.json").read_text(encoding="utf-8"))
    s = GoogleCSEScraper()
    products = s._parse_search_response(raw)
    amazon = [p for p in products if "amazon" in p.url]
    assert amazon, "Expected at least one Amazon result"
    assert amazon[0].sku == "gc-B0CMDJZ1TX"


def test_google_cse_hepsiburada_pm_sku():
    """Hepsiburada /pm-<id> pattern must be recognised."""
    raw = json.loads((FIX / "google_cse_search.json").read_text(encoding="utf-8"))
    s = GoogleCSEScraper()
    products = s._parse_search_response(raw)
    hb = [p for p in products if "hepsiburada" in p.url]
    assert hb, "Expected at least one hepsiburada result"
    assert hb[0].sku == "gc-123456789"


def test_technopat_leaves_sku_none_sets_category():
    html = (FIX / "technopat_search.html").read_text(encoding="utf-8")
    s = TechnopatScraper()
    products = s._parse_search_html(html)
    assert len(products) >= 1
    assert all(p.sku is None for p in products), "Forum products must have sku=None"
    assert all("Forum" in (p.category_path or "") for p in products), (
        f"Expected 'Forum' in category_path, got: {[p.category_path for p in products]}"
    )


def test_technopat_category_path_value():
    html = (FIX / "technopat_search.html").read_text(encoding="utf-8")
    s = TechnopatScraper()
    products = s._parse_search_html(html)
    assert all(p.category_path == "Forum > Technopat" for p in products)


def test_donanim_haber_leaves_sku_none_sets_category():
    html = (FIX / "donanim_haber_search.html").read_text(encoding="utf-8")
    s = DonanimHaberScraper()
    products = s._parse_search_html(html)
    assert len(products) >= 1
    assert all(p.sku is None for p in products), "Forum products must have sku=None"
    assert all("Forum" in (p.category_path or "") for p in products), (
        f"Expected 'Forum' in category_path, got: {[p.category_path for p in products]}"
    )


def test_donanim_haber_category_path_value():
    html = (FIX / "donanim_haber_search.html").read_text(encoding="utf-8")
    s = DonanimHaberScraper()
    products = s._parse_search_html(html)
    assert all(p.category_path == "Forum > Donanim Haber" for p in products)
