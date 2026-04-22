"""Batch 5 SKU tests: sahibinden, arabam, direnc_net scrapers."""

from pathlib import Path

from src.shopping.scrapers.sahibinden import SahibindenScraper
from src.shopping.scrapers.arabam import ArabamScraper
from src.shopping.scrapers.direnc_net import DirencNetScraper

FIX = Path(__file__).parent / "fixtures"


def test_sahibinden_search_populates_sku():
    html = (FIX / "sahibinden_search.html").read_text(encoding="utf-8")
    s = SahibindenScraper()
    products = s._parse_search_html(html)
    skus = [p.sku for p in products if p.sku]
    assert len(skus) >= 1 and all(sku.startswith("sh-") for sku in skus)


def test_sahibinden_sku_values():
    html = (FIX / "sahibinden_search.html").read_text(encoding="utf-8")
    s = SahibindenScraper()
    products = s._parse_search_html(html)
    sku_set = {p.sku for p in products if p.sku}
    assert "sh-987654321" in sku_set
    assert "sh-111222333" in sku_set


def test_arabam_search_populates_sku():
    html = (FIX / "arabam_search.html").read_text(encoding="utf-8")
    s = ArabamScraper()
    products = s._parse_search_html(html)
    skus = [p.sku for p in products if p.sku]
    assert len(skus) >= 1 and all(sku.startswith("ab-") for sku in skus)


def test_arabam_sku_values():
    html = (FIX / "arabam_search.html").read_text(encoding="utf-8")
    s = ArabamScraper()
    products = s._parse_search_html(html)
    sku_set = {p.sku for p in products if p.sku}
    assert "ab-55667788" in sku_set
    assert "ab-99887766" in sku_set


def test_direnc_net_search_populates_sku():
    html = (FIX / "direnc_net_search.html").read_text(encoding="utf-8")
    s = DirencNetScraper()
    products = s._parse_search_html(html)
    skus = [p.sku for p in products if p.sku]
    assert len(skus) >= 1 and all(sku.startswith("dn-") for sku in skus)


def test_direnc_net_sku_values():
    html = (FIX / "direnc_net_search.html").read_text(encoding="utf-8")
    s = DirencNetScraper()
    products = s._parse_search_html(html)
    sku_set = {p.sku for p in products if p.sku}
    assert "dn-12345" in sku_set
    assert "dn-67890" in sku_set
