from pathlib import Path
from src.shopping.scrapers.akakce import AkakceScraper
from src.shopping.scrapers.epey import EpeyScraper
from src.shopping.scrapers.kitapyurdu import KitapyurduScraper

FIX = Path(__file__).parent / "fixtures"


def test_akakce_search_populates_sku():
    html = (FIX / "akakce_search.html").read_text(encoding="utf-8")
    s = AkakceScraper()
    products = s._parse_search_html(html)
    skus = [p.sku for p in products if p.sku]
    assert len(skus) >= 1
    assert all(sku.startswith("ak-") for sku in skus)


def test_epey_search_populates_sku():
    html = (FIX / "epey_search.html").read_text(encoding="utf-8")
    s = EpeyScraper()
    products = s._parse_search_html(html)
    skus = [p.sku for p in products if p.sku]
    assert len(skus) >= 1
    assert all(sku.startswith("ep-") for sku in skus)


def test_kitapyurdu_search_populates_sku():
    html = (FIX / "kitapyurdu_search.html").read_text(encoding="utf-8")
    s = KitapyurduScraper()
    products = s._parse_search_html(html)
    skus = [p.sku for p in products if p.sku]
    assert len(skus) >= 1
    assert all(sku.startswith("ky-") for sku in skus)
