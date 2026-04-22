from pathlib import Path
from src.shopping.scrapers.dr_com_tr import DrComTrScraper
from src.shopping.scrapers.decathlon_tr import DecathlonTrScraper
from src.shopping.scrapers.home_improvement import KoctasScraper, IKEAScraper

FIX = Path(__file__).parent / "fixtures"


def test_dr_search_populates_sku():
    html = (FIX / "dr_com_tr_search.html").read_text(encoding="utf-8")
    s = DrComTrScraper()
    products = s._parse_search_html(html)
    skus = [p.sku for p in products if p.sku]
    assert len(skus) >= 1
    assert all(sku.startswith("dr-") for sku in skus)


def test_decathlon_search_populates_sku():
    html = (FIX / "decathlon_tr_search.html").read_text(encoding="utf-8")
    s = DecathlonTrScraper()
    products = s._parse_search_html(html)
    skus = [p.sku for p in products if p.sku]
    assert len(skus) >= 1
    assert all(sku.startswith("dc-") for sku in skus)


def test_koctas_search_populates_sku():
    html = (FIX / "koctas_search.html").read_text(encoding="utf-8")
    s = KoctasScraper()
    products = s._parse_search_html(html)
    skus = [p.sku for p in products if p.sku]
    assert len(skus) >= 1
    assert all(sku.startswith("kc-") for sku in skus)


def test_ikea_search_populates_sku():
    html = (FIX / "ikea_search.html").read_text(encoding="utf-8")
    s = IKEAScraper()
    products = s._parse_search_html(html)
    skus = [p.sku for p in products if p.sku]
    assert len(skus) >= 1
    assert all(sku.startswith("ik-") for sku in skus)
