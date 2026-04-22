"""Batch 4 SKU tests: Migros and Getir scrapers (grocery.py, API-based)."""

import json
from pathlib import Path

from src.shopping.scrapers.grocery import GetirScraper, MigrosScraper

FIX = Path(__file__).parent / "fixtures"


def test_migros_search_populates_sku():
    raw = json.loads((FIX / "migros_search.json").read_text(encoding="utf-8"))
    s = MigrosScraper()
    products = s._parse_search_response(raw)
    skus = [p.sku for p in products if p.sku]
    assert len(skus) >= 1
    assert all(sku.startswith("mg-") for sku in skus)


def test_migros_sku_values():
    raw = json.loads((FIX / "migros_search.json").read_text(encoding="utf-8"))
    s = MigrosScraper()
    products = s._parse_search_response(raw)
    sku_map = {p.name: p.sku for p in products}
    assert sku_map.get("Pınar Süt 1 Lt") == "mg-12345"
    assert sku_map.get("Sek Tereyağ 500 G") == "mg-12346" or sku_map.get("Sek Tereyag 500 G") == "mg-12346" or any(
        v == "mg-12346" for v in sku_map.values()
    )


def test_getir_search_populates_sku():
    raw = json.loads((FIX / "getir_search.json").read_text(encoding="utf-8"))
    s = GetirScraper()
    products = s._parse_search_response(raw)
    skus = [p.sku for p in products if p.sku]
    assert len(skus) >= 1
    assert all(sku.startswith("gt-") for sku in skus)


def test_getir_sku_values():
    raw = json.loads((FIX / "getir_search.json").read_text(encoding="utf-8"))
    s = GetirScraper()
    products = s._parse_search_response(raw)
    skus = {p.sku for p in products if p.sku}
    assert "gt-gt001" in skus
    assert "gt-gt002" in skus
    assert "gt-gt003" in skus


def test_migros_bare_list():
    """_parse_search_response also accepts a bare list."""
    items = [
        {"id": "99", "name": "Test Ürün 200 ml", "shownPrice": 10.0, "inStock": True}
    ]
    s = MigrosScraper()
    products = s._parse_search_response(items)
    assert len(products) == 1
    assert products[0].sku == "mg-99"


def test_getir_bare_list():
    """_parse_search_response also accepts a bare list."""
    items = [
        {"id": "abc", "name": "Test Ürün 500 g", "price": 15.0, "inStock": True, "slug": "test-urun"}
    ]
    s = GetirScraper()
    products = s._parse_search_response(items)
    assert len(products) == 1
    assert products[0].sku == "gt-abc"
