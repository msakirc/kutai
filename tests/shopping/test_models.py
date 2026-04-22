from src.shopping.models import Product


def test_product_has_sku_field_default_none():
    p = Product(name="x", url="u", source="trendyol")
    assert p.sku is None


def test_product_sku_accepts_string():
    p = Product(name="x", url="u", source="trendyol", sku="HBCV00004X9ZCH")
    assert p.sku == "HBCV00004X9ZCH"
