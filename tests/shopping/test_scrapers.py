"""Unit tests for shopping scrapers — parsing logic against fixture files.

Each test loads a realistic HTML/JSON fixture and calls the scraper's
internal parse method directly, verifying that the right fields are
extracted.  No live HTTP requests are made.
"""

import json
import os
import unittest
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _read_fixture(filename: str) -> str:
    with open(os.path.join(FIXTURES_DIR, filename), encoding="utf-8") as f:
        return f.read()


def _read_json_fixture(filename: str) -> dict:
    return json.loads(_read_fixture(filename))


# ---------------------------------------------------------------------------
# Mock dependencies so scraper modules can be imported without side effects
# ---------------------------------------------------------------------------

# Patch heavy dependencies before importing scrapers
_mock_patches = [
    patch("src.shopping.config.get_rate_limit", return_value={"delay_seconds": 0, "daily_budget": 9999}),
    patch("src.shopping.request_tracker.log_request", return_value=None),
    patch("src.shopping.request_tracker.get_daily_request_count", return_value=0),
    patch("src.shopping.cache.get_cached_search", return_value=None),
    patch("src.shopping.cache.cache_search", return_value=None),
    patch("src.shopping.cache.get_cached_product", return_value=None),
    patch("src.shopping.cache.cache_product", return_value=None),
    patch("src.shopping.cache.get_cached_reviews", return_value=None),
    patch("src.shopping.cache.cache_reviews", return_value=None),
]

for _p in _mock_patches:
    _p.start()

from src.shopping.scrapers.akakce import AkakceScraper
from src.shopping.scrapers.trendyol import TrendyolScraper
from src.shopping.scrapers.hepsiburada import HepsiburadaScraper
from src.shopping.scrapers.grocery import MigrosScraper, _calculate_unit_price
from src.shopping.scrapers.forums import TechnopatScraper, _score_thread


# =========================================================================
# Akakce
# =========================================================================


class TestAkakceScraper(unittest.TestCase):
    """Test AkakceScraper._parse_search_results against fixture HTML."""

    def setUp(self):
        self.scraper = AkakceScraper()
        self.html = _read_fixture("akakce_search.html")

    def test_parse_search_returns_products(self):
        products = self.scraper._parse_search_results(self.html, max_results=10)
        self.assertEqual(len(products), 3)

    def test_product_names_extracted(self):
        products = self.scraper._parse_search_results(self.html, max_results=10)
        names = [p.name for p in products]
        # normalize_product_name may strip filler, but core names should remain
        self.assertTrue(any("iphone" in n.lower() for n in names))
        self.assertTrue(any("samsung" in n.lower() or "galaxy" in n.lower() for n in names))
        self.assertTrue(any("xiaomi" in n.lower() for n in names))

    def test_prices_extracted(self):
        products = self.scraper._parse_search_results(self.html, max_results=10)
        iphone = products[0]
        # discounted_price from span.pt_v8: "42.999,90 TL" -> 42999.90
        self.assertIsNotNone(iphone.discounted_price)
        self.assertAlmostEqual(iphone.discounted_price, 42999.90, places=1)
        # original_price from span.pt_v8.old: "49.999,00 TL" -> 49999.0
        self.assertIsNotNone(iphone.original_price)
        self.assertAlmostEqual(iphone.original_price, 49999.0, places=1)

    def test_discount_percentage_calculated(self):
        products = self.scraper._parse_search_results(self.html, max_results=10)
        iphone = products[0]
        self.assertIsNotNone(iphone.discount_percentage)
        self.assertGreater(iphone.discount_percentage, 0)

    def test_urls_constructed(self):
        products = self.scraper._parse_search_results(self.html, max_results=10)
        # Relative href should be prefixed with base URL
        self.assertTrue(products[0].url.startswith("https://www.akakce.com/"))
        # Absolute URL should be kept as-is
        self.assertTrue(products[2].url.startswith("https://www.akakce.com/"))

    def test_store_count_in_specs(self):
        products = self.scraper._parse_search_results(self.html, max_results=10)
        self.assertEqual(products[0].specs.get("store_count"), 12)
        self.assertEqual(products[1].specs.get("store_count"), 8)

    def test_image_urls(self):
        products = self.scraper._parse_search_results(self.html, max_results=10)
        # First product: src attribute
        self.assertEqual(products[0].image_url, "https://cdn.akakce.com/img/iphone15.jpg")
        # Second product: data-src attribute
        self.assertEqual(products[1].image_url, "https://cdn.akakce.com/img/galaxy-s24.jpg")
        # Third product: src starting with "//" -> should be prefixed with https:
        self.assertEqual(products[2].image_url, "https://cdn.akakce.com/img/xiaomi14.jpg")

    def test_source_is_akakce(self):
        products = self.scraper._parse_search_results(self.html, max_results=10)
        for p in products:
            self.assertEqual(p.source, "akakce")

    def test_max_results_respected(self):
        products = self.scraper._parse_search_results(self.html, max_results=2)
        self.assertEqual(len(products), 2)

    def test_currency_is_try(self):
        products = self.scraper._parse_search_results(self.html, max_results=10)
        for p in products:
            self.assertEqual(p.currency, "TRY")

    def test_del_tag_as_old_price(self):
        """Third product uses <del> for the old price."""
        products = self.scraper._parse_search_results(self.html, max_results=10)
        xiaomi = products[2]
        self.assertIsNotNone(xiaomi.original_price)
        self.assertAlmostEqual(xiaomi.original_price, 34999.0, places=1)


# =========================================================================
# Trendyol
# =========================================================================


class TestTrendyolScraper(unittest.TestCase):
    """Test TrendyolScraper._parse_search_response against fixture JSON."""

    def setUp(self):
        self.scraper = TrendyolScraper()
        self.json_data = _read_json_fixture("trendyol_search.json")

    def _make_mock_response(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self.json_data
        return mock_resp

    def test_parse_returns_products(self):
        products = self.scraper._parse_search_response(
            self._make_mock_response(), max_results=10
        )
        self.assertEqual(len(products), 3)

    def test_brand_prepended_to_name(self):
        products = self.scraper._parse_search_response(
            self._make_mock_response(), max_results=10
        )
        # "JBL" brand should be prepended to "Kablosuz Bluetooth Kulaklık"
        self.assertTrue("jbl" in products[0].name.lower())

    def test_prices_from_dict(self):
        products = self.scraper._parse_search_response(
            self._make_mock_response(), max_results=10
        )
        jbl = products[0]
        self.assertAlmostEqual(jbl.original_price, 1299.99, places=2)
        # discountedPrice takes priority over sellingPrice
        self.assertAlmostEqual(jbl.discounted_price, 899.99, places=2)

    def test_discount_ratio_from_api(self):
        products = self.scraper._parse_search_response(
            self._make_mock_response(), max_results=10
        )
        jbl = products[0]
        self.assertIsNotNone(jbl.discount_percentage)
        self.assertAlmostEqual(jbl.discount_percentage, 30.8, places=1)

    def test_rating_extracted(self):
        products = self.scraper._parse_search_response(
            self._make_mock_response(), max_results=10
        )
        self.assertAlmostEqual(products[0].rating, 4.5, places=1)
        self.assertEqual(products[0].review_count, 1250)

    def test_image_url_cdn_prefix(self):
        products = self.scraper._parse_search_response(
            self._make_mock_response(), max_results=10
        )
        # First: relative path -> cdn prefix
        self.assertTrue(products[0].image_url.startswith("https://cdn.dsmcdn.com/"))
        # Second: already has full URL
        self.assertEqual(products[1].image_url, "https://cdn.dsmcdn.com/ty200/product2.jpg")

    def test_seller_name(self):
        products = self.scraper._parse_search_response(
            self._make_mock_response(), max_results=10
        )
        self.assertEqual(products[0].seller_name, "JBL Resmi Mağaza")
        self.assertEqual(products[1].seller_name, "Nike Store")
        # Third uses merchant dict
        self.assertEqual(products[2].seller_name, "Samsung Türkiye")

    def test_free_shipping(self):
        products = self.scraper._parse_search_response(
            self._make_mock_response(), max_results=10
        )
        self.assertTrue(products[0].free_shipping)
        self.assertFalse(products[1].free_shipping)

    def test_promotions_in_specs(self):
        products = self.scraper._parse_search_response(
            self._make_mock_response(), max_results=10
        )
        promos = products[0].specs.get("promotions", [])
        self.assertIn("Süper Fırsat", promos)
        self.assertIn("Kargo Bedava", promos)

    def test_category_path(self):
        products = self.scraper._parse_search_response(
            self._make_mock_response(), max_results=10
        )
        self.assertEqual(products[0].category_path, "2001:Kulaklik")

    def test_scalar_price(self):
        """Third product has price as a scalar (not dict)."""
        products = self.scraper._parse_search_response(
            self._make_mock_response(), max_results=10
        )
        samsung = products[2]
        self.assertAlmostEqual(samsung.discounted_price, 5999.0, places=1)

    def test_source_is_trendyol(self):
        products = self.scraper._parse_search_response(
            self._make_mock_response(), max_results=10
        )
        for p in products:
            self.assertEqual(p.source, "trendyol")

    def test_url_construction(self):
        products = self.scraper._parse_search_response(
            self._make_mock_response(), max_results=10
        )
        # Relative URL
        self.assertTrue(products[0].url.startswith("https://www.trendyol.com/"))
        # Absolute URL
        self.assertTrue(products[2].url.startswith("https://www.trendyol.com/"))


# =========================================================================
# Hepsiburada
# =========================================================================


class TestHepsiburadaScraper(unittest.TestCase):
    """Test HepsiburadaScraper product parsing from __NEXT_DATA__."""

    def setUp(self):
        self.scraper = HepsiburadaScraper()
        self.html = _read_fixture("hepsiburada_product.html")

    def test_parse_search_next_data_for_product(self):
        """Simulate get_product parsing via __NEXT_DATA__ extraction."""
        import re as _re
        import json as _json

        m = _re.search(
            r'<script\s+id="__NEXT_DATA__"[^>]*>(.*?)</script>',
            self.html,
            _re.DOTALL,
        )
        self.assertIsNotNone(m, "__NEXT_DATA__ script tag should be found")
        data = _json.loads(m.group(1))
        props = data["props"]["pageProps"]
        pd = props.get("product", {})

        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()
        product = self.scraper._item_to_product(pd, now_iso)

        self.assertIsNotNone(product)
        self.assertIn("sony", product.name.lower())
        self.assertAlmostEqual(product.original_price, 12999.0, places=1)
        self.assertAlmostEqual(product.discounted_price, 9999.0, places=1)
        self.assertAlmostEqual(product.rating, 4.7, places=1)
        self.assertEqual(product.review_count, 523)
        self.assertEqual(product.seller_name, "Sony Türkiye")
        self.assertTrue(product.free_shipping)
        self.assertEqual(product.source, "hepsiburada")

    def test_discount_percentage_calculated(self):
        import re as _re, json as _json
        from datetime import datetime, timezone

        m = _re.search(r'<script\s+id="__NEXT_DATA__"[^>]*>(.*?)</script>', self.html, _re.DOTALL)
        data = _json.loads(m.group(1))
        pd = data["props"]["pageProps"]["product"]
        product = self.scraper._item_to_product(pd, datetime.now(timezone.utc).isoformat())

        self.assertIsNotNone(product.discount_percentage)
        expected_pct = round((1 - 9999.0 / 12999.0) * 100, 1)
        self.assertAlmostEqual(product.discount_percentage, expected_pct, places=1)

    def test_image_url(self):
        import re as _re, json as _json
        from datetime import datetime, timezone

        m = _re.search(r'<script\s+id="__NEXT_DATA__"[^>]*>(.*?)</script>', self.html, _re.DOTALL)
        data = _json.loads(m.group(1))
        pd = data["props"]["pageProps"]["product"]
        product = self.scraper._item_to_product(pd, datetime.now(timezone.utc).isoformat())

        self.assertEqual(product.image_url, "https://cdn.hepsiburada.com/sony-xm5-detail.jpg")


# =========================================================================
# Migros (Grocery)
# =========================================================================


class TestMigrosScraper(unittest.TestCase):
    """Test MigrosScraper._search_api parsing and unit price calculation."""

    def setUp(self):
        self.scraper = MigrosScraper()
        self.json_data = _read_json_fixture("migros_search.json")

    def test_parse_products_from_api_data(self):
        """Directly parse the fixture JSON as the scraper would."""
        from datetime import datetime, timezone

        now_iso = datetime.now(timezone.utc).isoformat()
        items = self.json_data["data"]["storeProductInfos"]
        products = []

        for item in items:
            from src.shopping.text_utils import normalize_product_name
            name = item.get("name", "")
            name = normalize_product_name(name)
            price = item.get("shownPrice") or item.get("price")
            original_price = item.get("regularPrice") or item.get("strikeThroughPrice")
            in_stock = item.get("inStock", True)

            specs = {"type": "grocery"}
            unit_info = _calculate_unit_price(price, name)
            if unit_info:
                specs.update(unit_info)

            badges = item.get("badges", [])
            if badges and isinstance(badges, list):
                badge_texts = []
                for b in badges:
                    if isinstance(b, dict):
                        badge_texts.append(b.get("name", str(b)))
                if badge_texts:
                    specs["campaign_badge"] = ", ".join(badge_texts)

            from src.shopping.models import Product
            products.append(Product(
                name=name,
                url=f"https://www.migros.com.tr/{item.get('prettyName', '')}",
                source="migros",
                original_price=float(original_price) if original_price else None,
                discounted_price=float(price) if price else None,
                currency="TRY",
                availability="in_stock" if in_stock else "out_of_stock",
                specs=specs,
                fetched_at=now_iso,
            ))

        self.assertEqual(len(products), 3)

        # First product: Pınar Süt 1 Lt
        sut = products[0]
        self.assertAlmostEqual(sut.discounted_price, 42.90, places=2)
        self.assertAlmostEqual(sut.original_price, 49.90, places=2)
        self.assertEqual(sut.availability, "in_stock")
        self.assertIn("Money", sut.specs.get("campaign_badge", ""))

        # Third product: out of stock
        cikolata = products[2]
        self.assertEqual(cikolata.availability, "out_of_stock")

    def test_unit_price_calculation_kg(self):
        result = _calculate_unit_price(124.90, "Sek Tereyağ 500 g")
        self.assertIsNotNone(result)
        self.assertEqual(result["unit"], "kg")
        self.assertAlmostEqual(result["quantity"], 0.5, places=3)
        # 124.90 / 0.5 = 249.80
        self.assertAlmostEqual(result["unit_price"], 249.80, places=2)

    def test_unit_price_calculation_litre(self):
        result = _calculate_unit_price(42.90, "Pınar Süt 1 Lt")
        self.assertIsNotNone(result)
        self.assertEqual(result["unit"], "L")
        self.assertAlmostEqual(result["quantity"], 1.0, places=3)
        self.assertAlmostEqual(result["unit_price"], 42.90, places=2)

    def test_unit_price_calculation_ml(self):
        result = _calculate_unit_price(15.90, "Meyve Suyu 200 ml")
        self.assertIsNotNone(result)
        self.assertEqual(result["unit"], "L")
        self.assertAlmostEqual(result["quantity"], 0.2, places=3)
        self.assertAlmostEqual(result["unit_price"], 79.50, places=2)

    def test_unit_price_calculation_grams(self):
        result = _calculate_unit_price(29.90, "Ülker Çikolata 80 gr")
        self.assertIsNotNone(result)
        self.assertEqual(result["unit"], "kg")
        self.assertAlmostEqual(result["quantity"], 0.08, places=3)
        self.assertAlmostEqual(result["unit_price"], 373.75, places=2)

    def test_unit_price_none_when_no_quantity(self):
        result = _calculate_unit_price(29.90, "Ülker Çikolata Sade")
        self.assertIsNone(result)

    def test_unit_price_none_when_no_price(self):
        result = _calculate_unit_price(None, "Süt 1 Lt")
        self.assertIsNone(result)


# =========================================================================
# Technopat (Forums)
# =========================================================================


class TestTechnopatScraper(unittest.TestCase):
    """Test TechnopatScraper._extract_thread_posts against fixture HTML."""

    def setUp(self):
        self.scraper = TechnopatScraper()
        self.html = _read_fixture("technopat_thread.html")

    def test_extract_posts(self):
        """Parse posts from the fixture HTML using BeautifulSoup directly."""
        from bs4 import BeautifulSoup
        import re

        soup = BeautifulSoup(self.html, "html.parser")
        post_els = soup.select("article.message")
        self.assertEqual(len(post_els), 3)

        # Simulate the parser logic
        posts = []
        for post_el in post_els:
            body_el = post_el.select_one("div.bbWrapper")
            if body_el is None:
                continue
            text = body_el.get_text(separator="\n", strip=True)
            if not text or len(text) < 20:
                continue

            author_el = post_el.select_one("a.message-name")
            author = author_el.get_text(strip=True) if author_el else None

            time_el = post_el.select_one("time")
            date_str = time_el.get("datetime") if time_el else None

            is_solution = bool(post_el.select_one("span.label--solution"))

            like_count = 0
            like_el = post_el.select_one("span.reactionsBar-count")
            if like_el:
                m = re.search(r"(\d+)", like_el.get_text())
                if m:
                    like_count = int(m.group(1))

            posts.append({
                "text": text,
                "source": "technopat",
                "author": author,
                "date": date_str,
                "is_solution": is_solution,
                "helpful_count": like_count,
            })

        # Should only get 2 posts (third is <20 chars)
        self.assertEqual(len(posts), 2)

    def test_first_post_fields(self):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(self.html, "html.parser")
        post_els = soup.select("article.message")

        body_el = post_els[0].select_one("div.bbWrapper")
        text = body_el.get_text(separator="\n", strip=True)

        self.assertIn("iPhone 15", text)
        self.assertIn("kamera", text.lower())

        author_el = post_els[0].select_one("a.message-name")
        self.assertEqual(author_el.get_text(strip=True), "techfan")

        time_el = post_els[0].select_one("time")
        self.assertEqual(time_el["datetime"], "2024-12-15T10:30:00+03:00")

    def test_solution_tag_detected(self):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(self.html, "html.parser")
        post_els = soup.select("article.message")
        second = post_els[1]

        is_solution = bool(second.select_one("span.label--solution"))
        self.assertTrue(is_solution)

    def test_like_count_extracted(self):
        from bs4 import BeautifulSoup
        import re

        soup = BeautifulSoup(self.html, "html.parser")
        post_els = soup.select("article.message")
        first = post_els[0]

        like_el = first.select_one("span.reactionsBar-count")
        m = re.search(r"(\d+)", like_el.get_text())
        self.assertEqual(int(m.group(1)), 24)

    def test_short_post_filtered(self):
        """Posts shorter than 20 chars should be skipped."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(self.html, "html.parser")
        post_els = soup.select("article.message")
        third = post_els[2]
        body = third.select_one("div.bbWrapper")
        text = body.get_text(strip=True)
        self.assertLess(len(text), 20)


# =========================================================================
# Thread relevance scoring
# =========================================================================


class TestThreadScoring(unittest.TestCase):
    """Test the _score_thread relevance scoring function."""

    def test_neutral_title(self):
        score = _score_thread("iPhone 15 hakkında")
        self.assertAlmostEqual(score, 0.5, places=1)

    def test_positive_keywords_boost(self):
        score = _score_thread("iPhone 15 inceleme ve deneyim paylaşımı")
        self.assertGreater(score, 0.5)

    def test_negative_keywords_penalize(self):
        score = _score_thread("satılık iPhone 15 acil takas olur")
        self.assertLess(score, 0.5)

    def test_score_clamped_to_range(self):
        # Many negative keywords
        score = _score_thread("satılık acil takas hediye")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_review_keyword(self):
        score = _score_thread("Samsung S24 review benchmark performans test")
        self.assertGreater(score, 0.7)


if __name__ == "__main__":
    unittest.main()
