"""Tests for shopping tool functions."""
import asyncio
import json
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.shopping.models import Product


def _make_product(**overrides) -> Product:
    """Create a Product with sensible defaults, overridden by kwargs."""
    defaults = dict(
        name="Test Product",
        url="https://www.trendyol.com/product-p-123",
        source="trendyol",
        original_price=1000.0,
        discounted_price=800.0,
        currency="TRY",
        rating=4.5,
        review_count=120,
    )
    defaults.update(overrides)
    return Product(**defaults)


class TestShoppingSearchSerialization(unittest.TestCase):
    """_tool_shopping_search must return JSON with a 'products' list of dicts."""

    def test_returns_json_with_products_list(self):
        """Mock get_product_with_fallback to return Product instances.
        Verify the tool result is valid JSON containing a list of dicts."""
        products = [
            _make_product(name="Product A", original_price=500),
            _make_product(name="Product B", original_price=700),
        ]

        with patch(
            "src.shopping.resilience.fallback_chain.get_product_with_fallback",
            new_callable=AsyncMock,
            return_value=products,
        ):
            from src.tools import _optional_tools
            tool_fn = _optional_tools["shopping_search"]["function"]
            result = asyncio.get_event_loop().run_until_complete(tool_fn("test query"))

        parsed = json.loads(result)
        self.assertIn("products", parsed)
        self.assertIsInstance(parsed["products"], list)
        self.assertEqual(len(parsed["products"]), 2)
        # Each product must be a dict, not a repr string
        for p in parsed["products"]:
            self.assertIsInstance(p, dict, f"Expected dict, got {type(p)}: {p!r}")
            self.assertIn("name", p)
            self.assertIn("url", p)

    def test_empty_results(self):
        """When no products are found, products list should be empty."""
        with patch(
            "src.shopping.resilience.fallback_chain.get_product_with_fallback",
            new_callable=AsyncMock,
            return_value=[],
        ):
            from src.tools import _optional_tools
            tool_fn = _optional_tools["shopping_search"]["function"]
            result = asyncio.get_event_loop().run_until_complete(tool_fn("nonexistent"))

        parsed = json.loads(result)
        self.assertEqual(parsed["products"], [])
        self.assertEqual(parsed["product_count"], 0)


class TestShoppingCompareSerialization(unittest.TestCase):
    """_tool_shopping_compare must pass Product objects to score_products."""

    def test_compare_converts_dicts_to_products(self):
        """When given JSON dicts, the tool should convert them to Product objects
        before passing to score_products."""
        product_dicts = [
            {"name": "A", "url": "http://a.com", "source": "trendyol",
             "original_price": 500, "rating": 4.0, "review_count": 50},
            {"name": "B", "url": "http://b.com", "source": "hepsiburada",
             "original_price": 700, "rating": 4.5, "review_count": 100},
        ]

        captured_args = {}

        async def mock_score(products, **kwargs):
            # Capture the actual argument type
            captured_args["products"] = products
            return [{"product_name": "A", "value_score": 80, "rank": 1},
                    {"product_name": "B", "value_score": 70, "rank": 2}]

        async def mock_delivery(products, **kwargs):
            return []

        with patch(
            "src.tools._score_products", side_effect=mock_score,
        ), patch(
            "src.tools._compare_delivery", side_effect=mock_delivery,
        ):
            from src.tools import _optional_tools
            tool_fn = _optional_tools["shopping_compare"]["function"]
            result = asyncio.get_event_loop().run_until_complete(
                tool_fn(json.dumps(product_dicts))
            )

        # Verify score_products received Product objects, not dicts
        self.assertIn("products", captured_args)
        for p in captured_args["products"]:
            self.assertIsInstance(p, Product, f"Expected Product, got {type(p)}")

        # Verify the result is valid JSON
        parsed = json.loads(result)
        self.assertIn("scores", parsed)


class TestShoppingFetchReviews(unittest.TestCase):
    """Test the URL-to-scraper domain detection in _tool_shopping_fetch_reviews."""

    def test_trendyol_url_detected(self):
        """A Trendyol URL should auto-detect the trendyol scraper."""
        mock_scraper_instance = MagicMock()
        mock_scraper_instance.get_reviews = AsyncMock(return_value=[
            {"text": "Great product", "rating": 5}
        ])
        mock_scraper_cls = MagicMock(return_value=mock_scraper_instance)

        with patch(
            "src.shopping.scrapers.get_scraper",
            return_value=mock_scraper_cls,
        ) as mock_get:
            from src.tools import _optional_tools
            tool_fn = _optional_tools["shopping_fetch_reviews"]["function"]
            result = asyncio.get_event_loop().run_until_complete(
                tool_fn("https://www.trendyol.com/product-p-12345")
            )

        parsed = json.loads(result)
        self.assertEqual(parsed["source"], "trendyol")
        self.assertEqual(parsed["review_count"], 1)
        mock_get.assert_called_with("trendyol")
        mock_scraper_instance.get_reviews.assert_awaited_once()

    def test_hepsiburada_url_detected(self):
        mock_scraper_instance = MagicMock()
        mock_scraper_instance.get_reviews = AsyncMock(return_value=[])
        mock_scraper_cls = MagicMock(return_value=mock_scraper_instance)

        with patch(
            "src.shopping.scrapers.get_scraper",
            return_value=mock_scraper_cls,
        ):
            from src.tools import _optional_tools
            tool_fn = _optional_tools["shopping_fetch_reviews"]["function"]
            result = asyncio.get_event_loop().run_until_complete(
                tool_fn("https://www.hepsiburada.com/some-product-p-HBC123")
            )

        parsed = json.loads(result)
        self.assertEqual(parsed["source"], "hepsiburada")

    def test_amazon_tr_url_detected(self):
        mock_scraper_instance = MagicMock()
        mock_scraper_instance.get_reviews = AsyncMock(return_value=[])
        mock_scraper_cls = MagicMock(return_value=mock_scraper_instance)

        with patch(
            "src.shopping.scrapers.get_scraper",
            return_value=mock_scraper_cls,
        ):
            from src.tools import _optional_tools
            tool_fn = _optional_tools["shopping_fetch_reviews"]["function"]
            result = asyncio.get_event_loop().run_until_complete(
                tool_fn("https://www.amazon.com.tr/dp/B0CTEST123")
            )

        parsed = json.loads(result)
        self.assertEqual(parsed["source"], "amazon_tr")

    def test_unknown_url_returns_error(self):
        with patch(
            "src.shopping.scrapers.list_scrapers",
            return_value={"trendyol": MagicMock(), "hepsiburada": MagicMock()},
        ):
            from src.tools import _optional_tools
            tool_fn = _optional_tools["shopping_fetch_reviews"]["function"]
            result = asyncio.get_event_loop().run_until_complete(
                tool_fn("https://www.unknownshop.com/product/123")
            )

        parsed = json.loads(result)
        self.assertIn("error", parsed)
        self.assertIn("Could not detect scraper", parsed["error"])

    def test_explicit_source_overrides_detection(self):
        """When source is explicitly provided, it should be used directly."""
        mock_scraper_instance = MagicMock()
        mock_scraper_instance.get_reviews = AsyncMock(return_value=[])
        mock_scraper_cls = MagicMock(return_value=mock_scraper_instance)

        with patch(
            "src.shopping.scrapers.get_scraper",
            return_value=mock_scraper_cls,
        ) as mock_get:
            from src.tools import _optional_tools
            tool_fn = _optional_tools["shopping_fetch_reviews"]["function"]
            result = asyncio.get_event_loop().run_until_complete(
                tool_fn("https://www.anysite.com/product", source="akakce")
            )

        mock_get.assert_called_with("akakce")
        parsed = json.loads(result)
        self.assertEqual(parsed["source"], "akakce")


class TestScraperRegistry(unittest.TestCase):
    """Verify scraper registry completeness."""

    def test_expected_scrapers_registered(self):
        """All major Turkish shopping sites should have registered scrapers."""
        from src.shopping.scrapers import list_scrapers
        registry = list_scrapers()
        expected = [
            "akakce", "trendyol", "hepsiburada", "amazon_tr",
            "technopat", "donanimhaber", "eksisozluk", "sikayetvar",
            "getir", "migros", "koctas", "ikea",
        ]
        for name in expected:
            self.assertIn(name, registry, f"Scraper '{name}' missing from registry")

    def test_search_planner_sources_have_scrapers(self):
        """All source names used in search_planner must exist in the scraper registry."""
        from src.shopping.scrapers import list_scrapers
        from src.shopping.intelligence.search_planner import (
            _SOURCES_BY_CATEGORY,
            _DEFAULT_SOURCES,
        )
        registry = list_scrapers()

        # Collect all source names referenced by the planner
        all_sources = set(_DEFAULT_SOURCES)
        for sources in _SOURCES_BY_CATEGORY.values():
            all_sources.update(sources)

        # Sources like "google", "youtube" are external and not in scraper registry
        external_sources = {"google", "youtube"}
        testable = all_sources - external_sources

        for source in testable:
            self.assertIn(
                source, registry,
                f"Search planner references source '{source}' "
                f"but no scraper is registered for it"
            )

    def test_registry_values_are_classes(self):
        """Registry values should be classes (not instances)."""
        from src.shopping.scrapers import list_scrapers, BaseScraper
        registry = list_scrapers()
        for name, cls in registry.items():
            self.assertTrue(
                isinstance(cls, type),
                f"Registry entry '{name}' is {type(cls)}, expected a class"
            )


if __name__ == "__main__":
    unittest.main()
