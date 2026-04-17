"""Tests for ShoppingPipeline — mechanical shopping workflow step executor."""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from src.shopping.models import Product


# ---------------------------------------------------------------------------
# Async test helper
# ---------------------------------------------------------------------------

def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_fake_products():
    return [
        Product(
            name="Siemens S100 Inverter",
            url="https://www.trendyol.com/siemens-s100",
            source="trendyol",
            original_price=4500.0,
            discounted_price=3999.0,
        ),
        Product(
            name="Siemens S100 AC Drive",
            url="https://www.hepsiburada.com/siemens-s100",
            source="hepsiburada",
            original_price=4800.0,
        ),
    ]


# ═══════════════════════════════════════════════════════════════════════════
# TestShoppingPipeline
# ═══════════════════════════════════════════════════════════════════════════

class TestShoppingPipeline(unittest.TestCase):

    def _make_search_task(self, mission_id=99):
        return {
            "mission_id": mission_id,
            "context": {
                "step_name": "execute_product_search",
                "input_artifacts": ["user_query"],
            },
        }

    def _make_format_task(self, mission_id=99):
        return {
            "mission_id": mission_id,
            "context": {
                "step_name": "format_and_deliver",
                "input_artifacts": ["search_results"],
            },
        }

    # ------------------------------------------------------------------
    # test_search_step_returns_products
    # ------------------------------------------------------------------

    def test_search_step_returns_products(self):
        """Search step with products found → completed + JSON with products."""
        from src.workflows.shopping.pipeline import ShoppingPipeline

        fake_products = _make_fake_products()
        fake_community = [{"name": "Forum post", "url": "https://forum.example.com/1"}]

        with patch(
            "src.shopping.resilience.fallback_chain.get_product_with_fallback",
            new_callable=AsyncMock,
            return_value=fake_products,
        ), patch(
            "src.shopping.resilience.fallback_chain.get_community_data",
            new_callable=AsyncMock,
            return_value=fake_community,
        ), patch(
            "src.workflows.engine.artifacts.ArtifactStore.retrieve",
            new_callable=AsyncMock,
            return_value="siemens s100",
        ):
            result = run_async(ShoppingPipeline().run(self._make_search_task()))

        self.assertEqual(result["status"], "completed")
        data = json.loads(result["result"])
        self.assertEqual(data["product_count"], 2)
        self.assertEqual(len(data["products"]), 2)
        self.assertFalse(data["escalation_needed"])

    # ------------------------------------------------------------------
    # test_search_step_empty_sets_escalation
    # ------------------------------------------------------------------

    def test_search_step_empty_sets_escalation(self):
        """Search step with no products → escalation_needed is true."""
        from src.workflows.shopping.pipeline import ShoppingPipeline

        with patch(
            "src.shopping.resilience.fallback_chain.get_product_with_fallback",
            new_callable=AsyncMock,
            return_value=[],
        ), patch(
            "src.shopping.resilience.fallback_chain.get_community_data",
            new_callable=AsyncMock,
            return_value=[],
        ), patch(
            "src.workflows.engine.artifacts.ArtifactStore.retrieve",
            new_callable=AsyncMock,
            return_value="siemens s100",
        ):
            result = run_async(ShoppingPipeline().run(self._make_search_task()))

        self.assertEqual(result["status"], "completed")
        data = json.loads(result["result"])
        self.assertTrue(data["escalation_needed"])
        self.assertEqual(data["product_count"], 0)

    # ------------------------------------------------------------------
    # test_format_step_produces_text
    # ------------------------------------------------------------------

    def test_format_step_produces_text(self):
        """Format step converts search JSON into human-readable text."""
        from src.workflows.shopping.pipeline import ShoppingPipeline

        search_results = json.dumps({
            "products": [
                {
                    "name": "Siemens S100 Inverter",
                    "url": "https://www.trendyol.com/siemens-s100",
                    "source": "trendyol",
                    "original_price": 4500.0,
                    "discounted_price": 3999.0,
                }
            ],
            "community": [],
            "product_count": 1,
            "community_count": 0,
            "escalation_needed": False,
        })

        with patch(
            "src.workflows.engine.artifacts.ArtifactStore.retrieve",
            new_callable=AsyncMock,
            return_value=search_results,
        ):
            result = run_async(ShoppingPipeline().run(self._make_format_task()))

        self.assertEqual(result["status"], "completed")
        text = result["result"]
        # Result should be non-empty text, not raw JSON
        self.assertTrue(len(text) > 0)
        # It should not be parseable as JSON (it's human-readable text)
        is_json = True
        try:
            json.loads(text)
        except (json.JSONDecodeError, ValueError):
            is_json = False
        self.assertFalse(is_json, "format step should return text, not JSON")

    # ------------------------------------------------------------------
    # test_unknown_step_fails
    # ------------------------------------------------------------------

    def test_unknown_step_fails(self):
        """Unknown step name results in failed status."""
        from src.workflows.shopping.pipeline import ShoppingPipeline

        task = {
            "mission_id": 99,
            "context": {
                "step_name": "nonexistent",
                "input_artifacts": [],
            },
        }
        result = run_async(ShoppingPipeline().run(task))

        self.assertEqual(result["status"], "failed")
        self.assertIn("nonexistent", result["result"])

    # ------------------------------------------------------------------
    # test_returns_zero_cost
    # ------------------------------------------------------------------

    def test_returns_zero_cost(self):
        """Pipeline always returns zero cost and correct model tag."""
        from src.workflows.shopping.pipeline import ShoppingPipeline

        with patch(
            "src.shopping.resilience.fallback_chain.get_product_with_fallback",
            new_callable=AsyncMock,
            return_value=[],
        ), patch(
            "src.shopping.resilience.fallback_chain.get_community_data",
            new_callable=AsyncMock,
            return_value=[],
        ), patch(
            "src.workflows.engine.artifacts.ArtifactStore.retrieve",
            new_callable=AsyncMock,
            return_value="test query",
        ):
            result = run_async(ShoppingPipeline().run(self._make_search_task()))

        self.assertEqual(result["cost"], 0.0)
        self.assertEqual(result["model"], "shopping_pipeline")


class TestCommunityRelevanceFilter(unittest.TestCase):
    def test_strict_mode_returns_empty_when_no_match(self):
        from src.workflows.shopping.pipeline import _filter_relevant
        items = [
            {"name": "Ahmet'in cezaevi telefon sikayeti", "source": "sikayetvar"},
            {"name": "Cepte fatura sorunu", "source": "sikayetvar"},
        ]
        out = _filter_relevant(items, "siemens s100 kahve makinesi", strict=True)
        self.assertEqual(out, [])

    def test_strict_mode_keeps_matches(self):
        from src.workflows.shopping.pipeline import _filter_relevant
        items = [
            {"name": "Siemens S100 kahve makinesi arıza", "source": "sikayetvar"},
            {"name": "Tamamen alakasız bir post", "source": "teknopat"},
        ]
        out = _filter_relevant(items, "siemens s100 kahve", strict=True)
        names = [i["name"] for i in out]
        self.assertIn("Siemens S100 kahve makinesi arıza", names)
        self.assertNotIn("Tamamen alakasız bir post", names)
        self.assertEqual(len(out), 1)

    def test_default_mode_still_falls_back(self):
        """Non-strict mode keeps existing behavior — return original if nothing passes."""
        from src.workflows.shopping.pipeline import _filter_relevant
        items = [{"name": "Cezaevi telefon", "source": "sikayetvar"}]
        out = _filter_relevant(items, "coffee machine", strict=False)
        self.assertEqual(out, items)  # fallback to original


class TestFakeDiscountAnnotation(unittest.TestCase):
    def test_outlier_ratio_flagged(self):
        from src.workflows.shopping.pipeline import _annotate_fake_discounts
        group = {
            "products": [
                {"name": "X", "source": "trendyol",
                 "original_price": 10000, "discounted_price": 5000, "url": "a"},
                {"name": "X", "source": "hepsiburada",
                 "original_price": 5500, "discounted_price": 5000, "url": "b"},
                {"name": "X", "source": "amazon_tr",
                 "original_price": 5200, "discounted_price": 4800, "url": "c"},
            ],
        }
        flags = _annotate_fake_discounts([group])
        self.assertTrue(flags[("X", "trendyol", "a")]["is_suspicious_discount"])
        self.assertFalse(flags.get(("X", "hepsiburada", "b"), {}).get("is_suspicious_discount", False))

    def test_no_flag_when_consistent(self):
        from src.workflows.shopping.pipeline import _annotate_fake_discounts
        group = {
            "products": [
                {"name": "Y", "source": "trendyol",
                 "original_price": 5200, "discounted_price": 5000, "url": "a"},
                {"name": "Y", "source": "hepsiburada",
                 "original_price": 5500, "discounted_price": 5000, "url": "b"},
            ],
        }
        flags = _annotate_fake_discounts([group])
        for key, f in flags.items():
            self.assertFalse(f.get("is_suspicious_discount", False))

    def test_single_entry_group_skipped(self):
        from src.workflows.shopping.pipeline import _annotate_fake_discounts
        group = {"products": [
            {"name": "Z", "source": "trendyol",
             "original_price": 10000, "discounted_price": 1000, "url": "a"},
        ]}
        flags = _annotate_fake_discounts([group])
        self.assertEqual(flags, {})


class TestSiteRankSort(unittest.TestCase):
    def test_site_rank_beats_value_score(self):
        """Rank-0 from any site wins over rank-2 even with lower value_score."""
        rank0_low_score = {
            "name": "Siemens S100 Coffee Machine",
            "source": "trendyol",
            "site_rank": 0,
            "value_score": 60,
            "discounted_price": 4200,
        }
        rank2_high_score = {
            "name": "Siemens S100 spare brewing unit",
            "source": "amazon_tr",
            "site_rank": 2,
            "value_score": 80,
            "discounted_price": 4800,
        }
        items = [rank2_high_score, rank0_low_score]

        def _sort_key(p: dict):
            rank = p.get("site_rank")
            rank = rank if rank is not None else 999
            score = p.get("value_score") or 0
            price = p.get("discounted_price") or p.get("original_price") or 0
            return (rank, -score, price)

        items.sort(key=_sort_key)
        self.assertEqual(items[0]["name"], "Siemens S100 Coffee Machine")

    def test_missing_site_rank_sinks(self):
        """Products without site_rank come last."""
        ranked = {"name": "A", "site_rank": 3, "value_score": 50, "discounted_price": 100}
        unranked = {"name": "B", "value_score": 99, "discounted_price": 100}
        items = [unranked, ranked]

        def _sort_key(p: dict):
            rank = p.get("site_rank")
            rank = rank if rank is not None else 999
            score = p.get("value_score") or 0
            price = p.get("discounted_price") or p.get("original_price") or 0
            return (rank, -score, price)

        items.sort(key=_sort_key)
        self.assertEqual(items[0]["name"], "A")


if __name__ == "__main__":
    unittest.main()
