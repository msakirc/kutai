"""Tests for product_research workflow step handlers (deterministic)."""
from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from src.shopping.models import Product


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _fake_products():
    return [
        Product(
            name="Siemens S100 Inverter",
            url="https://www.trendyol.com/siemens-s100",
            source="trendyol",
            original_price=4500.0,
            discounted_price=3999.0,
            rating=4.5,
            review_count=42,
        ),
        Product(
            name="Siemens S100 AC Drive",
            url="https://www.hepsiburada.com/siemens-s100",
            source="hepsiburada",
            original_price=4800.0,
            discounted_price=4200.0,
        ),
    ]


class TestProductResearchHandlers(unittest.TestCase):

    def test_search_for_product_returns_products(self):
        from src.workflows.shopping.pipeline import ShoppingPipeline

        pipeline = ShoppingPipeline()
        task = {
            "mission_id": 77,
            "context": {
                "step_name": "search_for_product",
                "input_artifacts": ["user_query"],
            },
        }

        with patch(
            "src.workflows.shopping.pipeline._read_artifacts",
            new=AsyncMock(return_value={"user_query": "siemens s100"}),
        ), patch(
            "src.shopping.resilience.fallback_chain.get_product_with_fallback",
            new=AsyncMock(return_value=_fake_products()),
        ), patch(
            "src.shopping.resilience.fallback_chain.get_community_data",
            new=AsyncMock(return_value=[]),
        ):
            out = run_async(pipeline.run(task))

        self.assertEqual(out["status"], "completed")
        data = json.loads(out["result"])
        self.assertGreaterEqual(data["product_count"], 1)

    def test_enrich_product_returns_enrichment_dict(self):
        from src.workflows.shopping.pipeline import ShoppingPipeline

        pipeline = ShoppingPipeline()
        search_artifact = json.dumps({
            "products": [
                {"name": "X", "source": "trendyol",
                 "original_price": 5000, "discounted_price": 4500, "url": "a"},
                {"name": "X", "source": "hepsiburada",
                 "original_price": 5200, "discounted_price": 4800, "url": "b"},
            ],
            "community": [],
            "product_count": 2,
        })
        task = {
            "mission_id": 77,
            "context": {
                "step_name": "enrich_product_results",
                "input_artifacts": ["search_results", "user_query"],
            },
        }
        with patch(
            "src.workflows.shopping.pipeline._read_artifacts",
            new=AsyncMock(return_value={
                "search_results": search_artifact,
                "user_query": "X",
            }),
        ):
            out = run_async(pipeline.run(task))

        self.assertEqual(out["status"], "completed")
        data = json.loads(out["result"])
        self.assertIn("products", data)
        self.assertIn("cross_store_summary", data)
        self.assertIn("suspicious_discount_count", data["cross_store_summary"])

    def test_deliver_product_research_formats_for_telegram(self):
        from src.workflows.shopping.pipeline import ShoppingPipeline

        pipeline = ShoppingPipeline()
        enriched = json.dumps({
            "products": [
                {"name": "Siemens S100", "source": "trendyol",
                 "discounted_price": 3999, "url": "https://t.co/x",
                 "rating": 4.5, "review_count": 42},
            ],
            "community": [],
            "cross_store_summary": {
                "store_count": 1, "suspicious_discount_count": 0,
            },
        })
        task = {
            "mission_id": 77,
            "context": {
                "step_name": "deliver_product_research",
                "input_artifacts": ["enriched_product_data", "user_query"],
            },
        }
        with patch(
            "src.workflows.shopping.pipeline._read_artifacts",
            new=AsyncMock(return_value={
                "enriched_product_data": enriched,
                "user_query": "Siemens S100",
            }),
        ):
            out = run_async(pipeline.run(task))

        self.assertEqual(out["status"], "completed")
        self.assertIsInstance(out["result"], str)
        self.assertIn("Siemens S100", out["result"])
        self.assertIn("3999", out["result"].replace(".", "").replace(",", ""))

    def test_stub_review_synthesis_returns_placeholder(self):
        from src.workflows.shopping.pipeline import ShoppingPipeline

        pipeline = ShoppingPipeline()
        task = {
            "mission_id": 77,
            "context": {
                "step_name": "synthesize_product_reviews",
                "input_artifacts": ["enriched_product_data"],
            },
        }
        with patch(
            "src.workflows.shopping.pipeline._read_artifacts",
            new=AsyncMock(return_value={"enriched_product_data": "{}"}),
        ):
            out = run_async(pipeline.run(task))
        self.assertEqual(out["status"], "completed")
        data = json.loads(out["result"])
        self.assertEqual(data["status"], "disabled")
        self.assertIn("scraper", data["reason"].lower())
