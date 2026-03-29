# tests/test_deep_search_integration.py
"""Integration tests for the deep search pipeline."""

import asyncio
import importlib
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ws_mod = importlib.import_module("src.tools.web_search")


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestIntentInference(unittest.TestCase):

    def test_search_depth_deep(self):
        intent, params = _ws_mod._infer_search_intent({"search_depth": "deep"})
        self.assertEqual(intent, "research")
        self.assertTrue(params.use_deep_pipeline)

    def test_search_depth_quick(self):
        intent, params = _ws_mod._infer_search_intent({"search_depth": "quick"})
        self.assertEqual(intent, "factual")
        self.assertFalse(params.use_deep_pipeline)

    def test_search_depth_standard(self):
        intent, params = _ws_mod._infer_search_intent({"search_depth": "standard"})
        self.assertEqual(intent, "product")
        self.assertTrue(params.use_deep_pipeline)

    def test_shopping_sub_intent_compare(self):
        intent, _ = _ws_mod._infer_search_intent({"shopping_sub_intent": "compare"})
        self.assertEqual(intent, "product")

    def test_shopping_sub_intent_research(self):
        intent, _ = _ws_mod._infer_search_intent({"shopping_sub_intent": "research"})
        self.assertEqual(intent, "market")

    def test_shopping_sub_intent_purchase_advice(self):
        intent, _ = _ws_mod._infer_search_intent({"shopping_sub_intent": "purchase_advice"})
        self.assertEqual(intent, "reviews")

    def test_agent_type_researcher(self):
        intent, _ = _ws_mod._infer_search_intent({"agent_type": "researcher"})
        self.assertEqual(intent, "research")

    def test_agent_type_deal_analyst(self):
        intent, _ = _ws_mod._infer_search_intent({"agent_type": "deal_analyst"})
        self.assertEqual(intent, "market")

    def test_agent_type_assistant_defaults_factual(self):
        intent, _ = _ws_mod._infer_search_intent({"agent_type": "assistant"})
        self.assertEqual(intent, "factual")

    def test_no_hints_defaults_factual(self):
        intent, _ = _ws_mod._infer_search_intent({})
        self.assertEqual(intent, "factual")

    def test_params_have_required_fields(self):
        _, params = _ws_mod._infer_search_intent({"search_depth": "deep"})
        self.assertTrue(hasattr(params, "max_results"))
        self.assertTrue(hasattr(params, "total_budget"))
        self.assertTrue(hasattr(params, "use_deep_pipeline"))


if __name__ == "__main__":
    unittest.main()
