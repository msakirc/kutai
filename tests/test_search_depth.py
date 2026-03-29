# tests/test_search_depth.py
"""Tests for search_depth classification field."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.task_classifier import _classify_search_depth


class TestSearchDepthClassification(unittest.TestCase):

    def test_deep_for_analysis_keywords(self):
        self.assertEqual(_classify_search_depth("Analyze the market for robot vacuums"), "deep")
        self.assertEqual(_classify_search_depth("Research the best laptops in detail"), "deep")

    def test_standard_for_comparison_keywords(self):
        self.assertEqual(_classify_search_depth("Compare iPhone vs Samsung"), "standard")
        self.assertEqual(_classify_search_depth("What is the price of RTX 4070"), "standard")
        self.assertEqual(_classify_search_depth("DeLonghi kahve makinesi fiyat"), "standard")
        self.assertEqual(_classify_search_depth("Sony WH-1000XM5 review"), "standard")

    def test_quick_for_simple_queries(self):
        self.assertEqual(_classify_search_depth("What is Python"), "quick")
        self.assertEqual(_classify_search_depth("Hello"), "quick")

    def test_none_for_non_search_tasks(self):
        self.assertEqual(_classify_search_depth("Write a function to sort a list"), "none")
        self.assertEqual(_classify_search_depth("Fix the bug in main.py"), "none")

    def test_case_insensitive(self):
        self.assertEqual(_classify_search_depth("ANALYZE market trends"), "deep")
        self.assertEqual(_classify_search_depth("COMPARE products"), "standard")


class TestTimeSensitiveDetection(unittest.TestCase):
    """Time-sensitive queries must be upgraded to at least quick/standard."""

    def test_predicted_xi_is_at_least_quick(self):
        depth = _classify_search_depth("predicted xi for turkey next tuesday")
        self.assertIn(depth, ("quick", "standard", "deep"))

    def test_weather_is_quick(self):
        depth = _classify_search_depth("weather in istanbul tomorrow")
        self.assertIn(depth, ("quick", "standard"))

    def test_stock_price_is_standard(self):
        depth = _classify_search_depth("current price of AAPL stock")
        self.assertIn(depth, ("standard", "deep"))

    def test_turkish_time_sensitive(self):
        depth = _classify_search_depth("bugün dolar kuru ne")
        self.assertIn(depth, ("standard", "deep"))

    def test_match_lineup_is_standard(self):
        depth = _classify_search_depth("turkey vs kosovo lineup tonight")
        self.assertIn(depth, ("standard", "deep"))

    def test_non_time_sensitive_stays_none(self):
        depth = _classify_search_depth("write a python function to sort a list")
        self.assertEqual(depth, "none")

    def test_score_is_standard(self):
        depth = _classify_search_depth("what is the score of the match")
        self.assertIn(depth, ("standard", "deep"))

    def test_turkish_match_query(self):
        depth = _classify_search_depth("bu akşam maç kadrosu")
        self.assertIn(depth, ("standard", "deep"))

    def test_exchange_rate_turkish(self):
        depth = _classify_search_depth("altın fiyatı ne kadar")
        self.assertIn(depth, ("standard", "deep"))

    def test_latest_news(self):
        depth = _classify_search_depth("latest news about AI")
        self.assertIn(depth, ("quick", "standard", "deep"))

    def test_son_dakika(self):
        depth = _classify_search_depth("son dakika haberleri")
        self.assertIn(depth, ("quick", "standard", "deep"))

    def test_tomorrow_does_not_stay_none(self):
        """A query with 'tomorrow' must not be classified as 'none'."""
        depth = _classify_search_depth("what happens tomorrow")
        self.assertNotEqual(depth, "none")

    def test_deep_not_downgraded(self):
        """Time-sensitivity should not downgrade an already-deep classification."""
        depth = _classify_search_depth("analyze stock price trends in detail today")
        self.assertEqual(depth, "deep")


if __name__ == "__main__":
    unittest.main()
