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


if __name__ == "__main__":
    unittest.main()
