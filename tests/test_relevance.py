# tests/test_relevance.py
"""Tests for relevance scoring and budget allocation."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.content_extract import ExtractedContent
from src.tools.relevance import score_and_budget, BudgetedContent


def _make_content(text, url="", has_prices=False, has_reviews=False):
    return ExtractedContent(text=text, title="", url=url,
        word_count=len(text.split()), has_prices=has_prices, has_reviews=has_reviews)


class TestScoreAndBudget(unittest.TestCase):

    def test_returns_budgeted_content_list(self):
        contents = [
            _make_content("Coffee machines are great for morning routines and productivity"),
            _make_content("Python programming tutorial for beginners with examples"),
        ]
        result = score_and_budget(contents, query="coffee machines", total_budget=5000)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], BudgetedContent)

    def test_higher_relevance_gets_more_budget(self):
        contents = [
            _make_content("The best coffee machines of 2026 include DeLonghi and Philips models with automatic brewing"),
            _make_content("Weather forecast for Istanbul shows sunny skies and mild temperatures this week"),
        ]
        result = score_and_budget(contents, query="best coffee machines 2026", total_budget=6000)
        coffee = next(b for b in result if "coffee" in b.content.text.lower())
        weather = next(b for b in result if "weather" in b.content.text.lower())
        self.assertGreater(coffee.allocated_chars, weather.allocated_chars)

    def test_total_budget_respected(self):
        contents = [_make_content("Word " * 500), _make_content("Text " * 500), _make_content("Data " * 500)]
        result = score_and_budget(contents, query="word", total_budget=3000)
        total_allocated = sum(b.allocated_chars for b in result)
        self.assertLessEqual(total_allocated, 3000)

    def test_minimum_budget_per_page(self):
        contents = [
            _make_content("Highly relevant coffee machine review " * 20),
            _make_content("Completely irrelevant text " * 5),
        ]
        result = score_and_budget(contents, query="coffee machine review", total_budget=5000)
        min_budget = min(b.allocated_chars for b in result)
        self.assertGreaterEqual(min_budget, 200)

    def test_max_budget_cap_per_page(self):
        contents = [_make_content("Coffee " * 100), _make_content("Tea " * 10)]
        result = score_and_budget(contents, query="coffee", total_budget=10000)
        max_budget = max(b.allocated_chars for b in result)
        self.assertLessEqual(max_budget, 10000 * 0.4 + 1)

    def test_truncated_text_within_budget(self):
        long_text = "This is a sentence about coffee machines. " * 100
        contents = [_make_content(long_text)]
        result = score_and_budget(contents, query="coffee", total_budget=500)
        self.assertLessEqual(len(result[0].truncated_text), 520)

    def test_product_intent_boosts_price_pages(self):
        contents = [
            _make_content("The iPhone costs $799 and Samsung costs $699", has_prices=True),
            _make_content("Smartphones use lithium ion batteries and ARM processors"),
        ]
        result = score_and_budget(contents, query="smartphone", total_budget=5000, intent="product")
        price_page = next(b for b in result if b.content.has_prices)
        no_price = next(b for b in result if not b.content.has_prices)
        self.assertGreater(price_page.relevance_score, no_price.relevance_score)

    def test_reviews_intent_boosts_review_pages(self):
        contents = [
            _make_content("User rating: 4.5 out of 5 stars. 230 reviews.", has_reviews=True),
            _make_content("The product specifications include 8GB RAM and 256GB storage"),
        ]
        result = score_and_budget(contents, query="product", total_budget=5000, intent="reviews")
        review_page = next(b for b in result if b.content.has_reviews)
        spec_page = next(b for b in result if not b.content.has_reviews)
        self.assertGreater(review_page.relevance_score, spec_page.relevance_score)

    def test_empty_contents_returns_empty(self):
        result = score_and_budget([], query="anything", total_budget=5000)
        self.assertEqual(result, [])

    def test_single_content(self):
        contents = [_make_content("Single page about coffee machines and their features")]
        result = score_and_budget(contents, query="coffee", total_budget=5000)
        self.assertEqual(len(result), 1)
        self.assertGreater(result[0].allocated_chars, 0)


if __name__ == "__main__":
    unittest.main()
