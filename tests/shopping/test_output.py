"""Comprehensive tests for shopping output modules:
formatters, summary, product_cards.
"""

from __future__ import annotations

import json
import unittest


# ═══════════════════════════════════════════════════════════════════════════
# 1. TestFormatters
# ═══════════════════════════════════════════════════════════════════════════

class TestFormatPrice(unittest.TestCase):
    """Test format_price Turkish locale formatting."""

    def test_simple_price(self):
        from src.shopping.output.formatters import format_price
        self.assertEqual(format_price(1299.99), "1.299,99 TL")

    def test_zero_price(self):
        from src.shopping.output.formatters import format_price
        self.assertEqual(format_price(0), "0,00 TL")

    def test_large_price(self):
        from src.shopping.output.formatters import format_price
        self.assertEqual(format_price(1234567.89), "1.234.567,89 TL")

    def test_none_price(self):
        from src.shopping.output.formatters import format_price
        self.assertEqual(format_price(None), "")

    def test_negative_price(self):
        from src.shopping.output.formatters import format_price
        result = format_price(-500.0)
        self.assertTrue(result.startswith("-"))
        self.assertIn("500", result)

    def test_custom_currency(self):
        from src.shopping.output.formatters import format_price
        self.assertEqual(format_price(100, "USD"), "100,00 USD")

    def test_small_price(self):
        from src.shopping.output.formatters import format_price
        self.assertEqual(format_price(9.99), "9,99 TL")

    def test_thousands_separator(self):
        from src.shopping.output.formatters import format_price
        result = format_price(45000)
        self.assertEqual(result, "45.000,00 TL")


class TestValueBadge(unittest.TestCase):
    """Test _value_badge function."""

    def test_best_value(self):
        from src.shopping.output.formatters import _value_badge
        badge = _value_badge({"is_best_value": True})
        self.assertIn("Best Value", badge)

    def test_deal_badge(self):
        from src.shopping.output.formatters import _value_badge
        badge = _value_badge({"discount_percentage": 25})
        self.assertIn("Deal", badge)

    def test_warning_badge(self):
        from src.shopping.output.formatters import _value_badge
        badge = _value_badge({"has_warning": True})
        self.assertIn("Warning", badge)

    def test_suggestion_badge(self):
        from src.shopping.output.formatters import _value_badge
        badge = _value_badge({"is_suggestion": True})
        self.assertIn("Suggestion", badge)

    def test_no_badge(self):
        from src.shopping.output.formatters import _value_badge
        badge = _value_badge({})
        self.assertEqual(badge, "")


class TestPriceTrendBadge(unittest.TestCase):
    """Test _price_trend_badge."""

    def test_empty_history(self):
        from src.shopping.output.formatters import _price_trend_badge
        self.assertEqual(_price_trend_badge([]), "")

    def test_single_entry(self):
        from src.shopping.output.formatters import _price_trend_badge
        self.assertEqual(_price_trend_badge([{"price": 100}]), "")

    def test_near_low(self):
        from src.shopping.output.formatters import _price_trend_badge
        history = [{"price": 200}, {"price": 150}, {"price": 100}, {"price": 102}]
        badge = _price_trend_badge(history)
        self.assertIn("Low", badge)

    def test_price_rising(self):
        from src.shopping.output.formatters import _price_trend_badge
        history = [{"price": 100}, {"price": 150}, {"price": 195}]
        badge = _price_trend_badge(history)
        self.assertIn("Rising", badge)

    def test_stable_no_badge(self):
        from src.shopping.output.formatters import _price_trend_badge
        history = [{"price": 100}, {"price": 100}]
        badge = _price_trend_badge(history)
        self.assertEqual(badge, "")


class TestComparisonTable(unittest.TestCase):
    """Test format_comparison_table in various modes."""

    def _sample_products(self):
        return [
            {
                "name": "Samsung Galaxy S24",
                "discounted_price": 35000,
                "source": "trendyol",
                "rating": 4.5,
                "review_count": 120,
                "currency": "TL",
            },
            {
                "name": "iPhone 15",
                "discounted_price": 40000,
                "source": "hepsiburada",
                "rating": 4.7,
                "review_count": 200,
                "currency": "TL",
            },
        ]

    def test_telegram_format(self):
        from src.shopping.output.formatters import format_comparison_table
        result = format_comparison_table(self._sample_products(), "telegram")
        self.assertIn("Samsung Galaxy S24", result)
        self.assertIn("iPhone 15", result)

    def test_json_format(self):
        from src.shopping.output.formatters import format_comparison_table
        result = format_comparison_table(self._sample_products(), "json")
        parsed = json.loads(result)
        self.assertEqual(parsed["count"], 2)
        self.assertIn("products", parsed)

    def test_terminal_format(self):
        from src.shopping.output.formatters import format_comparison_table
        result = format_comparison_table(self._sample_products(), "terminal")
        self.assertIn("Product", result)
        self.assertIn("Price", result)

    def test_empty_products(self):
        from src.shopping.output.formatters import format_comparison_table
        result = format_comparison_table([], "telegram")
        self.assertIn("No products", result)

    def test_json_format_fields(self):
        from src.shopping.output.formatters import format_comparison_table
        result = format_comparison_table(self._sample_products(), "json")
        parsed = json.loads(result)
        product = parsed["products"][0]
        self.assertIn("name", product)
        self.assertIn("price", product)
        self.assertIn("source", product)
        self.assertIn("value_badge", product)

    def test_unknown_format_defaults_telegram(self):
        from src.shopping.output.formatters import format_comparison_table
        result = format_comparison_table(self._sample_products(), "unknown")
        # Should default to telegram format
        self.assertIn("**", result)


class TestTruncate(unittest.TestCase):
    def test_short_text(self):
        from src.shopping.output.formatters import _truncate
        self.assertEqual(_truncate("hello", 10), "hello")

    def test_long_text(self):
        from src.shopping.output.formatters import _truncate
        result = _truncate("a" * 100, 20)
        self.assertEqual(len(result), 20)

    def test_empty(self):
        from src.shopping.output.formatters import _truncate
        self.assertEqual(_truncate("", 10), "")


class TestInstallmentFormatting(unittest.TestCase):
    def test_format_installment_options(self):
        from src.shopping.output.formatters import format_installment_options
        options = [
            {"months": 12, "monthly_amount": 833, "bank": "Garanti", "interest_rate": 1.89},
            {"months": 6, "monthly_amount": 1666},
        ]
        result = format_installment_options(options)
        self.assertIn("12 ay", result)
        self.assertIn("6 ay", result)
        self.assertIn("Garanti", result)

    def test_empty_options(self):
        from src.shopping.output.formatters import format_installment_options
        self.assertEqual(format_installment_options([]), "")


# ═══════════════════════════════════════════════════════════════════════════
# 2. TestSummary
# ═══════════════════════════════════════════════════════════════════════════

class TestSummary(unittest.TestCase):
    """Test recommendation summary formatting."""

    def test_format_top_pick(self):
        from src.shopping.output.summary import format_top_pick
        result = format_top_pick({
            "name": "Samsung Galaxy S24",
            "discounted_price": 35000,
            "source": "trendyol",
            "reason": "Best value for money",
        })
        self.assertIn("Samsung Galaxy S24", result)
        self.assertIn("Best value for money", result)

    def test_format_top_pick_empty(self):
        from src.shopping.output.summary import format_top_pick
        self.assertEqual(format_top_pick({}), "")

    def test_format_budget_option(self):
        from src.shopping.output.summary import format_budget_option
        result = format_budget_option({
            "name": "Xiaomi Phone",
            "price": 15000,
            "reason": "Good enough specs",
        })
        self.assertIn("Xiaomi Phone", result)

    def test_format_alternatives(self):
        from src.shopping.output.summary import format_alternatives
        result = format_alternatives([
            {"name": "Alt 1", "price": 20000, "why_not": "Slower"},
            {"name": "Alt 2", "price": 25000, "why_not": "Heavier"},
        ])
        self.assertIn("Alt 1", result)
        self.assertIn("Slower", result)

    def test_format_warnings(self):
        from src.shopping.output.summary import format_warnings
        result = format_warnings(["Battery issues reported", "High return rate"])
        self.assertIn("Warnings", result)
        self.assertIn("Battery issues reported", result)

    def test_format_warnings_empty(self):
        from src.shopping.output.summary import format_warnings
        self.assertEqual(format_warnings([]), "")

    def test_format_timing_buy(self):
        from src.shopping.output.summary import format_timing_advice
        result = format_timing_advice({"action": "buy", "reason": "Price at lowest"})
        self.assertIn("Buy Now", result)

    def test_format_timing_wait(self):
        from src.shopping.output.summary import format_timing_advice
        result = format_timing_advice({"action": "wait", "reason": "Sale coming", "until": "2025-11-25"})
        self.assertIn("Wait", result)
        self.assertIn("Sale coming", result)

    def test_format_timing_empty(self):
        from src.shopping.output.summary import format_timing_advice
        self.assertEqual(format_timing_advice({}), "")

    def test_adapt_to_complexity_simple(self):
        from src.shopping.output.summary import _adapt_to_complexity
        results = {
            "top_pick": {"name": "Best Phone", "price": 30000, "url": "http://example.com"},
            "alternatives": [],
            "warnings": [],
            "confidence": 0.9,
        }
        compact = _adapt_to_complexity(results)
        self.assertNotEqual(compact, "")
        self.assertIn("Best Phone", compact)

    def test_adapt_to_complexity_complex(self):
        from src.shopping.output.summary import _adapt_to_complexity
        results = {
            "top_pick": {"name": "Phone"},
            "alternatives": [{"name": "A"}, {"name": "B"}],
            "warnings": ["Watch out"],
            "confidence": 0.5,
        }
        compact = _adapt_to_complexity(results)
        self.assertEqual(compact, "")  # complex query, no compact form

    def test_format_recommendation_summary_full(self):
        from src.shopping.output.summary import format_recommendation_summary
        results = {
            "top_pick": {"name": "Samsung S24", "price": 35000},
            "budget_option": {"name": "Xiaomi", "price": 15000},
            "alternatives": [
                {"name": "iPhone", "price": 40000, "why_not": "Expensive"},
            ],
            "warnings": ["Battery issues"],
            "timing": {"action": "buy", "reason": "Good price"},
            "confidence": 0.5,
            "sources": 3,
        }
        result = format_recommendation_summary(results, "telegram")
        self.assertIn("Samsung S24", result)
        self.assertIn("Xiaomi", result)
        self.assertIn("Battery issues", result)

    def test_confidence_indicator_high(self):
        from src.shopping.output.summary import _confidence_indicator
        result = _confidence_indicator(5)
        self.assertIn("5 sources", result)

    def test_confidence_indicator_low(self):
        from src.shopping.output.summary import _confidence_indicator
        result = _confidence_indicator(1)
        self.assertIn("Limited", result)

    def test_confidence_indicator_zero(self):
        from src.shopping.output.summary import _confidence_indicator
        result = _confidence_indicator(0)
        self.assertIn("No source", result)


# ═══════════════════════════════════════════════════════════════════════════
# 3. TestProductCards
# ═══════════════════════════════════════════════════════════════════════════

class TestProductCards(unittest.TestCase):
    """Test product card formatting."""

    def test_format_product_card_structure(self):
        from src.shopping.output.product_cards import format_product_card
        product = {
            "name": "Samsung Galaxy S24",
            "original_price": 40000,
            "discounted_price": 35000,
            "rating": 4.5,
            "review_count": 120,
            "source": "trendyol",
            "url": "https://trendyol.com/samsung-s24",
        }
        card = format_product_card(product)
        self.assertIn("text", card)
        self.assertIn("reply_markup", card)
        self.assertIn("Samsung Galaxy S24", card["text"])

    def test_format_product_card_no_discount(self):
        from src.shopping.output.product_cards import format_product_card
        product = {"name": "Test", "price": 5000}
        card = format_product_card(product)
        self.assertIn("text", card)
        self.assertIn("5.000", card["text"])

    def test_format_product_card_buttons(self):
        from src.shopping.output.product_cards import format_product_card
        product = {"name": "Test", "url": "https://x.com", "id": "123"}
        card = format_product_card(product)
        buttons = card["reply_markup"]
        self.assertTrue(len(buttons) >= 1)
        # First row should have Open Link button
        url_buttons = [b for row in buttons for b in row if "url" in b]
        self.assertTrue(len(url_buttons) > 0)

    def test_format_discount(self):
        from src.shopping.output.product_cards import _format_discount
        result = _format_discount(4000, 2800)
        self.assertIn("4.000", result)
        self.assertIn("2.800", result)
        self.assertIn("-30%", result)

    def test_format_discount_no_original(self):
        from src.shopping.output.product_cards import _format_discount
        result = _format_discount(None, 2800)
        self.assertIn("2.800", result)

    def test_format_discount_no_discount(self):
        from src.shopping.output.product_cards import _format_discount
        result = _format_discount(3000, 3500)
        self.assertIn("3.500", result)
        self.assertNotIn("-", result)

    def test_format_rating_stars(self):
        from src.shopping.output.product_cards import _format_rating_stars
        result = _format_rating_stars(4.0)
        self.assertIn("\u2605", result)
        self.assertIn("4.0", result)

    def test_format_rating_stars_none(self):
        from src.shopping.output.product_cards import _format_rating_stars
        self.assertEqual(_format_rating_stars(None), "")

    def test_format_rating_stars_clamped(self):
        from src.shopping.output.product_cards import _format_rating_stars
        result = _format_rating_stars(6.0)
        self.assertEqual(result.count("\u2605"), 5)  # clamped to 5

    def test_batch_cards(self):
        from src.shopping.output.product_cards import format_product_cards_batch
        products = [
            {"name": "P1", "price": 1000},
            {"name": "P2", "price": 2000},
        ]
        cards = format_product_cards_batch(products)
        self.assertEqual(len(cards), 2)

    def test_deal_card(self):
        from src.shopping.output.product_cards import format_deal_card
        product = {
            "name": "Hot Product",
            "original_price": 5000,
            "discounted_price": 3000,
            "source": "trendyol",
            "url": "https://x.com",
        }
        card = format_deal_card(product, 40.0)
        self.assertIn("DEAL", card["text"])
        self.assertIn("40", card["text"])

    def test_combo_card(self):
        from src.shopping.output.product_cards import format_combo_card
        combo = {
            "products": [
                {"name": "CPU", "original_price": 3000},
                {"name": "GPU", "original_price": 5000},
            ],
            "total_price": 8000,
            "compatibility_notes": ["Socket match: LGA1700"],
        }
        card = format_combo_card(combo)
        self.assertIn("Combo", card["text"])
        self.assertIn("CPU", card["text"])
        self.assertIn("8.000", card["text"])
        self.assertIn("Socket match", card["text"])


if __name__ == "__main__":
    unittest.main()
