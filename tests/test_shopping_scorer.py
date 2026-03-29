"""Tests for the shopping value scorer."""
import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.shopping.models import Product
from src.shopping.intelligence.value_scorer import (
    score_products,
    _effective_price,
    _total_cost,
    _normalize,
    _normalize_positive,
    _seller_score,
    _warranty_score,
    _rating_score,
    _availability_score,
)


def _make_product(**overrides) -> Product:
    """Create a Product with sensible defaults."""
    defaults = dict(
        name="Test Product",
        url="https://example.com/p/1",
        source="trendyol",
        original_price=1000.0,
        currency="TRY",
    )
    defaults.update(overrides)
    return Product(**defaults)


class TestEffectivePrice(unittest.TestCase):
    def test_discounted_price_preferred(self):
        p = _make_product(original_price=1000, discounted_price=800)
        self.assertEqual(_effective_price(p), 800.0)

    def test_original_price_when_no_discount(self):
        p = _make_product(original_price=1000, discounted_price=None)
        self.assertEqual(_effective_price(p), 1000.0)

    def test_zero_when_both_none(self):
        p = _make_product(original_price=None, discounted_price=None)
        self.assertEqual(_effective_price(p), 0.0)


class TestTotalCost(unittest.TestCase):
    def test_includes_shipping(self):
        p = _make_product(original_price=1000, shipping_cost=50)
        self.assertEqual(_total_cost(p), 1050.0)

    def test_no_shipping(self):
        p = _make_product(original_price=1000, shipping_cost=None)
        self.assertEqual(_total_cost(p), 1000.0)


class TestNormalize(unittest.TestCase):
    def test_min_value_scores_100(self):
        # Normalize is inverted (lower value = higher score)
        self.assertEqual(_normalize(10, 10, 100), 100.0)

    def test_max_value_scores_0(self):
        self.assertEqual(_normalize(100, 10, 100), 0.0)

    def test_mid_value(self):
        self.assertAlmostEqual(_normalize(55, 10, 100), 50.0, places=1)

    def test_equal_min_max_returns_50(self):
        self.assertEqual(_normalize(42, 42, 42), 50.0)


class TestNormalizePositive(unittest.TestCase):
    def test_max_value_scores_100(self):
        self.assertEqual(_normalize_positive(100, 100), 100.0)

    def test_zero_scores_0(self):
        self.assertEqual(_normalize_positive(0, 100), 0.0)

    def test_zero_max_returns_50(self):
        self.assertEqual(_normalize_positive(50, 0), 50.0)


class TestSellerScore(unittest.TestCase):
    def test_unknown_seller(self):
        p = _make_product(seller_rating=None)
        self.assertEqual(_seller_score(p), 50.0)

    def test_high_rated_seller(self):
        p = _make_product(seller_rating=4.8, seller_review_count=500)
        score = _seller_score(p)
        self.assertGreater(score, 70)

    def test_low_rated_seller(self):
        p = _make_product(seller_rating=1.0, seller_review_count=5)
        score = _seller_score(p)
        self.assertLess(score, 50)


class TestWarrantyScore(unittest.TestCase):
    def test_no_warranty_info(self):
        p = _make_product(warranty_months=None)
        self.assertEqual(_warranty_score(p), 30.0)

    def test_no_warranty(self):
        p = _make_product(warranty_months=0)
        self.assertEqual(_warranty_score(p), 0.0)

    def test_24_months(self):
        p = _make_product(warranty_months=24)
        score = _warranty_score(p)
        self.assertEqual(score, 80.0)

    def test_longer_warranty_scores_higher(self):
        p12 = _make_product(warranty_months=12)
        p24 = _make_product(warranty_months=24)
        self.assertGreater(_warranty_score(p24), _warranty_score(p12))


class TestRatingScore(unittest.TestCase):
    def test_unrated_product(self):
        p = _make_product(rating=None)
        self.assertEqual(_rating_score(p), 40.0)

    def test_high_rated_many_reviews(self):
        p = _make_product(rating=4.9, review_count=1000)
        score = _rating_score(p)
        self.assertGreater(score, 80)

    def test_high_rated_few_reviews_lower_than_many(self):
        """Bayesian adjustment: same rating with fewer reviews should score lower."""
        p_few = _make_product(rating=5.0, review_count=2)
        p_many = _make_product(rating=5.0, review_count=500)
        self.assertGreater(_rating_score(p_many), _rating_score(p_few))


class TestAvailabilityScore(unittest.TestCase):
    def test_in_stock(self):
        p = _make_product(availability="in_stock")
        self.assertEqual(_availability_score(p), 100.0)

    def test_out_of_stock(self):
        p = _make_product(availability="out_of_stock")
        self.assertEqual(_availability_score(p), 0.0)


class TestScoreProducts(unittest.TestCase):
    """Integration tests for the full score_products pipeline."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_empty_list(self):
        result = self._run(score_products([]))
        self.assertEqual(result, [])

    def test_single_product(self):
        products = [_make_product(name="Solo", original_price=500, rating=4.0, review_count=50)]
        result = self._run(score_products(products))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["product_name"], "Solo")
        self.assertEqual(result[0]["rank"], 1)
        self.assertIn("value_score", result[0])
        self.assertIn("breakdown", result[0])
        self.assertIn("perspectives", result[0])

    def test_cheaper_higher_rated_scores_better(self):
        """A cheaper product with higher rating should score better than
        an expensive product with lower rating (all else equal)."""
        good = _make_product(
            name="Good Deal",
            original_price=500,
            rating=4.8,
            review_count=200,
            seller_rating=4.5,
            seller_review_count=1000,
        )
        bad = _make_product(
            name="Bad Deal",
            original_price=2000,
            rating=2.5,
            review_count=10,
            seller_rating=2.0,
            seller_review_count=5,
        )
        result = self._run(score_products([good, bad]))
        scores = {r["product_name"]: r["value_score"] for r in result}
        self.assertGreater(scores["Good Deal"], scores["Bad Deal"])

    def test_ranks_are_correct(self):
        """Ranks should be 1-indexed and ordered by descending score."""
        products = [
            _make_product(name="C", original_price=1500, rating=3.0, review_count=10),
            _make_product(name="A", original_price=300, rating=4.9, review_count=500),
            _make_product(name="B", original_price=800, rating=4.0, review_count=100),
        ]
        result = self._run(score_products(products))
        ranks = {r["product_name"]: r["rank"] for r in result}
        # A should rank best (cheapest + highest rated)
        self.assertEqual(ranks["A"], 1)
        # All ranks should be unique 1..3
        self.assertEqual(sorted(ranks.values()), [1, 2, 3])

    def test_missing_optional_fields(self):
        """Products with None for optional fields should still score without error."""
        p = Product(
            name="Bare Minimum",
            url="https://example.com",
            source="test",
        )
        result = self._run(score_products([p]))
        self.assertEqual(len(result), 1)
        self.assertGreaterEqual(result[0]["value_score"], 0)
        self.assertLessEqual(result[0]["value_score"], 100)

    def test_category_weights_affect_scoring(self):
        """Different categories should produce different scores for the same product
        (because weights differ)."""
        products = [
            _make_product(
                name="P",
                original_price=1000,
                rating=4.0,
                review_count=100,
                warranty_months=24,
                shipping_cost=50,
            ),
        ]
        result_electronics = self._run(score_products(products, category="electronics"))
        result_grocery = self._run(score_products(products, category="grocery"))
        # Scores should differ because weights are different
        # (grocery has 0 warranty weight, electronics has 0.15)
        self.assertNotEqual(
            result_electronics[0]["value_score"],
            result_grocery[0]["value_score"],
        )

    def test_score_bounds(self):
        """All scores should be within 0-100."""
        products = [
            _make_product(name="X", original_price=100, rating=5.0, review_count=10000),
            _make_product(name="Y", original_price=99999, rating=0.1, review_count=1),
        ]
        result = self._run(score_products(products))
        for r in result:
            self.assertGreaterEqual(r["value_score"], 0)
            self.assertLessEqual(r["value_score"], 100)
            for key, val in r["breakdown"].items():
                self.assertGreaterEqual(val, 0, f"breakdown.{key} below 0")
                self.assertLessEqual(val, 100, f"breakdown.{key} above 100")


if __name__ == "__main__":
    unittest.main()
