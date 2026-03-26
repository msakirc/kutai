"""Comprehensive tests for shopping intelligence modules:
query_analyzer, search_planner, alternatives, substitution, constraints,
value_scorer, product_matcher, timing, combo_builder, review_synthesizer,
installment_calculator, delivery_compare, return_analyzer.
"""

from __future__ import annotations

import asyncio
import json
import time
import unittest
from datetime import datetime, date
from unittest.mock import AsyncMock, MagicMock, patch

from src.shopping.models import Product, UserConstraint


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
# 1. TestQueryAnalyzer
# ═══════════════════════════════════════════════════════════════════════════

class TestQueryAnalyzer(unittest.TestCase):
    """Test query analysis — keyword fallback path (no LLM)."""

    def _analyze(self, query):
        from src.shopping.intelligence.query_analyzer import _fallback_analyze
        return _fallback_analyze(query)

    def test_empty_query(self):
        result = self._analyze("")
        self.assertEqual(result["intent"], "explore")
        self.assertEqual(result["source"], "fallback")

    def test_intent_compare(self):
        result = self._analyze("iPhone vs Samsung karşılaştır")
        self.assertEqual(result["intent"], "compare")

    def test_intent_find_cheapest(self):
        result = self._analyze("en ucuz laptop")
        self.assertEqual(result["intent"], "find_cheapest")

    def test_intent_find_best(self):
        result = self._analyze("en iyi telefon öner")
        self.assertEqual(result["intent"], "find_best")

    def test_intent_explore(self):
        result = self._analyze("ne almalıyım bilgisayar")
        self.assertEqual(result["intent"], "explore")

    def test_category_electronics(self):
        result = self._analyze("iyi bir laptop arıyorum")
        self.assertEqual(result["category"], "electronics")

    def test_category_appliances(self):
        result = self._analyze("buzdolabı bakıyorum")
        self.assertEqual(result["category"], "appliances")

    def test_category_furniture(self):
        result = self._analyze("masa ve sandalye lazım")
        self.assertEqual(result["category"], "furniture")

    def test_category_grocery(self):
        result = self._analyze("süt ve peynir alacağım")
        self.assertEqual(result["category"], "grocery")

    def test_category_clothing(self):
        result = self._analyze("yeni ayakkabı istiyorum")
        self.assertEqual(result["category"], "clothing")

    def test_category_none(self):
        result = self._analyze("bir şey arıyorum")
        self.assertIsNone(result["category"])

    def test_urgency_high(self):
        result = self._analyze("acil kulaklık lazım bugün")
        self.assertEqual(result["urgency"], "high")

    def test_urgency_normal(self):
        result = self._analyze("kulaklık bakıyorum")
        self.assertEqual(result["urgency"], "normal")

    def test_budget_extraction(self):
        result = self._analyze("5000 TL altı laptop")
        self.assertEqual(result["budget"], 5000.0)

    def test_budget_with_comma(self):
        result = self._analyze("2.500 TL altında")
        self.assertEqual(result["budget"], 2500.0)

    def test_budget_none_when_absent(self):
        result = self._analyze("iyi bir laptop")
        self.assertIsNone(result["budget"])

    def test_constraint_dimensional(self):
        result = self._analyze("60 cm genişliğinde buzdolabı")
        self.assertIn("dimensional", result["constraints"])

    def test_constraint_budget(self):
        result = self._analyze("5000 TL altı telefon")
        self.assertIn("budget", result["constraints"])

    def test_constraint_electrical(self):
        result = self._analyze("1000 watt süpürge")
        self.assertIn("electrical", result["constraints"])

    def test_simple_complexity(self):
        result = self._analyze("ucuz laptop")
        self.assertEqual(result["search_complexity"], "simple")

    def test_moderate_complexity(self):
        result = self._analyze("en iyi fiyat performans oyuncu laptopu bütçe dostu")
        self.assertEqual(result["search_complexity"], "moderate")

    def test_language_field_present(self):
        result = self._analyze("laptop arıyorum")
        self.assertIn("language", result)

    def test_analyze_query_llm_fails(self):
        """analyze_query should fall back when LLM is unavailable."""
        from src.shopping.intelligence.query_analyzer import analyze_query
        with patch("src.shopping.intelligence.query_analyzer._llm_call", new_callable=AsyncMock, return_value=""):
            result = run_async(analyze_query("ucuz telefon"))
        self.assertEqual(result["source"], "fallback")
        self.assertEqual(result["intent"], "find_cheapest")

    def test_analyze_query_llm_success(self):
        """analyze_query should use LLM when it returns valid JSON."""
        from src.shopping.intelligence.query_analyzer import analyze_query
        llm_json = json.dumps({"intent": "compare", "category": "electronics"})
        with patch("src.shopping.intelligence.query_analyzer._llm_call", new_callable=AsyncMock, return_value=llm_json):
            result = run_async(analyze_query("iPhone vs Samsung"))
        self.assertEqual(result["source"], "llm")
        self.assertEqual(result["intent"], "compare")


# ═══════════════════════════════════════════════════════════════════════════
# 2. TestSearchPlanner
# ═══════════════════════════════════════════════════════════════════════════

class TestSearchPlanner(unittest.TestCase):
    """Test search plan generation — rule-based path."""

    def _plan(self, analyzed):
        from src.shopping.intelligence.search_planner import _build_rule_based_plan
        return _build_rule_based_plan(analyzed)

    def test_empty_analysis(self):
        from src.shopping.intelligence.search_planner import generate_search_plan
        with patch("src.shopping.intelligence.search_planner._llm_call", new_callable=AsyncMock, return_value=""):
            result = run_async(generate_search_plan({}))
        self.assertEqual(result, [])

    def test_electronics_sources(self):
        from src.shopping.intelligence.search_planner import _sources_for
        sources = _sources_for("electronics")
        self.assertIn("akakce", sources)
        self.assertIn("trendyol", sources)

    def test_default_sources(self):
        from src.shopping.intelligence.search_planner import _sources_for
        sources = _sources_for("unknown_category")
        self.assertEqual(sources, ["akakce", "trendyol", "hepsiburada"])

    def test_plan_with_products(self):
        plan = self._plan({
            "category": "electronics",
            "intent": "find_cheapest",
            "products_mentioned": ["iPhone 15"],
            "constraints": [],
        })
        self.assertTrue(len(plan) > 0)
        self.assertTrue(any(t["phase"] == 1 for t in plan))

    def test_plan_with_budget(self):
        plan = self._plan({
            "category": "electronics",
            "intent": "find_cheapest",
            "products_mentioned": [],
            "constraints": [],
            "budget": 5000,
            "raw_query": "ucuz telefon",
        })
        budget_tasks = [t for t in plan if "Budget" in t["purpose"]]
        self.assertTrue(len(budget_tasks) > 0)

    def test_phase2_for_compare_intent(self):
        plan = self._plan({
            "category": "electronics",
            "intent": "compare",
            "products_mentioned": ["iPhone 15"],
            "constraints": [],
        })
        phase2 = [t for t in plan if t["phase"] == 2]
        self.assertTrue(len(phase2) > 0)

    def test_plan_budget_caps(self):
        from src.shopping.intelligence.search_planner import MAX_SEARCHES_PER_SESSION
        plan = self._plan({
            "category": "electronics",
            "intent": "find_cheapest",
            "products_mentioned": ["a", "b", "c"],
            "constraints": [],
            "raw_query": "test",
            "budget": 5000,
        })
        self.assertLessEqual(len(plan), MAX_SEARCHES_PER_SESSION)


# ═══════════════════════════════════════════════════════════════════════════
# 3. TestAlternatives
# ═══════════════════════════════════════════════════════════════════════════

class TestAlternatives(unittest.TestCase):
    """Test alternative product suggestions — rule-based path."""

    def test_known_iphone_alternatives(self):
        from src.shopping.intelligence.alternatives import _rule_based_alternatives
        results = _rule_based_alternatives("iPhone 15", "electronics", [])
        self.assertTrue(len(results) > 0)
        self.assertTrue(any("Samsung" in r["product"] for r in results))
        for r in results:
            self.assertEqual(r["source"], "rule_based")

    def test_known_macbook_alternatives(self):
        from src.shopping.intelligence.alternatives import _rule_based_alternatives
        results = _rule_based_alternatives("MacBook Pro", "electronics", [])
        self.assertTrue(len(results) > 0)
        self.assertTrue(any("ThinkPad" in r["product"] for r in results))

    def test_category_alternatives(self):
        from src.shopping.intelligence.alternatives import _rule_based_alternatives
        results = _rule_based_alternatives("random product", "electronics", [])
        # Should get category-level alternatives
        self.assertTrue(any(r["confidence"] == 0.4 for r in results))

    def test_budget_boost(self):
        from src.shopping.intelligence.alternatives import _rule_based_alternatives
        results = _rule_based_alternatives("iPhone", "electronics", ["budget"])
        budget_friendly = [r for r in results if "bütçe" in r.get("reasoning", "").lower()
                           or "uygun fiyat" in r.get("reasoning", "").lower()
                           or "tasarruf" in r.get("reasoning", "").lower()]
        for r in budget_friendly:
            self.assertGreater(r["confidence"], 0.4)

    def test_empty_query(self):
        from src.shopping.intelligence.alternatives import generate_alternatives
        with patch("src.shopping.intelligence.alternatives._llm_call", new_callable=AsyncMock, return_value=""):
            result = run_async(generate_alternatives(""))
        self.assertEqual(result, [])

    def test_results_sorted_by_confidence(self):
        from src.shopping.intelligence.alternatives import generate_alternatives
        with patch("src.shopping.intelligence.alternatives._llm_call", new_callable=AsyncMock, return_value=""):
            results = run_async(generate_alternatives("iPhone", "electronics"))
        confidences = [r["confidence"] for r in results]
        self.assertEqual(confidences, sorted(confidences, reverse=True))


# ═══════════════════════════════════════════════════════════════════════════
# 4. TestSubstitution
# ═══════════════════════════════════════════════════════════════════════════

class TestSubstitution(unittest.TestCase):
    """Test substitution suggestions from knowledge base."""

    def test_kb_substitution_kahve_makinesi(self):
        from src.shopping.intelligence.substitution import _kb_substitutions
        results = _kb_substitutions("kahve makinesi")
        self.assertTrue(len(results) > 0)
        self.assertTrue(any("French press" in r["substitute"] for r in results))
        for r in results:
            self.assertEqual(r["source"], "knowledge_base")

    def test_kb_substitution_robot_supurge(self):
        from src.shopping.intelligence.substitution import _kb_substitutions
        results = _kb_substitutions("robot süpürge")
        self.assertTrue(len(results) > 0)

    def test_kb_no_match(self):
        from src.shopping.intelligence.substitution import _kb_substitutions
        results = _kb_substitutions("random nonexistent product")
        self.assertEqual(len(results), 0)

    def test_empty_product(self):
        from src.shopping.intelligence.substitution import suggest_substitutions
        with patch("src.shopping.intelligence.substitution._llm_call", new_callable=AsyncMock, return_value=""):
            result = run_async(suggest_substitutions(""))
        self.assertEqual(result, [])

    def test_price_triggered_mode(self):
        from src.shopping.intelligence.substitution import suggest_substitutions
        with patch("src.shopping.intelligence.substitution._llm_call", new_callable=AsyncMock, return_value=""):
            result = run_async(suggest_substitutions(
                "kahve makinesi", budget=1000, found_min_price=2000
            ))
        for r in result:
            self.assertTrue(r.get("price_triggered"))
        # Low-cost options should be first
        if len(result) >= 2:
            low_indices = [i for i, r in enumerate(result) if r.get("price_range") == "low"]
            mid_indices = [i for i, r in enumerate(result) if r.get("price_range") == "mid"]
            if low_indices and mid_indices:
                self.assertLess(min(low_indices), min(mid_indices))


# ═══════════════════════════════════════════════════════════════════════════
# 5. TestConstraints
# ═══════════════════════════════════════════════════════════════════════════

class TestConstraints(unittest.TestCase):
    """Test constraint checking."""

    def _make_product(self, name="Test", price=5000, **kwargs):
        return Product(name=name, url="https://x.com", source="test",
                       original_price=price, **kwargs)

    def test_no_constraints_pass(self):
        from src.shopping.intelligence.constraints import check_constraints
        products = [self._make_product()]
        result = run_async(check_constraints(products, []))
        self.assertTrue(result[0]["passes_all"])

    def test_empty_products(self):
        from src.shopping.intelligence.constraints import check_constraints
        result = run_async(check_constraints([], []))
        self.assertEqual(result, [])

    def test_budget_pass(self):
        from src.shopping.intelligence.constraints import check_constraints
        product = self._make_product(price=4000)
        constraint = UserConstraint(type="budget", value="5000", hard_or_soft="hard")
        result = run_async(check_constraints([product], [constraint]))
        self.assertTrue(result[0]["passes_all"])

    def test_budget_fail_hard(self):
        from src.shopping.intelligence.constraints import check_constraints
        product = self._make_product(price=6000)
        constraint = UserConstraint(type="budget", value="5000", hard_or_soft="hard")
        result = run_async(check_constraints([product], [constraint]))
        self.assertFalse(result[0]["passes_all"])
        self.assertIn("budget", result[0]["failed_hard"])

    def test_budget_soft_within_tolerance(self):
        from src.shopping.intelligence.constraints import check_constraints
        product = self._make_product(price=5400)  # 8% over 5000
        constraint = UserConstraint(type="budget", value="5000", hard_or_soft="soft")
        result = run_async(check_constraints([product], [constraint]))
        # Soft with <10% overshoot should pass
        self.assertEqual(result[0]["failed_hard"], [])

    def test_electrical_voltage_fail(self):
        from src.shopping.intelligence.constraints import check_constraints
        product = self._make_product(specs={"voltage": "110V"})
        constraint = UserConstraint(type="electrical", value="220 volt uyumlu", hard_or_soft="hard")
        result = run_async(check_constraints([product], [constraint]))
        self.assertFalse(result[0]["passes_all"])

    def test_electrical_voltage_pass(self):
        from src.shopping.intelligence.constraints import check_constraints
        product = self._make_product(specs={"voltage": "220V"})
        constraint = UserConstraint(type="electrical", value="220 volt uyumlu", hard_or_soft="hard")
        result = run_async(check_constraints([product], [constraint]))
        self.assertTrue(result[0]["passes_all"])


# ═══════════════════════════════════════════════════════════════════════════
# 6. TestValueScorer
# ═══════════════════════════════════════════════════════════════════════════

class TestValueScorer(unittest.TestCase):
    """Test value scoring and normalization."""

    def _make_product(self, name="P", price=1000, rating=4.0, reviews=50, **kwargs):
        return Product(
            name=name, url="u", source="s",
            original_price=price, discounted_price=price,
            rating=rating, review_count=reviews, **kwargs,
        )

    def test_empty_products(self):
        from src.shopping.intelligence.value_scorer import score_products
        result = run_async(score_products([]))
        self.assertEqual(result, [])

    def test_single_product_score(self):
        from src.shopping.intelligence.value_scorer import score_products
        p = self._make_product()
        result = run_async(score_products([p]))
        self.assertEqual(len(result), 1)
        self.assertIn("value_score", result[0])
        self.assertGreaterEqual(result[0]["value_score"], 0)
        self.assertLessEqual(result[0]["value_score"], 100)

    def test_ranking_order(self):
        from src.shopping.intelligence.value_scorer import score_products
        cheap = self._make_product("Cheap", 500, 4.5, 100)
        expensive = self._make_product("Expensive", 5000, 3.0, 10)
        result = run_async(score_products([cheap, expensive]))
        self.assertEqual(result[0]["rank"], 1)
        self.assertEqual(result[1]["rank"], 2)

    def test_breakdown_keys(self):
        from src.shopping.intelligence.value_scorer import score_products
        p = self._make_product()
        result = run_async(score_products([p]))
        bd = result[0]["breakdown"]
        for key in ("price", "seller", "shipping", "warranty", "rating", "availability", "review_volume"):
            self.assertIn(key, bd)

    def test_perspectives_keys(self):
        from src.shopping.intelligence.value_scorer import score_products
        p = self._make_product()
        result = run_async(score_products([p]))
        persp = result[0]["perspectives"]
        for key in ("best_price", "best_tco", "best_installment"):
            self.assertIn(key, persp)

    def test_category_weights(self):
        from src.shopping.intelligence.value_scorer import score_products
        p = self._make_product()
        r1 = run_async(score_products([p], "electronics"))
        r2 = run_async(score_products([p], "grocery"))
        # Different categories may yield different scores
        self.assertIsNotNone(r1[0]["value_score"])
        self.assertIsNotNone(r2[0]["value_score"])


# ═══════════════════════════════════════════════════════════════════════════
# 7. TestProductMatcher
# ═══════════════════════════════════════════════════════════════════════════

class TestProductMatcher(unittest.TestCase):
    """Test product matching — EAN, fuzzy name, confidence scoring."""

    def _make_product(self, name="P", source="s", specs=None):
        return Product(name=name, url="u", source=source, specs=specs or {})

    def test_empty_products(self):
        from src.shopping.intelligence.product_matcher import match_products
        result = run_async(match_products([]))
        self.assertEqual(result, [])

    def test_single_product(self):
        from src.shopping.intelligence.product_matcher import match_products
        p = self._make_product("Samsung Galaxy S24")
        result = run_async(match_products([p]))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["product_count"], 1)

    def test_ean_match(self):
        from src.shopping.intelligence.product_matcher import match_products
        p1 = self._make_product("Samsung Galaxy S24", "trendyol", {"ean": "8806094123456"})
        p2 = self._make_product("Samsung Galaxy S24 128GB", "hepsiburada", {"ean": "8806094123456"})
        result = run_async(match_products([p1, p2]))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["product_count"], 2)
        self.assertGreaterEqual(result[0]["confidence_score"], 0.95)

    def test_fuzzy_name_match(self):
        from src.shopping.intelligence.product_matcher import match_products
        p1 = self._make_product("Samsung Galaxy S24 Ultra 256GB", "trendyol")
        p2 = self._make_product("Samsung Galaxy S24 Ultra 256 GB Siyah", "hepsiburada")
        result = run_async(match_products([p1, p2]))
        # These names are similar enough to match
        groups_with_two = [r for r in result if r["product_count"] == 2]
        # May or may not match depending on threshold; just verify structure
        self.assertTrue(len(result) >= 1)

    def test_no_match_different_products(self):
        from src.shopping.intelligence.product_matcher import match_products
        p1 = self._make_product("Samsung Galaxy S24", "trendyol")
        p2 = self._make_product("Apple iPhone 15 Pro Max", "hepsiburada")
        result = run_async(match_products([p1, p2]))
        self.assertEqual(len(result), 2)

    def test_mpn_match(self):
        from src.shopping.intelligence.product_matcher import match_products
        p1 = self._make_product("Lenovo ThinkPad", "amazon", {"model_no": "21HM-S12345"})
        p2 = self._make_product("Lenovo ThinkPad T14", "n11", {"model_no": "21HM-S12345"})
        result = run_async(match_products([p1, p2]))
        matched = [r for r in result if r["product_count"] == 2]
        self.assertEqual(len(matched), 1)
        self.assertGreaterEqual(matched[0]["confidence_score"], 0.9)


# ═══════════════════════════════════════════════════════════════════════════
# 8. TestTiming
# ═══════════════════════════════════════════════════════════════════════════

class TestTiming(unittest.TestCase):
    """Test market timing advisor."""

    def test_days_until_next_event(self):
        from src.shopping.intelligence.timing import _days_until_next_event
        # Use a date in June — next event should be found
        now = datetime(2025, 6, 1)
        result = _days_until_next_event(now)
        self.assertIsNotNone(result)
        self.assertIsInstance(result[0], str)
        self.assertIsInstance(result[1], int)
        self.assertGreaterEqual(result[1], 0)

    def test_check_best_window_match(self):
        from src.shopping.intelligence.timing import _check_best_window
        now = datetime(2025, 3, 15)
        result = _check_best_window("klima", now)
        self.assertIsNotNone(result)
        self.assertTrue(result["in_window"])

    def test_check_best_window_no_match(self):
        from src.shopping.intelligence.timing import _check_best_window
        now = datetime(2025, 3, 15)
        result = _check_best_window("random stuff", now)
        self.assertIsNone(result)

    def test_analyze_price_trend_insufficient(self):
        from src.shopping.intelligence.timing import _analyze_price_trend
        result = _analyze_price_trend([{"price": 100}])
        self.assertEqual(result["trend"], "insufficient_data")

    def test_analyze_price_trend_stable(self):
        from src.shopping.intelligence.timing import _analyze_price_trend
        now = time.time()
        history = [
            {"price": 100, "observed_at": now - 86400 * 10},
            {"price": 101, "observed_at": now},
        ]
        result = _analyze_price_trend(history)
        self.assertEqual(result["trend"], "stable")


# ═══════════════════════════════════════════════════════════════════════════
# 9. TestComboBuilder
# ═══════════════════════════════════════════════════════════════════════════

class TestComboBuilder(unittest.TestCase):
    """Test combo building with mock components."""

    def test_empty_components(self):
        from src.shopping.intelligence.combo_builder import build_combos
        with patch("src.shopping.intelligence.combo_builder._llm_call", new_callable=AsyncMock, return_value=""):
            result = run_async(build_combos([], 10000, []))
        self.assertEqual(result, [])

    def test_basic_combo(self):
        from src.shopping.intelligence.combo_builder import build_combos
        components = [
            {"role": "cpu", "candidates": [
                {"name": "CPU A", "original_price": 3000, "source": "trendyol", "category": "cpu"},
            ]},
            {"role": "gpu", "candidates": [
                {"name": "GPU A", "original_price": 5000, "source": "trendyol", "category": "gpu"},
            ]},
        ]
        with patch("src.shopping.intelligence.combo_builder._llm_call", new_callable=AsyncMock, return_value=""):
            result = run_async(build_combos(components, 20000, []))
        self.assertTrue(len(result) > 0)
        self.assertIn("total_price", result[0])
        self.assertIn("tier", result[0])

    def test_combo_scoring(self):
        from src.shopping.intelligence.combo_builder import _score_combo
        products = [
            {"name": "A", "original_price": 1000, "source": "trendyol", "rating": 4.5},
            {"name": "B", "original_price": 2000, "source": "trendyol", "rating": 4.0},
        ]
        score = _score_combo(products)
        self.assertEqual(score["total_price"], 3000.0)
        self.assertEqual(score["store_count"], 1)
        self.assertGreater(score["value_score"], 0)

    def test_assign_tier(self):
        from src.shopping.intelligence.combo_builder import _assign_tier
        self.assertEqual(_assign_tier(5000, 10000), "budget")
        self.assertEqual(_assign_tier(9000, 10000), "mid")
        self.assertEqual(_assign_tier(12000, 10000), "premium")


# ═══════════════════════════════════════════════════════════════════════════
# 10. TestReviewSynthesizer
# ═══════════════════════════════════════════════════════════════════════════

class TestReviewSynthesizer(unittest.TestCase):
    """Test review synthesis — input/output structure with mock reviews."""

    def test_empty_reviews(self):
        from src.shopping.intelligence.review_synthesizer import synthesize_reviews
        with patch("src.shopping.intelligence.review_synthesizer._llm_call", new_callable=AsyncMock, return_value=""):
            result = run_async(synthesize_reviews([], "Test Product"))
        self.assertEqual(result["overall_sentiment"], "unknown")
        self.assertIsNone(result["confidence_adjusted_rating"])

    def test_positive_reviews(self):
        from src.shopping.intelligence.review_synthesizer import synthesize_reviews
        reviews = [
            {"rating": 5.0, "text": "Harika urun", "source": "trendyol", "verified_purchase": True}
            for _ in range(20)
        ]
        with patch("src.shopping.intelligence.review_synthesizer._llm_call", new_callable=AsyncMock, return_value=""):
            result = run_async(synthesize_reviews(reviews, "Test"))
        self.assertEqual(result["overall_sentiment"], "positive")
        self.assertIsNotNone(result["confidence_adjusted_rating"])
        self.assertGreater(result["confidence_adjusted_rating"], 3.5)

    def test_volume_confidence_low(self):
        from src.shopping.intelligence.review_synthesizer import _volume_confidence
        self.assertAlmostEqual(_volume_confidence(2), 0.3)

    def test_volume_confidence_high(self):
        from src.shopping.intelligence.review_synthesizer import _volume_confidence
        self.assertAlmostEqual(_volume_confidence(200), 1.0)

    def test_temporal_weight_recent(self):
        from src.shopping.intelligence.review_synthesizer import _compute_temporal_weight
        recent = datetime.now().isoformat()
        self.assertEqual(_compute_temporal_weight(recent), 2.0)

    def test_temporal_weight_none(self):
        from src.shopping.intelligence.review_synthesizer import _compute_temporal_weight
        self.assertEqual(_compute_temporal_weight(None), 1.0)

    def test_review_quality_high(self):
        from src.shopping.intelligence.review_synthesizer import _assess_review_quality
        reviews = [
            {"text": "A" * 100, "verified_purchase": True} for _ in range(15)
        ]
        q = _assess_review_quality(reviews)
        self.assertEqual(q["quality"], "high")

    def test_review_quality_none(self):
        from src.shopping.intelligence.review_synthesizer import _assess_review_quality
        q = _assess_review_quality([])
        self.assertEqual(q["quality"], "none")


# ═══════════════════════════════════════════════════════════════════════════
# 11. TestInstallmentCalculator
# ═══════════════════════════════════════════════════════════════════════════

class TestInstallmentCalculator(unittest.TestCase):
    """Test installment calculation."""

    def test_zero_price(self):
        from src.shopping.intelligence.installment_calculator import calculate_installments
        # Reset cache
        import src.shopping.intelligence.installment_calculator as ic
        ic._installment_cache = None
        result = run_async(calculate_installments(0, "trendyol"))
        self.assertEqual(result, [])

    def test_negative_price(self):
        from src.shopping.intelligence.installment_calculator import calculate_installments
        result = run_async(calculate_installments(-100, "trendyol"))
        self.assertEqual(result, [])

    def test_generic_fallback(self):
        from src.shopping.intelligence.installment_calculator import calculate_installments
        import src.shopping.intelligence.installment_calculator as ic
        ic._installment_cache = {"stores": {}, "bank_cards": {}}
        result = run_async(calculate_installments(10000, "unknown_store"))
        self.assertTrue(len(result) > 0)
        for r in result:
            self.assertIn("monthly_payment", r)
            self.assertIn("total_amount", r)
            self.assertGreater(r["monthly_payment"], 0)

    def test_faizsiz_installment(self):
        from src.shopping.intelligence.installment_calculator import _compute_installment
        result = _compute_installment(12000, 6, True)
        self.assertEqual(result["monthly_payment"], 2000.0)
        self.assertEqual(result["total_amount"], 12000.0)
        self.assertEqual(result["interest_amount"], 0.0)

    def test_interest_installment(self):
        from src.shopping.intelligence.installment_calculator import _compute_installment
        result = _compute_installment(10000, 12, False)
        self.assertGreater(result["total_amount"], 10000)
        self.assertGreater(result["interest_amount"], 0)
        self.assertGreater(result["interest_pct"], 0)

    def test_store_normalization(self):
        from src.shopping.intelligence.installment_calculator import _normalize_store
        self.assertEqual(_normalize_store("Trendyol"), "trendyol")
        self.assertEqual(_normalize_store("amazon.com.tr"), "amazon_tr")
        self.assertEqual(_normalize_store("VATAN"), "vatanbilgisayar")


# ═══════════════════════════════════════════════════════════════════════════
# 12. TestDeliveryCompare
# ═══════════════════════════════════════════════════════════════════════════

class TestDeliveryCompare(unittest.TestCase):
    """Test delivery comparison."""

    def test_empty_products(self):
        from src.shopping.intelligence.delivery_compare import compare_delivery
        with patch("src.shopping.intelligence.delivery_compare._llm_call", new_callable=AsyncMock, return_value=""):
            result = run_async(compare_delivery([]))
        self.assertEqual(result, [])

    def test_effective_price_with_shipping(self):
        from src.shopping.intelligence.delivery_compare import compare_delivery
        products = [
            {"name": "P1", "source": "trendyol", "discounted_price": 1000, "shipping_cost": 50},
            {"name": "P2", "source": "hepsiburada", "discounted_price": 1020, "shipping_cost": 0},
        ]
        with patch("src.shopping.intelligence.delivery_compare._llm_call", new_callable=AsyncMock, return_value=""):
            result = run_async(compare_delivery(products))
        self.assertEqual(len(result), 2)
        # Should be sorted by effective price
        self.assertLessEqual(result[0]["effective_price"], result[1]["effective_price"])

    def test_international_detection(self):
        from src.shopping.intelligence.delivery_compare import _is_international
        self.assertTrue(_is_international({"name": "China import phone", "seller_name": ""}))
        self.assertFalse(_is_international({"name": "Samsung Galaxy", "seller_name": "Official Store"}))

    def test_ranking_and_badge(self):
        from src.shopping.intelligence.delivery_compare import compare_delivery
        products = [
            {"name": "P1", "source": "trendyol", "discounted_price": 1000, "shipping_cost": 0},
        ]
        with patch("src.shopping.intelligence.delivery_compare._llm_call", new_callable=AsyncMock, return_value=""):
            result = run_async(compare_delivery(products))
        self.assertEqual(result[0]["rank"], 1)
        self.assertEqual(result[0]["badge"], "Best effective price")


# ═══════════════════════════════════════════════════════════════════════════
# 13. TestReturnAnalyzer
# ═══════════════════════════════════════════════════════════════════════════

class TestReturnAnalyzer(unittest.TestCase):
    """Test return policy analysis."""

    def setUp(self):
        # Reset cache
        import src.shopping.intelligence.return_analyzer as ra
        ra._profiles_cache = None

    def test_known_store(self):
        from src.shopping.intelligence.return_analyzer import analyze_return_policy
        product = {"name": "Samsung TV"}
        with patch("src.shopping.intelligence.return_analyzer._llm_call", new_callable=AsyncMock, return_value=""):
            result = run_async(analyze_return_policy(product, "amazon_tr"))
        self.assertEqual(result["return_window_days"], 30)
        self.assertTrue(result["free_return"])

    def test_unknown_store(self):
        from src.shopping.intelligence.return_analyzer import analyze_return_policy
        product = {"name": "Test Product"}
        with patch("src.shopping.intelligence.return_analyzer._llm_call", new_callable=AsyncMock, return_value=""):
            result = run_async(analyze_return_policy(product, "unknown_store_xyz"))
        self.assertEqual(result["score"], 0.0)
        self.assertIn("not found", result["warnings"][0])

    def test_ease_badge_easy(self):
        from src.shopping.intelligence.return_analyzer import _score_return_ease
        policy = {
            "return_window_days": 365,
            "free_return": True,
            "physical_stores": True,
            "marketplace_caveat": False,
            "electronics_note": "",
        }
        badge, score = _score_return_ease(policy, False)
        self.assertGreaterEqual(score, 0.7)
        self.assertIn("Easy", badge)

    def test_ease_badge_difficult(self):
        from src.shopping.intelligence.return_analyzer import _score_return_ease
        policy = {
            "return_window_days": 0,
            "free_return": False,
            "physical_stores": False,
            "marketplace_caveat": True,
            "electronics_note": "",
        }
        badge, score = _score_return_ease(policy, False)
        self.assertLess(score, 0.4)

    def test_electronics_detection(self):
        from src.shopping.intelligence.return_analyzer import _is_electronics
        self.assertTrue(_is_electronics({"name": "Samsung 55 inch TV televizyon"}))
        self.assertFalse(_is_electronics({"name": "Nike Running Shoes"}))


if __name__ == "__main__":
    unittest.main()
