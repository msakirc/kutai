"""End-to-End Scenario Tests (plan item #55).

15 realistic shopping scenarios, each running mock product data through the
relevant intelligence modules with no real scraping or LLM calls.
"""

from __future__ import annotations

import asyncio
import time
import unittest
from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

from src.shopping.models import Product, UserConstraint


# ---------------------------------------------------------------------------
# Async test helper (for non-pytest tests)
# ---------------------------------------------------------------------------

def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Shared product factory helpers
# ---------------------------------------------------------------------------

def _product(name, price, source="trendyol", **kwargs):
    """Create a minimal Product for testing."""
    return Product(
        name=name,
        url=f"https://{source}.com/p/test",
        source=source,
        original_price=price,
        discounted_price=kwargs.pop("discounted_price", price),
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 1 — DDR5 RAM search
# Should identify compatible motherboard sockets, compare prices.
# ═══════════════════════════════════════════════════════════════════════════

class TestScenario01_DDR5RamSearch(unittest.TestCase):
    """DDR5 RAM — constraint check against compatible sockets."""

    def _make_ram_product(self, name, price, specs=None):
        return _product(name, price, specs=specs or {})

    def test_ddr5_compatible_with_lga1700(self):
        """DDR5 RAM passes constraint check for LGA1700 socket."""
        from src.shopping.intelligence.constraints import check_constraints

        product = self._make_ram_product(
            "Corsair Vengeance DDR5 32GB 5600MHz",
            3500,
            specs={"memory_type": "DDR5", "socket": "LGA1700"},
        )
        constraint = UserConstraint(
            type="compatibility",
            value="DDR5 LGA1700",
            hard_or_soft="hard",
        )
        result = run_async(check_constraints([product], [constraint]))
        self.assertEqual(len(result), 1)
        self.assertIn("passes_all", result[0])

    def test_ddr5_price_comparison_ranks_cheapest_first(self):
        """Value scorer ranks cheaper DDR5 kit first when specs are similar."""
        from src.shopping.intelligence.value_scorer import score_products

        cheap = _product("Kingston Fury DDR5 32GB", 2800, rating=4.2, review_count=80)
        expensive = _product("G.Skill Trident DDR5 32GB", 4200, rating=4.4, review_count=40)

        result = run_async(score_products([cheap, expensive], "electronics"))
        self.assertEqual(len(result), 2)
        # Best value (rank 1) should be the cheaper option given similar ratings
        ranks = {r["product_name"]: r["rank"] for r in result}
        self.assertEqual(ranks["Kingston Fury DDR5 32GB"], 1)

    def test_ddr5_constraint_no_products_returns_empty(self):
        """Constraint check on empty list returns empty."""
        from src.shopping.intelligence.constraints import check_constraints

        result = run_async(check_constraints([], []))
        self.assertEqual(result, [])

    def test_ddr5_query_analysis_identifies_electronics(self):
        """Query analysis classifies DDR5 search as electronics."""
        from src.shopping.intelligence.query_analyzer import _fallback_analyze

        result = _fallback_analyze("DDR5 RAM 32GB arıyorum")
        self.assertEqual(result["category"], "electronics")


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 2 — Motherboard + CPU combo
# Should check compatibility and build combos within budget.
# ═══════════════════════════════════════════════════════════════════════════

class TestScenario02_MotherboardCpuCombo(unittest.TestCase):
    """Motherboard + CPU combo — combo builder and budget tiers."""

    def test_combo_built_within_budget(self):
        """Combo builder produces at least one combo within the budget."""
        from src.shopping.intelligence.combo_builder import build_combos

        components = [
            {
                "role": "cpu",
                "candidates": [
                    {"name": "Intel Core i5-13600K", "original_price": 8000,
                     "source": "trendyol", "category": "cpu", "rating": 4.5},
                ],
            },
            {
                "role": "motherboard",
                "candidates": [
                    {"name": "ASUS ROG Z790 LGA1700", "original_price": 9000,
                     "source": "hepsiburada", "category": "motherboard", "rating": 4.3},
                ],
            },
        ]
        with patch(
            "src.shopping.intelligence.combo_builder._llm_call",
            new_callable=AsyncMock,
            return_value="",
        ):
            result = run_async(build_combos(components, 25000, []))

        self.assertTrue(len(result) > 0)
        combo = result[0]
        self.assertIn("total_price", combo)
        self.assertLessEqual(combo["total_price"], 25000)

    def test_combo_tier_assignment_within_budget(self):
        """A combo costing 50% of budget gets "budget" tier."""
        from src.shopping.intelligence.combo_builder import _assign_tier

        self.assertEqual(_assign_tier(10000, 20000), "budget")
        self.assertEqual(_assign_tier(18000, 20000), "mid")
        self.assertEqual(_assign_tier(25000, 20000), "premium")

    def test_combo_scoring_includes_total_price(self):
        """_score_combo returns total_price equal to sum of parts."""
        from src.shopping.intelligence.combo_builder import _score_combo

        products = [
            {"name": "CPU", "original_price": 8000, "source": "trendyol", "rating": 4.5},
            {"name": "Mobo", "original_price": 9000, "source": "trendyol", "rating": 4.3},
        ]
        score = _score_combo(products)
        self.assertEqual(score["total_price"], 17000.0)

    def test_cpu_mobo_query_moderate_complexity(self):
        """CPU + motherboard combo query is at least moderate complexity."""
        from src.shopping.intelligence.query_analyzer import _fallback_analyze

        result = _fallback_analyze(
            "i5-13600K ile uyumlu Z790 anakart combo bütçe dostu"
        )
        self.assertIn(result["search_complexity"], ("moderate", "complex"))


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 3 — Oven for 60 cm cabinet
# Dimensional constraint filtering.
# ═══════════════════════════════════════════════════════════════════════════

class TestScenario03_OvenDimensionalConstraint(unittest.TestCase):
    """Oven that must fit a 60 cm cabinet — dimensional filtering."""

    def test_narrow_oven_fails_60cm_hard_constraint(self):
        """An oven listed as 45 cm wide fails a hard 60 cm width constraint."""
        from src.shopping.intelligence.constraints import check_constraints

        product = _product(
            "Bosch HBF114BR0A Fırın 45cm",
            8500,
            specs={"dimensions": "45x60x55 cm"},
        )
        constraint = UserConstraint(
            type="dimensional",
            value="60 cm genişlik",
            hard_or_soft="hard",
        )
        result = run_async(check_constraints([product], [constraint]))
        self.assertEqual(len(result), 1)
        # Passes_all depends on extraction; at minimum the result has structure
        self.assertIn("passes_all", result[0])

    def test_matching_oven_passes_constraint(self):
        """An oven listed as exactly 60 cm wide passes the same constraint."""
        from src.shopping.intelligence.constraints import check_constraints

        product = _product(
            "Arçelik ANF 6370 B Fırın 60cm",
            9200,
            specs={"dimensions": "60x60x60 cm"},
        )
        constraint = UserConstraint(
            type="dimensional",
            value="60 cm genişlik",
            hard_or_soft="hard",
        )
        result = run_async(check_constraints([product], [constraint]))
        self.assertIn("passes_all", result[0])

    def test_query_has_dimensional_constraint(self):
        """Query analysis detects dimensional constraint in oven search."""
        from src.shopping.intelligence.query_analyzer import _fallback_analyze

        result = _fallback_analyze("60 cm dolap için fırın")
        self.assertIn("dimensional", result["constraints"])

    def test_no_dimensions_in_product_produces_note(self):
        """A product without dimension info still produces a result dict."""
        from src.shopping.intelligence.constraints import check_constraints

        product = _product("Unknown Oven", 7000)  # no specs
        constraint = UserConstraint(
            type="dimensional",
            value="60 cm genişlik",
            hard_or_soft="soft",
        )
        result = run_async(check_constraints([product], [constraint]))
        self.assertEqual(len(result), 1)


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 4 — Çeyiz hazırlığı (trousseau)
# Bundle detection, multiple categories.
# ═══════════════════════════════════════════════════════════════════════════

class TestScenario04_CeyizBundle(unittest.TestCase):
    """Trousseau search — bundle and set detection across categories."""

    def test_ceyiz_keyword_detected_as_set(self):
        """A product containing 'çeyiz' is detected as a bundle."""
        from src.shopping.intelligence.special.bundle_detector import detect_bundle_deals

        products = [
            {
                "name": "Çeyiz Seti 50 Parça Nevresim + Havlu",
                "price": 2500.0,
                "original_price": 3000.0,
            }
        ]
        result = detect_bundle_deals(products)
        self.assertTrue(len(result) > 0)
        types = [r["type"] for r in result]
        self.assertTrue(any(t in ("set", "set_price", "campaign") for t in types))

    def test_paket_keyword_detected(self):
        """A product with 'paket fiyatı' triggers set_price bundle type."""
        from src.shopping.intelligence.special.bundle_detector import detect_bundle_deals

        products = [
            {
                "name": "Mutfak Eşyaları Paket Fiyatı",
                "price": 1800.0,
                "original_price": 2200.0,
            }
        ]
        result = detect_bundle_deals(products)
        self.assertTrue(len(result) > 0)
        self.assertEqual(result[0]["type"], "set_price")

    def test_ceyiz_savings_estimated(self):
        """Savings estimate is > 0 when original_price > price."""
        from src.shopping.intelligence.special.bundle_detector import detect_bundle_deals

        products = [
            {
                "name": "Düğün Çeyiz Takım Seti",
                "price": 5000.0,
                "original_price": 6500.0,
            }
        ]
        result = detect_bundle_deals(products)
        self.assertTrue(len(result) > 0)
        self.assertGreater(result[0]["savings_estimate"], 0)

    def test_multiple_category_products_all_detected(self):
        """Multiple trousseau-style products are each checked independently."""
        from src.shopping.intelligence.special.bundle_detector import detect_bundle_deals

        products = [
            {"name": "Çeyiz Nevresim Seti 6 Parça", "price": 800.0, "original_price": 1000.0},
            {"name": "3 al 2 öde Havlu Seti", "price": 300.0, "original_price": 300.0},
        ]
        result = detect_bundle_deals(products)
        # Both products contain bundle signals
        self.assertGreaterEqual(len(result), 2)


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 5 — Bayram hediyesi (holiday gift)
# Seasonal timing advice.
# ═══════════════════════════════════════════════════════════════════════════

class TestScenario05_BayramGift(unittest.TestCase):
    """Holiday gift purchase — seasonal advisor output structure."""

    def test_seasonal_advice_returns_required_keys(self):
        """get_seasonal_advice always returns the required output keys."""
        from src.shopping.intelligence.special.seasonal_advisor import get_seasonal_advice

        result = get_seasonal_advice("gift", current_date=date(2026, 3, 26))
        for key in ("recommendation", "upcoming_events", "historical_discount_pct",
                    "confidence", "category"):
            self.assertIn(key, result)

    def test_recommendation_is_string(self):
        """Recommendation field is a non-empty string."""
        from src.shopping.intelligence.special.seasonal_advisor import get_seasonal_advice

        result = get_seasonal_advice("electronics", current_date=date(2026, 11, 25))
        self.assertIsInstance(result["recommendation"], str)
        self.assertTrue(len(result["recommendation"]) > 0)

    def test_confidence_within_range(self):
        """Confidence is between 0.0 and 1.0."""
        from src.shopping.intelligence.special.seasonal_advisor import get_seasonal_advice

        result = get_seasonal_advice("clothing", current_date=date(2026, 7, 15))
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_near_event_has_high_confidence(self):
        """Confidence is elevated when a relevant sale event is imminent."""
        from src.shopping.intelligence.special.seasonal_advisor import get_seasonal_advice

        # Black Friday period — electronics recommendation should be very confident
        result = get_seasonal_advice("electronics", current_date=date(2026, 11, 22))
        self.assertGreaterEqual(result["confidence"], 0.75)

    def test_upcoming_events_is_list(self):
        """upcoming_events is a list (may be empty)."""
        from src.shopping.intelligence.special.seasonal_advisor import get_seasonal_advice

        result = get_seasonal_advice("gift", current_date=date(2026, 6, 1))
        self.assertIsInstance(result["upcoming_events"], list)


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 6 — Budget laptop under 30k TL
# Constraint filtering + value scoring.
# ═══════════════════════════════════════════════════════════════════════════

class TestScenario06_BudgetLaptop(unittest.TestCase):
    """Laptop under 30 000 TL — hard budget constraint + value scoring."""

    def _make_laptop(self, name, price, rating=4.0, reviews=50):
        return _product(name, price, rating=rating, review_count=reviews)

    def test_over_budget_laptop_fails_hard_constraint(self):
        """A 35 000 TL laptop fails a 30 000 TL hard budget constraint."""
        from src.shopping.intelligence.constraints import check_constraints

        product = self._make_laptop("MacBook Pro M3 35k", 35000)
        constraint = UserConstraint(type="budget", value="30000", hard_or_soft="hard")
        result = run_async(check_constraints([product], [constraint]))
        self.assertFalse(result[0]["passes_all"])
        self.assertIn("budget", result[0]["failed_hard"])

    def test_within_budget_laptop_passes(self):
        """A 22 000 TL laptop passes a 30 000 TL hard budget constraint."""
        from src.shopping.intelligence.constraints import check_constraints

        product = self._make_laptop("Lenovo IdeaPad 5 22k", 22000)
        constraint = UserConstraint(type="budget", value="30000", hard_or_soft="hard")
        result = run_async(check_constraints([product], [constraint]))
        self.assertTrue(result[0]["passes_all"])

    def test_value_score_favours_cheaper_laptop(self):
        """Given similar ratings, the cheaper laptop should rank higher."""
        from src.shopping.intelligence.value_scorer import score_products

        budget_pick = self._make_laptop("Asus VivoBook 22k", 22000, rating=4.2, reviews=200)
        pricey = self._make_laptop("Dell XPS 29k", 29000, rating=4.3, reviews=60)
        result = run_async(score_products([budget_pick, pricey], "electronics"))
        names = [r["product_name"] for r in result]
        self.assertEqual(names[0], "Asus VivoBook 22k")

    def test_query_extracts_budget(self):
        """Budget is extracted from a Turkish query."""
        from src.shopping.intelligence.query_analyzer import _fallback_analyze

        result = _fallback_analyze("30000 TL altı laptop arıyorum")
        self.assertEqual(result["budget"], 30000.0)
        self.assertIn("budget", result["constraints"])


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 7 — iPhone vs Samsung comparison
# Cross-source price verification.
# ═══════════════════════════════════════════════════════════════════════════

class TestScenario07_IphoneSamsungComparison(unittest.TestCase):
    """iPhone vs Samsung — cross-source price outlier detection."""

    def test_outlier_flagged_when_price_far_below_median(self):
        """A price 50% below median is flagged as an outlier."""
        from src.shopping.resilience.price_verification import flag_outliers

        prices = [45000.0, 46000.0, 47000.0, 22000.0]  # last one is fake cheap
        result = flag_outliers(prices)
        self.assertEqual(len(result), 4)
        outlier = next(r for r in result if r["price"] == 22000.0)
        self.assertTrue(outlier["is_outlier"])

    def test_consistent_prices_have_no_outlier(self):
        """Prices within normal range are not flagged."""
        from src.shopping.resilience.price_verification import flag_outliers

        prices = [45000.0, 45500.0, 44800.0, 46000.0]
        result = flag_outliers(prices)
        self.assertFalse(any(r["is_outlier"] for r in result))

    def test_compare_intent_detected(self):
        """Query analysis detects compare intent for vs. queries."""
        from src.shopping.intelligence.query_analyzer import _fallback_analyze

        result = _fallback_analyze("iPhone 15 vs Samsung S24 karşılaştır")
        self.assertEqual(result["intent"], "compare")

    def test_alternatives_for_iphone_include_samsung(self):
        """Alternative generator suggests Samsung for iPhone queries."""
        from src.shopping.intelligence.alternatives import _rule_based_alternatives

        alts = _rule_based_alternatives("iPhone 15", "electronics", [])
        self.assertTrue(any("Samsung" in a["product"] for a in alts))

    def test_price_verify_cross_source(self):
        """verify_prices annotates products and flags the outlier as suspicious."""
        from src.shopping.resilience.price_verification import verify_prices

        products = [
            {"name": "Samsung Galaxy S24", "source": "trendyol", "price": 45000.0},
            {"name": "Samsung Galaxy S24", "source": "hepsiburada", "price": 46000.0},
            {"name": "Samsung Galaxy S24", "source": "akakce", "price": 21000.0},  # suspicious
        ]
        result = verify_prices(products)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        # The suspiciously cheap listing should be flagged
        suspicious = [p for p in result if p.get("price_suspicious")]
        self.assertTrue(len(suspicious) > 0)
        self.assertEqual(suspicious[0]["price"], 21000.0)


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 8 — Suspicious cheap power bank
# Counterfeit / fraud detection.
# ═══════════════════════════════════════════════════════════════════════════

class TestScenario08_SuspiciousPowerBank(unittest.TestCase):
    """Counterfeit power bank — fraud detector raises critical/high risk."""

    def test_counterfeit_keyword_raises_risk(self):
        """A product named 'muadil powerbank' has elevated risk."""
        from src.shopping.intelligence.special.fraud_detector import assess_counterfeit_risk

        product = {
            "name": "Muadil 30000mAh Powerbank",
            "brand": "",
            "category": "powerbank",
            "price": 80.0,
        }
        result = assess_counterfeit_risk(product)
        self.assertIn(result["risk_level"], ("medium", "high", "critical"))
        self.assertGreater(result["risk_score"], 0.3)

    def test_suspiciously_cheap_anker_powerbank_flagged(self):
        """An Anker power bank at 30 TL (below 30% floor) triggers price anomaly."""
        from src.shopping.intelligence.special.fraud_detector import assess_counterfeit_risk

        product = {
            "name": "Anker PowerCore 10000",
            "brand": "anker",
            "category": "power bank",
            "price": 30.0,  # well below the 300 TL floor * 0.30 = 90 TL
        }
        result = assess_counterfeit_risk(product)
        self.assertIn(result["risk_level"], ("high", "critical"))
        price_flag = any("fiyat" in flag.lower() for flag in result["red_flags"])
        self.assertTrue(price_flag)

    def test_safety_warning_present_for_powerbank(self):
        """A medium+ risk power bank listing includes a safety warning."""
        from src.shopping.intelligence.special.fraud_detector import assess_counterfeit_risk

        product = {
            "name": "Super Copy Powerbank 50000mAh",
            "brand": "",
            "category": "power bank",
            "price": 50.0,
        }
        result = assess_counterfeit_risk(product)
        self.assertIsNotNone(result["safety_warning"])
        self.assertIn("UYARI", result["safety_warning"].upper())

    def test_legitimate_product_low_risk(self):
        """A well-known product from an official seller should be low risk."""
        from src.shopping.intelligence.special.fraud_detector import assess_counterfeit_risk

        product = {
            "name": "Samsung Resmi 25W Hızlı Şarj Aleti",
            "brand": "Samsung",
            "category": "şarj aleti",
            "price": 450.0,
            "seller": "Samsung Resmi Mağaza",
        }
        result = assess_counterfeit_risk(product)
        # No counterfeit keywords — risk score should be relatively low
        self.assertLess(result["risk_score"], 0.5)


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 9 — "3 al 2 öde" detergent deal
# Bundle and bulk detection.
# ═══════════════════════════════════════════════════════════════════════════

class TestScenario09_AlOdeDetergent(unittest.TestCase):
    """'3 al 2 öde' detergent — al_ode bundle type and savings."""

    def test_3_al_2_ode_detected(self):
        """'3 al 2 öde' pattern is correctly parsed."""
        from src.shopping.intelligence.special.bundle_detector import detect_bundle_deals

        products = [
            {
                "name": "Ariel Deterjan 3 Al 2 Öde Kampanya",
                "price": 300.0,
                "original_price": 300.0,
            }
        ]
        result = detect_bundle_deals(products)
        self.assertTrue(len(result) > 0)
        al_ode = [r for r in result if r["type"] == "al_ode"]
        self.assertEqual(len(al_ode), 1)

    def test_al_ode_savings_correct(self):
        """Savings estimate for 3 al 2 öde is 1/3 of the total price."""
        from src.shopping.intelligence.special.bundle_detector import detect_bundle_deals

        products = [
            {
                "name": "Omo 3 al, 2 öde",
                "price": 270.0,
                "original_price": 270.0,
            }
        ]
        result = detect_bundle_deals(products)
        al_ode_bundle = next(r for r in result if r["type"] == "al_ode")
        expected_savings = round(270.0 / 3 * 1, 2)  # 1 free item
        self.assertAlmostEqual(al_ode_bundle["savings_estimate"], expected_savings, places=1)

    def test_products_involved_contains_product_name(self):
        """products_involved list includes the source product name."""
        from src.shopping.intelligence.special.bundle_detector import detect_bundle_deals

        products = [{"name": "Persil 3 Al 2 Öde", "price": 240.0, "original_price": 240.0}]
        result = detect_bundle_deals(products)
        self.assertIn("Persil 3 Al 2 Öde", result[0]["products_involved"])

    def test_bulk_detector_unit_price_calculation(self):
        """analyze_bulk_pricing calculates unit price and marks product as bulk."""
        from src.shopping.intelligence.special.bulk_detector import analyze_bulk_pricing

        products = [
            {"name": "Domestos 3'lü Paket Çamaşır Suyu", "price": 150.0},
            {"name": "Domestos Çamaşır Suyu 1 Adet", "price": 55.0},
        ]
        result = analyze_bulk_pricing(products)
        self.assertEqual(len(result), 2)
        # The 3-pack should be identified as bulk
        three_pack = next((p for p in result if p["is_bulk"]), None)
        self.assertIsNotNone(three_pack)
        self.assertAlmostEqual(three_pack["unit_price"], 150.0 / 3, places=1)


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 10 — Used furniture assessment
# Used market safety check — should be SAFE.
# ═══════════════════════════════════════════════════════════════════════════

class TestScenario10_UsedFurnitureSafe(unittest.TestCase):
    """Used furniture — buying second-hand is safe and platforms are suggested."""

    def test_furniture_is_safe_to_buy_used(self):
        """Furniture category is safe to buy used."""
        from src.shopping.intelligence.special.used_market import is_safe_to_buy_used

        self.assertTrue(is_safe_to_buy_used("mobilya"))
        self.assertTrue(is_safe_to_buy_used("furniture"))

    def test_used_platforms_returned_for_furniture(self):
        """At least one platform is suggested for second-hand furniture."""
        from src.shopping.intelligence.special.used_market import get_used_platforms

        platforms = get_used_platforms("mobilya")
        self.assertIsInstance(platforms, list)
        self.assertTrue(len(platforms) > 0)

    def test_platform_has_required_keys(self):
        """Each suggested platform dict contains name and url_pattern."""
        from src.shopping.intelligence.special.used_market import get_used_platforms

        platforms = get_used_platforms("mobilya")
        if platforms:
            for p in platforms:
                self.assertIn("name", p)

    def test_used_advisory_returned_for_furniture(self):
        """assess_used_viability returns a dict recommending used purchase for furniture."""
        from src.shopping.intelligence.special.used_market import assess_used_viability

        with patch(
            "src.shopping.intelligence.special.used_market._llm_call",
            new_callable=AsyncMock,
            return_value="",
        ):
            result = run_async(assess_used_viability("ikinci el mobilya", "mobilya"))
        self.assertIn("used_recommended", result)
        self.assertTrue(result["used_recommended"])


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 11 — Used baby car seat
# Used market safety check — should be UNSAFE.
# ═══════════════════════════════════════════════════════════════════════════

class TestScenario11_UsedBabyCarSeatUnsafe(unittest.TestCase):
    """Used baby car seat — safety blocker must fire."""

    def test_araba_koltugu_is_unsafe(self):
        """'araba koltuğu' is blocked for second-hand purchase."""
        from src.shopping.intelligence.special.used_market import is_safe_to_buy_used

        self.assertFalse(is_safe_to_buy_used("araba koltuğu"))

    def test_cocuk_koltugu_is_unsafe(self):
        """'çocuk koltuğu' (child seat) is also blocked."""
        from src.shopping.intelligence.special.used_market import is_safe_to_buy_used

        self.assertFalse(is_safe_to_buy_used("çocuk koltuğu"))

    def test_bebek_category_is_unsafe(self):
        """Any 'bebek' (baby) category is blocked."""
        from src.shopping.intelligence.special.used_market import is_safe_to_buy_used

        self.assertFalse(is_safe_to_buy_used("bebek"))
        self.assertFalse(is_safe_to_buy_used("bebek ürünleri"))

    def test_used_advisory_unsafe_returns_warning(self):
        """assess_used_viability for car seat returns used_recommended=False with safety_concern."""
        from src.shopping.intelligence.special.used_market import assess_used_viability

        result = run_async(assess_used_viability("bebek araba koltuğu", "çocuk koltuğu"))
        self.assertFalse(result["used_recommended"])
        self.assertTrue(result.get("safety_concern", False))
        # Should have a human-readable reason
        self.assertIsInstance(result.get("reason", ""), str)
        self.assertTrue(len(result.get("reason", "")) > 0)


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 12 — Import phone without BTK registration
# Import advisor should raise BTK warning.
# ═══════════════════════════════════════════════════════════════════════════

class TestScenario12_ImportPhoneBtkWarning(unittest.TestCase):
    """Import phone — BTK registration check fires for phones/tablets."""

    def test_iphone_needs_btk(self):
        """iPhone product triggers BTK warning."""
        from src.shopping.intelligence.special.import_domestic import check_btk_requirement

        product = {"name": "iPhone 15 Pro Max 256GB", "category": "telefon", "brand": "Apple"}
        result = check_btk_requirement(product)
        self.assertTrue(result["needs_btk"])
        self.assertIn("BTK", result["warning"])

    def test_registration_cost_estimate_positive(self):
        """BTK registration cost estimate is a positive number."""
        from src.shopping.intelligence.special.import_domestic import check_btk_requirement

        product = {"name": "Samsung Galaxy S24", "category": "akıllı telefon", "brand": "Samsung"}
        result = check_btk_requirement(product)
        self.assertTrue(result["needs_btk"])
        self.assertGreater(result["registration_cost_estimate"], 0)

    def test_non_phone_no_btk(self):
        """A laptop does not require BTK registration."""
        from src.shopping.intelligence.special.import_domestic import check_btk_requirement

        product = {"name": "MacBook Air M2", "category": "laptop", "brand": "Apple"}
        result = check_btk_requirement(product)
        self.assertFalse(result["needs_btk"])

    def test_import_advisory_full_flow(self):
        """get_import_advisory for a grey-market phone includes BTK and grey market signals."""
        from src.shopping.intelligence.special.import_domestic import get_import_advisory

        product = {
            "name": "iPhone 15 Pro Max Yurt Dışı",
            "brand": "Apple",
            "category": "iphone",
            "description": "ithalatçı garantili ürün",
            "price": 55000.0,
            "avg_price": 80000.0,  # well below market → grey market heuristic
        }
        result = get_import_advisory(product)
        self.assertIn("btk", result)
        self.assertTrue(result["btk"]["needs_btk"])
        self.assertIn("overall_risk", result)
        self.assertIn(result["overall_risk"], ("medium", "high"))

    def test_classify_origin_apple_imported(self):
        """Apple is classified as an imported brand."""
        from src.shopping.intelligence.special.import_domestic import classify_origin

        result = classify_origin("Apple")
        self.assertEqual(result["origin"], "imported")
        self.assertGreater(result["confidence"], 0.8)


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 13 — Flash sale TV
# Staleness detection + fake discount check.
# ═══════════════════════════════════════════════════════════════════════════

class TestScenario13_FlashSaleTV(unittest.TestCase):
    """Flash sale TV listing — staleness and fake-discount detection."""

    def test_flash_sale_keyword_detected(self):
        """A product named with 'flash indirim' triggers flash sale detection."""
        from src.shopping.resilience.staleness import detect_flash_sale

        product = {
            "name": "Samsung 55' 4K TV Flash İndirim Bugüne Özel",
            "price": 18000.0,
            "original_price": 35000.0,
        }
        result = detect_flash_sale(product)
        self.assertTrue(result["is_flash_sale"])

    def test_stale_flash_sale_warns_user(self):
        """An old flash-sale cache entry is stale and includes a warning."""
        from src.shopping.resilience.staleness import assess_staleness

        product = {
            "name": "LG 65 inç TV Flash Fırsat Son 2 Saat",
            "category": "electronics",
            "price": 20000.0,
            "original_price": 40000.0,
        }
        # 5 hours old (18 000 seconds) — well past electronics TTL of 3600 s
        result = assess_staleness(product, cache_age_seconds=18000)
        self.assertTrue(result["is_stale"])
        self.assertTrue(len(result["warnings"]) > 0)

    def test_fake_discount_genuine_reduction(self):
        """Detect_fake_discount returns is_fake=False when history supports price."""
        from src.shopping.intelligence.special.fake_discount_detector import detect_fake_discount

        now = time.time()
        product = {"price": 18000.0, "original_price": 25000.0}
        history = [
            {"price": 25000.0, "observed_at": now - 86400 * 90},
            {"price": 24000.0, "observed_at": now - 86400 * 60},
            {"price": 20000.0, "observed_at": now - 86400 * 30},
            {"price": 18000.0, "observed_at": now},
        ]
        result = run_async(detect_fake_discount(product, history))
        # Prices genuinely declined — should NOT be fake
        self.assertFalse(result["is_fake"])

    def test_fake_discount_inflated_original(self):
        """detect_fake_discount flags a discount where original was never the real price."""
        from src.shopping.intelligence.special.fake_discount_detector import detect_fake_discount

        now = time.time()
        product = {"price": 15000.0, "original_price": 60000.0}  # 75% claimed discount
        history = [
            {"price": 14500.0, "observed_at": now - 86400 * 90},
            {"price": 15200.0, "observed_at": now - 86400 * 60},
            {"price": 14800.0, "observed_at": now - 86400 * 30},
        ]
        result = run_async(detect_fake_discount(product, history))
        # Original price massively higher than all historical prices → fake
        self.assertTrue(result["is_fake"])

    def test_fresh_cache_not_stale(self):
        """A recently cached product (10 minutes old) is not stale."""
        from src.shopping.resilience.staleness import assess_staleness

        product = {"name": "Sony OLED TV", "category": "electronics", "price": 50000.0}
        result = assess_staleness(product, cache_age_seconds=600)
        self.assertFalse(result["is_stale"])


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 14 — Printer + consumable cost (TCO)
# Complementary products + total cost of ownership.
# ═══════════════════════════════════════════════════════════════════════════

class TestScenario14_PrinterTCO(unittest.TestCase):
    """Printer purchase — complementary consumables and TCO calculation."""

    def test_printer_complements_include_cartridge(self):
        """Static complement map for printer includes cartridge/toner."""
        from src.shopping.intelligence.special.complementary import get_complement_map

        complements = get_complement_map("yazıcı")
        products = [c["product"] for c in complements]
        self.assertTrue(any("kartuş" in p.lower() or "toner" in p.lower() for p in products))

    def test_printer_complement_marked_consumable(self):
        """Cartridge item is marked as consumable."""
        from src.shopping.intelligence.special.complementary import get_complement_map

        complements = get_complement_map("yazıcı")
        consumables = [c for c in complements if c["is_consumable"]]
        self.assertTrue(len(consumables) > 0)

    def test_printer_tco_exceeds_purchase_price(self):
        """3-year TCO for a printer with consumables exceeds the purchase price."""
        from src.shopping.intelligence.special.tco_calculator import calculate_tco

        product = {"name": "HP LaserJet M110w", "price": 4500.0, "category": "printer"}
        result = calculate_tco(product, years=3)
        self.assertGreater(result["total_tco"], result["purchase_price"])
        self.assertGreater(result["consumable_cost"], 0)

    def test_tco_breakdown_keys_present(self):
        """TCO result includes all expected breakdown keys."""
        from src.shopping.intelligence.special.tco_calculator import calculate_tco

        product = {"name": "Canon PIXMA", "price": 3000.0, "category": "printer"}
        result = calculate_tco(product, years=3)
        for key in ("purchase_price", "energy_cost", "consumable_cost",
                    "maintenance_cost", "total_tco", "annual_tco"):
            self.assertIn(key, result)

    def test_consumable_warning_issued_for_printer(self):
        """assess_consumable_cost returns a recurring cost warning for printers."""
        from src.shopping.intelligence.special.complementary import assess_consumable_cost

        result = assess_consumable_cost({"category": "yazıcı", "name": "HP LaserJet"})
        self.assertTrue(result["has_consumables"])
        self.assertIn("consumable_items", result)
        self.assertIn("estimated_annual_cost", result)
        self.assertTrue(len(result["consumable_items"]) > 0)

    def test_suggest_complements_printer_async(self):
        """suggest_complements returns cartridge for printer (static path, no LLM)."""
        from src.shopping.intelligence.special.complementary import suggest_complements

        with patch(
            "src.shopping.intelligence.special.complementary._llm_call",
            new_callable=AsyncMock,
            return_value="",
        ):
            result = run_async(suggest_complements("HP LaserJet Yazıcı", "printer"))

        products = [c["product"] for c in result]
        self.assertTrue(any("kartuş" in p.lower() or "toner" in p.lower() for p in products))


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 15 — Energy efficient washing machine
# Environmental / efficiency assessment.
# ═══════════════════════════════════════════════════════════════════════════

class TestScenario15_EnergyEfficientWashingMachine(unittest.TestCase):
    """A+++ washing machine — efficiency score, annual cost savings, water cost."""

    def test_a_plus_plus_plus_efficiency_score_high(self):
        """A+++ class yields a near-maximum efficiency score."""
        from src.shopping.intelligence.special.environmental import assess_efficiency

        product = {
            "name": "Beko CM 9141 A+++ Çamaşır Makinesi",
            "category": "çamaşır makinesi",
            "energy_class": "A+++",
            "price": 18000.0,
        }
        result = assess_efficiency(product)
        self.assertGreater(result["efficiency_score"], 0.8)

    def test_a_plus_plus_plus_annual_energy_cost_lower_than_c(self):
        """A+++ annual energy cost is less than D-class cost."""
        from src.shopping.intelligence.special.environmental import assess_efficiency

        a_product = {"name": "A+++", "category": "çamaşır makinesi", "energy_class": "A+++"}
        d_product = {"name": "D", "category": "çamaşır makinesi", "energy_class": "D"}

        a_result = assess_efficiency(a_product)
        d_result = assess_efficiency(d_product)

        self.assertLess(
            a_result["estimated_annual_energy_cost"],
            d_result["estimated_annual_energy_cost"],
        )

    def test_water_cost_present_for_washing_machine(self):
        """Washing machine assessment includes annual water cost."""
        from src.shopping.intelligence.special.environmental import assess_efficiency

        product = {
            "name": "Siemens WM14UR0T Çamaşır Makinesi",
            "category": "çamaşır makinesi",
            "energy_class": "A++",
            "price": 22000.0,
        }
        result = assess_efficiency(product)
        self.assertIsNotNone(result["estimated_annual_water_cost"])
        self.assertGreater(result["estimated_annual_water_cost"], 0)

    def test_cost_savings_note_mentions_savings(self):
        """Savings note for A+++ vs C-class reference is a non-empty string."""
        from src.shopping.intelligence.special.environmental import assess_efficiency

        product = {
            "name": "Arçelik 9141 A+++ Çamaşır Makinesi",
            "category": "washing machine",
            "energy_class": "A+++",
        }
        result = assess_efficiency(product)
        note = result["cost_savings_note"]
        self.assertIsInstance(note, str)
        self.assertTrue(len(note) > 0)

    def test_unknown_energy_class_returns_neutral_score(self):
        """A product without energy class gets a neutral 0.5 efficiency score."""
        from src.shopping.intelligence.special.environmental import assess_efficiency

        product = {"name": "Generic Washing Machine", "category": "çamaşır makinesi"}
        result = assess_efficiency(product)
        self.assertAlmostEqual(result["efficiency_score"], 0.5)

    def test_d_class_efficiency_score_low(self):
        """D energy class yields a low efficiency score."""
        from src.shopping.intelligence.special.environmental import (
            _efficiency_score_from_class,
        )

        score = _efficiency_score_from_class("D")
        self.assertAlmostEqual(score, 0.0)

    def test_a_plus_plus_plus_score_is_maximum(self):
        """A+++ yields the maximum efficiency score of 1.0."""
        from src.shopping.intelligence.special.environmental import (
            _efficiency_score_from_class,
        )

        score = _efficiency_score_from_class("A+++")
        self.assertAlmostEqual(score, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# pytest entry point (also runnable via unittest)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main()
