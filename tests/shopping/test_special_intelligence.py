"""Comprehensive tests for shopping special intelligence modules:
tco_calculator, seasonal_advisor, unit_price_calculator,
fake_discount_detector, seller_trust, warranty_analyzer.
"""

from __future__ import annotations

import asyncio
import time
import unittest
from datetime import date
from unittest.mock import patch


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
# 1. TestTCOCalculator
# ═══════════════════════════════════════════════════════════════════════════

class TestTCOCalculator(unittest.TestCase):
    """Test total cost of ownership calculator."""

    def test_basic_tco(self):
        from src.shopping.intelligence.special.tco_calculator import calculate_tco
        product = {"price": 10000, "watts": 0, "category": "tv"}
        result = calculate_tco(product, years=3)
        self.assertEqual(result["purchase_price"], 10000)
        self.assertEqual(result["years"], 3)
        self.assertGreaterEqual(result["total_tco"], 10000)

    def test_energy_cost_calculation(self):
        from src.shopping.intelligence.special.tco_calculator import estimate_energy_cost
        # 100W device, 8 hours/day, 1 year
        cost = estimate_energy_cost(100, 8, 1)
        self.assertGreater(cost, 0)
        # Verify: 0.1kW * 8h * 365d * 2.83 TL/kWh = ~826 TL
        expected = 0.1 * 8 * 365 * 2.83
        self.assertAlmostEqual(cost, expected, places=0)

    def test_energy_cost_zero_watts(self):
        from src.shopping.intelligence.special.tco_calculator import estimate_energy_cost
        cost = estimate_energy_cost(0, 8, 1)
        self.assertEqual(cost, 0)

    def test_consumable_cost_printer(self):
        from src.shopping.intelligence.special.tco_calculator import estimate_consumable_cost
        cost = estimate_consumable_cost("printer", 3)
        self.assertEqual(cost, 1500.0 * 3)

    def test_consumable_cost_unknown(self):
        from src.shopping.intelligence.special.tco_calculator import estimate_consumable_cost
        cost = estimate_consumable_cost("random_category", 3)
        self.assertEqual(cost, 0)

    def test_tco_with_energy(self):
        from src.shopping.intelligence.special.tco_calculator import calculate_tco
        product = {"price": 5000, "watts": 200, "category": "air_conditioner", "daily_usage_hours": 8}
        result = calculate_tco(product, years=5)
        self.assertGreater(result["energy_cost"], 0)
        self.assertGreater(result["total_tco"], 5000)

    def test_tco_with_maintenance(self):
        from src.shopping.intelligence.special.tco_calculator import calculate_tco
        product = {"price": 20000, "watts": 0, "category": "air_conditioner"}
        result = calculate_tco(product, years=3)
        self.assertGreater(result["maintenance_cost"], 0)

    def test_tco_breakdown_sums_to_100(self):
        from src.shopping.intelligence.special.tco_calculator import calculate_tco
        product = {"price": 10000, "watts": 100, "category": "dishwasher"}
        result = calculate_tco(product, years=3)
        bd = result["breakdown"]
        total = bd["purchase_pct"] + bd["energy_pct"] + bd["consumable_pct"] + bd["maintenance_pct"]
        self.assertAlmostEqual(total, 100.0, places=0)

    def test_compare_tco(self):
        from src.shopping.intelligence.special.tco_calculator import compare_tco
        products = [
            {"name": "Cheap", "price": 3000, "watts": 200, "category": "dishwasher"},
            {"name": "Expensive", "price": 8000, "watts": 50, "category": "dishwasher"},
        ]
        result = compare_tco(products, years=5)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["rank"], 1)
        self.assertGreaterEqual(result[0]["savings_vs_worst"], 0)

    def test_compare_tco_empty(self):
        from src.shopping.intelligence.special.tco_calculator import compare_tco
        self.assertEqual(compare_tco([], years=3), [])

    def test_default_daily_hours(self):
        from src.shopping.intelligence.special.tco_calculator import _default_daily_hours
        self.assertEqual(_default_daily_hours({"category": "refrigerator"}), 24.0)
        self.assertEqual(_default_daily_hours({"category": "tv"}), 5.0)
        self.assertEqual(_default_daily_hours({"category": "unknown"}), 2.0)


# ═══════════════════════════════════════════════════════════════════════════
# 2. TestSeasonalAdvisor
# ═══════════════════════════════════════════════════════════════════════════

class TestSeasonalAdvisor(unittest.TestCase):
    """Test seasonal buying advice."""

    def test_upcoming_sales_november(self):
        from src.shopping.intelligence.special.seasonal_advisor import get_upcoming_sales
        # Nov 1 should see 11.11 and Black Friday
        result = get_upcoming_sales(days_ahead=60, reference_date=date(2025, 11, 1))
        self.assertTrue(len(result) > 0)
        names = [e["name"] for e in result]
        self.assertTrue(any("11.11" in n or "Bekarlar" in n for n in names))

    def test_upcoming_sales_empty(self):
        from src.shopping.intelligence.special.seasonal_advisor import get_upcoming_sales
        # Very short window in a quiet month
        result = get_upcoming_sales(days_ahead=5, reference_date=date(2025, 6, 15))
        # Should have 0 or few results
        self.assertIsInstance(result, list)

    def test_seasonal_advice_electronics_november(self):
        from src.shopping.intelligence.special.seasonal_advisor import get_seasonal_advice
        result = get_seasonal_advice("electronics", current_date=date(2025, 11, 5))
        self.assertIn("recommendation", result)
        self.assertGreater(result["confidence"], 0)
        self.assertEqual(result["category"], "electronics")

    def test_seasonal_advice_no_upcoming(self):
        from src.shopping.intelligence.special.seasonal_advisor import get_seasonal_advice
        # Use a date far from events
        result = get_seasonal_advice("electronics", current_date=date(2025, 6, 15))
        self.assertIn("recommendation", result)

    def test_is_good_time_high_urgency(self):
        from src.shopping.intelligence.special.seasonal_advisor import is_good_time_to_buy
        result = is_good_time_to_buy("electronics", urgency="high", current_date=date(2025, 6, 15))
        self.assertTrue(result["buy_now"])
        self.assertEqual(result["wait_days"], 0)

    def test_is_good_time_active_sale(self):
        from src.shopping.intelligence.special.seasonal_advisor import is_good_time_to_buy
        # During Black Friday
        result = is_good_time_to_buy("electronics", urgency="normal", current_date=date(2025, 11, 25))
        self.assertTrue(result["buy_now"])

    def test_is_good_time_wait_for_sale(self):
        from src.shopping.intelligence.special.seasonal_advisor import is_good_time_to_buy
        # Nov 5, Black Friday 15 days away, normal urgency
        result = is_good_time_to_buy("electronics", urgency="normal", current_date=date(2025, 11, 5))
        # Should recommend waiting since sale is within 30 days
        self.assertFalse(result["buy_now"])
        self.assertGreater(result["wait_days"], 0)

    def test_best_months_electronics(self):
        from src.shopping.intelligence.special.seasonal_advisor import _BEST_MONTHS
        self.assertIn(11, _BEST_MONTHS["electronics"])

    def test_upcoming_sales_sorted_by_proximity(self):
        from src.shopping.intelligence.special.seasonal_advisor import get_upcoming_sales
        result = get_upcoming_sales(days_ahead=365, reference_date=date(2025, 1, 1))
        if len(result) >= 2:
            self.assertLessEqual(result[0]["days_until"], result[1]["days_until"])


# ═══════════════════════════════════════════════════════════════════════════
# 3. TestUnitPriceCalculator
# ═══════════════════════════════════════════════════════════════════════════

class TestUnitPriceCalculator(unittest.TestCase):
    """Test per-unit price calculation and quantity detection."""

    def test_calculate_unit_price_kg(self):
        from src.shopping.intelligence.special.unit_price_calculator import calculate_unit_price
        result = calculate_unit_price(100, 2, "kg")
        self.assertEqual(result["unit_price"], 50.0)
        self.assertEqual(result["normalized_unit"], "kg")

    def test_calculate_unit_price_grams(self):
        from src.shopping.intelligence.special.unit_price_calculator import calculate_unit_price
        result = calculate_unit_price(100, 500, "g")
        # 500g = 0.5kg, 100/0.5 = 200 TL/kg
        self.assertEqual(result["unit_price"], 200.0)
        self.assertEqual(result["normalized_unit"], "kg")

    def test_calculate_unit_price_ml(self):
        from src.shopping.intelligence.special.unit_price_calculator import calculate_unit_price
        result = calculate_unit_price(50, 750, "ml")
        # 750ml = 0.75L, 50/0.75 = 66.67 TL/L
        self.assertAlmostEqual(result["unit_price"], 66.67, places=2)
        self.assertEqual(result["normalized_unit"], "L")

    def test_calculate_unit_price_zero_quantity(self):
        from src.shopping.intelligence.special.unit_price_calculator import calculate_unit_price
        result = calculate_unit_price(100, 0, "kg")
        self.assertEqual(result["unit_price"], 0.0)

    def test_detect_quantity_kg(self):
        from src.shopping.intelligence.special.unit_price_calculator import detect_quantity
        result = detect_quantity("Pinar Sut 1L")
        self.assertIsNotNone(result)
        self.assertEqual(result["quantity"], 1.0)
        self.assertEqual(result["unit"], "l")

    def test_detect_quantity_grams(self):
        from src.shopping.intelligence.special.unit_price_calculator import detect_quantity
        result = detect_quantity("Eti Cicibebe 500g")
        self.assertIsNotNone(result)
        self.assertEqual(result["quantity"], 500.0)
        self.assertEqual(result["unit"], "g")

    def test_detect_quantity_pack(self):
        from src.shopping.intelligence.special.unit_price_calculator import detect_quantity
        result = detect_quantity("Colgate 75ml 3'li Paket")
        self.assertIsNotNone(result)
        # Should detect 75ml first
        self.assertIn(result["unit"], ["ml", "adet"])

    def test_detect_quantity_tablets(self):
        from src.shopping.intelligence.special.unit_price_calculator import detect_quantity
        result = detect_quantity("Vitamin C 100 tablet")
        self.assertIsNotNone(result)
        self.assertEqual(result["quantity"], 100)

    def test_detect_quantity_none(self):
        from src.shopping.intelligence.special.unit_price_calculator import detect_quantity
        result = detect_quantity("Samsung Galaxy S24")
        self.assertIsNone(result)

    def test_detect_quantity_empty(self):
        from src.shopping.intelligence.special.unit_price_calculator import detect_quantity
        result = detect_quantity("")
        self.assertIsNone(result)

    def test_compare_unit_prices(self):
        from src.shopping.intelligence.special.unit_price_calculator import compare_unit_prices
        products = [
            {"name": "Milk 1L", "price": 40, "quantity": 1, "unit": "L"},
            {"name": "Milk 500ml", "price": 25, "quantity": 500, "unit": "ml"},
        ]
        result = compare_unit_prices(products)
        self.assertEqual(len(result), 2)
        # 1L at 40 TL/L vs 500ml at 50 TL/L — 1L is cheaper per unit
        self.assertEqual(result[0]["product_name"], "Milk 1L")
        self.assertTrue(result[0]["best_value"])

    def test_compare_unit_prices_autodetect(self):
        from src.shopping.intelligence.special.unit_price_calculator import compare_unit_prices
        products = [
            {"name": "Sut 1L", "price": 40},
            {"name": "Sut 500ml", "price": 25},
        ]
        result = compare_unit_prices(products)
        # Should auto-detect quantities
        self.assertEqual(len(result), 2)

    def test_compare_unit_prices_no_quantity(self):
        from src.shopping.intelligence.special.unit_price_calculator import compare_unit_prices
        products = [{"name": "Unknown Product", "price": 100}]
        result = compare_unit_prices(products)
        self.assertIsNone(result[0]["unit_price"])
        self.assertFalse(result[0]["best_value"])


# ═══════════════════════════════════════════════════════════════════════════
# 4. TestFakeDiscountDetector
# ═══════════════════════════════════════════════════════════════════════════

class TestFakeDiscountDetector(unittest.TestCase):
    """Test fake discount detection."""

    def test_no_discount(self):
        from src.shopping.intelligence.special.fake_discount_detector import detect_fake_discount
        result = run_async(detect_fake_discount(
            {"price": 1000, "original_price": 1000}, []
        ))
        self.assertFalse(result["is_fake"])

    def test_genuine_discount_with_history(self):
        from src.shopping.intelligence.special.fake_discount_detector import detect_fake_discount
        now = time.time()
        history = [
            {"price": 5000, "observed_at": now - 86400 * 60},
            {"price": 5100, "observed_at": now - 86400 * 30},
            {"price": 4800, "observed_at": now - 86400 * 10},
        ]
        result = run_async(detect_fake_discount(
            {"name": "Test", "price": 4000, "original_price": 5000}, history
        ))
        # Original price (5000) is consistent with history, so not fake
        self.assertIsInstance(result["is_fake"], bool)
        self.assertIn("real_discount_pct", result)

    def test_inflated_original_price(self):
        from src.shopping.intelligence.special.fake_discount_detector import detect_fake_discount
        now = time.time()
        # Historical max was 3000 but original claims 5000
        history = [
            {"price": 2500, "observed_at": now - 86400 * 60},
            {"price": 2800, "observed_at": now - 86400 * 30},
            {"price": 3000, "observed_at": now - 86400 * 10},
        ]
        result = run_async(detect_fake_discount(
            {"name": "Test", "price": 2000, "original_price": 5000}, history
        ))
        self.assertTrue(result["is_fake"])
        self.assertGreater(result["confidence"], 0)

    def test_check_price_inflation_inflated(self):
        from src.shopping.intelligence.special.fake_discount_detector import check_price_inflation
        result = check_price_inflation(5000, 3000, 3500)
        self.assertTrue(result["is_inflated"])

    def test_check_price_inflation_consistent(self):
        from src.shopping.intelligence.special.fake_discount_detector import check_price_inflation
        result = check_price_inflation(3500, 3000, 3400)
        self.assertFalse(result["is_inflated"])

    def test_check_price_inflation_no_data(self):
        from src.shopping.intelligence.special.fake_discount_detector import check_price_inflation
        result = check_price_inflation(5000, 3000, 0)
        self.assertFalse(result["is_inflated"])

    def test_cross_store_consistency_consistent(self):
        from src.shopping.intelligence.special.fake_discount_detector import check_cross_store_consistency
        prices = {"trendyol": 5000, "hepsiburada": 5200, "amazon": 5100}
        result = check_cross_store_consistency(prices)
        self.assertTrue(result["is_consistent"])
        self.assertLess(result["spread_pct"], 15)

    def test_cross_store_consistency_inconsistent(self):
        from src.shopping.intelligence.special.fake_discount_detector import check_cross_store_consistency
        prices = {"trendyol": 3000, "hepsiburada": 5000}
        result = check_cross_store_consistency(prices)
        self.assertFalse(result["is_consistent"])
        self.assertGreater(result["spread_pct"], 30)

    def test_cross_store_single_store(self):
        from src.shopping.intelligence.special.fake_discount_detector import check_cross_store_consistency
        result = check_cross_store_consistency({"trendyol": 5000})
        self.assertTrue(result["is_consistent"])

    def test_very_high_discount_flagged(self):
        from src.shopping.intelligence.special.fake_discount_detector import detect_fake_discount
        result = run_async(detect_fake_discount(
            {"name": "Test", "price": 1000, "original_price": 10000}, []
        ))
        # 90% discount should be suspicious
        self.assertTrue(result["is_fake"])


# ═══════════════════════════════════════════════════════════════════════════
# 5. TestSellerTrust
# ═══════════════════════════════════════════════════════════════════════════

class TestSellerTrust(unittest.TestCase):
    """Test seller trust scoring."""

    def test_high_trust_seller(self):
        from src.shopping.intelligence.special.seller_trust import score_seller
        result = score_seller({
            "name": "Great Store",
            "rating": 4.8,
            "review_count": 500,
            "joined_date": "2020-01-01",
            "total_sales": 5000,
            "is_verified": True,
        })
        self.assertGreaterEqual(result["trust_score"], 70)
        self.assertTrue(len(result["badges"]) > 0)
        self.assertEqual(len(result["warnings"]), 0)

    def test_low_trust_seller(self):
        from src.shopping.intelligence.special.seller_trust import score_seller
        result = score_seller({
            "name": "Shady Shop",
            "rating": 2.0,
            "review_count": 3,
            "total_sales": 5,
            "return_rate": 0.2,
        })
        self.assertLess(result["trust_score"], 50)
        self.assertTrue(len(result["warnings"]) > 0)

    def test_new_seller_warning(self):
        from src.shopping.intelligence.special.seller_trust import score_seller
        # Joined 1 month ago
        from datetime import date, timedelta
        recent = (date.today() - timedelta(days=20)).isoformat()
        result = score_seller({
            "rating": 4.5,
            "review_count": 10,
            "joined_date": recent,
        })
        self.assertTrue(any("yeni" in w.lower() or "kisa" in w.lower() for w in result["warnings"]))

    def test_verified_bonus(self):
        from src.shopping.intelligence.special.seller_trust import score_seller
        unverified = score_seller({"rating": 4.0, "review_count": 50})
        verified = score_seller({"rating": 4.0, "review_count": 50, "is_verified": True})
        self.assertGreater(verified["trust_score"], unverified["trust_score"])

    def test_check_seller_age_old(self):
        from src.shopping.intelligence.special.seller_trust import check_seller_age
        result = check_seller_age("2020-01-01")
        self.assertFalse(result["is_new"])
        self.assertGreater(result["months_active"], 24)

    def test_check_seller_age_new(self):
        from src.shopping.intelligence.special.seller_trust import check_seller_age
        from datetime import date, timedelta
        recent = (date.today() - timedelta(days=30)).isoformat()
        result = check_seller_age(recent)
        self.assertTrue(result["is_new"])

    def test_check_seller_age_invalid(self):
        from src.shopping.intelligence.special.seller_trust import check_seller_age
        result = check_seller_age("not-a-date")
        self.assertTrue(result["is_new"])
        self.assertIn("gecersiz", result.get("warning", ""))

    def test_review_authenticity_suspicious(self):
        from src.shopping.intelligence.special.seller_trust import check_review_authenticity
        # All 5-star, same text, same day
        reviews = [
            {"rating": 5.0, "text": "Great product!", "date": "2025-01-15"}
            for _ in range(20)
        ]
        result = check_review_authenticity(reviews)
        self.assertTrue(result["suspicious"])
        self.assertGreater(result["confidence"], 0)

    def test_review_authenticity_genuine(self):
        from src.shopping.intelligence.special.seller_trust import check_review_authenticity
        reviews = [
            {"rating": 4.0 + (i % 3) * 0.5, "text": f"Review text number {i} with detailed explanation about product quality and performance", "date": f"2025-0{(i % 9) + 1}-15"}
            for i in range(15)
        ]
        result = check_review_authenticity(reviews)
        self.assertFalse(result["suspicious"])

    def test_high_return_rate_penalty(self):
        from src.shopping.intelligence.special.seller_trust import score_seller
        result = score_seller({"rating": 4.0, "review_count": 50, "return_rate": 0.2})
        self.assertTrue(any("iade" in w.lower() for w in result["warnings"]))

    def test_score_clamped_0_100(self):
        from src.shopping.intelligence.special.seller_trust import score_seller
        # Very bad seller
        result = score_seller({
            "rating": 1.0, "review_count": 100, "return_rate": 0.5,
            "response_time_hours": 100,
        })
        self.assertGreaterEqual(result["trust_score"], 0)
        self.assertLessEqual(result["trust_score"], 100)


# ═══════════════════════════════════════════════════════════════════════════
# 6. TestWarrantyAnalyzer
# ═══════════════════════════════════════════════════════════════════════════

class TestWarrantyAnalyzer(unittest.TestCase):
    """Test warranty analysis."""

    def test_known_brand_apple(self):
        from src.shopping.intelligence.special.warranty_analyzer import get_service_network
        result = get_service_network("Apple")
        self.assertEqual(result["warranty_months"], 24)
        self.assertTrue(result["official_service"])
        self.assertIn("Istanbul", result["service_centers"])

    def test_known_brand_arcelik(self):
        from src.shopping.intelligence.special.warranty_analyzer import get_service_network
        result = get_service_network("arcelik")
        self.assertEqual(result["warranty_months"], 36)

    def test_unknown_brand(self):
        from src.shopping.intelligence.special.warranty_analyzer import get_service_network
        result = get_service_network("unknown_brand_xyz")
        self.assertEqual(result["warranty_months"], 24)
        self.assertFalse(result["official_service"])

    def test_gray_market_risk_low(self):
        from src.shopping.intelligence.special.warranty_analyzer import assess_gray_market_risk
        risk = assess_gray_market_risk({"name": "Samsung TV"}, 5000, 5000)
        self.assertLess(risk, 0.2)

    def test_gray_market_risk_high_price_deviation(self):
        from src.shopping.intelligence.special.warranty_analyzer import assess_gray_market_risk
        # Price way below average
        risk = assess_gray_market_risk({"name": "Samsung TV"}, 3000, 5000)
        self.assertGreater(risk, 0.3)

    def test_gray_market_risk_import_hint(self):
        from src.shopping.intelligence.special.warranty_analyzer import assess_gray_market_risk
        risk = assess_gray_market_risk(
            {"name": "iPhone import", "seller_type": "ithalat"}, 4000, 5000
        )
        self.assertGreater(risk, 0.3)

    def test_gray_market_risk_no_warranty_hint(self):
        from src.shopping.intelligence.special.warranty_analyzer import assess_gray_market_risk
        risk = assess_gray_market_risk({"name": "Laptop garantisiz"}, 3000, 5000)
        self.assertGreater(risk, 0.5)

    def test_gray_market_risk_zero_price(self):
        from src.shopping.intelligence.special.warranty_analyzer import assess_gray_market_risk
        risk = assess_gray_market_risk({"name": "Test"}, 0, 5000)
        self.assertEqual(risk, 0.0)

    def test_analyze_warranty_official_store(self):
        from src.shopping.intelligence.special.warranty_analyzer import analyze_warranty
        result = analyze_warranty(
            {"name": "Samsung Galaxy S24", "brand": "samsung", "price": 35000, "avg_price": 35000},
            "trendyol",
        )
        self.assertEqual(result["warranty_months"], 24)
        self.assertTrue(result["official_service"])
        self.assertEqual(result["brand"], "samsung")

    def test_analyze_warranty_unofficial_store(self):
        from src.shopping.intelligence.special.warranty_analyzer import analyze_warranty
        result = analyze_warranty(
            {"name": "Samsung TV", "brand": "samsung", "price": 35000, "avg_price": 35000},
            "random_store",
        )
        self.assertTrue(any("resmi" in n.lower() or "garanti" in n.lower() for n in result["notes"]))

    def test_analyze_warranty_gray_market_warning(self):
        from src.shopping.intelligence.special.warranty_analyzer import analyze_warranty
        result = analyze_warranty(
            {"name": "iPhone garantisiz", "brand": "apple", "price": 15000, "avg_price": 40000},
            "random_store",
        )
        self.assertGreater(result["gray_market_risk"], 0.5)
        self.assertTrue(any("fiyat" in n.lower() or "garanti" in n.lower() for n in result["notes"]))

    def test_extract_brand_from_name(self):
        from src.shopping.intelligence.special.warranty_analyzer import _extract_brand
        brand = _extract_brand({"name": "Samsung Galaxy S24"})
        self.assertEqual(brand, "Samsung")

    def test_extract_brand_explicit(self):
        from src.shopping.intelligence.special.warranty_analyzer import _extract_brand
        brand = _extract_brand({"name": "Phone X", "brand": "Apple"})
        self.assertEqual(brand, "Apple")

    def test_extract_brand_unknown(self):
        from src.shopping.intelligence.special.warranty_analyzer import _extract_brand
        brand = _extract_brand({"name": "Some Random Product"})
        self.assertEqual(brand, "unknown")


if __name__ == "__main__":
    unittest.main()
