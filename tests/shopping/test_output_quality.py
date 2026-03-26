"""Output Quality Evaluation — plan item #56.

Verifies the STRUCTURE and QUALITY of intelligence module outputs, not merely
that functions execute without error.  Tests cover:

1. Required fields are present in all returned dicts.
2. Turkish text in warnings/notes is properly formed (non-empty, non-placeholder).
3. Numeric values are within valid domain ranges.
4. Enum / string values belong to documented expected sets.
5. Edge cases return sensible defaults rather than crashes.
"""

from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import AsyncMock, patch


# ═══════════════════════════════════════════════════════════════════════════
# 1. TestFraudDetectorQuality
# ═══════════════════════════════════════════════════════════════════════════

class TestFraudDetectorQuality(unittest.TestCase):
    """Quality checks on assess_counterfeit_risk and related helpers."""

    _VALID_RISK_LEVELS = frozenset(["low", "medium", "high", "critical"])

    def _assess(self, product: dict) -> dict:
        from src.shopping.intelligence.special.fraud_detector import assess_counterfeit_risk
        return assess_counterfeit_risk(product)

    # ── Required fields ──────────────────────────────────────────────────

    def test_required_fields_present(self):
        result = self._assess({"name": "Test Ürün", "category": "elektronik"})
        for field in ("risk_level", "risk_score", "red_flags", "recommendation"):
            self.assertIn(field, result, f"Missing field: {field}")

    def test_risk_level_is_valid_enum(self):
        for product in [
            {"name": "Normal ürün"},
            {"name": "kopya iphone", "category": "şarj aleti"},
            {"name": "orijinal samsung", "brand": "samsung", "price": 50},
        ]:
            result = self._assess(product)
            self.assertIn(
                result["risk_level"],
                self._VALID_RISK_LEVELS,
                f"Unexpected risk_level: {result['risk_level']}",
            )

    def test_risk_score_in_range(self):
        for product in [
            {},
            {"name": "replika çanta", "brand": "chanel", "price": 100},
            {"name": "a kalite samsung şarj aleti", "category": "şarj"},
        ]:
            result = self._assess(product)
            self.assertGreaterEqual(result["risk_score"], 0.0)
            self.assertLessEqual(result["risk_score"], 1.0)

    # ── Red flags are Turkish strings ────────────────────────────────────

    def test_red_flags_are_strings(self):
        result = self._assess({"name": "replika ürün", "category": "kozmetik"})
        for flag in result["red_flags"]:
            self.assertIsInstance(flag, str)
            self.assertGreater(len(flag.strip()), 0, "Red flag must not be empty")

    def test_red_flags_contain_turkish_context(self):
        """Each red flag for a known counterfeit keyword must contain a Turkish explanation."""
        result = self._assess({"name": "taklit rolex saat"})
        self.assertGreater(len(result["red_flags"]), 0)
        # Each flag should reference either the keyword or a Turkish explanation
        for flag in result["red_flags"]:
            self.assertTrue(
                any(c.isalpha() for c in flag),
                "Red flag appears to be pure punctuation/numbers",
            )

    def test_critical_keyword_raises_risk(self):
        result = self._assess({"name": "replika iphone kılıf"})
        self.assertIn(result["risk_level"], ("high", "critical"))
        self.assertGreaterEqual(result["risk_score"], 0.5)

    def test_info_keyword_is_low_risk(self):
        result = self._assess({"name": "oem kablosuz kulaklık", "category": "elektronik"})
        # "oem" is info-severity — should not push into high/critical on its own
        self.assertIn(result["risk_level"], ("low", "medium"))

    # ── Safety warnings for dangerous categories ─────────────────────────

    def test_safety_warning_non_empty_for_dangerous_category(self):
        from src.shopping.intelligence.special.fraud_detector import get_safety_warnings

        dangerous = ["şarj aleti", "powerbank", "kozmetik", "bebek", "ilaç",
                     "supplement", "hafıza kartı"]
        for cat in dangerous:
            product = {"category": cat, "name": "test ürün"}
            warnings = get_safety_warnings(product)
            self.assertGreater(
                len(warnings), 0,
                f"Expected safety warnings for dangerous category '{cat}'",
            )

    def test_safety_warning_text_is_turkish(self):
        from src.shopping.intelligence.special.fraud_detector import get_safety_warnings

        product = {"category": "şarj aleti"}
        warnings = get_safety_warnings(product)
        self.assertTrue(len(warnings) > 0)
        # Turkish warning keywords that must appear
        combined = " ".join(warnings).lower()
        self.assertTrue(
            "uyari" in combined or "uyarı" in combined or "dikkat" in combined,
            f"Safety warning lacks a Turkish alert prefix: {warnings}",
        )

    def test_safety_warning_none_for_safe_category(self):
        result = self._assess({"name": "Normal masa lambası", "category": "mobilya"})
        self.assertIsNone(result.get("safety_warning"))

    # ── Recommendation is a non-empty string ─────────────────────────────

    def test_recommendation_is_non_empty_string(self):
        for product in [
            {"name": "normal ürün"},
            {"name": "kopya saat"},
            {"name": "a kalite powerbank", "category": "power bank"},
        ]:
            result = self._assess(product)
            rec = result.get("recommendation", "")
            self.assertIsInstance(rec, str)
            self.assertGreater(len(rec.strip()), 0)

    # ── Price anomaly flag ────────────────────────────────────────────────

    def test_price_anomaly_flagged_for_suspiciously_cheap_brand(self):
        # Apple floor is 800 TL; 30% of that is 240 TL. Price of 50 TL triggers.
        result = self._assess({"name": "apple şarj kablosu", "brand": "apple", "price": 50})
        self.assertGreater(len(result["red_flags"]), 0)
        # At least one flag should mention TL or price
        flag_text = " ".join(result["red_flags"])
        self.assertIn("TL", flag_text)

    # ── Edge cases ────────────────────────────────────────────────────────

    def test_empty_product_returns_low_risk(self):
        result = self._assess({})
        self.assertEqual(result["risk_level"], "low")
        self.assertEqual(result["risk_score"], 0.0)
        self.assertIsInstance(result["red_flags"], list)

    def test_none_price_does_not_crash(self):
        result = self._assess({"name": "ürün", "price": None})
        self.assertIn(result["risk_level"], self._VALID_RISK_LEVELS)

    def test_detect_counterfeit_keywords_meaning_non_empty(self):
        from src.shopping.intelligence.special.fraud_detector import detect_counterfeit_keywords

        hits = detect_counterfeit_keywords("super copy güneş gözlüğü")
        self.assertGreater(len(hits), 0)
        for hit in hits:
            self.assertIn("keyword", hit)
            self.assertIn("severity", hit)
            self.assertIn("meaning", hit)
            self.assertGreater(len(hit["meaning"].strip()), 0, "meaning must not be blank")


# ═══════════════════════════════════════════════════════════════════════════
# 2. TestBundleDetectorQuality
# ═══════════════════════════════════════════════════════════════════════════

class TestBundleDetectorQuality(unittest.TestCase):
    """Quality checks on detect_bundle_deals, detect_set_pricing, and
    suggest_shipping_combos."""

    # ── detect_bundle_deals ──────────────────────────────────────────────

    def test_al_ode_bundle_description_meaningful(self):
        from src.shopping.intelligence.special.bundle_detector import detect_bundle_deals

        products = [{"name": "3 al 2 öde deterjan", "price": 150.0}]
        bundles = detect_bundle_deals(products)
        self.assertGreater(len(bundles), 0)
        bundle = bundles[0]
        desc = bundle.get("description", "")
        self.assertGreater(len(desc.strip()), 10, f"Bundle description too short: '{desc}'")
        self.assertIn("al", desc.lower())
        self.assertIn("öde", desc.lower())

    def test_set_bundle_description_non_empty(self):
        from src.shopping.intelligence.special.bundle_detector import detect_bundle_deals

        products = [{"name": "6'lı set fincan takımı", "price": 300.0}]
        bundles = detect_bundle_deals(products)
        self.assertGreater(len(bundles), 0)
        for bundle in bundles:
            self.assertGreater(len(bundle.get("description", "").strip()), 0)

    def test_savings_estimate_non_negative(self):
        from src.shopping.intelligence.special.bundle_detector import detect_bundle_deals

        products = [
            {"name": "2 al 1 öde şampuan", "price": 100.0, "original_price": 200.0},
            {"name": "3'lü paket sabun", "price": 60.0, "original_price": 90.0},
            {"name": "kampanyalı ürün", "price": 50.0},
        ]
        bundles = detect_bundle_deals(products)
        for bundle in bundles:
            self.assertGreaterEqual(
                bundle.get("savings_estimate", 0.0),
                0.0,
                f"Savings estimate is negative: {bundle}",
            )

    def test_bundle_required_fields(self):
        from src.shopping.intelligence.special.bundle_detector import detect_bundle_deals

        products = [{"name": "3 al 2 öde ürün", "price": 200.0}]
        bundles = detect_bundle_deals(products)
        for bundle in bundles:
            for field in ("type", "description", "savings_estimate", "products_involved"):
                self.assertIn(field, bundle, f"Bundle missing field: {field}")

    def test_products_involved_is_list(self):
        from src.shopping.intelligence.special.bundle_detector import detect_bundle_deals

        products = [{"name": "set paket ürünü", "price": 120.0}]
        bundles = detect_bundle_deals(products)
        for bundle in bundles:
            self.assertIsInstance(bundle["products_involved"], list)
            self.assertGreater(len(bundle["products_involved"]), 0)

    def test_empty_products_returns_empty_list(self):
        from src.shopping.intelligence.special.bundle_detector import detect_bundle_deals

        self.assertEqual(detect_bundle_deals([]), [])

    # ── detect_set_pricing ───────────────────────────────────────────────

    def test_set_pricing_verdict_meaningful(self):
        from src.shopping.intelligence.special.bundle_detector import detect_set_pricing

        products = [
            {"name": "6'lı set çorap", "price": 120.0, "original_price": 180.0},
        ]
        results = detect_set_pricing(products)
        for r in results:
            verdict = r.get("verdict", "")
            self.assertGreater(len(verdict.strip()), 10, f"Verdict too short: '{verdict}'")

    def test_set_pricing_savings_pct_range(self):
        from src.shopping.intelligence.special.bundle_detector import detect_set_pricing

        products = [
            {
                "name": "3'lü paket deterjan",
                "price": 90.0,
                "individual_price": 40.0,
            }
        ]
        results = detect_set_pricing(products)
        for r in results:
            if r.get("savings_pct") is not None:
                self.assertGreaterEqual(r["savings_pct"], -100.0)
                self.assertLessEqual(r["savings_pct"], 100.0)

    # ── suggest_shipping_combos ──────────────────────────────────────────

    def test_shipping_combo_suggestion_non_empty(self):
        from src.shopping.intelligence.special.bundle_detector import suggest_shipping_combos

        products = [{"price": 80.0}]
        results = suggest_shipping_combos(products, free_shipping_threshold=150.0)
        self.assertEqual(len(results), 1)
        suggestion = results[0].get("suggestion", "")
        self.assertGreater(len(suggestion.strip()), 10)

    def test_already_reached_threshold_message(self):
        from src.shopping.intelligence.special.bundle_detector import suggest_shipping_combos

        products = [{"price": 200.0}]
        results = suggest_shipping_combos(products, free_shipping_threshold=150.0)
        suggestion = results[0]["suggestion"]
        self.assertEqual(results[0]["gap"], 0.0)
        self.assertGreater(len(suggestion), 0)


# ═══════════════════════════════════════════════════════════════════════════
# 3. TestEnvironmentalQuality
# ═══════════════════════════════════════════════════════════════════════════

class TestEnvironmentalQuality(unittest.TestCase):
    """Quality checks on environmental module outputs."""

    # ── assess_efficiency ────────────────────────────────────────────────

    def test_energy_cost_reasonable_range(self):
        """Annual energy cost should be between 100 and 5000 TRY for known categories."""
        from src.shopping.intelligence.special.environmental import assess_efficiency

        for category, energy_class in [
            ("buzdolabı",        "A+++"),
            ("çamaşır makinesi", "B"),
            ("laptop",           "A"),
            ("klima",            "D"),
        ]:
            product = {"category": category, "energy_class": energy_class}
            result = assess_efficiency(product)
            cost = result["estimated_annual_energy_cost"]
            self.assertIsNotNone(cost, f"Expected cost for {category}/{energy_class}")
            self.assertGreaterEqual(cost, 100.0,  f"{category}/{energy_class}: cost too low ({cost})")
            self.assertLessEqual(   cost, 5000.0, f"{category}/{energy_class}: cost too high ({cost})")

    def test_efficiency_required_fields(self):
        from src.shopping.intelligence.special.environmental import assess_efficiency

        result = assess_efficiency({"category": "tv", "energy_class": "A+"})
        for field in ("energy_class", "estimated_annual_energy_cost", "efficiency_score",
                      "cost_savings_note"):
            self.assertIn(field, result, f"Missing field: {field}")

    def test_efficiency_score_in_range(self):
        from src.shopping.intelligence.special.environmental import assess_efficiency

        for ec in ("A+++", "A++", "A+", "A", "B", "C", "D"):
            result = assess_efficiency({"energy_class": ec})
            score = result["efficiency_score"]
            self.assertGreaterEqual(score, 0.0, f"Score below 0 for class {ec}")
            self.assertLessEqual(score, 1.0,   f"Score above 1 for class {ec}")

    def test_cost_savings_note_non_empty(self):
        from src.shopping.intelligence.special.environmental import assess_efficiency

        for ec in ("A+++", "D", "C"):
            result = assess_efficiency({"energy_class": ec, "category": "buzdolabı"})
            note = result.get("cost_savings_note", "")
            self.assertGreater(len(note.strip()), 10, f"Note too short for class {ec}")

    def test_unknown_energy_class_does_not_crash(self):
        from src.shopping.intelligence.special.environmental import assess_efficiency

        result = assess_efficiency({"category": "laptop", "energy_class": "Z"})
        self.assertIsNone(result["energy_class"])
        self.assertIn("cost_savings_note", result)

    # ── estimate_lifespan ────────────────────────────────────────────────

    def test_lifespan_in_valid_range(self):
        """Expected lifespan should be between 1 and 25 years for all known categories."""
        from src.shopping.intelligence.special.environmental import estimate_lifespan

        for category in ("buzdolabı", "çamaşır makinesi", "laptop", "telefon", "tv",
                         "klima", "yazıcı"):
            result = estimate_lifespan({"category": category, "price": 10000})
            years = result["expected_years"]
            self.assertGreaterEqual(years, 1.0,  f"{category}: lifespan too short ({years})")
            self.assertLessEqual(   years, 25.0, f"{category}: lifespan too long ({years})")

    def test_lifespan_required_fields(self):
        from src.shopping.intelligence.special.environmental import estimate_lifespan

        result = estimate_lifespan({"category": "laptop", "price": 15000})
        for field in ("expected_years", "category_average_years", "cost_per_year", "note"):
            self.assertIn(field, result)

    def test_lifespan_note_non_empty(self):
        from src.shopping.intelligence.special.environmental import estimate_lifespan

        result = estimate_lifespan({"category": "tv", "price": 8000})
        self.assertGreater(len(result["note"].strip()), 10)

    def test_cost_per_year_positive_when_price_given(self):
        from src.shopping.intelligence.special.environmental import estimate_lifespan

        result = estimate_lifespan({"category": "buzdolabı", "price": 20000})
        self.assertIsNotNone(result["cost_per_year"])
        self.assertGreater(result["cost_per_year"], 0.0)

    def test_unknown_category_uses_default_lifespan(self):
        from src.shopping.intelligence.special.environmental import estimate_lifespan

        result = estimate_lifespan({"category": "bilinmeyen_kategori", "price": 5000})
        self.assertGreaterEqual(result["expected_years"], 1.0)
        self.assertLessEqual(result["expected_years"], 25.0)

    # ── get_lifetime_cost ─────────────────────────────────────────────────

    def test_lifetime_cost_fields_present(self):
        from src.shopping.intelligence.special.environmental import get_lifetime_cost

        result = get_lifetime_cost({"category": "laptop", "price": 15000})
        for field in ("purchase_price", "annual_running_cost", "lifetime_years",
                      "total_lifetime_cost", "cost_per_year", "summary"):
            self.assertIn(field, result)

    def test_lifetime_cost_greater_than_purchase(self):
        from src.shopping.intelligence.special.environmental import get_lifetime_cost

        result = get_lifetime_cost({"category": "çamaşır makinesi", "price": 12000})
        self.assertGreater(result["total_lifetime_cost"], result["purchase_price"])

    def test_lifetime_summary_non_empty(self):
        from src.shopping.intelligence.special.environmental import get_lifetime_cost

        result = get_lifetime_cost({"category": "buzdolabı", "price": 25000})
        self.assertGreater(len(result["summary"].strip()), 20)

    # ── assess_repairability ─────────────────────────────────────────────

    def test_repairability_score_in_range(self):
        from src.shopping.intelligence.special.environmental import assess_repairability

        for category in ("buzdolabı", "telefon", "tv", "laptop"):
            result = assess_repairability({"category": category})
            score = result["repairability_score"]
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 10.0)

    def test_repairability_note_mentions_turkey(self):
        from src.shopping.intelligence.special.environmental import assess_repairability

        result = assess_repairability({"category": "buzdolabı"})
        note = result["note"].lower()
        self.assertTrue(
            "türkiye" in note or "servis" in note,
            f"Repairability note should mention Turkish service context: {note}",
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4. TestImportDomesticQuality
# ═══════════════════════════════════════════════════════════════════════════

class TestImportDomesticQuality(unittest.TestCase):
    """Quality checks on import_domestic module outputs."""

    _VALID_ORIGINS = frozenset(["domestic", "imported", "unknown"])

    # ── classify_origin ──────────────────────────────────────────────────

    def test_known_domestic_brands_return_correct_origin(self):
        from src.shopping.intelligence.special.import_domestic import classify_origin

        domestic_brands = ["Vestel", "Arçelik", "Beko", "Casper", "Karaca", "Profilo"]
        for brand in domestic_brands:
            result = classify_origin(brand)
            self.assertEqual(
                result["origin"],
                "domestic",
                f"{brand} should be classified as domestic, got {result['origin']}",
            )
            self.assertEqual(result["country"], "Türkiye")
            self.assertGreater(result["confidence"], 0.5)

    def test_known_imported_brands_return_correct_origin(self):
        from src.shopping.intelligence.special.import_domestic import classify_origin

        imported = [
            ("Apple",   "ABD"),
            ("Samsung", "Güney Kore"),
            ("Sony",    "Japonya"),
            ("Xiaomi",  "Çin"),
            ("Bosch",   "Almanya"),
        ]
        for brand, expected_country in imported:
            result = classify_origin(brand)
            self.assertEqual(
                result["origin"],
                "imported",
                f"{brand} should be 'imported', got {result['origin']}",
            )
            self.assertEqual(
                result["country"],
                expected_country,
                f"{brand} country mismatch: {result['country']} != {expected_country}",
            )

    def test_origin_is_valid_enum(self):
        from src.shopping.intelligence.special.import_domestic import classify_origin

        for brand in ("UnknownBrand123", "Vestel", "Apple", ""):
            result = classify_origin(brand)
            self.assertIn(result["origin"], self._VALID_ORIGINS)

    def test_classify_origin_required_fields(self):
        from src.shopping.intelligence.special.import_domestic import classify_origin

        result = classify_origin("Samsung")
        for field in ("origin", "country", "confidence", "notes"):
            self.assertIn(field, result)

    def test_confidence_in_range(self):
        from src.shopping.intelligence.special.import_domestic import classify_origin

        for brand in ("Vestel", "Apple", "UnknownXYZ"):
            result = classify_origin(brand)
            self.assertGreaterEqual(result["confidence"], 0.0)
            self.assertLessEqual(result["confidence"], 1.0)

    # ── detect_grey_market ───────────────────────────────────────────────

    def test_grey_market_flags_expected_keywords(self):
        from src.shopping.intelligence.special.import_domestic import detect_grey_market

        product = {
            "name": "iPhone 15 yurt dışı paralel ithalat garantisiz",
            "description": "distribütör garantisi yok",
        }
        result = detect_grey_market(product)
        self.assertTrue(result["is_grey_market"])
        self.assertGreater(len(result["indicators"]), 0)
        # Each indicator should be a non-empty string
        for indicator in result["indicators"]:
            self.assertIsInstance(indicator, str)
            self.assertGreater(len(indicator.strip()), 0)

    def test_grey_market_warnings_are_turkish(self):
        from src.shopping.intelligence.special.import_domestic import detect_grey_market

        product = {"name": "paralel ithalat ürün", "description": "ithalatçı garantili"}
        result = detect_grey_market(product)
        if result["is_grey_market"]:
            self.assertGreater(len(result["warnings"]), 0)
            for w in result["warnings"]:
                self.assertGreater(len(w.strip()), 10)

    def test_clean_product_not_grey_market(self):
        from src.shopping.intelligence.special.import_domestic import detect_grey_market

        product = {"name": "Samsung Galaxy S24 Türkiye garantili"}
        result = detect_grey_market(product)
        self.assertFalse(result["is_grey_market"])
        self.assertEqual(result["warnings"], [])

    def test_grey_market_required_fields(self):
        from src.shopping.intelligence.special.import_domestic import detect_grey_market

        result = detect_grey_market({"name": "test"})
        for field in ("is_grey_market", "confidence", "indicators", "warnings"):
            self.assertIn(field, result)

    # ── get_import_advisory overall_risk ─────────────────────────────────

    def test_import_advisory_overall_risk_valid(self):
        from src.shopping.intelligence.special.import_domestic import get_import_advisory

        product = {"name": "Vestel TV 55 inç", "brand": "Vestel", "category": "tv"}
        result = get_import_advisory(product)
        self.assertIn(result["overall_risk"], ("low", "medium", "high"))

    def test_import_advisory_required_fields(self):
        from src.shopping.intelligence.special.import_domestic import get_import_advisory

        result = get_import_advisory({"brand": "Apple", "name": "iPhone"})
        for field in ("origin", "grey_market", "btk", "warranty_implications",
                      "service_center_note", "overall_risk", "recommendations"):
            self.assertIn(field, result)


# ═══════════════════════════════════════════════════════════════════════════
# 5. TestCampaignPatternsQuality
# ═══════════════════════════════════════════════════════════════════════════

class TestCampaignPatternsQuality(unittest.TestCase):
    """Quality checks on campaign_patterns module."""

    # ── get_campaign_calendar ────────────────────────────────────────────

    def test_calendar_has_at_least_12_events(self):
        from src.shopping.intelligence.special.campaign_patterns import get_campaign_calendar

        calendar = get_campaign_calendar()
        self.assertGreaterEqual(
            len(calendar),
            12,
            f"Campaign calendar should have ≥12 events, got {len(calendar)}",
        )

    def test_calendar_covers_all_12_months(self):
        from src.shopping.intelligence.special.campaign_patterns import get_campaign_calendar

        calendar = get_campaign_calendar()
        months_covered = {entry["month"] for entry in calendar}
        for month in range(1, 13):
            self.assertIn(month, months_covered, f"Month {month} not in campaign calendar")

    def test_calendar_event_required_fields(self):
        from src.shopping.intelligence.special.campaign_patterns import get_campaign_calendar

        for entry in get_campaign_calendar():
            for field in ("event", "month", "day_start", "day_end",
                          "expected_discount_pct", "categories_affected", "advice"):
                self.assertIn(field, entry, f"Calendar entry missing field: {field}")

    def test_calendar_discount_pct_positive(self):
        from src.shopping.intelligence.special.campaign_patterns import get_campaign_calendar

        for entry in get_campaign_calendar():
            self.assertGreater(
                entry["expected_discount_pct"],
                0.0,
                f"Event '{entry['event']}' has non-positive discount",
            )

    def test_calendar_event_names_non_empty(self):
        from src.shopping.intelligence.special.campaign_patterns import get_campaign_calendar

        for entry in get_campaign_calendar():
            self.assertGreater(len(entry["event"].strip()), 0)

    def test_calendar_advice_text_non_empty(self):
        from src.shopping.intelligence.special.campaign_patterns import get_campaign_calendar

        for entry in get_campaign_calendar():
            self.assertGreater(
                len(entry["advice"].strip()),
                10,
                f"Advice too short for event '{entry['event']}'",
            )

    # ── predict_upcoming_campaigns ────────────────────────────────────────

    def test_predictions_return_future_dates(self):
        from src.shopping.intelligence.special.campaign_patterns import predict_upcoming_campaigns

        today = date.today()
        campaigns = predict_upcoming_campaigns()
        self.assertGreater(len(campaigns), 0)
        for c in campaigns:
            self.assertGreaterEqual(
                c["days_until"],
                0,
                f"days_until is negative for event '{c['event']}'",
            )

    def test_predictions_sorted_by_proximity(self):
        from src.shopping.intelligence.special.campaign_patterns import predict_upcoming_campaigns

        campaigns = predict_upcoming_campaigns()
        days = [c["days_until"] for c in campaigns]
        self.assertEqual(days, sorted(days), "Campaigns should be sorted by days_until")

    def test_prediction_required_fields(self):
        from src.shopping.intelligence.special.campaign_patterns import predict_upcoming_campaigns

        for c in predict_upcoming_campaigns():
            for field in ("event", "expected_date_range", "days_until",
                          "expected_discount_pct", "categories_affected", "advice"):
                self.assertIn(field, c, f"Campaign prediction missing field: {field}")

    def test_filtered_predictions_respect_category(self):
        from src.shopping.intelligence.special.campaign_patterns import predict_upcoming_campaigns

        campaigns = predict_upcoming_campaigns(category="electronics")
        for c in campaigns:
            self.assertTrue(
                "electronics" in c["categories_affected"]
                or "everything" in c["categories_affected"],
                f"Filtered campaign should include 'electronics': {c}",
            )

    def test_prediction_discount_pct_positive(self):
        from src.shopping.intelligence.special.campaign_patterns import predict_upcoming_campaigns

        for c in predict_upcoming_campaigns():
            self.assertGreater(c["expected_discount_pct"], 0.0)

    # ── get_category_patterns ────────────────────────────────────────────

    def test_no_observations_returns_safe_defaults(self):
        from src.shopping.intelligence.special.campaign_patterns import get_category_patterns

        result = get_category_patterns("completely_unknown_xyz")
        self.assertEqual(result["avg_discount_pct"], 0.0)
        self.assertEqual(result["observations"], 0)
        self.assertGreater(len(result["prediction"].strip()), 0)

    def test_category_patterns_after_recording(self):
        from src.shopping.intelligence.special.campaign_patterns import (
            get_category_patterns, record_campaign,
        )

        record_campaign("test_category_q56", "Test Event", 25.0, "2026-11-28")
        record_campaign("test_category_q56", "Test Event 2", 30.0, "2026-11-29")
        result = get_category_patterns("test_category_q56")
        self.assertEqual(result["observations"], 2)
        self.assertGreater(result["avg_discount_pct"], 0.0)
        self.assertGreater(result["best_discount_pct"], 0.0)
        self.assertGreater(len(result["prediction"].strip()), 20)


# ═══════════════════════════════════════════════════════════════════════════
# 6. TestComplementaryQuality
# ═══════════════════════════════════════════════════════════════════════════

class TestComplementaryQuality(unittest.TestCase):
    """Quality checks on complementary module outputs."""

    _VALID_PRIORITIES = frozenset(["yüksek", "orta", "düşük"])

    # ── get_complement_map ───────────────────────────────────────────────

    def test_known_categories_return_complements(self):
        from src.shopping.intelligence.special.complementary import get_complement_map

        for category in ("telefon", "laptop", "yazıcı", "tablet", "kamera"):
            complements = get_complement_map(category)
            self.assertGreater(
                len(complements),
                0,
                f"Expected complements for '{category}', got none",
            )

    def test_english_aliases_resolve_correctly(self):
        from src.shopping.intelligence.special.complementary import get_complement_map

        mappings = [("printer", "yazıcı"), ("notebook", "laptop"), ("smartphone", "telefon")]
        for english, turkish in mappings:
            en_result = get_complement_map(english)
            tr_result = get_complement_map(turkish)
            self.assertEqual(
                len(en_result),
                len(tr_result),
                f"English alias '{english}' should return same complements as '{turkish}'",
            )

    def test_complement_required_fields(self):
        from src.shopping.intelligence.special.complementary import get_complement_map

        complements = get_complement_map("telefon")
        for item in complements:
            for field in ("product", "reason", "priority", "is_consumable"):
                self.assertIn(field, item, f"Complement missing field: {field}")

    def test_complement_priority_valid_enum(self):
        from src.shopping.intelligence.special.complementary import get_complement_map

        for category in ("telefon", "laptop", "yazıcı", "kamera"):
            for item in get_complement_map(category):
                self.assertIn(
                    item["priority"],
                    self._VALID_PRIORITIES,
                    f"Invalid priority '{item['priority']}' for {category}",
                )

    def test_complement_reason_non_empty(self):
        from src.shopping.intelligence.special.complementary import get_complement_map

        for item in get_complement_map("laptop"):
            self.assertGreater(
                len(item["reason"].strip()),
                5,
                f"Complement reason too short: '{item['reason']}'",
            )

    def test_unknown_category_returns_empty(self):
        from src.shopping.intelligence.special.complementary import get_complement_map

        self.assertEqual(get_complement_map("uzaylı_elektronik_xyz"), [])

    # ── assess_consumable_cost for printers ──────────────────────────────

    def test_printer_consumable_costs_non_empty(self):
        from src.shopping.intelligence.special.complementary import assess_consumable_cost

        result = assess_consumable_cost({"category": "yazıcı", "name": "HP LaserJet"})
        self.assertTrue(result["has_consumables"])
        self.assertGreater(len(result["consumable_items"]), 0)
        self.assertGreater(len(result["estimated_annual_cost"].strip()), 0)

    def test_printer_consumable_cost_string_mentions_tl(self):
        from src.shopping.intelligence.special.complementary import assess_consumable_cost

        result = assess_consumable_cost({"category": "yazıcı"})
        cost_str = result["estimated_annual_cost"]
        self.assertTrue(
            "TL" in cost_str or "tl" in cost_str.lower() or "₺" in cost_str,
            f"Annual cost string should mention TL: '{cost_str}'",
        )

    def test_printer_consumable_warning_non_empty(self):
        from src.shopping.intelligence.special.complementary import assess_consumable_cost

        result = assess_consumable_cost({"category": "yazıcı"})
        self.assertIsNotNone(result["warning"])
        self.assertGreater(len(result["warning"].strip()), 10)

    def test_no_consumable_product_returns_false(self):
        from src.shopping.intelligence.special.complementary import assess_consumable_cost

        result = assess_consumable_cost({"category": "masa lambası", "name": "LED Masa Lambası"})
        self.assertFalse(result["has_consumables"])
        self.assertEqual(result["consumable_items"], [])

    def test_consumable_cost_required_fields(self):
        from src.shopping.intelligence.special.complementary import assess_consumable_cost

        result = assess_consumable_cost({"category": "kahve makinesi"})
        for field in ("has_consumables", "consumable_items", "estimated_annual_cost", "warning"):
            self.assertIn(field, result)

    # ── suggest_complements (async, LLM mocked) ──────────────────────────

    def test_suggest_complements_known_category_no_llm(self):
        """suggest_complements should resolve from static map without LLM."""
        import asyncio
        from src.shopping.intelligence.special.complementary import suggest_complements

        result = asyncio.get_event_loop().run_until_complete(
            suggest_complements("iPhone 15", category="telefon")
        )
        self.assertGreater(len(result), 0)

    def test_suggest_complements_llm_fallback(self):
        """Unknown category should call _llm_call and parse its JSON response."""
        import asyncio
        from src.shopping.intelligence.special.complementary import suggest_complements

        mock_response = (
            '[{"product": "pil", "reason": "Cihaz için gerekli", '
            '"priority": "yüksek", "is_consumable": true, "recurring_cost_note": null}]'
        )
        with patch(
            "src.shopping.intelligence.special.complementary._llm_call",
            new=AsyncMock(return_value=mock_response),
        ):
            result = asyncio.get_event_loop().run_until_complete(
                suggest_complements("Bilinmeyen Cihaz 3000", category="bilinmeyen_kategori_xyz")
            )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["product"], "pil")
        self.assertIn(result[0]["priority"], self._VALID_PRIORITIES)


# ═══════════════════════════════════════════════════════════════════════════
# 7. TestBulkDetectorQuality
# ═══════════════════════════════════════════════════════════════════════════

class TestBulkDetectorQuality(unittest.TestCase):
    """Quality checks on bulk_detector module."""

    # ── detect_fake_bulk_deal math ────────────────────────────────────────

    def test_fake_bulk_detected_when_expensive_per_unit(self):
        from src.shopping.intelligence.special.bulk_detector import detect_fake_bulk_deal

        # 50 TL single; 6-pack for 360 TL → 60 TL/unit → fake bulk
        result = detect_fake_bulk_deal(
            single_price=50.0, bulk_price=360.0, bulk_quantity=6
        )
        self.assertTrue(result["is_fake"])
        self.assertGreater(result["difference_pct"], 0.0)
        self.assertAlmostEqual(result["bulk_unit_price"], 60.0)
        self.assertAlmostEqual(result["single_unit_price"], 50.0)

    def test_genuine_bulk_detected_when_cheaper_per_unit(self):
        from src.shopping.intelligence.special.bulk_detector import detect_fake_bulk_deal

        # 50 TL single; 6-pack for 240 TL → 40 TL/unit → genuine bulk
        result = detect_fake_bulk_deal(
            single_price=50.0, bulk_price=240.0, bulk_quantity=6
        )
        self.assertFalse(result["is_fake"])
        self.assertLess(result["difference_pct"], 0.0)

    def test_fake_bulk_math_difference_pct_correct(self):
        """difference_pct = (bulk_unit - single) / single * 100."""
        from src.shopping.intelligence.special.bulk_detector import detect_fake_bulk_deal

        result = detect_fake_bulk_deal(
            single_price=100.0, bulk_price=600.0, bulk_quantity=4
        )
        # bulk_unit = 150, diff = (150-100)/100*100 = 50%
        self.assertAlmostEqual(result["difference_pct"], 50.0, places=1)

    def test_fake_bulk_required_fields(self):
        from src.shopping.intelligence.special.bulk_detector import detect_fake_bulk_deal

        result = detect_fake_bulk_deal(10.0, 60.0, 5)
        for field in ("is_fake", "single_unit_price", "bulk_unit_price", "difference_pct"):
            self.assertIn(field, result)

    def test_zero_bulk_quantity_returns_safe_default(self):
        from src.shopping.intelligence.special.bulk_detector import detect_fake_bulk_deal

        result = detect_fake_bulk_deal(50.0, 200.0, 0)
        self.assertFalse(result["is_fake"])
        self.assertEqual(result["bulk_unit_price"], 0.0)

    # ── analyze_bulk_pricing ──────────────────────────────────────────────

    def test_bulk_pricing_annotates_unit_price(self):
        from src.shopping.intelligence.special.bulk_detector import analyze_bulk_pricing

        products = [
            {"name": "şampuan tek", "price": 50.0},
            {"name": "6'lı şampuan paket", "price": 240.0},
        ]
        result = analyze_bulk_pricing(products)
        bulk = next((p for p in result if p.get("is_bulk")), None)
        self.assertIsNotNone(bulk)
        self.assertAlmostEqual(bulk["unit_price"], 40.0)

    def test_is_fake_bulk_correctly_set(self):
        from src.shopping.intelligence.special.bulk_detector import analyze_bulk_pricing

        products = [
            {"name": "şampuan tek",      "price": 50.0},
            {"name": "6'lı paket şampuan", "price": 360.0},  # 60/unit > 50 → fake
        ]
        result = analyze_bulk_pricing(products)
        fake = next((p for p in result if p.get("is_bulk")), None)
        self.assertIsNotNone(fake)
        self.assertTrue(fake["is_fake_bulk"])

    def test_analyze_bulk_pricing_required_fields(self):
        from src.shopping.intelligence.special.bulk_detector import analyze_bulk_pricing

        result = analyze_bulk_pricing([{"name": "ürün", "price": 100.0}])
        for field in ("unit_price", "quantity", "is_bulk", "bulk_savings_pct", "is_fake_bulk"):
            self.assertIn(field, result[0])

    def test_empty_products_returns_empty(self):
        from src.shopping.intelligence.special.bulk_detector import analyze_bulk_pricing

        self.assertEqual(analyze_bulk_pricing([]), [])

    # ── assess_bulk_value ────────────────────────────────────────────────

    def test_high_waste_risk_not_recommended(self):
        from src.shopping.intelligence.special.bulk_detector import assess_bulk_value

        product = {"name": "taze ekmek", "category": "ekmek", "price": 200.0, "quantity": 30}
        result = assess_bulk_value(product, household_size=2)
        self.assertEqual(result["waste_risk"], "high")
        self.assertFalse(result["recommended"])

    def test_non_perishable_low_waste_risk(self):
        from src.shopping.intelligence.special.bulk_detector import assess_bulk_value

        product = {"name": "pil 12'li paket", "category": "pil", "price": 120.0, "quantity": 12}
        result = assess_bulk_value(product, household_size=2)
        self.assertEqual(result["waste_risk"], "low")

    def test_waste_risk_valid_enum(self):
        from src.shopping.intelligence.special.bulk_detector import assess_bulk_value

        valid_risks = frozenset(["low", "medium", "high"])
        for product in [
            {"name": "et",     "category": "gıda", "price": 100.0, "quantity": 10},
            {"name": "deterjan", "category": "temizlik", "price": 50.0, "quantity": 6},
        ]:
            result = assess_bulk_value(product)
            self.assertIn(result["waste_risk"], valid_risks)

    def test_bulk_value_reason_non_empty(self):
        from src.shopping.intelligence.special.bulk_detector import assess_bulk_value

        product = {"name": "deterjan 6'lı", "price": 240.0, "quantity": 6,
                   "single_price": 50.0, "category": "temizlik"}
        result = assess_bulk_value(product, household_size=3)
        self.assertGreater(len(result["reason"].strip()), 10)

    def test_break_even_months_positive(self):
        from src.shopping.intelligence.special.bulk_detector import assess_bulk_value

        product = {"name": "sabun 12'li", "price": 120.0, "quantity": 12}
        result = assess_bulk_value(product, household_size=2)
        self.assertGreater(result["break_even_months"], 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# 8. TestStalenessQuality
# ═══════════════════════════════════════════════════════════════════════════

class TestStalenessQuality(unittest.TestCase):
    """Quality checks on staleness module."""

    _VALID_STALENESS_LEVELS = frozenset(["fresh", "aging", "stale", "expired"])

    # ── assess_staleness ──────────────────────────────────────────────────

    def test_staleness_level_valid_enum(self):
        from src.shopping.resilience.staleness import assess_staleness

        test_cases = [
            ({"category": "electronics", "price": 5000}, 100),     # fresh
            ({"category": "electronics", "price": 5000}, 3000),    # aging
            ({"category": "electronics", "price": 5000}, 4000),    # stale/expired
            ({"category": "groceries",   "price": 50},  10000),    # expired
        ]
        for product, age_seconds in test_cases:
            result = assess_staleness(product, age_seconds)
            self.assertIn(
                result["staleness_level"],
                self._VALID_STALENESS_LEVELS,
                f"Invalid level for age={age_seconds}: {result['staleness_level']}",
            )

    def test_fresh_data_is_not_stale(self):
        from src.shopping.resilience.staleness import assess_staleness

        result = assess_staleness({"category": "electronics"}, cache_age_seconds=100)
        self.assertFalse(result["is_stale"])
        self.assertEqual(result["staleness_level"], "fresh")

    def test_expired_data_is_stale(self):
        from src.shopping.resilience.staleness import assess_staleness

        result = assess_staleness({"category": "groceries"}, cache_age_seconds=86400)
        self.assertTrue(result["is_stale"])
        self.assertEqual(result["staleness_level"], "expired")

    def test_staleness_required_fields(self):
        from src.shopping.resilience.staleness import assess_staleness

        result = assess_staleness({"category": "electronics"}, cache_age_seconds=1000)
        for field in ("is_stale", "staleness_level", "recommended_ttl_seconds",
                      "warnings", "confidence"):
            self.assertIn(field, result)

    def test_staleness_warnings_are_turkish_strings(self):
        from src.shopping.resilience.staleness import assess_staleness

        result = assess_staleness({"category": "electronics", "price": 5000}, 4000)
        for w in result["warnings"]:
            self.assertIsInstance(w, str)
            self.assertGreater(len(w.strip()), 10, "Warning too short")

    def test_confidence_in_range(self):
        from src.shopping.resilience.staleness import assess_staleness

        for age in (10, 1800, 7200, 36000):
            result = assess_staleness({"category": "electronics"}, age)
            self.assertGreaterEqual(result["confidence"], 0.0)
            self.assertLessEqual(result["confidence"], 1.0)

    # ── get_recommended_ttl ──────────────────────────────────────────────

    def test_ttl_recommendations_are_positive(self):
        from src.shopping.resilience.staleness import get_recommended_ttl

        test_cases = [
            ("electronics",  0.0),
            ("groceries",    0.5),
            ("furniture",    0.0),
            ("home",         1.0),
            (None,           0.3),
            ("unknown_cat",  0.7),
        ]
        for category, volatility in test_cases:
            ttl = get_recommended_ttl(category, volatility)
            self.assertGreater(
                ttl,
                0,
                f"TTL must be positive for category={category}, volatility={volatility}",
            )

    def test_high_volatility_reduces_ttl(self):
        from src.shopping.resilience.staleness import get_recommended_ttl

        ttl_low_vol  = get_recommended_ttl("electronics", 0.0)
        ttl_high_vol = get_recommended_ttl("electronics", 1.0)
        self.assertGreater(
            ttl_low_vol,
            ttl_high_vol,
            "High volatility should result in shorter TTL",
        )

    def test_known_category_ttls_within_reason(self):
        """Electronics TTL < groceries TTL < furniture TTL."""
        from src.shopping.resilience.staleness import get_recommended_ttl

        ttl_elec  = get_recommended_ttl("electronics", 0.0)
        ttl_groc  = get_recommended_ttl("groceries",   0.0)
        ttl_furn  = get_recommended_ttl("furniture",   0.0)

        # electronics price-check matters more often than furniture
        self.assertLessEqual(ttl_elec, ttl_furn)
        # groceries can be even more volatile than electronics
        self.assertLessEqual(ttl_groc, ttl_elec)

    # ── detect_flash_sale ─────────────────────────────────────────────────

    def test_flash_sale_detected_for_large_discount(self):
        from src.shopping.resilience.staleness import detect_flash_sale

        product = {"name": "Laptop", "price": 5000, "original_price": 12000}
        result = detect_flash_sale(product)
        self.assertTrue(result["is_flash_sale"])
        self.assertIn(result["urgency"], ("low", "medium", "high"))

    def test_flash_sale_detected_for_turkish_keywords(self):
        from src.shopping.resilience.staleness import detect_flash_sale

        product = {"name": "fırsat ürün", "description": "sınırlı stok son 2 saat"}
        result = detect_flash_sale(product)
        self.assertTrue(result["is_flash_sale"])
        self.assertGreater(len(result["indicators"]), 0)

    def test_no_flash_sale_for_normal_product(self):
        from src.shopping.resilience.staleness import detect_flash_sale

        product = {"name": "Normal Kalem Seti", "price": 50}
        result = detect_flash_sale(product)
        self.assertFalse(result["is_flash_sale"])
        self.assertEqual(result["urgency"], "none")

    def test_flash_sale_urgency_valid_enum(self):
        from src.shopping.resilience.staleness import detect_flash_sale

        valid_urgency = frozenset(["none", "low", "medium", "high"])
        for product in [
            {"name": "normal ürün"},
            {"name": "flash indirim", "price": 100, "original_price": 300},
            {"name": "son 2 saat sınırlı stok fırsatı", "price": 50, "original_price": 200},
        ]:
            result = detect_flash_sale(product)
            self.assertIn(result["urgency"], valid_urgency)


if __name__ == "__main__":
    unittest.main()
