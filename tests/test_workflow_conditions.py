"""Tests for the conditional group evaluator."""

import json
import unittest

from src.workflows.engine.conditions import evaluate_condition, resolve_group


class TestEvaluateCondition(unittest.TestCase):
    """Tests for evaluate_condition against all supported DSL patterns."""

    # --- length(field) >= N ---

    def test_evaluate_competitor_count_true(self):
        """length(competitors) >= 3 should be True when list has 3+ items."""
        artifact = json.dumps({
            "competitors": [
                {"name": "Comp A"},
                {"name": "Comp B"},
                {"name": "Comp C"},
            ]
        })
        self.assertTrue(
            evaluate_condition("length(competitors) >= 3", artifact)
        )

    def test_evaluate_competitor_count_false(self):
        """length(competitors) >= 3 should be False when list has 1 item."""
        artifact = json.dumps({
            "competitors": [{"name": "Only One"}]
        })
        self.assertFalse(
            evaluate_condition("length(competitors) >= 3", artifact)
        )

    # --- any(item.field == 'value') ---

    def test_evaluate_any_category(self):
        """any(req.category == 'realtime') True when a matching item exists."""
        artifact = json.dumps([
            {"category": "auth", "name": "Login"},
            {"category": "realtime", "name": "Live chat"},
        ])
        self.assertTrue(
            evaluate_condition("any(req.category == 'realtime')", artifact)
        )

    def test_evaluate_any_category_false(self):
        """any(req.category == 'realtime') False when no item matches."""
        artifact = json.dumps([
            {"category": "auth", "name": "Login"},
            {"category": "crud", "name": "User profile"},
        ])
        self.assertFalse(
            evaluate_condition("any(req.category == 'realtime')", artifact)
        )

    # --- field != 'value' ---

    def test_evaluate_pricing_model(self):
        """pricing_model != 'free' should be True for 'freemium'."""
        artifact = json.dumps({"pricing_model": "freemium"})
        self.assertTrue(
            evaluate_condition("pricing_model != 'free'", artifact)
        )

    def test_evaluate_pricing_model_free(self):
        """pricing_model != 'free' should be False when value is 'free'."""
        artifact = json.dumps({"pricing_model": "free"})
        self.assertFalse(
            evaluate_condition("pricing_model != 'free'", artifact)
        )

    # --- platforms_include('value') ---

    def test_evaluate_platforms_include(self):
        """platforms_include('ios') True when 'ios' is in platforms list."""
        artifact = json.dumps({"platforms": ["web", "ios"]})
        self.assertTrue(
            evaluate_condition("platforms_include('ios')", artifact)
        )

    def test_evaluate_platforms_include_false(self):
        """platforms_include('ios') False when platforms has no ios/android."""
        artifact = json.dumps({"platforms": ["web"]})
        self.assertFalse(
            evaluate_condition("platforms_include('ios')", artifact)
        )

    # --- field == true/false (boolean) ---

    def test_evaluate_boolean_field(self):
        """has_public_web_pages == true should be True when field is True."""
        artifact = json.dumps({"has_public_web_pages": True})
        self.assertTrue(
            evaluate_condition("has_public_web_pages == true", artifact)
        )

    def test_evaluate_boolean_field_false(self):
        """has_public_web_pages == true should be False when field is False."""
        artifact = json.dumps({"has_public_web_pages": False})
        self.assertFalse(
            evaluate_condition("has_public_web_pages == true", artifact)
        )

    # --- OR expression ---

    def test_evaluate_or_expression(self):
        """OR expression: True when at least one side matches."""
        artifact = json.dumps({"platforms": ["web", "android"]})
        self.assertTrue(
            evaluate_condition(
                "platforms_include('ios') OR platforms_include('android')",
                artifact,
            )
        )

    def test_evaluate_or_expression_neither(self):
        """OR expression: False when neither side matches."""
        artifact = json.dumps({"platforms": ["web"]})
        self.assertFalse(
            evaluate_condition(
                "platforms_include('ios') OR platforms_include('android')",
                artifact,
            )
        )

    # --- unknown / malformed ---

    def test_evaluate_unknown_returns_false(self):
        """Unrecognized expressions should safely return False."""
        self.assertFalse(
            evaluate_condition("some_unknown_expression!!!", "{}")
        )

    def test_evaluate_invalid_json_returns_false(self):
        """Non-JSON artifact string should not crash; returns False."""
        self.assertFalse(
            evaluate_condition("length(competitors) >= 3", "not json at all")
        )


class TestResolveGroup(unittest.TestCase):
    """Tests for resolve_group which maps condition results to step lists."""

    def test_resolve_group_true(self):
        """When condition is True, returns (if_true, if_false) as included/excluded."""
        group = {
            "group_id": "competitor_deep_dive",
            "condition_check": "length(competitors) >= 3",
            "if_true": ["1.5", "1.6"],
            "if_false": ["1.5_lite"],
            "fallback_steps": [{"id": "1.5_lite"}],
        }
        artifact = json.dumps({
            "competitors": [{"n": 1}, {"n": 2}, {"n": 3}]
        })
        included, excluded = resolve_group(group, artifact)
        self.assertEqual(included, ["1.5", "1.6"])
        self.assertEqual(excluded, ["1.5_lite"])

    def test_resolve_group_false_with_fallback(self):
        """When condition is False, returns (if_false + fallback_ids, if_true) as included/excluded."""
        group = {
            "group_id": "competitor_deep_dive",
            "condition_check": "length(competitors) >= 3",
            "if_true": ["1.5", "1.6"],
            "if_false": ["1.5_lite"],
            "fallback_steps": [{"id": "1.5_lite"}, {"id": "1.5_extra"}],
        }
        artifact = json.dumps({
            "competitors": [{"n": 1}]
        })
        included, excluded = resolve_group(group, artifact)
        # included = if_false + fallback ids
        self.assertIn("1.5_lite", included)
        self.assertIn("1.5_extra", included)
        # excluded = if_true
        self.assertEqual(excluded, ["1.5", "1.6"])

    def test_resolve_group_no_fallback(self):
        """When condition is False and no fallback_steps, included is just if_false."""
        group = {
            "group_id": "payment_flow",
            "condition_check": "pricing_model != 'free'",
            "if_true": ["13.30", "14.4"],
            "if_false": [],
        }
        artifact = json.dumps({"pricing_model": "free"})
        included, excluded = resolve_group(group, artifact)
        self.assertEqual(included, [])
        self.assertEqual(excluded, ["13.30", "14.4"])


if __name__ == "__main__":
    unittest.main()
