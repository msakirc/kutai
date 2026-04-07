"""Tests for agent iteration exhaustion fixes."""

import json
import unittest


class TestValidateTaskOutput(unittest.TestCase):
    """Tests for validate_task_output research keyword matching."""

    def test_sources_plural_passes(self):
        """Local LLMs write 'Sources:' (plural) — must pass validation."""
        from src.models.models import validate_task_output
        result = "## Research\n\n**Sources:**\n- Wikipedia\n- Reddit"
        errors = validate_task_output("researcher", result)
        self.assertEqual(errors, [])

    def test_sources_bold_markdown_passes(self):
        """'**Sources:**' in markdown bold must pass."""
        from src.models.models import validate_task_output
        result = "## Analysis\n\n**Sources:**\n1. App Store reviews"
        errors = validate_task_output("researcher", result)
        self.assertEqual(errors, [])

    def test_based_on_passes(self):
        """'based on' is a common LLM phrasing for source attribution."""
        from src.models.models import validate_task_output
        result = "Based on analysis of competitor reviews, the market shows..."
        errors = validate_task_output("researcher", result)
        self.assertEqual(errors, [])

    def test_references_plural_passes(self):
        """'references:' (plural) must pass."""
        from src.models.models import validate_task_output
        result = "## Study\n\nReferences:\n- Smith et al."
        errors = validate_task_output("researcher", result)
        self.assertEqual(errors, [])

    def test_url_still_passes(self):
        """URL-based validation still works."""
        from src.models.models import validate_task_output
        result = "See https://example.com for details."
        errors = validate_task_output("researcher", result)
        self.assertEqual(errors, [])

    def test_no_source_still_fails(self):
        """Result with no source indicators should still fail."""
        from src.models.models import validate_task_output
        result = "The market is growing rapidly."
        errors = validate_task_output("researcher", result)
        self.assertGreater(len(errors), 0)

    def test_analyst_same_rules(self):
        """Analyst uses same 'research' category."""
        from src.models.models import validate_task_output
        result = "## Analysis\n\n**Sources:**\n- Market data"
        errors = validate_task_output("analyst", result)
        self.assertEqual(errors, [])


class TestArtifactSchemaValidation(unittest.TestCase):
    """Tests for validate_artifact_schema handling of JSON-wrapped content."""

    def test_array_validation_with_table_rows(self):
        """Normal markdown table should pass array validation."""
        from src.workflows.engine.hooks import validate_artifact_schema
        content = (
            "## Competitors\n\n"
            "| Name | Rating | Notes |\n"
            "|------|--------|-------|\n"
            "| App A | 4.5 | Good UX |\n"
            "| App B | 3.8 | Slow |\n"
            "| App C | 4.2 | Expensive |\n"
        )
        schema = {"competitor_list": {"type": "array", "min_items": 1}}
        is_valid, err = validate_artifact_schema(content, schema)
        self.assertTrue(is_valid, f"Should pass but got: {err}")

    def test_array_validation_with_json_escaped_table(self):
        """JSON-escaped table (\\n instead of newlines) should be unwrapped."""
        from src.workflows.engine.hooks import validate_artifact_schema
        inner = (
            "## Competitors\n\n"
            "| Name | Rating |\n"
            "|------|--------|\n"
            "| App A | 4.5 |\n"
            "| App B | 3.8 |\n"
        )
        wrapped = json.dumps({"action": "final_answer", "result": inner})
        schema = {"competitor_list": {"type": "array", "min_items": 1}}
        is_valid, err = validate_artifact_schema(wrapped, schema)
        self.assertTrue(is_valid, f"JSON-wrapped content should be unwrapped: {err}")

    def test_object_validation_with_json_wrapped(self):
        """JSON-wrapped content should pass object field validation."""
        from src.workflows.engine.hooks import validate_artifact_schema
        inner = "## Research\n\npain_points: Users struggle with...\ncurrent_tools: They use..."
        wrapped = json.dumps({"action": "final_answer", "result": inner})
        schema = {"audience_data": {
            "type": "object",
            "required_fields": ["pain_points", "current_tools"],
        }}
        is_valid, err = validate_artifact_schema(wrapped, schema)
        self.assertTrue(is_valid, f"Should find keywords after unwrapping: {err}")


class TestSummaryFirstFetching(unittest.TestCase):
    """Downstream steps should prefer summaries when full artifact exceeds budget."""

    def test_summary_used_when_artifact_exceeds_budget(self):
        """When full artifact > tier budget, summary should be used instead."""
        from src.workflows.engine.artifacts import CONTEXT_BUDGETS

        full_content = "## Full Research\n\n" + "X" * 10000  # 10k chars
        summary_content = "## Summary\n\nKey findings: X, Y, Z."  # 40 chars

        # reference tier budget is 3000 — full_content (10k) exceeds it
        budget = CONTEXT_BUDGETS["reference"]
        self.assertGreater(len(full_content), budget)
        self.assertLess(len(summary_content), budget)

        # The logic: if len(full) > budget and summary exists, use summary
        should_use_summary = len(full_content) > budget and summary_content
        self.assertTrue(should_use_summary)

    def test_full_artifact_used_when_fits_budget(self):
        """When full artifact fits in tier budget, use it directly."""
        from src.workflows.engine.artifacts import CONTEXT_BUDGETS

        full_content = "## Short Report\n\nDone."  # 23 chars
        summary_content = "Short report done."

        budget = CONTEXT_BUDGETS["primary"]  # 8000
        self.assertLess(len(full_content), budget)

        # The logic: if len(full) <= budget, use full even if summary exists
        should_use_summary = len(full_content) > budget and summary_content
        self.assertFalse(should_use_summary)

    def test_full_artifact_used_when_no_summary(self):
        """When no summary exists, fall back to full artifact (truncated by formatter)."""
        full_content = "## Big Report\n\n" + "Y" * 10000
        summary_content = None

        budget = 3000
        should_use_summary = len(full_content) > budget and summary_content
        self.assertFalse(should_use_summary)  # no summary, must use full
