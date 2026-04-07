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
