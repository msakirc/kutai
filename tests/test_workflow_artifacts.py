"""Tests for the workflow artifact store (v2 context_strategy support)."""

import asyncio
import unittest

from src.workflows.engine.artifacts import (
    ArtifactStore,
    CONTEXT_BUDGETS,
    format_artifacts_for_prompt,
)


class TestArtifactStore(unittest.TestCase):
    """Tests for ArtifactStore with use_db=False (in-memory only)."""

    def setUp(self):
        self.store = ArtifactStore(use_db=False)

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_store_and_retrieve(self):
        """Basic roundtrip: store then retrieve."""
        self._run(self.store.store(1, "spec", "the spec content"))
        result = self._run(self.store.retrieve(1, "spec"))
        self.assertEqual(result, "the spec content")

    def test_retrieve_missing_returns_none(self):
        """Retrieving a non-existent artifact returns None."""
        result = self._run(self.store.retrieve(999, "nope"))
        self.assertIsNone(result)

    def test_collect_multiple(self):
        """Batch retrieve with some missing artifacts."""
        self._run(self.store.store(1, "a", "aaa"))
        self._run(self.store.store(1, "b", "bbb"))
        result = self._run(self.store.collect(1, ["a", "b", "c"]))
        self.assertEqual(result, {"a": "aaa", "b": "bbb", "c": None})

    def test_has_artifact(self):
        """has() returns True/False correctly."""
        self._run(self.store.store(1, "x", "val"))
        self.assertTrue(self._run(self.store.has(1, "x")))
        self.assertFalse(self._run(self.store.has(1, "y")))

    def test_list_artifacts(self):
        """list_artifacts returns cached artifact names."""
        self._run(self.store.store(5, "alpha", "a"))
        self._run(self.store.store(5, "beta", "b"))
        names = self._run(self.store.list_artifacts(5))
        self.assertEqual(sorted(names), ["alpha", "beta"])

    def test_list_artifacts_empty(self):
        """list_artifacts for unknown mission returns empty list."""
        names = self._run(self.store.list_artifacts(999))
        self.assertEqual(names, [])


class TestFormatArtifacts(unittest.TestCase):
    """Tests for format_artifacts_for_prompt."""

    def test_format_artifacts_for_prompt(self):
        """Basic formatting: header per artifact, joined by separator."""
        artifacts = {"spec": "spec content", "plan": "plan content"}
        result = format_artifacts_for_prompt(artifacts)
        self.assertIn("### spec", result)
        self.assertIn("spec content", result)
        self.assertIn("### plan", result)
        self.assertIn("plan content", result)
        self.assertIn("---", result)

    def test_format_with_context_strategy(self):
        """Context strategy applies different budgets per tier."""
        long_content = "x" * 10000
        artifacts = {
            "primary_art": long_content,
            "ref_art": long_content,
            "full_art": long_content,
            "other_art": long_content,
        }
        strategy = {
            "primary": ["primary_art"],
            "reference": ["ref_art"],
            "full_only_if_needed": ["full_art"],
        }
        result = format_artifacts_for_prompt(artifacts, context_strategy=strategy)

        # Primary tier gets 8000 char budget, so content should be truncated to ~8000
        self.assertIn("### primary_art", result)
        self.assertIn("### ref_art", result)
        self.assertIn("### full_art", result)
        # "other_art" is uncategorized, gets default budget
        self.assertIn("### other_art", result)

        # Verify truncation happened (content was 10000 chars, budgets are smaller)
        # The reference artifact content should be shorter than primary
        primary_section = result.split("### primary_art")[1].split("---")[0]
        ref_section = result.split("### ref_art")[1].split("---")[0]
        self.assertGreater(len(primary_section), len(ref_section))

    def test_format_empty_artifacts(self):
        """Empty artifacts dict returns empty string."""
        result = format_artifacts_for_prompt({})
        self.assertEqual(result, "")

    def test_format_respects_max_total(self):
        """Final output is truncated to max_total."""
        artifacts = {"big": "y" * 50000}
        result = format_artifacts_for_prompt(artifacts, max_total=1000)
        self.assertLessEqual(len(result), 1000)

    def test_context_budgets_values(self):
        """CONTEXT_BUDGETS has the expected tiers and values."""
        self.assertEqual(CONTEXT_BUDGETS["primary"], 8000)
        self.assertEqual(CONTEXT_BUDGETS["reference"], 3000)
        self.assertEqual(CONTEXT_BUDGETS["full_only_if_needed"], 1500)
        self.assertEqual(CONTEXT_BUDGETS["default"], 6000)


if __name__ == "__main__":
    unittest.main()
