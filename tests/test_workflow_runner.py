"""Tests for the workflow runner utility functions.

The runner does heavy DB integration, so we focus on testing
the pure utility functions: resolve_dependencies and build_step_description.
"""

import logging
import unittest

from src.workflows.engine.runner import (
    build_step_description,
    resolve_dependencies,
)


class TestResolveDependencies(unittest.TestCase):
    """Tests for mapping step IDs to DB task IDs."""

    def test_resolve_dependencies(self):
        """Known step IDs are mapped to their task IDs."""
        step_to_task = {"0.1": 10, "0.2": 20, "1.1": 30}
        result = resolve_dependencies(["0.1", "1.1"], step_to_task)
        self.assertEqual(result, [10, 30])

    def test_resolve_skips_missing(self):
        """Unknown step IDs are logged and skipped."""
        step_to_task = {"0.1": 10}
        with self.assertLogs("workflows.engine.runner", level="WARNING") as cm:
            result = resolve_dependencies(["0.1", "MISSING_STEP"], step_to_task)
        self.assertEqual(result, [10])
        self.assertTrue(any("MISSING_STEP" in msg for msg in cm.output))

    def test_resolve_empty_list(self):
        """Empty dependency list returns empty result."""
        result = resolve_dependencies([], {"0.1": 10})
        self.assertEqual(result, [])

    def test_resolve_all_missing(self):
        """All missing step IDs returns empty result with warnings."""
        with self.assertLogs("workflows.engine.runner", level="WARNING"):
            result = resolve_dependencies(["X", "Y"], {"0.1": 10})
        self.assertEqual(result, [])


class TestBuildStepDescription(unittest.TestCase):
    """Tests for combining instruction, artifacts, and done_when."""

    def test_build_step_description(self):
        """Instruction + artifacts are combined."""
        desc = build_step_description(
            instruction="Analyze the idea.",
            input_artifacts=["idea", "market_data"],
            artifact_contents={"idea": "Build a chat app", "market_data": "TAM $1B"},
        )
        self.assertIn("Analyze the idea.", desc)
        self.assertIn("idea", desc)
        self.assertIn("Build a chat app", desc)
        self.assertIn("market_data", desc)
        self.assertIn("TAM $1B", desc)

    def test_build_step_description_with_done_when(self):
        """done_when section is appended."""
        desc = build_step_description(
            instruction="Write the spec.",
            input_artifacts=[],
            artifact_contents={},
            done_when="Spec document is complete and reviewed.",
        )
        self.assertIn("Write the spec.", desc)
        self.assertIn("Done when", desc)
        self.assertIn("Spec document is complete and reviewed.", desc)

    def test_build_step_description_missing_artifacts(self):
        """Missing artifacts are noted in the description."""
        desc = build_step_description(
            instruction="Review the design.",
            input_artifacts=["design_doc", "missing_artifact"],
            artifact_contents={"design_doc": "The design is..."},
        )
        self.assertIn("Review the design.", desc)
        self.assertIn("design_doc", desc)
        self.assertIn("missing_artifact", desc)
        # Should note it's missing/not available
        self.assertIn("not available", desc.lower())

    def test_build_step_description_no_artifacts(self):
        """With no artifacts, just the instruction is returned."""
        desc = build_step_description(
            instruction="Do the thing.",
            input_artifacts=[],
            artifact_contents={},
        )
        self.assertIn("Do the thing.", desc)
        # Should be relatively short — just the instruction
        self.assertNotIn("###", desc)

    def test_build_step_description_empty_done_when(self):
        """Empty done_when string does not add a section."""
        desc = build_step_description(
            instruction="Do something.",
            input_artifacts=[],
            artifact_contents={},
            done_when="",
        )
        self.assertNotIn("Done when", desc)


if __name__ == "__main__":
    unittest.main()
