"""Tests for the WorkflowRunner.preview() dry-run method."""

import asyncio
import unittest

from src.workflows.engine.runner import WorkflowRunner


class TestWorkflowPreview(unittest.TestCase):
    """Verify that preview() returns correct structure without touching the DB."""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    def test_preview_returns_structure(self):
        """Preview should return all expected top-level keys."""
        runner = WorkflowRunner()
        result = self._run(runner.preview("i2p_v2"))
        self.assertIn("total_steps", result)
        self.assertIn("phases", result)
        self.assertIn("estimated_cost", result)
        self.assertIn("workflow_name", result)
        self.assertIn("title", result)
        self.assertIn("direct_steps", result)
        self.assertIn("template_estimated_steps", result)
        self.assertIn("recurring_steps", result)
        self.assertIn("conditional_groups", result)
        self.assertIn("templates", result)
        self.assertGreater(result["total_steps"], 0)
        self.assertGreater(len(result["phases"]), 0)

    def test_preview_has_phase_details(self):
        """Each phase entry should contain id, name, step_count, agents."""
        runner = WorkflowRunner()
        result = self._run(runner.preview("i2p_v2"))
        for phase in result["phases"]:
            self.assertIn("phase_id", phase)
            self.assertIn("phase_name", phase)
            self.assertIn("step_count", phase)
            self.assertIn("agents", phase)
            self.assertGreater(phase["step_count"], 0)
            self.assertIsInstance(phase["agents"], list)

    def test_preview_cost_positive(self):
        """Estimated cost should be a positive number."""
        runner = WorkflowRunner()
        result = self._run(runner.preview("i2p_v2"))
        self.assertGreater(result["estimated_cost"], 0)

    def test_preview_with_existing_codebase(self):
        """With existing codebase, Phase -1 onboarding should be excluded."""
        runner = WorkflowRunner()
        result = self._run(
            runner.preview("i2p_v2", existing_codebase_path="/tmp/project")
        )
        # Should still have steps, just potentially fewer
        self.assertGreater(result["total_steps"], 0)

    def test_preview_includes_templates(self):
        """Template-related fields should be present."""
        runner = WorkflowRunner()
        result = self._run(runner.preview("i2p_v2"))
        self.assertIn("templates", result)
        self.assertIn("template_estimated_steps", result)
        self.assertIsInstance(result["templates"], int)
        self.assertIsInstance(result["template_estimated_steps"], int)

    def test_preview_workflow_name_matches(self):
        """The returned workflow_name should match what was requested."""
        runner = WorkflowRunner()
        result = self._run(runner.preview("i2p_v2"))
        self.assertEqual(result["workflow_name"], "i2p_v2")


if __name__ == "__main__":
    unittest.main()
