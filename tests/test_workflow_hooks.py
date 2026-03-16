"""Tests for workflow execution hooks (pre/post orchestrator integration)."""

import asyncio
import json
import unittest
from unittest.mock import patch, AsyncMock

from src.workflows.engine.hooks import (
    get_artifact_store,
    is_workflow_step,
    extract_output_artifact_names,
    enrich_task_description,
    pre_execute_workflow_step,
    post_execute_workflow_step,
)
from src.workflows.engine.artifacts import ArtifactStore


class TestIsWorkflowStep(unittest.TestCase):
    """Tests for is_workflow_step detection."""

    def test_is_workflow_step(self):
        """Returns True when is_workflow_step is set."""
        self.assertTrue(is_workflow_step({"is_workflow_step": True}))

    def test_is_workflow_step_false(self):
        """Returns False for empty dict or missing key."""
        self.assertFalse(is_workflow_step({}))
        self.assertFalse(is_workflow_step({"other_key": True}))

    def test_is_workflow_step_explicit_false(self):
        """Returns False when explicitly set to False."""
        self.assertFalse(is_workflow_step({"is_workflow_step": False}))


class TestExtractOutputArtifactNames(unittest.TestCase):
    """Tests for extract_output_artifact_names."""

    def test_extract_output_artifact_names(self):
        """Returns list from context output_artifacts."""
        ctx = {"output_artifacts": ["spec", "plan"]}
        self.assertEqual(extract_output_artifact_names(ctx), ["spec", "plan"])

    def test_extract_output_artifact_names_missing(self):
        """Returns empty list when key is absent."""
        self.assertEqual(extract_output_artifact_names({}), [])

    def test_extract_single_artifact(self):
        """Single item list works."""
        ctx = {"output_artifacts": ["result"]}
        self.assertEqual(extract_output_artifact_names(ctx), ["result"])


class TestEnrichTaskDescription(unittest.TestCase):
    """Tests for enrich_task_description."""

    def test_enrich_task_description(self):
        """Adds artifact content to the task description."""
        task = {
            "description": "Implement the feature",
            "context": json.dumps({
                "is_workflow_step": True,
            }),
        }
        artifacts = {"spec": "The specification content"}
        result = enrich_task_description(task, artifacts)
        self.assertIn("Implement the feature", result)
        self.assertIn("spec", result)
        self.assertIn("The specification content", result)

    def test_enrich_with_context_strategy(self):
        """Passes context_strategy through to format_artifacts_for_prompt."""
        task = {
            "description": "Do the work",
            "context": json.dumps({
                "is_workflow_step": True,
                "context_strategy": {
                    "primary": ["spec"],
                    "reference": ["plan"],
                },
            }),
        }
        artifacts = {"spec": "spec content", "plan": "plan content"}
        result = enrich_task_description(task, artifacts)
        self.assertIn("Do the work", result)
        self.assertIn("### spec", result)
        self.assertIn("### plan", result)

    def test_enrich_with_done_when(self):
        """Appends done_when section to the description."""
        task = {
            "description": "Build feature X",
            "context": json.dumps({
                "is_workflow_step": True,
                "done_when": "All tests pass and code is reviewed",
            }),
        }
        artifacts = {}
        result = enrich_task_description(task, artifacts)
        self.assertIn("Build feature X", result)
        self.assertIn("## Done When", result)
        self.assertIn("All tests pass and code is reviewed", result)

    def test_enrich_no_artifacts(self):
        """Returns instruction (+ done_when if present) when no artifacts."""
        task = {
            "description": "Simple task",
            "context": json.dumps({
                "is_workflow_step": True,
            }),
        }
        result = enrich_task_description(task, {})
        self.assertIn("Simple task", result)
        # Should not have artifact headers
        self.assertNotIn("###", result)

    def test_enrich_no_artifacts_with_done_when(self):
        """No artifacts but done_when still appends."""
        task = {
            "description": "Simple task",
            "context": json.dumps({
                "is_workflow_step": True,
                "done_when": "Task is done",
            }),
        }
        result = enrich_task_description(task, {})
        self.assertIn("Simple task", result)
        self.assertIn("## Done When", result)
        self.assertIn("Task is done", result)

    def test_enrich_with_dict_context(self):
        """Handles context already parsed as dict."""
        task = {
            "description": "Dict context task",
            "context": {
                "is_workflow_step": True,
                "done_when": "Done condition",
            },
        }
        result = enrich_task_description(task, {})
        self.assertIn("Dict context task", result)
        self.assertIn("## Done When", result)


class TestGetArtifactStore(unittest.TestCase):
    """Tests for the module-level singleton."""

    def test_get_artifact_store_returns_instance(self):
        """Returns an ArtifactStore instance."""
        store = get_artifact_store()
        self.assertIsInstance(store, ArtifactStore)

    def test_get_artifact_store_singleton(self):
        """Returns the same instance on repeated calls."""
        store1 = get_artifact_store()
        store2 = get_artifact_store()
        self.assertIs(store1, store2)


class TestPreExecuteWorkflowStep(unittest.TestCase):
    """Tests for pre_execute_workflow_step."""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    def test_non_workflow_step_unchanged(self):
        """Non-workflow tasks pass through unchanged."""
        task = {"description": "Normal task", "context": "{}"}
        result = self._run(pre_execute_workflow_step(task))
        self.assertEqual(result, task)

    def test_workflow_step_enriches_description(self):
        """Workflow step has its description enriched with artifacts."""
        store = get_artifact_store()
        self._run(store.store(42, "spec", "Spec content here"))

        task = {
            "description": "Implement feature",
            "context": json.dumps({
                "is_workflow_step": True,
                "goal_id": 42,
                "input_artifacts": ["spec"],
            }),
        }
        result = self._run(pre_execute_workflow_step(task))
        self.assertIn("Spec content here", result["description"])


class TestPostExecuteWorkflowStep(unittest.TestCase):
    """Tests for post_execute_workflow_step."""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    def test_non_workflow_step_noop(self):
        """Non-workflow tasks are ignored."""
        task = {"context": "{}"}
        # Should not raise
        self._run(post_execute_workflow_step(task, {"result": "result"}))

    def test_stores_single_output_artifact(self):
        """Single output artifact stores the full result."""
        store = get_artifact_store()
        task = {
            "context": json.dumps({
                "is_workflow_step": True,
                "goal_id": 100,
                "output_artifacts": ["analysis"],
            }),
        }
        result = {"result": "Analysis result text"}
        self._run(post_execute_workflow_step(task, result))

        stored = self._run(store.retrieve(100, "analysis"))
        self.assertEqual(stored, "Analysis result text")

    def test_stores_multiple_output_artifacts(self):
        """Multiple output artifacts each get the full result."""
        store = get_artifact_store()
        task = {
            "context": json.dumps({
                "is_workflow_step": True,
                "goal_id": 101,
                "output_artifacts": ["doc_a", "doc_b"],
            }),
        }
        result = {"result": "Combined output"}
        self._run(post_execute_workflow_step(task, result))

        stored_a = self._run(store.retrieve(101, "doc_a"))
        stored_b = self._run(store.retrieve(101, "doc_b"))
        self.assertEqual(stored_a, "Combined output")
        self.assertEqual(stored_b, "Combined output")


if __name__ == "__main__":
    unittest.main()
