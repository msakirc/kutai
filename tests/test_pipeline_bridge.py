"""Tests for the pipeline delegation bridge.

Verifies that workflow template steps tagged for pipeline delegation
are correctly identified and packaged into CodingPipeline-compatible tasks.
"""

import unittest

from src.workflows.engine.pipeline_bridge import (
    should_delegate_to_pipeline,
    extract_feature_context,
    build_pipeline_task,
    PIPELINE_DELEGATE_STEPS,
    PIPELINE_DELEGATE_AGENTS,
)


class TestShouldDelegateToPipeline(unittest.TestCase):
    """Tests for should_delegate_to_pipeline."""

    def test_should_delegate_backend_steps(self):
        """feat.3 through feat.7 with implementer agent should delegate."""
        for n in range(3, 8):
            step_id = f"8.F-001.feat.{n}"
            self.assertTrue(
                should_delegate_to_pipeline(step_id, "implementer"),
                f"feat.{n} with implementer should delegate",
            )

    def test_should_delegate_frontend_steps(self):
        """feat.10 through feat.18 with implementer agent should delegate."""
        for n in range(10, 19):
            step_id = f"8.F-001.feat.{n}"
            self.assertTrue(
                should_delegate_to_pipeline(step_id, "implementer"),
                f"feat.{n} with implementer should delegate",
            )

    def test_should_delegate_coder_agent(self):
        """coder agent on a delegate step should also delegate."""
        self.assertTrue(
            should_delegate_to_pipeline("8.F-001.feat.5", "coder")
        )

    def test_should_not_delegate_planner(self):
        """feat.1 with planner agent should NOT delegate."""
        self.assertFalse(
            should_delegate_to_pipeline("8.F-001.feat.1", "planner")
        )

    def test_should_not_delegate_test_generator(self):
        """feat.8 with test_generator agent should NOT delegate."""
        self.assertFalse(
            should_delegate_to_pipeline("8.F-001.feat.8", "test_generator")
        )

    def test_should_not_delegate_reviewer(self):
        """feat.28 with reviewer agent should NOT delegate."""
        self.assertFalse(
            should_delegate_to_pipeline("8.F-001.feat.28", "reviewer")
        )

    def test_should_not_delegate_wrong_agent_right_step(self):
        """A delegate step with a non-delegate agent should NOT delegate."""
        self.assertFalse(
            should_delegate_to_pipeline("8.F-001.feat.5", "reviewer")
        )

    def test_should_not_delegate_right_agent_wrong_step(self):
        """An implementer on a non-delegate step should NOT delegate."""
        self.assertFalse(
            should_delegate_to_pipeline("8.F-001.feat.1", "implementer")
        )


class TestExtractFeatureContext(unittest.TestCase):
    """Tests for extract_feature_context."""

    def test_extract_from_nested_id(self):
        """'8.F-001.feat.5' should extract feature_id 'F-001'."""
        step_context = {
            "step_id": "8.F-001.feat.5",
            "workflow_context": {"feature_name": "User Login"},
        }
        feature_id, feature_name = extract_feature_context(step_context)
        self.assertEqual(feature_id, "F-001")
        self.assertEqual(feature_name, "User Login")

    def test_extract_falls_back_to_feature_id(self):
        """When workflow_context has no feature_name, fall back to feature_id."""
        step_context = {
            "step_id": "8.F-002.feat.3",
            "workflow_context": {},
        }
        feature_id, feature_name = extract_feature_context(step_context)
        self.assertEqual(feature_id, "F-002")
        self.assertEqual(feature_name, "F-002")

    def test_extract_no_workflow_context(self):
        """When workflow_context is missing entirely, fall back to feature_id."""
        step_context = {
            "step_id": "8.F-003.feat.10",
        }
        feature_id, feature_name = extract_feature_context(step_context)
        self.assertEqual(feature_id, "F-003")
        self.assertEqual(feature_name, "F-003")

    def test_extract_complex_feature_id(self):
        """Feature IDs with multiple segments should be extracted correctly."""
        step_context = {
            "step_id": "8.F-ABC-123.feat.7",
            "workflow_context": {"feature_name": "Complex Feature"},
        }
        feature_id, feature_name = extract_feature_context(step_context)
        self.assertEqual(feature_id, "F-ABC-123")
        self.assertEqual(feature_name, "Complex Feature")


class TestBuildPipelineTask(unittest.TestCase):
    """Tests for build_pipeline_task."""

    def test_build_pipeline_task(self):
        """All fields should be set correctly in the returned dict."""
        task = build_pipeline_task(
            step_title="Implement API endpoints",
            step_instruction="Create REST endpoints for user CRUD.",
            goal_id="G-001",
            feature_name="User Management",
            artifact_context="Existing models: User, Role",
        )
        self.assertEqual(task["title"], "Implement API endpoints")
        self.assertIn("Feature: User Management", task["description"])
        self.assertIn("Create REST endpoints for user CRUD.", task["description"])
        self.assertIn("Existing models: User, Role", task["description"])
        self.assertEqual(task["goal_id"], "G-001")
        self.assertEqual(task["context"]["pipeline_mode"], "feature")
        self.assertTrue(task["context"]["prefer_quality"])

    def test_build_pipeline_task_no_context(self):
        """Empty artifact_context should still produce a valid task."""
        task = build_pipeline_task(
            step_title="Build UI component",
            step_instruction="Create a React form.",
            goal_id="G-002",
            feature_name="Settings Page",
            artifact_context="",
        )
        self.assertEqual(task["title"], "Build UI component")
        self.assertIn("Feature: Settings Page", task["description"])
        self.assertIn("Create a React form.", task["description"])
        self.assertEqual(task["goal_id"], "G-002")
        self.assertEqual(task["context"]["pipeline_mode"], "feature")
        self.assertTrue(task["context"]["prefer_quality"])

    def test_build_pipeline_task_default_context(self):
        """When artifact_context is omitted, it defaults to empty string."""
        task = build_pipeline_task(
            step_title="Title",
            step_instruction="Instruction",
            goal_id="G-003",
            feature_name="Feature",
        )
        self.assertIn("Feature: Feature", task["description"])
        self.assertIn("Instruction", task["description"])


if __name__ == "__main__":
    unittest.main()
