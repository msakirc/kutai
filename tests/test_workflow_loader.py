"""Tests for the workflow definition loader (v2-aware)."""

import unittest

from src.workflows.engine.loader import WorkflowDefinition, load_workflow, validate_dependencies


class TestWorkflowLoader(unittest.TestCase):
    """Tests for loading and validating v2 workflow definitions."""

    @classmethod
    def setUpClass(cls):
        """Load the v2 workflow once for all tests."""
        cls.wf = load_workflow("i2p_v3")

    def test_load_v2_workflow(self):
        """Verify plan_id and version are correct."""
        self.assertEqual(self.wf.plan_id, "i2p_v3")
        self.assertEqual(self.wf.version, "2.0")

    def test_workflow_has_17_phases(self):
        """v2 has 17 phases: Phase -1 through Phase 15."""
        self.assertEqual(len(self.wf.phases), 17)
        phase_ids = [p["id"] for p in self.wf.phases]
        self.assertIn("phase_-1", phase_ids)
        self.assertIn("phase_0", phase_ids)
        self.assertIn("phase_15", phase_ids)

    def test_steps_have_required_fields(self):
        """Every step must have id, agent, instruction, and depends_on."""
        required = {"id", "agent", "instruction", "depends_on"}
        for step in self.wf.steps:
            for field in required:
                self.assertIn(
                    field,
                    step,
                    f"Step {step.get('id', '???')} missing required field '{field}'",
                )

    def test_workflow_has_templates(self):
        """At least one template exists and the main template has 31 steps."""
        self.assertGreaterEqual(len(self.wf.templates), 1)
        main_tpl = self.wf.get_template("feature_implementation_template")
        self.assertIsNotNone(main_tpl)
        self.assertEqual(len(main_tpl["steps"]), 31)

    def test_workflow_has_conditional_groups(self):
        """v2 defines 7 conditional groups with expected IDs."""
        self.assertEqual(len(self.wf.conditional_groups), 7)
        expected_ids = {
            "competitor_deep_dive",
            "realtime_features",
            "payment_flow",
            "mobile_app_submission",
            "file_upload_security",
            "seo_implementation",
            "email_launch",
        }
        actual_ids = {cg["group_id"] for cg in self.wf.conditional_groups}
        self.assertEqual(actual_ids, expected_ids)

    def test_workflow_has_review_and_revision_policies(self):
        """Metadata must contain review_policy and revision_policy."""
        self.assertIn("review_policy", self.wf.metadata)
        self.assertIn("revision_policy", self.wf.metadata)
        self.assertEqual(
            self.wf.metadata["review_policy"]["max_review_cycles"], 3
        )
        self.assertIn(
            "allowed_revisions", self.wf.metadata["revision_policy"]
        )

    def test_workflow_has_onboarding_policy(self):
        """Metadata must contain onboarding_policy with key fields."""
        self.assertIn("onboarding_policy", self.wf.metadata)
        policy = self.wf.metadata["onboarding_policy"]
        self.assertTrue(policy["never_rewrite_existing_code_unless_asked"])
        self.assertTrue(policy["match_existing_patterns_over_plan_defaults"])

    def test_load_nonexistent_raises(self):
        """Loading a workflow that doesn't exist raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_workflow("nonexistent_workflow_v99")

    def test_dependency_graph_valid(self):
        """All depends_on references (including conditional fallback steps) must resolve."""
        errors = validate_dependencies(self.wf)
        self.assertEqual(errors, [], f"Broken dependencies found: {errors}")

    def test_get_recurring_steps(self):
        """There should be more than 10 recurring steps in v2."""
        recurring = self.wf.get_recurring_steps()
        self.assertGreater(len(recurring), 10)

    def test_get_step(self):
        """get_step returns the correct step by ID."""
        step = self.wf.get_step("0.1")
        self.assertIsNotNone(step)
        self.assertEqual(step["id"], "0.1")

    def test_get_step_missing(self):
        """get_step returns None for nonexistent step."""
        self.assertIsNone(self.wf.get_step("999.999"))

    def test_get_phase_steps(self):
        """get_phase_steps returns steps belonging to the given phase."""
        phase_0_steps = self.wf.get_phase_steps("phase_0")
        self.assertGreater(len(phase_0_steps), 0)
        for step in phase_0_steps:
            self.assertEqual(step["phase"], "phase_0")

    def test_get_phase(self):
        """get_phase returns the phase dict for a given phase_id."""
        phase = self.wf.get_phase("phase_-1")
        self.assertIsNotNone(phase)
        self.assertEqual(phase["id"], "phase_-1")
        self.assertIsNone(self.wf.get_phase("phase_99"))

    def test_get_conditional_group(self):
        """get_conditional_group returns the correct group."""
        cg = self.wf.get_conditional_group("competitor_deep_dive")
        self.assertIsNotNone(cg)
        self.assertEqual(cg["group_id"], "competitor_deep_dive")
        self.assertIsNone(self.wf.get_conditional_group("nonexistent"))


if __name__ == "__main__":
    unittest.main()
