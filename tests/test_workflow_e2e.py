"""End-to-end integration tests for the workflow engine.

These tests verify that the loader, expander, conditions, artifacts,
pipeline_bridge, status, and dispatch modules work together correctly.
Only DB calls are mocked — all engine internals use real logic.
"""

import asyncio
import json
import unittest

from src.workflows.engine.artifacts import ArtifactStore
from src.workflows.engine.conditions import evaluate_condition, resolve_group
from src.workflows.engine.dispatch import should_start_workflow
from src.workflows.engine.expander import (
    expand_template,
    filter_steps_for_context,
)
from src.workflows.engine.loader import load_workflow
from src.workflows.engine.pipeline_bridge import (
    PIPELINE_DELEGATE_STEPS,
    should_delegate_to_pipeline,
)
from src.workflows.engine.status import compute_phase_progress, format_status_message


class TestLoaderPlusExpanderStepCount(unittest.TestCase):
    """Test 1: Load v2, verify step counts and Phase -1 filtering."""

    @classmethod
    def setUpClass(cls):
        cls.wf = load_workflow("idea_to_product_v2")

    def test_total_step_count_exceeds_100(self):
        self.assertGreater(len(self.wf.steps), 100)

    def test_phase_minus1_excluded_for_greenfield(self):
        all_steps = self.wf.steps
        greenfield = filter_steps_for_context(all_steps, has_existing_codebase=False)
        phase_m1 = [s for s in all_steps if s.get("phase") == "phase_-1"]

        self.assertGreater(len(phase_m1), 0, "v2 should have Phase -1 steps")

        # Verify no Phase -1 steps remain in greenfield
        for s in greenfield:
            self.assertNotEqual(
                s.get("phase"), "phase_-1",
                f"Step {s['id']} is Phase -1 but was not filtered",
            )

        # Verify count identity
        self.assertEqual(len(greenfield), len(all_steps) - len(phase_m1))

    def test_all_steps_included_when_existing_codebase(self):
        all_steps = self.wf.steps
        with_existing = filter_steps_for_context(all_steps, has_existing_codebase=True)
        self.assertEqual(len(with_existing), len(all_steps))


class TestConditionalGroupEvaluationAllConditions(unittest.TestCase):
    """Test 2: Evaluate all 7 real condition expressions from v2 JSON."""

    def test_length_competitors_gte_3_true(self):
        artifact = json.dumps({"competitors": ["A", "B", "C"]})
        self.assertTrue(evaluate_condition("length(competitors) >= 3", artifact))

    def test_length_competitors_gte_3_false(self):
        artifact = json.dumps({"competitors": ["A"]})
        self.assertFalse(evaluate_condition("length(competitors) >= 3", artifact))

    def test_any_req_category_realtime_true(self):
        artifact = json.dumps([
            {"category": "crud"},
            {"category": "realtime"},
        ])
        self.assertTrue(evaluate_condition("any(req.category == 'realtime')", artifact))

    def test_any_req_category_realtime_false(self):
        artifact = json.dumps([{"category": "crud"}, {"category": "auth"}])
        self.assertFalse(evaluate_condition("any(req.category == 'realtime')", artifact))

    def test_pricing_model_not_free_true(self):
        artifact = json.dumps({"pricing_model": "freemium"})
        self.assertTrue(evaluate_condition("pricing_model != 'free'", artifact))

    def test_pricing_model_not_free_false(self):
        artifact = json.dumps({"pricing_model": "free"})
        self.assertFalse(evaluate_condition("pricing_model != 'free'", artifact))

    def test_platforms_include_ios_or_android_true(self):
        artifact = json.dumps({"platforms": ["ios", "web"]})
        self.assertTrue(
            evaluate_condition(
                "platforms_include('ios') OR platforms_include('android')", artifact
            )
        )

    def test_platforms_include_ios_or_android_false(self):
        artifact = json.dumps({"platforms": ["web"]})
        self.assertFalse(
            evaluate_condition(
                "platforms_include('ios') OR platforms_include('android')", artifact
            )
        )

    def test_any_req_category_file_upload_true(self):
        artifact = json.dumps([{"category": "file_upload"}])
        self.assertTrue(
            evaluate_condition("any(req.category == 'file_upload')", artifact)
        )

    def test_any_req_category_file_upload_false(self):
        artifact = json.dumps([{"category": "crud"}])
        self.assertFalse(
            evaluate_condition("any(req.category == 'file_upload')", artifact)
        )

    def test_has_public_web_pages_true(self):
        artifact = json.dumps({"has_public_web_pages": True})
        self.assertTrue(
            evaluate_condition("has_public_web_pages == true", artifact)
        )

    def test_has_public_web_pages_false(self):
        artifact = json.dumps({"has_public_web_pages": False})
        self.assertFalse(
            evaluate_condition("has_public_web_pages == true", artifact)
        )

    def test_email_list_exists_true(self):
        artifact = json.dumps({"email_list_exists": True})
        self.assertTrue(evaluate_condition("email_list_exists == true", artifact))

    def test_email_list_exists_false(self):
        artifact = json.dumps({"email_list_exists": False})
        self.assertFalse(evaluate_condition("email_list_exists == true", artifact))

    def test_resolve_group_true_path(self):
        """When condition is true, if_true steps are included, if_false excluded."""
        group = {
            "group_id": "test",
            "condition_check": "length(competitors) >= 3",
            "if_true": ["1.5", "1.6"],
            "if_false": ["1.5_lite"],
            "fallback_steps": [{"id": "1.5_lite"}],
        }
        artifact = json.dumps({"competitors": ["A", "B", "C"]})
        included, excluded = resolve_group(group, artifact)
        self.assertEqual(included, ["1.5", "1.6"])
        self.assertEqual(excluded, ["1.5_lite"])

    def test_resolve_group_false_path(self):
        """When condition is false, if_false + fallback included, if_true excluded."""
        group = {
            "group_id": "test",
            "condition_check": "length(competitors) >= 3",
            "if_true": ["1.5", "1.6"],
            "if_false": ["1.5_lite"],
            "fallback_steps": [{"id": "1.5_lite"}],
        }
        artifact = json.dumps({"competitors": ["A"]})
        included, excluded = resolve_group(group, artifact)
        self.assertIn("1.5_lite", included)
        self.assertEqual(excluded, ["1.5", "1.6"])


class TestTemplateExpansionFull(unittest.TestCase):
    """Test 3: Expand the feature_implementation_template with sample params."""

    @classmethod
    def setUpClass(cls):
        cls.wf = load_workflow("idea_to_product_v2")
        cls.template = cls.wf.get_template("feature_implementation_template")

    def test_template_exists(self):
        self.assertIsNotNone(self.template)

    def test_expansion_produces_31_steps(self):
        expanded = expand_template(
            self.template,
            params={"feature_id": "F-001", "feature_name": "User Auth"},
            prefix="8.F-001",
        )
        self.assertEqual(len(expanded), 31)

    def test_expanded_ids_have_correct_prefix(self):
        expanded = expand_template(
            self.template,
            params={"feature_id": "F-001", "feature_name": "User Auth"},
            prefix="8.F-001",
        )
        for step in expanded:
            self.assertTrue(
                step["id"].startswith("8.F-001."),
                f"Step ID {step['id']} should start with '8.F-001.'",
            )

    def test_parameter_substitution_applied(self):
        expanded = expand_template(
            self.template,
            params={"feature_id": "F-001", "feature_name": "User Auth"},
            prefix="8.F-001",
        )
        # The template uses {feature_name} placeholders in instructions
        instructions = " ".join(s.get("instruction", "") for s in expanded)
        self.assertIn("User Auth", instructions)
        # No raw {feature_name} placeholders should remain
        self.assertNotIn("{feature_name}", instructions)

    def test_expanded_steps_have_required_fields(self):
        expanded = expand_template(
            self.template,
            params={"feature_id": "F-001", "feature_name": "User Auth"},
            prefix="8.F-001",
        )
        for step in expanded:
            self.assertIn("id", step)
            self.assertIn("name", step)
            self.assertIn("agent", step)
            self.assertIn("instruction", step)


class TestArtifactStoreRoundtrip(unittest.TestCase):
    """Test 4: ArtifactStore with use_db=False — pure in-memory operations."""

    def setUp(self):
        self.store = ArtifactStore(use_db=False)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    def test_store_and_retrieve(self):
        self._run(self.store.store(1, "prd", "Product Requirements Doc"))
        result = self._run(self.store.retrieve(1, "prd"))
        self.assertEqual(result, "Product Requirements Doc")

    def test_retrieve_missing_returns_none(self):
        result = self._run(self.store.retrieve(1, "nonexistent"))
        self.assertIsNone(result)

    def test_has_existing(self):
        self._run(self.store.store(1, "spec", "The spec"))
        self.assertTrue(self._run(self.store.has(1, "spec")))

    def test_has_missing(self):
        self.assertFalse(self._run(self.store.has(1, "missing")))

    def test_collect_mixed(self):
        self._run(self.store.store(1, "alpha", "A"))
        self._run(self.store.store(1, "beta", "B"))
        result = self._run(
            self.store.collect(1, ["alpha", "beta", "gamma"])
        )
        self.assertEqual(result["alpha"], "A")
        self.assertEqual(result["beta"], "B")
        self.assertIsNone(result["gamma"])

    def test_multiple_goals_isolated(self):
        self._run(self.store.store(1, "doc", "Goal 1 doc"))
        self._run(self.store.store(2, "doc", "Goal 2 doc"))
        self.assertEqual(self._run(self.store.retrieve(1, "doc")), "Goal 1 doc")
        self.assertEqual(self._run(self.store.retrieve(2, "doc")), "Goal 2 doc")

    def test_list_artifacts(self):
        self._run(self.store.store(1, "x", "1"))
        self._run(self.store.store(1, "y", "2"))
        names = self._run(self.store.list_artifacts(1))
        self.assertCountEqual(names, ["x", "y"])


class TestPipelineBridgeDelegationMatrix(unittest.TestCase):
    """Test 5: Verify pipeline delegation for all 31 template steps."""

    @classmethod
    def setUpClass(cls):
        wf = load_workflow("idea_to_product_v2")
        cls.template = wf.get_template("feature_implementation_template")
        cls.expanded = expand_template(
            cls.template,
            params={"feature_id": "F-001", "feature_name": "User Auth"},
            prefix="8.F-001",
        )

    def test_expected_delegation_set(self):
        """Verify the exact set of steps that should delegate."""
        delegated = set()
        not_delegated = set()

        for step in self.expanded:
            step_id = step["id"]
            agent = step.get("agent", "executor")
            if should_delegate_to_pipeline(step_id, agent):
                delegated.add(step_id)
            else:
                not_delegated.add(step_id)

        # Expected delegation: feat.3-7, feat.10-18 (all have implementer agent)
        expected_feat_nums = {3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18}
        expected_ids = {f"8.F-001.feat.{n}" for n in expected_feat_nums}

        self.assertEqual(delegated, expected_ids)

    def test_non_implementer_steps_never_delegate(self):
        """Steps with non-implementer agents should never delegate."""
        non_implementer_agents = {"planner", "test_generator", "executor",
                                  "fixer", "reviewer", "writer", "visual_reviewer"}
        for step in self.expanded:
            agent = step.get("agent", "executor")
            if agent in non_implementer_agents:
                self.assertFalse(
                    should_delegate_to_pipeline(step["id"], agent),
                    f"Step {step['id']} (agent={agent}) should NOT delegate",
                )

    def test_total_delegation_count(self):
        """14 steps should delegate (feat.3-7 = 5 + feat.10-18 = 9)."""
        count = sum(
            1
            for step in self.expanded
            if should_delegate_to_pipeline(step["id"], step.get("agent", "executor"))
        )
        self.assertEqual(count, 14)


class TestStatusWithRealWorkflow(unittest.TestCase):
    """Test 6: Load v2, create fake tasks, compute and format status."""

    @classmethod
    def setUpClass(cls):
        cls.wf = load_workflow("idea_to_product_v2")

    def _make_fake_tasks(self):
        """Create fake tasks from real workflow steps with mixed statuses."""
        tasks = []
        for i, step in enumerate(self.wf.steps):
            phase = step.get("phase", "phase_0")
            # Cycle through statuses to create realistic distribution
            if i % 5 == 0:
                status = "completed"
            elif i % 5 == 1:
                status = "in_progress"
            else:
                status = "pending"

            tasks.append({
                "id": i + 1,
                "title": f"[{step['id']}] {step.get('name', '')}",
                "status": status,
                "context": {
                    "workflow_phase": phase,
                    "workflow_step_id": step["id"],
                    "status": status,
                },
            })
        return tasks

    def test_compute_phase_progress_covers_all_phases(self):
        tasks = self._make_fake_tasks()
        progress = compute_phase_progress(tasks)

        # All phases from the workflow should be represented
        workflow_phases = {s.get("phase") for s in self.wf.steps}
        for phase in workflow_phases:
            self.assertIn(
                phase, progress,
                f"Phase {phase} should be in progress report",
            )

    def test_phase_totals_sum_to_step_count(self):
        tasks = self._make_fake_tasks()
        progress = compute_phase_progress(tasks)
        total = sum(info["total"] for info in progress.values())
        self.assertEqual(total, len(self.wf.steps))

    def test_phase_completed_counts_correct(self):
        tasks = self._make_fake_tasks()
        progress = compute_phase_progress(tasks)
        total_completed = sum(info["completed"] for info in progress.values())
        expected_completed = sum(
            1 for t in tasks if t.get("status") == "completed"
        )
        self.assertEqual(total_completed, expected_completed)

    def test_format_status_message_structure(self):
        tasks = self._make_fake_tasks()
        progress = compute_phase_progress(tasks)
        msg = format_status_message("idea_to_product_v2", 42, progress)

        # Should contain the workflow ID and goal ID
        self.assertIn("idea_to_product_v2", msg)
        self.assertIn("42", msg)

        # Should contain phase names
        self.assertIn("Idea Capture", msg)
        self.assertIn("Core Implementation", msg)

    def test_format_status_phases_ordered(self):
        tasks = self._make_fake_tasks()
        progress = compute_phase_progress(tasks)
        msg = format_status_message("idea_to_product_v2", 42, progress)

        # Phase -1 should appear before Phase 0 which should appear before Phase 1
        lines = msg.split("\n")
        phase_lines = [l for l in lines if any(c in l for c in ["Onboarding", "Idea Capture", "Market"])]
        if len(phase_lines) >= 2:
            idx_onboard = next((i for i, l in enumerate(lines) if "Onboarding" in l), -1)
            idx_idea = next((i for i, l in enumerate(lines) if "Idea Capture" in l), -1)
            if idx_onboard >= 0 and idx_idea >= 0:
                self.assertLess(idx_onboard, idx_idea)


class TestDispatchClassification(unittest.TestCase):
    """Test 7: Verify dispatch correctly classifies product vs coding messages."""

    def test_product_messages_trigger_workflow(self):
        product_messages = [
            "Build me a SaaS product for managing invoices",
            "Create an app for tracking fitness goals",
            "I have an idea for a platform to connect freelancers",
            "Make a tool for automating email responses",
            "Develop a web application for project management",
            "idea.to.product: build an analytics dashboard",
            "I want to launch a product for small businesses",
            "Build a full product for inventory management",
            "Create a startup website for food delivery",
            "MVP build for a task management service",
        ]
        for msg in product_messages:
            self.assertTrue(
                should_start_workflow(msg),
                f"Should trigger workflow: {msg!r}",
            )

    def test_coding_messages_do_not_trigger_workflow(self):
        coding_messages = [
            "Fix the bug in the login page",
            "Add a unit test for the user model",
            "Refactor the database connection code",
            "What does this error mean?",
            "Help me write a Python function",
            "How do I use async/await in JavaScript?",
            "Review this pull request",
            "Update the README file",
            "Run the test suite",
            "Debug the API endpoint",
        ]
        for msg in coding_messages:
            self.assertFalse(
                should_start_workflow(msg),
                f"Should NOT trigger workflow: {msg!r}",
            )


if __name__ == "__main__":
    unittest.main()
