"""Tests for the workflow expander (steps to DB tasks)."""

import unittest

from src.workflows.engine.expander import (
    AGENT_MAP,
    expand_steps_to_tasks,
    expand_template,
    filter_steps_for_context,
    map_agent_type,
)


class TestMapAgentType(unittest.TestCase):
    """Tests for agent name mapping."""

    def test_router_maps_to_executor(self):
        self.assertEqual(map_agent_type("router"), "executor")

    def test_passthrough_agents(self):
        for agent in (
            "planner", "architect", "coder", "implementer", "fixer",
            "test_generator", "reviewer", "researcher", "writer",
            "analyst", "executor", "error_recovery", "visual_reviewer",
            "summarizer", "assistant",
        ):
            self.assertEqual(map_agent_type(agent), agent, f"{agent} should pass through")

    def test_unknown_agent_passes_through(self):
        self.assertEqual(map_agent_type("some_new_agent"), "some_new_agent")


class TestFilterStepsForContext(unittest.TestCase):
    """Tests for Phase -1 conditional inclusion."""

    def _make_steps(self):
        return [
            {"id": "-1.1", "phase": "phase_-1", "name": "discovery"},
            {"id": "-1.2", "phase": "phase_-1", "name": "assessment"},
            {"id": "0.1", "phase": "phase_0", "name": "kick_off"},
            {"id": "1.1", "phase": "phase_1", "name": "research"},
        ]

    def test_phase_minus1_excluded_without_codebase(self):
        steps = self._make_steps()
        filtered = filter_steps_for_context(steps, has_existing_codebase=False)
        phase_ids = [s["id"] for s in filtered]
        self.assertNotIn("-1.1", phase_ids)
        self.assertNotIn("-1.2", phase_ids)
        self.assertIn("0.1", phase_ids)
        self.assertIn("1.1", phase_ids)

    def test_phase_minus1_included_with_codebase(self):
        steps = self._make_steps()
        filtered = filter_steps_for_context(steps, has_existing_codebase=True)
        phase_ids = [s["id"] for s in filtered]
        self.assertIn("-1.1", phase_ids)
        self.assertIn("-1.2", phase_ids)
        self.assertIn("0.1", phase_ids)
        self.assertEqual(len(filtered), 4)


class TestExpandStepsToTasks(unittest.TestCase):
    """Tests for converting steps into task dicts."""

    def _make_step(self, **overrides):
        base = {
            "id": "0.1",
            "phase": "phase_0",
            "name": "kick_off",
            "agent": "planner",
            "depends_on": [],
            "instruction": "Do kickoff.",
            "input_artifacts": ["idea"],
            "output_artifacts": ["project_brief"],
            "may_need_clarification": False,
        }
        base.update(overrides)
        return base

    def test_expand_single_phase(self):
        """2 steps with dependency preservation."""
        steps = [
            self._make_step(id="0.1", name="step_a", depends_on=[]),
            self._make_step(id="0.2", name="step_b", depends_on=["0.1"]),
        ]
        tasks = expand_steps_to_tasks(steps, goal_id="g1")
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["title"], "[0.1] step_a")
        self.assertEqual(tasks[1]["title"], "[0.2] step_b")
        self.assertEqual(tasks[1]["depends_on_steps"], ["0.1"])
        self.assertEqual(tasks[0]["goal_id"], "g1")

    def test_expand_maps_v2_agents(self):
        """router maps to executor, others pass through."""
        steps = [
            self._make_step(id="1.1", agent="router"),
            self._make_step(id="1.2", agent="planner"),
        ]
        tasks = expand_steps_to_tasks(steps, goal_id="g1")
        self.assertEqual(tasks[0]["agent_type"], "executor")
        self.assertEqual(tasks[1]["agent_type"], "planner")

    def test_expand_preserves_step_type(self):
        """Recurring type and trigger are preserved in context."""
        step = self._make_step(
            id="8.1",
            type="recurring",
            trigger="Execute at sprint boundary",
        )
        tasks = expand_steps_to_tasks([step], goal_id="g1")
        ctx = tasks[0]["context"]
        self.assertEqual(ctx["step_type"], "recurring")
        self.assertEqual(ctx["trigger"], "Execute at sprint boundary")

    def test_expand_preserves_may_need_clarification(self):
        step = self._make_step(may_need_clarification=True)
        tasks = expand_steps_to_tasks([step], goal_id="g1")
        self.assertTrue(tasks[0]["context"]["may_need_clarification"])

    def test_expand_with_initial_context(self):
        """Initial context is propagated into workflow_context."""
        step = self._make_step()
        tasks = expand_steps_to_tasks([step], goal_id="g1", initial_context={"user_idea": "build X"})
        self.assertEqual(tasks[0]["context"]["workflow_context"], {"user_idea": "build X"})

    def test_expand_without_initial_context(self):
        """Without initial_context, workflow_context is absent."""
        step = self._make_step()
        tasks = expand_steps_to_tasks([step], goal_id="g1")
        self.assertNotIn("workflow_context", tasks[0]["context"])

    def test_expand_context_fields(self):
        """Core context fields are always present."""
        step = self._make_step(
            id="2.1",
            phase="phase_2",
            done_when="Brief completed",
            condition="Only if needed",
        )
        tasks = expand_steps_to_tasks([step], goal_id="g1")
        ctx = tasks[0]["context"]
        self.assertEqual(ctx["workflow_step_id"], "2.1")
        self.assertEqual(ctx["workflow_phase"], "phase_2")
        self.assertEqual(ctx["input_artifacts"], ["idea"])
        self.assertEqual(ctx["output_artifacts"], ["project_brief"])
        self.assertFalse(ctx["may_need_clarification"])
        self.assertTrue(ctx["is_workflow_step"])
        self.assertEqual(ctx["done_when"], "Brief completed")
        self.assertEqual(ctx["condition"], "Only if needed")

    def test_tier_is_auto(self):
        step = self._make_step()
        tasks = expand_steps_to_tasks([step], goal_id="g1")
        self.assertEqual(tasks[0]["tier"], "auto")

    def test_priority_from_phase(self):
        """Earlier phases get higher priority: Phase -1/0 = 10, Phase 15 = 1."""
        steps = [
            self._make_step(id="-1.1", phase="phase_-1"),
            self._make_step(id="0.1", phase="phase_0"),
            self._make_step(id="7.1", phase="phase_7"),
            self._make_step(id="15.1", phase="phase_15"),
        ]
        tasks = expand_steps_to_tasks(steps, goal_id="g1")
        priorities = {t["context"]["workflow_step_id"]: t["priority"] for t in tasks}
        self.assertEqual(priorities["-1.1"], 10)
        self.assertEqual(priorities["0.1"], 10)
        self.assertEqual(priorities["15.1"], 1)
        # Phase 7 should be between 1 and 10
        self.assertGreater(priorities["7.1"], 1)
        self.assertLess(priorities["7.1"], 10)


class TestExpandTemplate(unittest.TestCase):
    """Tests for template expansion into concrete steps."""

    def _make_template(self, **overrides):
        base = {
            "template_id": "feat_impl",
            "name": "Feature Implementation",
            "parameters": {"feature_name": "Name"},
            "context_artifacts": ["prd_final", "design_handoff"],
            "context_strategy": {
                "primary": ["prd_final"],
                "reference": ["design_handoff"],
            },
            "steps": [
                {
                    "template_step_id": "feat.1",
                    "name": "spec_review",
                    "agent": "planner",
                    "instruction": "Review spec for '{feature_name}'.",
                    "output_artifacts": ["feature_spec"],
                },
                {
                    "template_step_id": "feat.2",
                    "name": "implement",
                    "agent": "implementer",
                    "instruction": "Implement '{feature_name}' feature.",
                    "output_artifacts": ["impl_files"],
                },
            ],
        }
        base.update(overrides)
        return base

    def test_expand_template_produces_concrete_steps(self):
        """ID format is prefix.template_step_id, params are substituted."""
        tpl = self._make_template()
        params = {"feature_name": "login"}
        steps = expand_template(tpl, params, prefix="f1")
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0]["id"], "f1.feat.1")
        self.assertEqual(steps[1]["id"], "f1.feat.2")
        self.assertIn("login", steps[0]["instruction"])
        self.assertIn("login", steps[1]["instruction"])

    def test_expand_template_with_conditions(self):
        """Condition field is preserved in expanded steps."""
        tpl = self._make_template()
        tpl["steps"][1]["condition"] = "Only if needed"
        steps = expand_template(tpl, {"feature_name": "auth"}, prefix="f2")
        self.assertIsNone(steps[0].get("condition"))
        self.assertEqual(steps[1]["condition"], "Only if needed")

    def test_expand_template_propagates_context_strategy(self):
        """context_strategy from template is propagated to each step."""
        tpl = self._make_template()
        steps = expand_template(tpl, {"feature_name": "dashboard"}, prefix="f3")
        for step in steps:
            self.assertEqual(
                step["context_strategy"],
                {"primary": ["prd_final"], "reference": ["design_handoff"]},
            )

    def test_expand_template_input_artifacts_default(self):
        """Steps without input_artifacts get template's context_artifacts."""
        tpl = self._make_template()
        steps = expand_template(tpl, {"feature_name": "x"}, prefix="f4")
        # Neither template step defines input_artifacts, so they default
        for step in steps:
            self.assertEqual(step["input_artifacts"], ["prd_final", "design_handoff"])

    def test_expand_template_input_artifacts_override(self):
        """Steps with their own input_artifacts keep them."""
        tpl = self._make_template()
        tpl["steps"][0]["input_artifacts"] = ["custom_artifact"]
        steps = expand_template(tpl, {"feature_name": "x"}, prefix="f5")
        self.assertEqual(steps[0]["input_artifacts"], ["custom_artifact"])
        self.assertEqual(steps[1]["input_artifacts"], ["prd_final", "design_handoff"])

    def test_expand_template_empty_prefix(self):
        """With empty prefix, IDs are just the template_step_id."""
        tpl = self._make_template()
        steps = expand_template(tpl, {"feature_name": "x"}, prefix="")
        self.assertEqual(steps[0]["id"], "feat.1")


if __name__ == "__main__":
    unittest.main()
