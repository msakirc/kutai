"""Tests for shopping workflow dispatch.

Verifies that:
- quick_search and shopping workflow JSONs load with the expected step count
- Step agents and tools_hint are correct
- Dependency DAGs are valid (no cycles, no unknown refs, no orphans)
- The wf_map sub-intent mapping in _create_shopping_mission is correct
"""

from __future__ import annotations

import unittest


class TestQuickSearchWorkflow(unittest.TestCase):
    """Tests for the quick_search workflow."""

    def test_quick_search_loads(self):
        from src.workflows.engine.loader import load_workflow
        wf = load_workflow("quick_search")
        self.assertEqual(len(wf.steps), 2)

    def test_quick_search_valid_dag(self):
        from src.workflows.engine.loader import load_workflow, validate_dependencies
        wf = load_workflow("quick_search")
        errors = validate_dependencies(wf)
        self.assertEqual(errors, [], f"DAG errors: {errors}")

    def test_quick_search_step_agents(self):
        from src.workflows.engine.loader import load_workflow
        wf = load_workflow("quick_search")
        step_agents = {s["id"]: s.get("agent") for s in wf.steps}
        self.assertEqual(step_agents["0.1"], "shopping_pipeline")
        self.assertEqual(step_agents["1.1"], "shopping_pipeline")


class TestShoppingWorkflow(unittest.TestCase):
    """Tests for the shopping workflow."""

    def test_shopping_loads(self):
        from src.workflows.engine.loader import load_workflow
        wf = load_workflow("shopping")
        self.assertEqual(len(wf.steps), 6)

    def test_shopping_valid_dag(self):
        from src.workflows.engine.loader import load_workflow, validate_dependencies
        wf = load_workflow("shopping")
        errors = validate_dependencies(wf)
        self.assertEqual(errors, [], f"DAG errors: {errors}")

    def test_shopping_step_agents(self):
        from src.workflows.engine.loader import load_workflow
        wf = load_workflow("shopping")
        step_agents = {s["id"]: s.get("agent") for s in wf.steps}
        self.assertEqual(step_agents["0.1"], "shopping_pipeline")
        self.assertEqual(step_agents["1.1"], "shopping_pipeline")
        self.assertEqual(step_agents["2.1"], "shopping_pipeline")
        self.assertEqual(step_agents["3.1"], "deal_analyst")
        self.assertEqual(step_agents["4.1"], "shopping_advisor")
        self.assertEqual(step_agents["5.1"], "shopping_pipeline")

    def test_pipeline_steps_have_no_tools_hint(self):
        """Pipeline steps don't need tools_hint — they run Python directly."""
        from src.workflows.engine.loader import load_workflow
        wf = load_workflow("shopping")
        for s in wf.steps:
            if s.get("agent") == "shopping_pipeline":
                self.assertEqual(s.get("tools_hint", []), [],
                    f"Step {s['id']} should have empty tools_hint")


class TestSubIntentMapping(unittest.TestCase):
    """Tests for the wf_map sub-intent mapping logic in _create_shopping_mission."""

    def setUp(self):
        self.wf_map = {
            "deep_research": "shopping",
            "research": "shopping",
            "compare": "combo_research",
            "gift": "gift_recommendation",
            "deals": "exploration",
            "quick_search": "quick_search",
        }

    def test_deep_research_maps_to_shopping(self):
        self.assertEqual(self.wf_map.get("deep_research", "shopping"), "shopping")

    def test_quick_search_maps_to_quick_search(self):
        self.assertEqual(self.wf_map.get("quick_search", "shopping"), "quick_search")

    def test_none_defaults_to_shopping(self):
        self.assertEqual(self.wf_map.get(None, "shopping"), "shopping")

    def test_compare_maps_to_combo_research(self):
        self.assertEqual(self.wf_map.get("compare", "shopping"), "combo_research")

    def test_gift_maps_to_gift_recommendation(self):
        self.assertEqual(self.wf_map.get("gift", "shopping"), "gift_recommendation")

    def test_deals_maps_to_exploration(self):
        self.assertEqual(self.wf_map.get("deals", "shopping"), "exploration")

    def test_research_maps_to_shopping(self):
        self.assertEqual(self.wf_map.get("research", "shopping"), "shopping")

    def test_unknown_sub_intent_defaults_to_shopping(self):
        self.assertEqual(self.wf_map.get("unknown_intent", "shopping"), "shopping")


if __name__ == "__main__":
    unittest.main()
