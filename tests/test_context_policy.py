import pytest
from src.memory.context_policy import (
    get_context_policy,
    apply_heuristics,
    compute_layer_budgets,
    DEFAULT_POLICY,
)


class TestGetContextPolicy:
    def test_known_profile(self):
        policy = get_context_policy("executor")
        assert policy == {"deps", "skills", "api"}

    def test_shopping_advisor(self):
        policy = get_context_policy("shopping_advisor")
        assert policy == {"skills", "convo"}

    def test_reviewer_empty(self):
        policy = get_context_policy("reviewer")
        assert policy == set()

    def test_unknown_returns_default(self):
        policy = get_context_policy("nonexistent_agent_type")
        assert policy == DEFAULT_POLICY


class TestApplyHeuristics:
    def test_tools_hint_adds_skills_and_api(self):
        task = {"context": {"tools_hint": ["smart_search"]}}
        result = apply_heuristics(task, set())
        assert "skills" in result
        assert "api" in result

    def test_depends_on_adds_deps(self):
        task = {"depends_on": "[1, 2]"}
        result = apply_heuristics(task, set())
        assert "deps" in result

    def test_followup_adds_convo(self):
        task = {"context": {"is_followup": True}}
        result = apply_heuristics(task, set())
        assert "convo" in result

    def test_mission_adds_board(self):
        task = {"mission_id": 5}
        result = apply_heuristics(task, set())
        assert "board" in result

    def test_does_not_mutate_input(self):
        original = {"skills"}
        task = {"mission_id": 5}
        result = apply_heuristics(task, original)
        assert "board" not in original
        assert "board" in result

    def test_no_heuristics_returns_copy(self):
        policy = {"skills", "rag"}
        task = {}
        result = apply_heuristics(task, policy)
        assert result == policy
        assert result is not policy

    def test_string_context_parsed(self):
        """Task context may be a JSON string, not dict."""
        import json
        task = {"context": json.dumps({"tools_hint": ["web_search"]})}
        result = apply_heuristics(task, set())
        assert "skills" in result


class TestComputeLayerBudgets:
    def test_8k_executor(self):
        budgets = compute_layer_budgets(8192, {"deps", "skills", "api"})
        total = sum(budgets.values())
        assert total <= int(8192 * 0.40) + 1
        assert budgets["deps"] > budgets["skills"]
        assert budgets["skills"] > budgets["api"]

    def test_empty_layers(self):
        budgets = compute_layer_budgets(8192, set())
        assert budgets == {}

    def test_single_layer_gets_full_budget(self):
        budgets = compute_layer_budgets(8192, {"rag"})
        assert budgets["rag"] == int(8192 * 0.40)

    def test_32k_model_more_budget(self):
        budgets_8k = compute_layer_budgets(8192, {"deps", "skills"})
        budgets_32k = compute_layer_budgets(32768, {"deps", "skills"})
        assert budgets_32k["deps"] > budgets_8k["deps"]
