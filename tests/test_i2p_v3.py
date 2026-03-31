"""Tests for i2p v3 workflow features: difficulty routing, artifact schemas,
tools_hint, skip_when, JSON quality gates, and workflow loading."""

import json
import pytest

from src.workflows.engine.expander import (
    DIFFICULTY_MAP,
    expand_steps_to_tasks,
    expand_template,
    filter_skipped_steps,
)
from src.workflows.engine.hooks import validate_artifact_schema
from src.workflows.engine.loader import (
    load_workflow,
    validate_dependencies,
    validate_v3_fields,
)
from src.workflows.engine.quality_gates import (
    _evaluate_check_expression,
    evaluate_json_gate,
    format_gate_result,
)


# ── Difficulty mapping ────────────────────────────────────────────────────────


class TestDifficultyMapping:
    def test_difficulty_map_values(self):
        assert DIFFICULTY_MAP == {"easy": 3, "medium": 6, "hard": 8}

    def test_easy_step_context(self):
        steps = [
            {
                "id": "t1",
                "name": "test",
                "agent": "executor",
                "phase": "phase_0",
                "difficulty": "easy",
                "instruction": "do stuff",
            }
        ]
        tasks = expand_steps_to_tasks(steps, mission_id="1")
        ctx = tasks[0]["context"]
        assert ctx["difficulty"] == 3
        assert ctx.get("needs_thinking") is None or ctx.get("needs_thinking") is False

    def test_hard_step_sets_thinking_and_quality(self):
        steps = [
            {
                "id": "t2",
                "name": "test",
                "agent": "architect",
                "phase": "phase_4",
                "difficulty": "hard",
                "instruction": "design system",
            }
        ]
        tasks = expand_steps_to_tasks(steps, mission_id="1")
        ctx = tasks[0]["context"]
        assert ctx["difficulty"] == 8
        assert ctx["needs_thinking"] is True
        assert ctx["prefer_quality"] is True

    def test_medium_step_context(self):
        steps = [
            {
                "id": "t3",
                "name": "test",
                "agent": "coder",
                "phase": "phase_8",
                "difficulty": "medium",
                "instruction": "code it",
            }
        ]
        tasks = expand_steps_to_tasks(steps, mission_id="1")
        ctx = tasks[0]["context"]
        assert ctx["difficulty"] == 6

    def test_no_difficulty_field_no_key(self):
        """Steps without difficulty should not have the key in context."""
        steps = [
            {
                "id": "t4",
                "name": "test",
                "agent": "executor",
                "phase": "phase_0",
                "instruction": "old step",
            }
        ]
        tasks = expand_steps_to_tasks(steps, mission_id="1")
        ctx = tasks[0]["context"]
        assert "difficulty" not in ctx


# ── tools_hint ────────────────────────────────────────────────────────────────


class TestToolsHint:
    def test_tools_hint_propagated(self):
        steps = [
            {
                "id": "t5",
                "name": "test",
                "agent": "researcher",
                "phase": "phase_1",
                "tools_hint": ["web_search", "extract_url"],
                "instruction": "search",
            }
        ]
        tasks = expand_steps_to_tasks(steps, mission_id="1")
        assert tasks[0]["context"]["tools_hint"] == ["web_search", "extract_url"]

    def test_no_tools_hint_no_key(self):
        steps = [
            {
                "id": "t6",
                "name": "test",
                "agent": "executor",
                "phase": "phase_0",
                "instruction": "nope",
            }
        ]
        tasks = expand_steps_to_tasks(steps, mission_id="1")
        assert "tools_hint" not in tasks[0]["context"]

    def test_template_propagates_tools_hint(self):
        template = {
            "template_id": "test_tpl",
            "steps": [
                {
                    "template_step_id": "s1",
                    "name": "step1",
                    "agent": "coder",
                    "instruction": "code {feature_name}",
                    "tools_hint": ["shell", "write_file"],
                    "output_artifacts": [],
                }
            ],
        }
        expanded = expand_template(template, {"feature_name": "auth"}, prefix="8.F1")
        assert expanded[0].get("tools_hint") == ["shell", "write_file"]


# ── artifact_schema ───────────────────────────────────────────────────────────


class TestArtifactSchema:
    def test_propagated_to_context(self):
        schema = {"test_output": {"type": "object", "required_fields": ["name"]}}
        steps = [
            {
                "id": "t7",
                "name": "test",
                "agent": "executor",
                "phase": "phase_0",
                "artifact_schema": schema,
                "instruction": "go",
            }
        ]
        tasks = expand_steps_to_tasks(steps, mission_id="1")
        assert tasks[0]["context"]["artifact_schema"] == schema


class TestArtifactSchemaValidation:
    def test_valid_object(self):
        ok, err = validate_artifact_schema(
            '{"name": "x", "url": "y"}',
            {"art": {"type": "object", "required_fields": ["name", "url"]}},
        )
        assert ok
        assert err == ""

    def test_missing_required_field(self):
        ok, err = validate_artifact_schema(
            '{"name": "x"}',
            {"art": {"type": "object", "required_fields": ["name", "url"]}},
        )
        assert not ok
        assert "url" in err

    def test_not_json_object(self):
        ok, err = validate_artifact_schema(
            "just a string",
            {"art": {"type": "object", "required_fields": ["name"]}},
        )
        assert not ok

    def test_valid_array(self):
        ok, err = validate_artifact_schema(
            '[{"name": "a"}, {"name": "b"}]',
            {"art": {"type": "array", "min_items": 1, "item_fields": ["name"]}},
        )
        assert ok

    def test_array_too_few_items(self):
        ok, err = validate_artifact_schema(
            "[]",
            {"art": {"type": "array", "min_items": 1}},
        )
        assert not ok
        assert "0 items" in err

    def test_array_missing_item_fields(self):
        ok, err = validate_artifact_schema(
            '[{"name": "a"}, {"other": "b"}]',
            {"art": {"type": "array", "min_items": 1, "item_fields": ["name"]}},
        )
        assert not ok

    def test_valid_string(self):
        ok, err = validate_artifact_schema(
            "A substantial text output with content",
            {"art": {"type": "string", "min_length": 10}},
        )
        assert ok

    def test_string_too_short(self):
        ok, err = validate_artifact_schema(
            "hi",
            {"art": {"type": "string", "min_length": 10}},
        )
        assert not ok

    def test_valid_markdown(self):
        ok, err = validate_artifact_schema(
            "# Overview\nSome text\n## Problem Statement\nDetails",
            {"art": {"type": "markdown", "required_sections": ["Overview", "Problem Statement"]}},
        )
        assert ok

    def test_markdown_missing_section(self):
        ok, err = validate_artifact_schema(
            "# Overview\nSome text",
            {"art": {"type": "markdown", "required_sections": ["Overview", "Missing Section"]}},
        )
        assert not ok
        assert "Missing Section" in err

    def test_empty_schema_always_passes(self):
        ok, err = validate_artifact_schema("anything", {})
        assert ok

    def test_none_schema_always_passes(self):
        ok, err = validate_artifact_schema("anything", None)
        assert ok


# ── skip_when ─────────────────────────────────────────────────────────────────


class TestSkipWhen:
    def test_no_skip_conditions_keeps_all(self):
        steps = [{"id": "1"}, {"id": "2", "skip_when": ["no_mobile"]}]
        active, skipped = filter_skipped_steps(steps, set())
        assert len(active) == 2
        assert len(skipped) == 0

    def test_matching_condition_skips(self):
        steps = [
            {"id": "1"},
            {"id": "2", "skip_when": ["no_mobile"]},
            {"id": "3", "skip_when": ["no_payment"]},
        ]
        active, skipped = filter_skipped_steps(steps, {"no_mobile"})
        assert len(active) == 2
        assert len(skipped) == 1
        assert skipped[0]["id"] == "2"

    def test_multiple_conditions_any_match_skips(self):
        steps = [{"id": "1", "skip_when": ["no_mobile", "mvp_only"]}]
        active, skipped = filter_skipped_steps(steps, {"mvp_only"})
        assert len(active) == 0
        assert len(skipped) == 1

    def test_no_skip_when_field_never_skipped(self):
        steps = [{"id": "1"}, {"id": "2"}]
        active, skipped = filter_skipped_steps(steps, {"no_mobile", "no_payment"})
        assert len(active) == 2
        assert len(skipped) == 0


# ── Quality gates from JSON ──────────────────────────────────────────────────


class TestCheckExpression:
    def test_gte_pass(self):
        assert _evaluate_check_expression(
            "pass_rate >= 0.95", '{"pass_rate": 0.97}'
        )

    def test_gte_fail(self):
        assert not _evaluate_check_expression(
            "pass_rate >= 0.95", '{"pass_rate": 0.80}'
        )

    def test_eq_bool_true(self):
        assert _evaluate_check_expression("clean == true", '{"clean": true}')

    def test_eq_bool_false(self):
        assert not _evaluate_check_expression("clean == true", '{"clean": false}')

    def test_eq_string(self):
        assert _evaluate_check_expression(
            'status == "approved"', '{"status": "approved"}'
        )

    def test_empty_expression_passes(self):
        assert _evaluate_check_expression("", "anything")

    def test_unparseable_artifact_returns_false(self):
        assert not _evaluate_check_expression(
            "field >= 0.5", "not json at all"
        )


# ── Workflow loading ─────────────────────────────────────────────────────────


class TestV3WorkflowLoading:
    def test_v3_loads(self):
        wf = load_workflow("i2p_v3")
        assert wf.plan_id == "i2p_v3"
        assert wf.version == "3.0"

    def test_v3_step_count(self):
        wf = load_workflow("i2p_v3")
        assert 150 <= len(wf.steps) <= 200

    def test_v3_template_count(self):
        wf = load_workflow("i2p_v3")
        assert len(wf.templates) >= 1
        tpl = wf.get_template("feature_implementation_template")
        assert tpl is not None
        assert 10 <= len(tpl["steps"]) <= 20

    def test_v3_dag_valid(self):
        wf = load_workflow("i2p_v3")
        errors = validate_dependencies(wf)
        assert errors == [], f"DAG errors: {errors}"

    def test_v3_fields_valid(self):
        wf = load_workflow("i2p_v3")
        errors = validate_v3_fields(wf.steps)
        assert errors == [], f"V3 field errors: {errors}"

    def test_v3_has_quality_gates(self):
        wf = load_workflow("i2p_v3")
        gates = wf.metadata.get("quality_gates", {})
        assert "phase_9" in gates
        assert "phase_10" in gates
        assert "phase_13" in gates

    def test_v3_has_conditional_groups(self):
        wf = load_workflow("i2p_v3")
        assert len(wf.conditional_groups) >= 5

    def test_v3_has_spike_step(self):
        wf = load_workflow("i2p_v3")
        spike = wf.get_step("8.spike")
        assert spike is not None
        assert spike["difficulty"] == "hard"
        assert spike["name"] == "feasibility_spike"

    def test_v3_all_steps_have_difficulty(self):
        wf = load_workflow("i2p_v3")
        for step in wf.steps:
            assert "difficulty" in step, f"Step {step['id']} missing difficulty"
            assert step["difficulty"] in (
                "easy",
                "medium",
                "hard",
            ), f"Step {step['id']} has invalid difficulty: {step['difficulty']}"

    def test_v3_all_steps_have_tools_hint(self):
        wf = load_workflow("i2p_v3")
        for step in wf.steps:
            assert "tools_hint" in step, f"Step {step['id']} missing tools_hint"
            assert isinstance(
                step["tools_hint"], list
            ), f"Step {step['id']} tools_hint not a list"

    def test_v3_all_steps_have_artifact_schema(self):
        wf = load_workflow("i2p_v3")
        for step in wf.steps:
            assert (
                "artifact_schema" in step
            ), f"Step {step['id']} missing artifact_schema"

    def test_v2_backward_compatible(self):
        """v2 must still load without errors."""
        wf = load_workflow("i2p_v2")
        assert wf.plan_id == "i2p_v2"
        assert len(wf.steps) == 328

    def test_v3_difficulty_distribution(self):
        """Verify reasonable difficulty distribution."""
        wf = load_workflow("i2p_v3")
        from collections import Counter

        dist = Counter(s["difficulty"] for s in wf.steps)
        total = sum(dist.values())
        # Hard should be < 30%
        assert dist["hard"] / total < 0.30, f"Too many hard steps: {dist}"
        # Easy should be > 10%
        assert dist["easy"] / total > 0.10, f"Too few easy steps: {dist}"

    def test_v3_skip_when_steps_exist(self):
        """v3 should have some steps with skip_when."""
        wf = load_workflow("i2p_v3")
        skip_steps = [s for s in wf.steps if s.get("skip_when")]
        assert len(skip_steps) >= 10, f"Only {len(skip_steps)} steps have skip_when"
