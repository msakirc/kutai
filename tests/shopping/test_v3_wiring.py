"""shopping_v3.json structural wiring tests."""
from src.workflows.engine.loader import load_workflow


def test_v3_loads_with_plan_id():
    wf = load_workflow("shopping_v3")
    assert wf.plan_id == "shopping_v3"


def test_v3_producer_steps_use_agent_types():
    wf = load_workflow("shopping_v3")
    agents = {s["id"]: s["agent"] for s in wf.steps}
    assert agents["1.1b"] == "shopping_grouper"
    assert agents["1.1d"] == "shopping_labeler"
    assert agents["2.0b"] == "shopping_synthesizer"
    assert agents["2.2b"] == "shopping_synthesizer"


def test_v3_group_residual_skips_when_no_residuals():
    wf = load_workflow("shopping_v3")
    s11b = next(s for s in wf.steps if s["id"] == "1.1b")
    assert s11b.get("skip_when") == "group_input.has_residuals == 'false'"


def test_v3_synth_paths_are_mutually_exclusive():
    wf = load_workflow("shopping_v3")
    by_id = {s["id"]: s for s in wf.steps}
    assert by_id["2.0a"]["skip_when"] == "gate_result.gate.kind != 'chosen'"
    assert by_id["2.2a"]["skip_when"] == "clarify_choice.kind != 'variant'"


def test_v3_prep_apply_steps_route_to_pipeline_handler():
    wf = load_workflow("shopping_v3")
    by_id = {s["id"]: s for s in wf.steps}
    for sid in ("1.1a", "1.1c", "1.1e", "2.0a", "2.0c", "2.2a", "2.2c", "3.0"):
        assert by_id[sid]["agent"] == "shopping_pipeline_v2"
        # the pipeline dispatches by step.name -> _STEP_HANDLERS_V2 key
        assert by_id[sid]["name"] in {
            "group_prep", "group_apply_label_prep", "label_apply_filter_gate",
            "synth_prep", "synth_apply", "format_response",
        }


def test_v3_declares_no_file_produces():
    # Producer/prep/apply steps emit plain JSON artifacts, NOT declared file
    # produces — so materialize_produces is a no-op for them (spec risk 6).
    wf = load_workflow("shopping_v3")
    for s in wf.steps:
        assert "produces" not in s, f"{s['id']} unexpectedly declares produces"
